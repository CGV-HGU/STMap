import numpy as np
from llm_utils.gpt_request import gptv_response
from llm_utils.nav_prompt import GPT4V_PROMPT
from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *
import cv2
import ast
import time  # <-- 추가

class GPT4V_Planner:
    def __init__(self,dino_model,sam_model):
        self.gptv_trajectory = []
        self.dino_model = dino_model
        self.sam_model = sam_model
        self.detect_objects = ['bed','sofa','chair','plant','tv','toilet','floor']
        self.last_vis_rgb = None
        # ---- LLM/플래너/모듈별 시간 계측 저장소 ----
        self.llm_call_count = 0
        self.llm_durations = []          # 각 LLM 호출 소요시간(초)
    
    def reset(self,object_goal):
        # translation to align for the detection model
        if object_goal == 'tv_monitor':
            self.object_goal = 'tv'
        else:
            self.object_goal = object_goal

        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []
        self.action_history = []

        # ---- 에피소드 시작 시 계측 초기화 ----
        self.llm_call_count = 0
        self.llm_durations = []


    def concat_panoramic(self,images,angles):
        try:
            height,width = images[0].shape[0],images[0].shape[1]
        except:
            height,width = 480,640
        background_image = np.zeros((2*height + 3*10, 3*width + 4*10,3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
               continue
            copy_images[i] = cv2.putText(copy_images[i],"Angle %d"%angles[i],(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i//2) % 3
            background_image[10*(row+1)+row*height:10*(row+1)+row*height+height:,10*(col+1)+col*width:10*(col+1)+col*width+width,:] = copy_images[i]
        return background_image

    def _draw_detections(self, image, det, class_names, color=(0, 255, 0)):
        if det is None or det.xyxy is None:
            return image
        if det.xyxy.shape[0] == 0:
            return image
        out = image.copy()
        for box, cls_id, conf in zip(det.xyxy, det.class_id, det.confidence):
            if cls_id is None:
                continue
            idx = int(cls_id)
            label = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {float(conf):.2f}"
            cv2.putText(out, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out
    
    def make_plan(
        self,
        pano_images,
        context=None,
    ):
        direction, llm_goal_flag, scene_desc = self.query_gpt4v(pano_images, context=context)
        direction_image = pano_images[direction]
        debug_image = np.array(direction_image)
        target_bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),self.detect_objects,self.dino_model)
        try:
            target_idx = self.detect_objects.index(self.object_goal)
        except ValueError:
            target_idx = -1
        vis_bgr = draw_detections_bgr(direction_image, target_bbox, goal_idx=target_idx)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        target_visible = False
        try:
            target_visible = target_idx in target_bbox.class_id
        except ValueError:
            target_visible = False
        goal_flag = bool(llm_goal_flag)
        if goal_flag and not target_visible:
            goal_flag = False
        if goal_flag:
            bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),[self.object_goal],self.dino_model)
            debug_image = self._draw_detections(debug_image, bbox, [self.object_goal], color=(255, 0, 0))
        else:
            bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),['floor'],self.dino_model)
            debug_image = self._draw_detections(debug_image, bbox, ['floor'], color=(200, 200, 200))
        try:
            mask = sam_masking(direction_image,bbox.xyxy,self.sam_model)
        except:
            mask = np.ones_like(direction_image).mean(axis=-1)
        
        self.direction_image_trajectory.append(direction_image)
        self.direction_mask_trajectory.append(mask)

        debug_mask = np.zeros_like(debug_image)
        ys, xs = np.where(mask > 0)[0:2]
        if len(xs) == 0:
            h, w = debug_image.shape[:2]
            pixel_x, pixel_y = w // 2, h // 2
        else:
            pixel_y = int(ys.mean())
            pixel_x = int(xs.mean())
        debug_image = cv2.rectangle(debug_image,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,0,0),-1)
        debug_mask = cv2.rectangle(debug_mask,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,255,255),-1)
        debug_mask = debug_mask.mean(axis=-1)
        if vis_rgb is not None:
            cv2.drawMarker(
                vis_rgb,
                (pixel_x, pixel_y),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
            )
        self.last_vis_rgb = vis_rgb
        return direction_image,debug_mask,debug_image,vis_rgb,direction,goal_flag,scene_desc


    def query_gpt4v(self, pano_images, context=None):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = cv2.cvtColor(self.concat_panoramic(pano_images, angles), cv2.COLOR_BGR2RGB)

        cv2.imwrite("monitor-panoramic.jpg", inference_image)
        text_content = "<Target Object>:{}\n".format(self.object_goal)
        # NOTE: PixelNav original input uses only the target object.
        if context:
            text_content += "<Memory Context>:\n{}\n".format(context)
        if self.action_history:
            recent = ", ".join([f"Angle {a} ({s})" for a, s in self.action_history[-5:]])
            text_content += "<Recent Actions>: {}\n".format(recent)
        self.gptv_trajectory.append("\nInput:\n%s \n" % text_content)
        self.panoramic_trajectory.append(inference_image)

        raw_answer = None
        answer = None

        for _ in range(10):
            t0 = time.perf_counter()
            try:
                # ⬇️ 순수 API 호출만 타이밍
                raw_answer = gptv_response(text_content, inference_image, GPT4V_PROMPT)
            except Exception:
                raw_answer = None
            finally:
                # ✅ 성공/실패/빈응답 모두 “호출 1회”로 집계 + 시간 기록
                self.llm_call_count += 1
                self.llm_durations.append(time.perf_counter() - t0)

            # 이후는 파싱/검증 로직
            if not raw_answer:
                continue

            print("GPT-4V Output Response: %s" % raw_answer)

            if "{" not in raw_answer or "}" not in raw_answer:
                raw_answer = None
                continue

            try:
                blob = raw_answer[raw_answer.index("{"): raw_answer.index("}") + 1]
                answer = ast.literal_eval(blob)
            except Exception:
                raw_answer = None
                answer = None
                continue

            if 'Reason' in answer and 'Angle' in answer:
                try:
                    a = int(answer['Angle'])
                except Exception:
                    a = None
                if a is not None and a in set(int(x) for x in angles):
                    break  # 성공 조건

        self.gptv_trajectory.append("GPT-4V Answer:\n%s" % (raw_answer if raw_answer is not None else "<EMPTY>"))
        self.panoramic_trajectory.append(inference_image)

        try:
            # Scene comes from VLM output (e.g., Gemini 2.5 Flash Lite; "living_room", "corridor").
            scene_desc = "unknown"
            if isinstance(answer, dict):
                scene_desc = answer.get('Scene', scene_desc)

            chosen_angle = int(answer['Angle'])
            goal_flag = bool(answer.get('Flag', False))

            self.action_history.append((chosen_angle, scene_desc))
            idx = (int(chosen_angle // 30)) % max(1, len(pano_images))
            return idx, goal_flag, scene_desc
        except Exception:
            return np.random.randint(0, max(1, len(pano_images))), False, "unknown"

            
