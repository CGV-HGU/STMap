
import numpy as np
from llm_utils.gpt_request import gptv_response
from llm_utils.nav_prompt import GPT4V_PROMPT # We might need a new prompt
from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *
import cv2
import ast
import time
import json

# New Prompt for Metric-Free Planner
METRIC_FREE_PROMPT = """
You are a robotic navigation planner. You navigate by visual memory and semantic cues.
You do NOT use coordinates or distance values.

**Input:**
1. **Visual:** A 6-slot panoramic view (2 rows x 3 columns).
   - Top Row (left-to-right): Slot 0, Slot 1, Slot 2.
   - Bottom Row (left-to-right): Slot 3, Slot 4, Slot 5.
   - These indices (0-5) are your ONLY valid output for `angle_slot`.

2. **Context:** A semantic summary mapped 1:1 to your visual slots, plus topological categorization of exits.

**Output:**
A JSON object only.
{{
  "thought": "Directly compare target visibility, categorize exits (NEW vs SOURCE vs VISITED), and explain your choice based on navigation rules.",
  "angle_slot": <int 0..5>, 
  "goal_flag": <bool>,
  "why": "Brief explanation"
}}

**Navigation Rules (Strict Priority):**
1. **Goal Finding:** If the Target Object is clearly visible in any slot, set `goal_flag=true` and `angle_slot` to its slot.
2. **Exploration:** If the target is NOT visible, identify slots labeled `[NEW EXPLORATION]`. These are your HIGHEST priority exits. Pick one to enter a new space.
3. **Oscillation Prevention:** Do NOT go back through a slot labeled `[SOURCE/BACKTRACK]` unless YOU ARE STUCK (all other exits are VISITED or FAILED). Backtracking leads to infinite loops.
4. **Loop Breaking:** If you see a `[🚨 WARNING: OSCILLATION DETECTED]`, you MUST NOT pick the most recent direction. Force an exploration into a new or different previously explored door.
5. **Failures:** AVOID slots labeled `(FAILED xN)` unless they are the only remaining way forward.

**Context:**
{context}
"""

class GPT4V_Planner:
    def __init__(self,dino_model,sam_model):
        self.gptv_trajectory = []
        self.dino_model = dino_model
        self.sam_model = sam_model
        self.detect_objects = ['bed','sofa','chair','plant','tv','toilet','floor']
        self.last_vis_rgb = None
        self.llm_call_count = 0
        self.llm_durations = []

    def reset(self,object_goal):
        if object_goal == 'tv_monitor':
            self.object_goal = 'tv'
        else:
            self.object_goal = object_goal

        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []
        self.action_history = []
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

    def make_plan(
        self,
        pano_images, # Expecting 12 images
        context_text="", # Text from CircularMemory
    ):
        # Call LLM
        vlm_slot, goal_flag, desc, raw_json = self.query_gpt4v(pano_images, context_text)
        
        # Map VLM's 6-slot index (0..5) back to Original 12-slot index (odd only: 1,3,5...)
        # Reference: concat_panoramic uses indices 1, 3, 5, 7, 9, 11
        # VLM 0 -> Orig 1
        # VLM 1 -> Orig 3
        # ...
        try:
            direction_slot = int(vlm_slot) * 2 + 1
            direction_slot = direction_slot % 12 # Safety wrap
        except:
            direction_slot = 1
        
        # --- Visualization & Target Generation (Existing Logic adapted) ---
        direction_image = pano_images[direction_slot]
        debug_image = np.array(direction_image)
        
        # Detect goal object using DINO
        target_bbox = openset_detection(cv2.cvtColor(direction_image, cv2.COLOR_BGR2RGB), 
                                      self.detect_objects, self.dino_model)
        
        try:
            target_idx = self.detect_objects.index(self.object_goal)
        except ValueError:
            target_idx = -1
            
        vis_bgr = self._draw_detections(direction_image, target_bbox, goal_idx=target_idx) # Helper needed
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        
        # Verification of goal
        target_visible = False
        try:
            if target_bbox.class_id is not None:
                target_visible = target_idx in target_bbox.class_id
        except:
             pass

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

        return direction_image, debug_mask, debug_image, vis_rgb, vlm_slot, direction_slot, goal_flag, desc, raw_json


    def query_gpt4v(self, pano_images, context_text):
        # 1. Create Grid Image
        angles = np.arange(len(pano_images)) * (360 // len(pano_images))
        inference_image = cv2.cvtColor(self.concat_panoramic(pano_images, angles), cv2.COLOR_BGR2RGB)
        
        # 2. Prepare Prompt
        prompt = METRIC_FREE_PROMPT.format(context=context_text)
        
        # 3. Call VLM
        t0 = time.perf_counter()
        raw_answer = None
        try:
            raw_answer = gptv_response(prompt, inference_image) # Assumes gptv_response handles text+image
        except Exception as e:
            raw_answer = "{}"
            print(f"VLM Error: {e}")
            
        self.llm_call_count += 1
        self.llm_durations.append(time.perf_counter() - t0)
        
        print(f"GPT Output: {raw_answer}")
        
        # 4. Parse JSON
        angle_slot = 0
        goal_flag = False
        desc = "unknown"
        parsed = {}
        
        try:
            # Extract JSON block
            import re
            json_str = raw_answer
            match = re.search(r"\{.*\}", raw_answer, re.DOTALL)
            if match:
                json_str = match.group(0)
            
            parsed = json.loads(json_str)
            angle_slot = int(parsed.get("angle_slot", 0))
            goal_flag = bool(parsed.get("goal_flag", False))
            desc = parsed.get("why", "no reason")
            
            # Bound check
            angle_slot = max(0, min(angle_slot, 5)) # Enforce 0-5 range for 6 slots
            
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            angle_slot = np.random.randint(0, 6)
            
        return angle_slot, goal_flag, desc, parsed

    def _draw_detections(self, image, det, class_names=None, goal_idx=-1, color=(0, 255, 0)):
        # Simple detection viz helper
        out = image.copy()
        if det.xyxy is None: return out
        
        # Use detection's class names if not provided
        if class_names is None and hasattr(det, 'class_names'):
            class_names = det.class_names
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i, (box, cls, conf) in enumerate(zip(det.xyxy, det.class_id, det.confidence)):
            c = color
            # If goal_idx matches, use Red, otherwise Green (or passed color)
            # Logic in caller seems to pass Green(0,255,0) by default
            if goal_idx != -1 and int(cls) == int(goal_idx):
                c = (0, 0, 255) # Red for target
                
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            
            # Label
            label = ""
            if class_names is not None and int(cls) < len(class_names):
                label = f"{class_names[int(cls)]}"
            else:
                label = f"id:{int(cls)}"
            
            label += f" {conf:.2f}"
            
            # Draw Label
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
            cv2.putText(
                out, label,
                (x1 + 2, y1 - 4),
                font, 0.5,
                (255, 255, 255), 1,
                cv2.LINE_AA,
            )
            
        return out
