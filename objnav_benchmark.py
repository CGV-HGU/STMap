import habitat
import os
import argparse
import csv
import cv2
import imageio
import numpy as np
import time
from cv_utils.detection_tools import *
from tqdm import tqdm
from constants import *
from config_utils import hm3d_config
from gpt4v_planner import GPT4V_Planner
from policy_agent import Policy_Agent
from cv_utils.detection_tools import initialize_dino_model
from cv_utils.segmentation_tools import initialize_sam_model
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from omegaconf import OmegaConf, open_dict  # for NumSteps enable
from place_graph.manager import PlaceManager
from place_graph.visualizer import GraphVisualizer
from llm_utils.vertex_client_wrapper import VertexAIClient

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

MEMORY_DETECT_OBJECTS = [
    "door",
    "doorway",
    "entrance",
    "gate",
    "bed",
    "sofa",
    "chair",
    "table",
    "desk",
    "plant",
    "tv",
    "toilet",
]

def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'], 1024), cv2.COLOR_BGR2RGB)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    return parser.parse_known_args()[0]

def detect_mask(image, category, detect_model):
    det_result = openset_detection(image, category, detect_model)
    if det_result.xyxy.shape[0] > 0:
        goal_image = image
        goal_mask_xyxy = det_result.xyxy[np.argmax(det_result.confidence)]
        goal_mask_x = int((goal_mask_xyxy[0] + goal_mask_xyxy[2]) / 2)
        goal_mask_y = int((goal_mask_xyxy[1] + goal_mask_xyxy[3]) / 2)
        goal_mask = np.zeros((goal_image.shape[0], goal_image.shape[1]), np.uint8)
        goal_mask = cv2.rectangle(
            goal_mask,
            (goal_mask_x - 8, goal_mask_y - 8),
            (goal_mask_x + 8, goal_mask_y + 8),
            (255, 255, 255),
            -1,
        )
        return True, goal_image, goal_mask
    return False, [], []

def detect_memory_objects(image, detect_model, return_det=False):
    try:
        det = openset_detection(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), MEMORY_DETECT_OBJECTS, detect_model)
    except Exception:
        return ([], None) if return_det else []
    detected = []
    for cls_id, conf in zip(det.class_id, det.confidence):
        if cls_id is None:
            continue
        idx = int(cls_id)
        if 0 <= idx < len(MEMORY_DETECT_OBJECTS):
            detected.append({"label": MEMORY_DETECT_OBJECTS[idx], "conf": float(conf)})
    if return_det:
        return detected, det
    return detected

def get_agent_pose(sim):
    state = sim.get_agent_state()
    pos = state.position
    q = state.rotation
    yaw = -np.arctan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y ** 2 + q.z ** 2))
    return (float(pos[0]), float(pos[2]), float(yaw))

def update_place_memory(place_manager, sim, semantic_extractor, obs_image, detect_model):
    vlm_data = None
    if semantic_extractor is not None:
        try:
            _, buf = cv2.imencode(".jpg", obs_image[:, :, ::-1])
            vlm_data = semantic_extractor.analyze_scene(buf.tobytes())
        except Exception:
            vlm_data = None
    detected_objects, det = detect_memory_objects(obs_image, detect_model, return_det=True)
    pose = get_agent_pose(sim)
    place_manager.update(pose, vlm_data, detected_objects)
    return det

def draw_detections_on_frame(frame, det, class_names, color=(0, 255, 0)):
    if det is None or det.xyxy is None or det.xyxy.shape[0] == 0:
        return frame
    out = frame.copy()
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

args = get_args()
habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
print("scene_dataset =", habitat_config.habitat.simulator.scene_dataset)
print("scenes_dir    =", habitat_config.habitat.dataset.scenes_dir)
print("data_path     =", habitat_config.habitat.dataset.data_path)

# Enable NumSteps + set max steps
OmegaConf.set_readonly(habitat_config, False)
habitat_config.habitat.environment.max_episode_steps = args.max_episode_steps
try:
    from habitat.config.default_structured_configs import NumStepsMeasurementConfig
    with open_dict(habitat_config.habitat.task.measurements):
        if "num_steps" not in habitat_config.habitat.task.measurements:
            habitat_config.habitat.task.measurements.num_steps = NumStepsMeasurementConfig()
except Exception:
    with open_dict(habitat_config.habitat.task.measurements):
        if "num_steps" not in habitat_config.habitat.task.measurements:
            habitat_config.habitat.task.measurements.num_steps = {"type": "NumStepsMeasure"}

habitat_env = habitat.Env(habitat_config)
detection_model = initialize_dino_model()
segmentation_model = initialize_sam_model()

nav_planner = GPT4V_Planner(detection_model, segmentation_model)
nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT)
evaluation_metrics = []

# Semantic extractor (Vertex AI tuned model). Constants are in code, not env vars.
VERTEX_PROJECT_ID = "dogwood-method-480911-p3"
VERTEX_LOCATION = "us-central1"
VERTEX_ENDPOINT_ID = "5960655394069020672"
semantic_extractor = VertexAIClient(
    project=VERTEX_PROJECT_ID,
    location=VERTEX_LOCATION,
    endpoint_id=VERTEX_ENDPOINT_ID,
)

for i in tqdm(range(args.eval_episodes)):
    obs = habitat_env.reset()
    place_manager = PlaceManager()
    stm_visualizer = GraphVisualizer()

    # 시작 지오데식 거리(최단경로 기준)
    start_geodesic_m = float(habitat_env.get_metrics()['distance_to_goal'])

    # 유클리드 누적 이동거리 계측을 위해 env.step 래핑
    _stats = {
        "dist_m": 0.0,
        "prev": np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32),
    }
    _orig_step = habitat_env.step
    def _instrumented_step(action):
        obs_ = _orig_step(action)
        cur = np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32)
        _stats["dist_m"] += float(np.linalg.norm(cur - _stats["prev"]))
        _stats["prev"] = cur
        return obs_
    habitat_env.step = _instrumented_step

    dir = "./tmp/trajectory_%d" % i
    os.makedirs(dir, exist_ok=False)
    fps_writer = imageio.get_writer("%s/fps.mp4" % dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % dir, fps=4)
    stm_writer = imageio.get_writer("%s/stm.mp4" % dir, fps=4)
    heading_offset = 0

    nav_planner.reset(habitat_env.current_episode.object_category)
    episode_images_raw = []
    episode_video = []
    episode_topdowns = []
    episode_stm = []
    def record_step(obs_):
        episode_images_raw.append(obs_['rgb'])
        episode_video.append(obs_['rgb'].copy())
        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        episode_stm.append(stm_visualizer.draw_map(place_manager))
    def append_debug_frame(frame, repeat=2):
        if frame is None:
            return
        for _ in range(repeat):
            episode_video.append(frame.copy())
    record_step(obs)

    # 계산 시간(비디오 I/O 제외) 측정 시작
    episode_t0 = time.perf_counter()

    # 한 바퀴 관측 후 초기 플랜
    for _ in range(11):
        obs = habitat_env.step(3)
        record_step(obs)
    mem_det = update_place_memory(place_manager, habitat_env.sim, semantic_extractor, episode_images_raw[-1], detection_model)
    if mem_det is not None:
        episode_video[-1] = draw_detections_on_frame(episode_video[-1], mem_det, MEMORY_DETECT_OBJECTS, color=(0, 200, 255))
    nav_context = place_manager.get_nav_context()
    goal_image, goal_mask, debug_image, goal_rotate, goal_flag, _scene_desc = nav_planner.make_plan(
        episode_images_raw[-12:],
        context=nav_context,
    )
    append_debug_frame(debug_image)
    for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
        if goal_rotate <= 6:
            obs = habitat_env.step(3)
        else:
            obs = habitat_env.step(2)
        record_step(obs)
    nav_executor.reset(goal_image, goal_mask)

    while not habitat_env.episode_over:
        action, skill_image = nav_executor.step(obs['rgb'], habitat_env.sim.previous_step_collided)
        if action != 0 or goal_flag:
            if action == 4:
                heading_offset += 1
            elif action == 5:
                heading_offset -= 1
            obs = habitat_env.step(action)
            record_step(obs)
        else:
            if habitat_env.episode_over:
                break
            for _ in range(0, abs(heading_offset)):
                if habitat_env.episode_over:
                    break
                if heading_offset > 0:
                    obs = habitat_env.step(5)
                    heading_offset -= 1
                elif heading_offset < 0:
                    obs = habitat_env.step(4)
                    heading_offset += 1
                record_step(obs)

            # 다시 한 바퀴 관측 후 재플랜
            for _ in range(11):
                if habitat_env.episode_over:
                    break
                obs = habitat_env.step(3)
                record_step(obs)
            mem_det = update_place_memory(place_manager, habitat_env.sim, semantic_extractor, episode_images_raw[-1], detection_model)
            if mem_det is not None:
                episode_video[-1] = draw_detections_on_frame(episode_video[-1], mem_det, MEMORY_DETECT_OBJECTS, color=(0, 200, 255))
            nav_context = place_manager.get_nav_context()
            goal_image, goal_mask, debug_image, goal_rotate, goal_flag, _scene_desc = nav_planner.make_plan(
                episode_images_raw[-12:],
                context=nav_context,
            )
            append_debug_frame(debug_image)
            for j in range(min(11 - goal_rotate, goal_rotate + 1)):
                if habitat_env.episode_over:
                    break
                if goal_rotate <= 6:
                    obs = habitat_env.step(3)
                else:
                    obs = habitat_env.step(2)
                record_step(obs)
            nav_executor.reset(goal_image, goal_mask)

    # 계산 시간 측정 종료
    episode_t1 = time.perf_counter()
    episode_time_sec = episode_t1 - episode_t0

    # step 래핑 원복
    habitat_env.step = _orig_step

    for image in episode_video:
        fps_writer.append_data(image)
    for topdown in episode_topdowns:
        topdown_writer.append_data(topdown)
    for stm_frame in episode_stm:
        stm_writer.append_data(stm_frame)
    fps_writer.close()
    topdown_writer.close()
    stm_writer.close()

    # 최종 메트릭 스냅샷
    metrics_now = habitat_env.get_metrics()
    final_geodesic_m = float(metrics_now['distance_to_goal'])
    num_steps_val = int(metrics_now.get('num_steps', 0))

    # ==== 공통 지표만 기록 ====
    evaluation_metrics.append({
        'episode': i,
        'object_goal': habitat_env.current_episode.object_category,
        'success': metrics_now['success'],
        'spl': metrics_now['spl'],
        'start_distance_to_goal': start_geodesic_m,
        'final_distance_to_goal': final_geodesic_m,
        'llm_calls': int(nav_planner.llm_call_count),
        'llm_avg_time_sec': float(np.mean(nav_planner.llm_durations)) if len(nav_planner.llm_durations) > 0 else 0.0,
        'episode_time_sec': float(episode_time_sec),
        'num_steps': num_steps_val,
        'total_distance_m': float(_stats['dist_m']),
    })

    write_metrics(evaluation_metrics)
