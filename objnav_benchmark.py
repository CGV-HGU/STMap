
import habitat
import os
import argparse
import csv
import cv2
import imageio
import numpy as np
import time
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict

from constants import *
from config_utils import hm3d_config
from gpt4v_planner import GPT4V_Planner
from policy_agent import Policy_Agent
from cv_utils.detection_tools import initialize_dino_model, openset_detection
from cv_utils.segmentation_tools import initialize_sam_model
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

# New Imports for Metric-Free System
from place_graph.circular_memory import CircularMemory, Slot, K_SLOTS
from llm_utils.tokenizer import extract_tokens_and_description
from cv_utils.vocab import ALLOWED_VOCAB

# Constants
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_LOOK_UP = 4
ACTION_LOOK_DOWN = 5

def action_to_nav(action_id):
    if action_id == ACTION_FORWARD: return "forward"
    if action_id == ACTION_TURN_LEFT: return "turn_left"
    if action_id == ACTION_TURN_RIGHT: return "turn_right"
    if action_id == ACTION_STOP: return "stop"
    return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    return parser.parse_known_args()[0]

def write_metrics(metrics, path="objnav_hm3d.csv"):
    if not metrics: return
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def adjust_topdown(metrics):
    if 'top_down_map' in metrics:
        return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'], 1024), cv2.COLOR_BGR2RGB)
    return None

def main():
    args = get_args()
    habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
    
    # Enable NumSteps
    OmegaConf.set_readonly(habitat_config, False)
    habitat_config.habitat.environment.max_episode_steps = args.max_episode_steps
    # Note: Removed explicit NumStepsMeasure injection as it caused OmegaConf validation error.
    # Assuming habitat env provides 'num_steps' or we count manually.

    # Clear tmp directory
    import shutil
    if os.path.exists("./tmp"):
        shutil.rmtree("./tmp")
    os.makedirs("./tmp", exist_ok=True)

    habitat_env = habitat.Env(habitat_config)
    
    # Models
    detection_model = initialize_dino_model()
    segmentation_model = initialize_sam_model()
    planner = GPT4V_Planner(detection_model, segmentation_model)
    executor = Policy_Agent(model_path=POLICY_CHECKPOINT)

    evaluation_metrics = []

    for i in tqdm(range(args.eval_episodes)):
        obs = habitat_env.reset()
        
        # --- Episode Initialization ---
        planner.reset(habitat_env.current_episode.object_category)
        memory = CircularMemory() # Reset memory
        
        # Episode logging setup
        episode_dir = f"./tmp/trajectory_{i}"
        os.makedirs(episode_dir, exist_ok=True)
        
        fps_writer = imageio.get_writer(f"{episode_dir}/fps.mp4", fps=4)
        topdown_writer = imageio.get_writer(f"{episode_dir}/metric.mp4", fps=4)
        
        episode_video = []  # To store frames for manual append if needed
        episode_topdowns = []

        # Distance Tracking
        # 유클리드 누적 이동거리 계측을 위해 env.step 래핑
        _stats = {
            "dist_m": 0.0,
            "prev": np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32),
        }
        _orig_step = habitat_env.step
        
        def _instrumented_step(action):
            obs_ = _orig_step(action)
            cur = np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32)
            dist = float(np.linalg.norm(cur - _stats["prev"]))
            _stats["dist_m"] += dist
            _stats["prev"] = cur
            return obs_
        
        habitat_env.step = _instrumented_step
        
        step_count = 0
        heading_offset = 0  # Baseline: Track Look Up/Down actions
        event = "EPISODE_START" # Trigger
        active_attempt = None 
        
        # Timing
        episode_t0 = time.perf_counter()
        
        # Helper to record frame
        def record_frame(obs_rgb, metrics_now=None):
            if metrics_now is None:
                metrics_now = habitat_env.get_metrics()
            
            # FPS Video
            fps_writer.append_data(obs_rgb)
            
            # Topdown Video
            td_map = adjust_topdown(metrics_now)
            if td_map is not None:
                topdown_writer.append_data(td_map)

        # Record Initial Frame
        record_frame(obs['rgb'])

        # Main Loop
        while not habitat_env.episode_over:
            
            if event in ["EPISODE_START", "REPLAN"]:
                print(f"[Step {step_count}] Event: {event}")
                
                # Baseline: Restore heading before panorama
                for _ in range(0, abs(heading_offset)):
                    if habitat_env.episode_over:
                        break
                    if heading_offset > 0:
                        obs = habitat_env.step(ACTION_LOOK_DOWN)  # Undo Look Up
                        step_count += 1
                        record_frame(obs['rgb'])
                        heading_offset -= 1
                    elif heading_offset < 0:
                        obs = habitat_env.step(ACTION_LOOK_UP)  # Undo Look Down
                        step_count += 1
                        record_frame(obs['rgb'])
                        heading_offset += 1
                
                if habitat_env.episode_over: break
                
                # A. Scan 360 & Tokenize
                pano_images = []
                current_slots = []
                
                # Collect 12 images
                images_to_collect = []
                images_to_collect.append(obs['rgb']) # Image 0
                
                # Turn 11 times
                for _ in range(11):
                    obs = habitat_env.step(ACTION_TURN_RIGHT)
                    step_count += 1
                    images_to_collect.append(obs['rgb'])
                    record_frame(obs['rgb'])
                    if habitat_env.episode_over: break
                
                if habitat_env.episode_over: break
                
                pano_images = images_to_collect
                
                # Tokenize
                print("  -> Analyzine Scene (VLM)...")
                for slot_idx, img in enumerate(pano_images):
                    data = extract_tokens_and_description(img)
                    slot_obj = Slot(index=slot_idx, tokens=set(data["tokens"]), description=data["description"])
                    current_slots.append(slot_obj)
                
                # B. Memory Update
                pid, shift, score, revisit = memory.assign_place(current_slots)
                
                # Door Update Logic
                if active_attempt:
                     prev_pid = active_attempt["place_id"]
                     door_id = active_attempt["door_id"]
                     if door_id is not None:
                         if pid != prev_pid:
                             memory.update_door_status(prev_pid, door_id, "scene_changed", pid)
                             print(f"  -> Door {door_id} SUCCESS: {prev_pid} -> {pid}")
                         else:
                             memory.update_door_status(prev_pid, door_id, "stop_no_change")
                             print(f"  -> Door {door_id} FAILED: Stayed in {pid}")
                     active_attempt = None
                
                memory.manage_doors(pid, current_slots)
                print(f"  -> Place Assigned: {pid} (Revisit={revisit}, Score={score:.2f})")
                
                # C. Build Context & Plan
                context_text = memory.get_nav_context(habitat_env.current_episode.object_category)
                
                direction_image, goal_mask, debug_img, vis_rgb, angle_slot, goal_flag, desc, raw_json = \
                    planner.make_plan(pano_images, context_text)
                
                # Log Decision
                memory.dc_queue.append({
                    "place_id": pid,
                    "angle_slot": angle_slot,
                    "goal_flag": goal_flag,
                    "why": desc,
                    "result": "pending"
                })
                
                # Append Debug Frame to Video (Multiple times to make it visible)
                for _ in range(4): # 1 second approx
                    fps_writer.append_data(vis_rgb)
                    # For topdown, just repeat last map
                    td_map = adjust_topdown(habitat_env.get_metrics())
                    if td_map is not None:
                        topdown_writer.append_data(td_map)

                # D. Valid Door Binding
                target_slot = current_slots[angle_slot]
                chosen_door_id = None
                if target_slot.door_ids:
                    chosen_door_id = target_slot.door_ids[0]
                    
                active_attempt = {
                    "place_id": pid,
                    "door_id": chosen_door_id,
                    "start_scene_type": desc, 
                }
                
                # E. Turn to Target
                turns = (angle_slot - 11) % 12
                print(f"  -> Planning Turn: {turns} steps (Target Slot {angle_slot})")
                for _ in range(turns):
                    obs = habitat_env.step(ACTION_TURN_RIGHT)
                    step_count += 1
                    record_frame(obs['rgb'])
                    if habitat_env.episode_over: break
                
                if habitat_env.episode_over: break
                
                # Ensure mask is uint8
                if goal_mask.dtype == bool:
                    goal_mask = (goal_mask * 255).astype(np.uint8)
                
                executor.reset(direction_image, goal_mask)
                event = "EXECUTE"
            
            elif event == "EXECUTE":
                action, _, distance_pred = executor.step(obs['rgb'], habitat_env.sim.previous_step_collided)
                
                # --- Smart STOP Logic ---
                # PixelNav says we are close (distance_pred < 0.15 approx 1.5m) AND VLM said this is goal
                if goal_flag and distance_pred < 0.15:
                    # Verify with DINO in current view
                    # Use planner.dino_model (it holds the initialized DINO model)
                    
                    target_name = habitat_env.current_episode.object_category
                    if target_name == 'tv_monitor': target_name = 'tv' # Mapping
                    
                    bbox = openset_detection(obs['rgb'], [target_name], planner.dino_model, box_threshold=0.25)
                    
                    if len(bbox.xyxy) > 0:
                        print(f"  -> Smart STOP Triggered! (Dist={distance_pred:.2f}, DINO Found {target_name})")
                        obs = habitat_env.step(ACTION_STOP)
                        step_count += 1
                        record_frame(obs['rgb'])
                        break
                
                # Baseline: Track Look Up/Down actions
                if action == ACTION_LOOK_UP:
                    heading_offset += 1
                elif action == ACTION_LOOK_DOWN:
                    heading_offset -= 1
                
                if action == ACTION_STOP:
                    if not goal_flag: 
                        event = "REPLAN"
                    else:
                        print("  -> Goal Stop Triggered!")
                        obs = habitat_env.step(ACTION_STOP)  # Send STOP to Habitat!
                        step_count += 1
                        record_frame(obs['rgb'])
                        break
                else:
                    obs = habitat_env.step(action)
                    step_count += 1
                    record_frame(obs['rgb'])
            
            else:
                event = "REPLAN"

        # --- Episode Video Cleanup ---
        habitat_env.step = _orig_step # Restore step before getting final metrics
        fps_writer.close()
        topdown_writer.close()
        
        # --- Metrics Logging ---
        metrics_now = habitat_env.get_metrics()
        episode_t1 = time.perf_counter()
        
        start_geodesic = float(metrics_now.get('distance_to_goal', 0)) # Note: this is final dist, need initial? 
        # Actually habitat metrics 'distance_to_goal' is current distance.
        # Ideally we captured start distance at reset. 
        # But 'spl' calculation handles it internally.
        # We can just log what we have.
        
        evaluation_metrics.append({
            'episode': i,
            'object_goal': habitat_env.current_episode.object_category,
            'success': metrics_now['success'],
            'spl': metrics_now['spl'],
            'final_distance_to_goal': metrics_now['distance_to_goal'],
            'llm_calls': int(planner.llm_call_count),
            'llm_avg_time_sec': float(np.mean(planner.llm_durations)) if len(planner.llm_durations) > 0 else 0.0,
            'episode_time_sec': float(episode_t1 - episode_t0),
            'num_steps': step_count,
            'total_distance_m': float(_stats['dist_m']),
            'place_count': len(memory.places),
        })
        
        print(f"Episode {i} Summary: SPL={metrics_now['spl']:.2f}, Places={len(memory.places)}")
        write_metrics(evaluation_metrics)

if __name__ == "__main__":
    main()
