import math
import time
from collections import deque, Counter
from .data import Place, Anchor
from .reid import PlaceReIdentifier
from utils.logger import Logger

class PlaceManager:
    """
    Module 2: Place Manager.
    Manages the 2-Layer Hierarchical Graph (L1: Anchors, L2: Places).
    Refactored for Phase 2: Topological Logic.
    """
    def __init__(self):
        # Data Stores
        self.places = {}  # place_id -> Place
        self.anchors = {} # anchor_id -> Anchor
        
        # State Pointers
        self.current_anchor = None
        self.current_place = None
        
        # Robot State
        self.robot_pose = (0.0, 0.0, 0.0)
        
        # VLM Hysteresis
        self.vlm_buffer = deque(maxlen=5) 
        self.stable_vlm_context = "unknown"
        self._init_waits = 0
        
        # Logic Constants
        self.CORRIDOR_SPACING = 2.0
        self.ROOM_SPACING = 1.5
        self.CORNER_THRESHOLD = math.radians(30) # 30 degrees
        
        # Visualization Helpers
        self.edges = [] # List of explicit edges [(id1, id2), ...] for viz
        
        # Re-Identification
        self.reid = PlaceReIdentifier()
        
        self.last_door_time = 0
        
        # Dwell Time Policy (Phase 6)
        self.pending_scene_type = None
        self.pending_caption = None
        self.pending_start_time = 0
        self.DWELL_TIME_THRESHOLD = 3.0  # seconds

    def _check_l2_transition(self, context, caption=""):
        """
        Decides if we need to switch the Active Place (L2).
        Logic: 
        1. Corridor <-> Room (Classic)
        2. Room A -> Room B (Door Gated)
        3. Dwell Time Verification (New!)
        """
        if context == "unknown": 
            # Reset pending if VLM loses track
            self.pending_scene_type = None
            return
        
        current_type = self.current_place.place_type
        
        # Condition 0: Must be a different scene type
        if context == current_type:
            # Same type - reset any pending transition
            self.pending_scene_type = None
            return
        
        # Dwell Time Check: Has this new scene type been stable long enough?
        current_time = time.time()
        
        if self.pending_scene_type != context:
            # New scene type detected - start pending timer
            self.pending_scene_type = context
            self.pending_caption = caption
            self.pending_start_time = current_time
            Logger.info("Manager", f"Pending transition detected: {current_type} -> {context} (waiting...)")
            return
        
        # Same pending type - check if enough time has passed
        dwell_time = current_time - self.pending_start_time
        
        # Phase 9A: Adaptive Dwell Time based on VLM stability
        # Calculate how stable the VLM buffer is
        required_dwell_time = self.DWELL_TIME_THRESHOLD  # Default 3.0s
        
        if len(self.vlm_buffer) >= 3:
            counts = Counter(self.vlm_buffer)
            most_common_count = counts.most_common(1)[0][1]
            buffer_size = len(self.vlm_buffer)
            stability_ratio = most_common_count / buffer_size
            
            # Adjust dwell time based on stability
            if stability_ratio >= 0.9:  # Very stable (90%+ same)
                required_dwell_time = 1.5  # Fast response
            elif stability_ratio >= 0.7:  # Stable (70%+)
                required_dwell_time = 2.0  # Medium
            # else: use default 3.0s (unstable)
        
        if dwell_time < required_dwell_time:
            # Still pending, not enough time
            return
        
        # Dwell time satisfied! Now check other conditions
        Logger.info("Manager", f"Dwell time satisfied ({dwell_time:.1f}s / {required_dwell_time:.1f}s required). Evaluating transition conditions...")
        
        # Reset pending state
        self.pending_scene_type = None

        # Condition 1: Corridor Transition
        is_corridor_now = "corridor" in current_type
        is_corridor_next = "corridor" in context
        is_corridor_transition = (is_corridor_now != is_corridor_next)
        
        # Condition 2: Door Gated Transition (Room <-> Room)
        # Allow split if we saw a door recently (last 5 seconds)
        is_door_recent = (time.time() - self.last_door_time) < 5.0
        
        # Condition 3: Spatial Shift (Fallback for missed doors or adjacent rooms)
        # If we are far from the start of the current place, allow split
        # Distance threshold depends on VLM confidence
        dist_from_start = 0
        if self.current_place and self.current_place.anchors:
            start_aid = self.current_place.anchors[0]
            if start_aid in self.anchors:
                start_pose = self.anchors[start_aid].pose
                dist_from_start = math.sqrt((self.robot_pose[0]-start_pose[0])**2 + (self.robot_pose[1]-start_pose[1])**2)
        
        # Check VLM stability - if very confident, allow transition sooner
        vlm_is_stable = False
        if len(self.vlm_buffer) >= 4:
            counts = Counter(self.vlm_buffer)
            most_common_count = counts.most_common(1)[0][1]
            vlm_is_stable = most_common_count >= 4  # Very stable
        
        # Adaptive distance threshold
        # - High confidence VLM: 0.8m (adjacent rooms like Living->Dining)
        # - Normal: 1.5m (separate but close rooms)
        distance_threshold = 0.8 if vlm_is_stable else 1.5
        is_spatial_shift = dist_from_start > distance_threshold
        
        should_split = False
        
        if is_corridor_transition:
            should_split = True
            Logger.graph("Transition", f"Corridor Logic: {current_type} -> {context}")
        elif is_door_recent:
            should_split = True
            Logger.graph("Transition", f"Door Gated: {current_type} -> {context}")
        elif is_spatial_shift:
            should_split = True
            threshold_info = f"{distance_threshold}m (VLM stable)" if vlm_is_stable else f"{distance_threshold}m"
            Logger.graph("Transition", f"Spatial Shift > {threshold_info}: {current_type} -> {context}")
            
        if should_split:
            # Phase 7: Re-Identification (Loop Closure) [DISABLED by User Request]
            # Before creating new Place, check if we're returning to a known Place
            
            # Create a temporary "candidate" Place for Re-ID matching
            # candidate = Place(f"candidate_{context}", context)
            # if caption:
            #     candidate.caption = caption
            # Note: candidate has no objects yet, but that's okay - Re-ID can work with caption alone
            
            # Try to match against existing Places
            # matched_place_id = self.reid.identify(candidate, self.places)
            matched_place_id = None # Force No Match

            if matched_place_id:
                # Loop Closure! Reuse existing Place
                Logger.graph("Loop Closure", f"Switching to existing {matched_place_id}")
                matched_place = self.places[matched_place_id]
                
                # Update current Place pointer
                self.current_place = matched_place
                
                # Merge attributes
                if caption and not matched_place.caption:
                    matched_place.caption = caption
                    Logger.graph("Update", f"Updated caption for {matched_place_id}")
                
                # Create new Anchor in the matched Place
                self._create_anchor(matched_place_id, self.robot_pose)
                
            else:
                # No match - create new Place (original behavior)
                new_pid = self._create_place(context)
                if caption: 
                    self.places[new_pid].caption = caption
                self._create_anchor(new_pid, self.robot_pose)

    def get_current_pose(self):
        return self.robot_pose

    def _get_stable_context(self, raw_context):
        if not raw_context: return self.stable_vlm_context
        # Normalize
        norm_ctx = raw_context.lower().strip()
        # Remove trailing period
        if norm_ctx.endswith('.'): norm_ctx = norm_ctx[:-1]
        
        if "hallway" in norm_ctx: norm_ctx = norm_ctx.replace("hallway", "corridor")
        
        self.vlm_buffer.append(norm_ctx)
        if len(self.vlm_buffer) < 3: return norm_ctx
            
        counts = Counter(self.vlm_buffer)
        most_common, count = counts.most_common(1)[0]
        if count >= 3: return most_common
        return self.stable_vlm_context

    def update(self, current_pose, vlm_data, detected_objects):
        """
        Main Control Loop (30Hz approx)
        Args:
            current_pose: (x, y, yaw)
            vlm_data: Dict or None from VLM {"scene_type":..., "description":...}
            detected_objects: List of dicts from YOLO
        """
        # 1. Update State
        self.robot_pose = current_pose
        
        # Track Door Events (Global)
        if detected_objects:
             if any(obj['label'] in ['door', 'doorway', 'entrance', 'gate'] for obj in detected_objects):
                 self.last_door_time = time.time()
                 # print(f"[Manager] Door seen at {self.last_door_time}")
        
        # 2. Process VLM Data (If available)
        current_scene_type = "unknown"
        current_caption = ""
        
        if vlm_data:
            # Hysteresis on Scene Type
            current_scene_type = self._get_stable_context(vlm_data.get("scene_type", "unknown"))
            current_caption = vlm_data.get("description", "")
            # Note: vlm_data also has 'objects_to_look_for', we can expose this to the main loop if needed
        else:
            current_scene_type = self.stable_vlm_context
        
        # 3. Initialization
        # Fix: Don't initialize until we have a valid scene type from VLM
        if self.current_anchor is None:
            if current_scene_type == "unknown":
                self._init_waits += 1
                if self._init_waits < 3:
                    return
                # Fallback: allow graph init even if VLM stays unknown.
                current_scene_type = "unknown"
            self._init_waits = 0

            self._init_graph(current_scene_type)
            self.stable_vlm_context = current_scene_type
            return

        # 4. Detect Place Transition (L2 Logic)
        self._check_l2_transition(current_scene_type, current_caption)
        
        # 5. Check Anchor Creation (L1 Logic: Topology)
        self._check_l1_topology(detected_objects)
        
        # 6. Semantic Update (L2 Attributes)
        if self.current_place:
             # Update Caption if available and empty
             if current_caption and not self.current_place.caption:
                 self.current_place.caption = current_caption
                 Logger.graph("Update", f"Place {self.current_place.place_id} caption set: '{current_caption}'")
                 
             if detected_objects:
                for obj in detected_objects:
                    self.current_place.add_object_context(obj['label'])
                
        self.stable_vlm_context = current_scene_type
        
        # Performance: Periodic graph pruning (every 100 updates)
        if not hasattr(self, '_update_counter'):
            self._update_counter = 0
        self._update_counter += 1
        if self._update_counter % 100 == 0:
            self.prune_old_anchors(max_anchors_per_place=20)

    def _init_graph(self, context):
        pid = self._create_place(context)
        aid = self._create_anchor(pid, self.robot_pose)
        Logger.graph("Init", f"Graph Initialized. Start: {pid}/{aid}")

    def _check_l1_topology(self, detected_objects=None):
        """
        Decides if we need to drop a new Anchor (L1).
        Criteria: Distance OR Curvature OR Door Presence (Policy)
        """
        last_pose = self.current_anchor.pose
        dist = math.sqrt((self.robot_pose[0]-last_pose[0])**2 + (self.robot_pose[1]-last_pose[1])**2)
        
        # Curvature check
        angle_diff = abs(self.robot_pose[2] - last_pose[2])
        # Normalize angle
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        angle_diff = abs(angle_diff)
        
        # Thresholds
        is_corridor = "corridor" in self.current_place.place_type
        spacing = self.CORRIDOR_SPACING if is_corridor else self.ROOM_SPACING
        
        should_create = False
        
        # 1. Distance Trigger
        if dist > spacing:
            should_create = True
            
        # 2. Curvature Trigger
        if dist > 0.5 and angle_diff > self.CORNER_THRESHOLD:
            Logger.info("Manager", f"Corner Detected ({math.degrees(angle_diff):.0f} deg). Creating Anchor.")
            should_create = True
            
        # 3. Door Trigger (Policy)
        # If we see a door nearby, ensure we have an anchor
        # Reduced cooldown from 0.8m to 0.5m for better responsiveness
        door_labels = [
            'door', 'doorway', 'entrance', 'gate', 'exit',
            'open door', 'closed door', 'door frame', 'sliding door'
        ]
        
        if detected_objects and dist > 0.2: 
            has_door = any(obj['label'] in door_labels for obj in detected_objects)
            
            # Spatial Suppression: Don't create multiple door anchors for the same door
            if has_door:
                # Check distance to other DOOR anchors in this place
                is_duplicate = False
                for aid in self.current_place.anchors:
                    if aid not in self.anchors: continue
                    a = self.anchors[aid]
                    if getattr(a, 'is_door', False):
                        # Calculate dist
                        d_to_door = math.sqrt((self.robot_pose[0]-a.pose[0])**2 + (self.robot_pose[1]-a.pose[1])**2)
                        if d_to_door < 1.0:  # 1.0m suppression radius (relaxed)
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    Logger.info("Manager", f"Door Detected. Forcing Anchor.")
                    should_create = True
            
        if should_create:
            # Check if this creation was due to a Door Trigger (policy 3)
            # Re-evaluate the trigger conditions to pass correct flag
            is_door_creation = False
            if detected_objects and dist > 0.2:
                 if any(obj['label'] in door_labels for obj in detected_objects):
                     # apply same suppression check
                     is_duplicate = False
                     for aid in self.current_place.anchors:
                        if aid not in self.anchors: continue
                        a = self.anchors[aid]
                        if getattr(a, 'is_door', False):
                            d_to_door = math.sqrt((self.robot_pose[0]-a.pose[0])**2 + (self.robot_pose[1]-a.pose[1])**2)
                            if d_to_door < 1.0:
                                is_duplicate = True
                                break
                     if not is_duplicate:
                        is_door_creation = True

            self._create_anchor(self.current_place.place_id, self.robot_pose, is_door=is_door_creation)

    def _create_place(self, place_type):
        pid = f"place_{len(self.places)}"
        new_place = Place(pid, place_type)
        self.places[pid] = new_place
        self.current_place = new_place
        return pid

    def _create_anchor(self, place_id, pose, is_door=False):
        aid = f"node_{len(self.anchors)}"
        new_anchor = Anchor(aid, pose, place_id)
        new_anchor.is_door = is_door
        
        # Link Topology (Edge)
        if self.current_anchor:
            # Add neighbor link
            self.current_anchor.add_neighbor(aid)
            new_anchor.add_neighbor(self.current_anchor.anchor_id)
            
            # Store edge for viz
            self.edges.append({
                'from': self.current_anchor.anchor_id,
                'to': aid,
                'type': 'normal' 
            })
            
        # Link Hierarchy
        self.places[place_id].anchors.append(aid)
        
        self.anchors[aid] = new_anchor
        self.current_anchor = new_anchor
        
        # Debug
        Logger.graph("Anchor", f"+Anchor {aid} in {place_id}")
        return aid

    def add_manual_place(self):
        # Force a new place creation
        Logger.info("User", "Manual Place Triggered")
        self._create_place(self.stable_vlm_context)
        self._create_anchor(self.current_place.place_id, self.robot_pose)

    def prune_old_anchors(self, max_anchors_per_place=20):
        """
        Performance: Limit graph growth by removing old anchors.
        Keeps only the most recent N anchors per Place.
        """
        total_pruned = 0
        for place in self.places.values():
            if len(place.anchors) > max_anchors_per_place:
                # Sort by timestamp (oldest first)
                sorted_anchor_ids = sorted(
                    place.anchors,
                    key=lambda aid: self.anchors[aid].timestamp if aid in self.anchors else 0
                )
                
                # Remove oldest anchors
                to_remove = sorted_anchor_ids[:-max_anchors_per_place]
                for aid in to_remove:
                    if aid in self.anchors:
                        # Remove from global dict
                        del self.anchors[aid]
                        # Remove from place's anchor list
                        place.anchors.remove(aid)
                        total_pruned += 1
        
        if total_pruned > 0:
            Logger.info("Manager", f"Pruned {total_pruned} old anchors (max={max_anchors_per_place}/place)")
        
        return total_pruned

    def get_nav_context(self):
        """
        Generates a natural language summary of the current spatial context.
        Used for VLM Prompt Injection.
        """
        if not self.current_place:
            return "You are in an unknown area."
        
        # 1. Current Place Info & Exploration Status
        anchor_count = len(self.current_place.anchors)
        expl_status = "New/Unexplored"
        if anchor_count > 10:  # Lowered from 15 to 10 to trigger "Escape" sooner
            expl_status = "Extensively Explored"
        elif anchor_count > 4: # Lowered from 5
            expl_status = "Partially Explored"
            
        place_info = f"You are currently in {self.current_place.place_type} (Place ID: {self.current_place.place_id}).\n"
        place_info += f"Exploration Status: {expl_status} ({anchor_count} nodes mapped here)." 

        
        # 2. Description (Caption)
        desc_info = ""
        if self.current_place.caption:
            desc_info = f"Description: {self.current_place.caption}"
            
        # 3. Object Context
        obj_info = ""
        objs = list(self.current_place.object_signature.keys())
        if objs:
            # Sort by count or just list top 5
            top_objs = objs[:10] 
            obj_info = f"Visible objects nearby: {', '.join(top_objs)}."
            
        # 4. History (Visited Places)
        visited = []
        for p in self.places.values():
            if p.place_id != self.current_place.place_id:
                visited.append(p.place_type)
        # Deduplicate
        visited = list(set(visited))
        
        history_info = "You have previously explored: " + (", ".join(visited) if visited else "None") + "."
        
        # Combine
        lines = [place_info]
        if desc_info: lines.append(desc_info)
        if obj_info: lines.append(obj_info)
        lines.append(history_info)
        
        return "\n".join(lines)

    def get_neighbor_directions(self, current_yaw):
        """
        Calculates relative angles to connected anchors.
        Returns a string list of directions.
        current_yaw: in radians
        """
        if not self.current_anchor:
            return "No known connections."
            
        params = []
        
        cx, cy, _ = self.current_anchor.pose
        
        for nid in self.current_anchor.neighbors:
            if nid not in self.anchors: continue
            neighbor = self.anchors[nid]
            nx, ny, _ = neighbor.pose
            
            # Vector
            dx = nx - cx
            dy = ny - cy
            
            # Global Angle
            global_angle = math.atan2(dy, dx)
            
            # Relative Angle
            rel_angle = global_angle - current_yaw
            
            # Normalize
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
            
            deg = math.degrees(rel_angle)
            
            # Target Place Info
            n_place = self.places[neighbor.place_id]
            
            status = "Visited" if neighbor.place_id != self.current_place.place_id else "Same Room"
            
            info = f"- Angle {deg:.0f} degrees leads to {n_place.place_type} ({status}, Node {nid})"
            params.append(info)
            
        if not params:
            return "No mapped connections from this point."
            
        return "\n".join(params)

    def save_to_json(self, filename="place_graph.json"):
        import json
        data = {
            "places": {},
            "anchors": {},
            "edges": self.edges
        }
        for pid, p in self.places.items():
            data["places"][pid] = {
                "type": p.place_type,
                "objects": list(p.object_signature.keys()),
                "anchor_count": len(p.anchors)
            }
        for aid, a in self.anchors.items():
            data["anchors"][aid] = {
                "place_id": a.place_id,
                "pose": a.pose,
                "neighbors": a.neighbors
            }
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
                
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        Logger.info("IO", f"Saved {len(self.anchors)} anchors and {len(self.places)} places to {filename}")
