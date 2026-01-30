
import numpy as np
import uuid
import heapq
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional

# Constants
K_SLOTS = 6  # Optimized for Scene VLM (odd-indexed slots: 1, 3, 5, 7, 9, 11)

@dataclass
class DoorInfo:
    """
    Episode-local door state.
    """
    door_id: int
    place_id: str  # The place this door belongs to
    slot_index: int # The slot index in the circular list (0..5)
    visited: bool = False
    attempts: int = 0
    leads_to_place_id: Optional[str] = None
    last_result: str = "none" # "scene_changed", "stop_no_change", "failed"

@dataclass
class Slot:
    """
    A single angular slot (60-degree sector in memory, mapped from 30-degree slices).
    """
    index: int
    tokens: Set[str] # Valid object tokens
    landmarks: Set[str] = field(default_factory=set) # Unique visual markers
    description: str = "" # VLM description for Planner context
    
    # Embedded door info (for the Planner to see)
    # We store the ID here for easy access. Details are in DoorMemory.
    door_ids: List[int] = field(default_factory=list) 

    def to_planner_string(self, door_memory: Dict[Tuple[str, int], DoorInfo], place_id: str) -> str:
        """
        Format: "[sofa, table]" or "[door(id=1, visited=False)]"
        """
        items = sorted(list(self.tokens))
        
        # In the token set, 'exit' is just 'exit'. 
        # But for display, we want to augment 'exit' with its status if present.
        display_items = []
        has_exit_token = "exit" in self.tokens
        
        for t in items:
            if t == "exit":
                continue # Handle exits separately
            display_items.append(t)
            
        if has_exit_token and self.door_ids:
            for did in self.door_ids:
                # Look up visited status
                key = (place_id, did)
                visited = False
                if key in door_memory:
                    visited = door_memory[key].visited
                
                attempts = door_memory[key].attempts if key in door_memory else 0
                last_result = door_memory[key].last_result if key in door_memory else "none"
                
                if attempts > 0 and last_result == "stop_no_change":
                    display_items.append(f"exit(id={did}, visited={visited}, FAILED x{attempts})")
                else:
                    display_items.append(f"exit(id={did}, visited={visited})")
        elif has_exit_token:
            # Fallback if no ID assigned yet (shouldn't happen in finalized scan)
            display_items.append("exit(unknown)")
            
        return f"[{', '.join(display_items)}]"

@dataclass
class Scan:
    """
    One 360-degree panoramic observation event (6 slots).
    """
    scan_id: str
    place_id: str
    slots: List[Slot]  # Fixed length K=6
    timestamp: float
    
    # Matching info
    matched_shift: int = 0
    match_score: float = 0.0
    is_revisit: bool = False

class CircularMemory:
    """
    The Engine. Manages PlaceDB, matching, and Door states.
    """
    def __init__(self):
        self.places: Dict[str, List[Scan]] = {} # place_id -> list of representative scans
        self.door_memory: Dict[Tuple[str, int], DoorInfo] = {} # (place_id, door_id) -> Info
        self.history_scans: deque = deque(maxlen=3) # Last 3 scans for context
        self.dc_queue: deque = deque(maxlen=5) # Planner decision history
        
        self.current_place_id: Optional[str] = None
        self.next_place_idx = 0
        self.next_door_idx = 0 # Global or per-place? Let's do per-place or global unique. Global is easier for debugging.
        
        # Oscillation Prevention: Tracking where we came from
        self.source_door_key: Optional[Tuple[str, int]] = None # (prev_place_id, prev_door_id) that led here

    def _get_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not set_a and not set_b:
            return 1.0 # Both empty = match
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _get_weighted_similarity(self, slot_a: Slot, slot_b: Slot) -> Tuple[float, float]:
        """
        Compute similarity and weight for two slots.
        Slots with landmarks or tokens carry more 'information weight'.
        """
        token_sim = self._get_jaccard(slot_a.tokens, slot_b.tokens)
        landmark_sim = self._get_jaccard(slot_a.landmarks, slot_b.landmarks)
        
        # 1. Determine Weight based on information content
        if slot_a.landmarks or slot_b.landmarks:
            weight = 5.0  # Landmarks are extremely unique
            score = (0.2 * token_sim) + (0.8 * landmark_sim)
        elif slot_a.tokens or slot_b.tokens:
            weight = 1.0  # Common objects are standard landmarks
            score = token_sim
        else:
            weight = 0.1  # Empty spaces (walls/floors) provide very little uniqueness
            score = 1.0   # But they match each other perfectly
            
        return score, weight

    def get_best_shift(self, scan_a: List[Slot], scan_b: List[Slot]) -> Tuple[int, float]:
        """
        Find best circular shift to align scan_b to scan_a.
        Returns (shift, score). slot[i] in A matches slot[(i+shift)%K] in B.
        """
        best_shift = 0
        best_score = -1.0
        
        # For each possible shift
        for shift in range(K_SLOTS):
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for i in range(K_SLOTS):
                b_idx = (i + shift) % K_SLOTS
                score, weight = self._get_weighted_similarity(scan_a[i], scan_b[b_idx])
                total_weighted_score += (score * weight)
                total_weight += weight
            
            avg_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            if avg_score > best_score:
                best_score = avg_score
                best_shift = shift
                
        return best_shift, best_score

    def assign_place(self, current_slots: List[Slot], threshold: float = 0.25) -> Tuple[str, int, float, bool]:
        """
        Match current scan against PlaceDB.
        Returns: (place_id, shift, score, is_revisit)
        """
        best_overall = (None, 0, -1.0) # (pid, shift, score)
        
        for pid, scans in self.places.items():
            # Match ONLY against the canonical scan (first scan) to keep shifts canonical.
            if not scans:
                continue
            ref_scan = scans[0]
            shift, score = self.get_best_shift(ref_scan.slots, current_slots)
            
            if score > best_overall[2]:
                best_overall = (pid, shift, score)
        
        pid, shift, score = best_overall
        
        is_revisit = False
        if score >= threshold and pid is not None:
            is_revisit = True
            final_pid = pid
            # Optional: Add current scan as new representative if it adds diversity?
            # For simplicity, we just use the first/best match.
        else:
            # Create new place
            final_pid = f"place_{self.next_place_idx}"
            self.next_place_idx += 1
            is_revisit = False
            shift = 0 # No shift relative to itself
            score = 1.0 # Perfect match with itself
            
            # Init empty list for this new place
            self.places[final_pid] = []

        # Store this scan
        new_scan = Scan(
            scan_id=str(uuid.uuid4())[:8],
            place_id=final_pid,
            slots=current_slots,
            timestamp=0.0, # Filled by caller if needed
            matched_shift=shift,
            match_score=score,
            is_revisit=is_revisit
        )
        self.places[final_pid].append(new_scan)
        self.current_place_id = final_pid
        
        return final_pid, shift, score, is_revisit
    
    def manage_doors(self, place_id: str, slots: List[Slot]):
        """
        Identify doors in the slots and assign IDs.
        For now, simplistic logic: 
        - each slot with 'door' gets a new ID if it hasn't been assigned.
        - Wait, we need to Re-identify doors if we are revisiting a place!
        """
        # If this is a new place, just assign new IDs to every door slot.
        # If revisiting, we need to map current slots to original reference slots using 'matched_shift'.
        # However, 'assign_place' already returned the shift relative to the reference.
        # But 'slots' here are raw. We need to know which reference slot they correspond to.
        
        # The scan object we just saved has 'matched_shift'.
        # scan a (ref) [i] <-> scan b (cur) [(i + shift) % K]
        # So cur[j] corresponds to ref[(j - shift) % K]
        
        # It's complicated to track persistent door IDs across multiple visits without a strict canonical frame.
        # Simplification for Single-Episode:
        # Just use (PlaceID, SlotIndex_in_Canonical_Frame) as the Door Key.
        
        # 1. Find the canonical frame (Scan 0 of this place).
        ref_scan = self.places[place_id][0]
        
        # 2. Get the shift of the CURRENT scan relative to REF.
        # We assume the last added scan is the current one.
        current_scan = self.places[place_id][-1] 
        shift = current_scan.matched_shift
        
        # 3. For each current slot j:
        for j, slot in enumerate(slots):
            if "exit" in slot.tokens:
                # Calculate canonical index i
                # ref[i] matches cur[j] where j = (i + shift) % K
                # => i = (j - shift) % K
                canonical_idx = (j - shift) % K_SLOTS
                
                # Check if we already have a door ID for (place_id, canonical_idx)
                # We can store a mapping or just search existing DoorMemory?
                # Let's search DoorMemory keys.
                # Key format: (place_id, door_id) ... wait, we need to link canonical_idx to door_id.
                
                # Let's assume door_id is unique per (place, canonical_idx).
                # We need a lookup: (pid, c_idx) -> door_id
                
                existing_did = None
                for (d_pid, d_did), info in self.door_memory.items():
                    if d_pid == place_id and info.slot_index == canonical_idx:
                        existing_did = d_did
                        break
                
                if existing_did is not None:
                    # Found existing door
                    slot.door_ids.append(existing_did)
                else:
                    # New door detected
                    new_did = self.next_door_idx
                    self.next_door_idx += 1
                    
                    self.door_memory[(place_id, new_did)] = DoorInfo(
                        door_id=new_did,
                        place_id=place_id,
                        slot_index=canonical_idx,
                        visited=False
                    )
                    slot.door_ids.append(new_did)
    
    def update_door_status(self, place_id: str, door_id: int, result_type: str, new_place_id: str = None):
        """
        Called after execution to update visited/result status.
        result_type: "scene_changed", "stop_no_change", etc.
        """
        key = (place_id, door_id)
        if key not in self.door_memory:
            return
            
        info = self.door_memory[key]
        info.attempts += 1
        info.last_result = result_type
        
        if result_type == "scene_changed":
            info.visited = True
            info.leads_to_place_id = new_place_id
            # Record this as the source door for the new place
            self.source_door_key = key
        
            
    def get_graph_summary(self) -> str:
        """
        Generate a concise text summary of the known graph topology.
        Format:
        Place A (Bedroom): Connected to Place B
        Place B (Hallway): Connected to Place A, Place C
        ...
        """
        if not self.places:
            return "Map is empty."
            
        summary_lines = []
        
        # Build adjacency list from door_memory
        adj = {}
        for (pid, did), info in self.door_memory.items():
            if info.visited and info.leads_to_place_id:
                if pid not in adj: adj[pid] = set()
                adj[pid].add(info.leads_to_place_id)
                
                # Assume undirected for visualization? Or directed?
                # Doors are directed edges. 
                # But physically, if A->B via door 1, usually B->A via some door.
        
        # We only list places that have outgoing connections or are the current place
        relevant_places = set(adj.keys()) | {self.current_place_id} if self.current_place_id else set()
        
        for pid in sorted(list(relevant_places)):
            neighbors = sorted(list(adj.get(pid, [])))
            
            # Add semantic label if available (from first slot description or similar? tricky)
            # For now just use ID.
            
            line = f"- {pid}"
            if pid == self.current_place_id:
                line += " (CURRENT)"
            
            if neighbors:
                line += f" is connected to: {', '.join(neighbors)}"
            else:
                line += " has no explored exits yet."
            
            summary_lines.append(line)
            
        return "\n".join(summary_lines)

    def get_forbidden_slots(self) -> List[int]:
        """
        Detect cycles and return a list of slot indices that should be STRICTLY avoided.
        Logic:
        1. If we are in Place A, and we came from Place B.
        2. If going to Place C would complete a cycle A->..->C->..->A.
        
        Actually, simpler logic for 'Oscillation Prevention':
        If we recently visited Place X, do not go back to Place X immediately unless forced.
        
        Refined Logic (Graph Cycle):
        - Perform BFS/DFS to see if a door leads to a place we visited in the current 'segment'.
        - But we don't know where a CLOSED door leads.
        - We only know where OPEN doors lead.
        
        So, 'Forbidden' applies to:
        1. [Oscillation] The door leading back to 'prev_place_id' IF we have other options.
           (Already handled by prompt advice, but we want STRICT masking).
        2. [Loop] If we are at A, and door D leads to B, and B is already in our recent history (e.g. A->C->B->A), 
           then taking D is a cycle.
           
        Let's implement a 'Recent History' based ban.
        """
        forbidden = []
        if not self.current_place_id:
            return []
            
        # 1. Backtrack Ban (Oscillation)
        # If we have [NEW EXPLORATION] slots, strictly ban the [SOURCE] slot to force exploration.
        has_new_exploration = False
        current_scan = self.places[self.current_place_id][-1]
        for slot in current_scan.slots:
            for did in slot.door_ids:
                info = self.door_memory.get((self.current_place_id, did))
                if info and not info.visited and info.attempts == 0:
                     has_new_exploration = True
        
        if has_new_exploration and self.source_door_key:
             # Find which slot contains the source door
             prev_pid, prev_did = self.source_door_key
             # We need to find the slot in CURRENT place that leads to prev_pid
             # Wait, source_door_key is (prev_place, prev_door). 
             # We need the door in CURRENT place that leads to prev_place.
             
             # Search door memory for leads_to_place_id == prev_pid?
             # Or rely on 'leads_to' being set?
             # Usually, when we traverse A->B, we set A.door.leads_to = B.
             # We DO NOT automatically set B.door.leads_to = A.
             
             # So we might not know which door leads back, unless we track 'entry_door' for the current place.
             # Let's rely on the 'manage_doors' or logic that sets source_door_key.
             # Actually, we don't track 'entry_door_id' for the current place in this class.
             pass 

        # 2. Cycle Detection
        # Traverse decision queue to find recently visited places
        # If a door leads to a place in dc_queue[-3:], ban it.
        recent_places = [d['place_id'] for d in self.dc_queue]
        
        for vlm_idx, slot in enumerate(current_scan.slots):
             for did in slot.door_ids:
                 info = self.door_memory.get((self.current_place_id, did))
                 if info and info.visited and info.leads_to_place_id:
                     if info.leads_to_place_id in recent_places:
                         forbidden.append(vlm_idx)
                         
        return sorted(list(set(forbidden)))

    def get_discouraged_slots(self) -> List[int]:
        """
        Detect cycles and return a list of slot indices that should be AVOIDED if possible.
        Renamed from 'forbidden' to 'discouraged' to reflect softer constraints.
        """
        # Re-use logic for now, but semantically softer.
        return self._detect_potential_cycles()

    def _detect_potential_cycles(self) -> List[int]:
        forbidden = []
        if not self.current_place_id:
            return []
            
        # 1. Backtrack Discouragement
        # If we have [NEW EXPLORATION] slots, discourage the [SOURCE] slot.
        has_new_exploration = False
        current_scan = self.places[self.current_place_id][-1]
        for slot in current_scan.slots:
            for did in slot.door_ids:
                info = self.door_memory.get((self.current_place_id, did))
                if info and not info.visited and info.attempts == 0:
                     has_new_exploration = True
        
        if has_new_exploration and self.source_door_key:
             # Just discouragement, relying on prompt advice. 
             # We won't explicitly add it to this list unless we want to emphasize it in [DISCOURAGED SLOTS].
             # Let's add it to emphasize.
             prev_pid, prev_did = self.source_door_key
             # Logic to find which local slot leads to prev_pid is tricky without entry_door mapping.
             # Check leads_to_place_id
             for vlm_idx, slot in enumerate(current_scan.slots):
                 for did in slot.door_ids:
                      info = self.door_memory.get((self.current_place_id, did))
                      if info and info.leads_to_place_id == prev_pid:
                           forbidden.append(vlm_idx)

        # 2. Cycle Detection (Strict Repetition)
        # Only discourage if it leads to a place visited in the last 3 steps (ping-pong prevention).
        recent_places = [d['place_id'] for d in self.dc_queue][-3:]
        
        for vlm_idx, slot in enumerate(current_scan.slots):
             for did in slot.door_ids:
                 info = self.door_memory.get((self.current_place_id, did))
                 if info and info.visited and info.leads_to_place_id:
                     if info.leads_to_place_id in recent_places:
                         forbidden.append(vlm_idx)
                         
        return sorted(list(set(forbidden)))

    def update_last_decision_result(self, result_type: str):

        """
        Update the 'result' field of the most recent decision in dc_queue.
        """
        if self.dc_queue:
            self.dc_queue[-1]["result"] = result_type
            
    def get_nav_context(self, target_object: str) -> Tuple[str, List[int]]:
        """
        Build the prompt text for Planner VLM with explicit topological categorization.
        Returns: (context_text, forbidden_slots)
        """
        if not self.current_place_id:
            return "No location context available.", []

            
        current_scan = self.places[self.current_place_id][-1]
        
        lines = []
        lines.append(f"Target Object: {target_object}")
        lines.append(f"Current Location: {self.current_place_id} (Revisit: {current_scan.is_revisit}, Match Score: {current_scan.match_score:.2f})")
        
        # --- Global Graph Summary ---
        lines.append("\n[GLOBAL MAP SUMMARY]")
        lines.append(self.get_graph_summary())

        
        # --- Current Slots (6 units) ---
        lines.append("\n[CURRENT OBSERVATION (6 visual slots)]")
        lines.append("Note: These indices (0-5) match the 6-grid image provided.")
        
        unvisited_vlm_slots = []
        source_vlm_slot = None
        
        # Categorize doors for the Planner
        for vlm_idx in range(K_SLOTS):
            slot = current_scan.slots[vlm_idx]
            
            door_labels = []
            if slot.door_ids:
                for did in slot.door_ids:
                    key = (self.current_place_id, did)
                    info = self.door_memory.get(key)
                    if not info: continue
                    
                    # 1. Check if this is the SOURCE (Where we just came from)
                    is_source = False
                    if self.source_door_key:
                        prev_pid, prev_did = self.source_door_key
                        # If this door leads back to prev_pid, it's a return path
                        if info.leads_to_place_id == prev_pid:
                            is_source = True
                    
                    label = f"DoorID:{did}"
                    if is_source:
                        label += " [SOURCE/BACKTRACK]"
                        source_vlm_slot = vlm_idx
                    elif not info.visited:
                        label += " [NEW EXPLORATION]"
                        unvisited_vlm_slots.append(vlm_idx)
                    else:
                        label += f" [EXPLORED - leads to {info.leads_to_place_id}]"
                        
                    if info.attempts > 0 and not info.visited:
                        label += f" (FAILED x{info.attempts})"
                    
                    door_labels.append(label)

            content = slot.to_planner_string(self.door_memory, self.current_place_id)
            # Remove the generic exit format from to_planner_string and use our enhanced labels
            if door_labels:
                # Basic tokens without the generic 'exit' tags to avoid clutter
                base_tokens = [t for t in slot.tokens if t != "exit"]
                content = f"[{', '.join(base_tokens + door_labels)}]"

            landmark_part = f" (Landmarks: {', '.join(slot.landmarks)})" if slot.landmarks else ""
            desc_part = f" - {slot.description}" if slot.description else ""
            lines.append(f"Slot {vlm_idx}: {content}{landmark_part}{desc_part}")

        # CRITICAL TOPOLOGICAL ADVICE
        lines.append("\n[NAVIGATION STRATEGY & TOPOLOGY]")
        if unvisited_vlm_slots:
            unique_unvisited = sorted(list(set(unvisited_vlm_slots)))
            lines.append(f"- 🚪 ACTION: Found UNVISITED doors at Slot(s): {', '.join(map(str, unique_unvisited))}.")
            lines.append("- If you don't see the target object nearby, prioritize these NEW exits to expand the map.")
        
        if source_vlm_slot is not None:
            lines.append(f"- 🔙 AVOID: Slot {source_vlm_slot} is the path you just came from ({self.source_door_key[0]}).")
            lines.append("  Do NOT go back through this door immediately unless you are stuck or have finished searching this room.")

        # --- Decision History ---
        if self.dc_queue:
            lines.append("\n[DECISION HISTORY]")
            for item in list(self.dc_queue)[-3:]:
                res = item['result'].upper()
                goal_str = " (Goal Check)" if item.get('goal_flag') else ""
                lines.append(f"- At {item['place_id']}{goal_str}:")
                lines.append(f"  Thought: {item['why']}")
                lines.append(f"  Result: {res}")

        # --- Loop Detection ---
        if len(self.dc_queue) >= 3:
            recent_places = [d["place_id"] for d in list(self.dc_queue)[-3:]]
            if recent_places[0] == recent_places[2] and recent_places[0] != recent_places[1]:
                lines.append("\n[🚨 WARNING: OSCILLATION DETECTED]")
                lines.append(f"- You are ping-ponging between {recent_places[0]} and {recent_places[1]}.")
                lines.append("- STOP this cycle. Choose a [NEW EXPLORATION] slot or a different explored door.")
                
        return "\n".join(lines), self.get_discouraged_slots()
