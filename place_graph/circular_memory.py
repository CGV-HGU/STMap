
import numpy as np
import uuid
import heapq
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional

# Constants
K_SLOTS = 12  # 360 degrees / 30 degrees = 12 slots

@dataclass
class DoorInfo:
    """
    Episode-local door state.
    """
    door_id: int
    place_id: str  # The place this door belongs to
    slot_index: int # The slot index in the circular list (0..11)
    visited: bool = False
    attempts: int = 0
    leads_to_place_id: Optional[str] = None
    last_result: str = "none" # "scene_changed", "stop_no_change", "failed"

@dataclass
class Slot:
    """
    A single angular slot (30-degree sector).
    """
    index: int
    tokens: Set[str] # Valid object tokens
    description: str = "" # VLM description for Planner context
    
    # Embedded door info (for the Planner to see)
    # We store the ID here for easy access. Details are in DoorMemory.
    door_ids: List[int] = field(default_factory=list) 

    def to_planner_string(self, door_memory: Dict[Tuple[str, int], DoorInfo], place_id: str) -> str:
        """
        Format: "[sofa, table]" or "[door(id=1, visited=False)]"
        """
        items = sorted(list(self.tokens))
        
        # In the token set, 'door' is just 'door'. 
        # But for display, we want to augment 'door' with its status if present.
        display_items = []
        has_door_token = "door" in self.tokens
        
        for t in items:
            if t == "door":
                continue # Handle doors separately
            display_items.append(t)
            
        if has_door_token and self.door_ids:
            for did in self.door_ids:
                # Look up visited status
                key = (place_id, did)
                visited = False
                if key in door_memory:
                    visited = door_memory[key].visited
                
                display_items.append(f"door(id={did}, visited={visited})")
        elif has_door_token:
            # Fallback if no ID assigned yet (shouldn't happen in finalized scan)
            display_items.append("door(unknown)")
            
        return f"[{', '.join(display_items)}]"

@dataclass
class Scan:
    """
    One 360-degree panoramic observation event.
    """
    scan_id: str
    place_id: str
    slots: List[Slot]  # Fixed length K=12
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

    def _get_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not set_a and not set_b:
            return 1.0 # Both empty = match
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def get_best_shift(self, scan_a: List[Slot], scan_b: List[Slot]) -> Tuple[int, float]:
        """
        Find best circular shift to align scan_b to scan_a.
        Returns (shift, score). slot[i] in A matches slot[(i+shift)%K] in B.
        """
        best_shift = 0
        best_score = -1.0
        
        # For each possible shift
        for shift in range(K_SLOTS):
            total_sim = 0.0
            
            for i in range(K_SLOTS):
                # A[i] vs B[(i + shift) % K]
                set_a = scan_a[i].tokens
                
                # Careful: We want to match visual features. 
                # Door tokens are visual features too ("door"), so we perform match on raw tokens.
                # Attributes like ID/visited are not part of visual matching.
                
                b_idx = (i + shift) % K_SLOTS
                set_b = scan_b[b_idx].tokens
                
                total_sim += self._get_jaccard(set_a, set_b)
            
            avg_score = total_sim / K_SLOTS
            if avg_score > best_score:
                best_score = avg_score
                best_shift = shift
                
        return best_shift, best_score

    def assign_place(self, current_slots: List[Slot], threshold: float = 0.35) -> Tuple[str, int, float, bool]:
        """
        Match current scan against PlaceDB.
        Returns: (place_id, shift, score, is_revisit)
        """
        best_overall = (None, 0, -1.0) # (pid, shift, score)
        
        for pid, scans in self.places.items():
            # Match against all representatives (or just the first/last? All is safer for now)
            for ref_scan in scans:
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
            if "door" in slot.tokens:
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
    
    def update_door_status(self, place_id: int, door_id: int, result_type: str, new_place_id: str = None):
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
        
    def get_nav_context(self, target_object: str) -> str:
        """
        Build the prompt text for Planner VLM.
        """
        if not self.current_place_id:
            return "No location context available."
            
        current_scan = self.places[self.current_place_id][-1]
        
        lines = []
        lines.append(f"Target Object: {target_object}")
        lines.append(f"Current Location: {self.current_place_id} (Revisit: {current_scan.is_revisit}, Match Score: {current_scan.match_score:.2f})")
        
        # Current Slots
        lines.append("\nCurrent 360 Scan (12 slots):")
        unvisited_slots = []
        for slot in current_scan.slots:
            content = slot.to_planner_string(self.door_memory, self.current_place_id)
            desc_part = f" - {slot.description}" if slot.description else ""
            lines.append(f"Slot {slot.index:02d}: {content}{desc_part}")
            
            # Check for unvisited doors in this slot
            if slot.door_ids:
                is_unvisited = False
                for did in slot.door_ids:
                    key = (self.current_place_id, did)
                    if key in self.door_memory and not self.door_memory[key].visited:
                        is_unvisited = True
                        break
                if is_unvisited:
                    unvisited_slots.append(f"Slot {slot.index}")

        # CRITICAL HINT for Exploration
        if unvisited_slots:
            lines.append("\n[CRITICAL HINT]")
            lines.append(f"- 🚪 Unvisited Doors available at: {', '.join(unvisited_slots)}")
            lines.append("- If goal is not visible, prioritize these slots to EXPLORE NEW AREAS and ESCAPE the current room.")
        else:
            lines.append("\n[HINT]")
            lines.append("- No unvisited doors in this view. You may need to revisit a door or look for a hallway.")
            
        # History
        if self.dc_queue:
            lines.append("\nRecent Decisions (Discovered Context):")
            for item in self.dc_queue:
                lines.append(f"- At {item['place_id']}, chose slot {item['angle_slot']} (Goal={item['goal_flag']}) -> Result: {item['result']}")
                
        return "\n".join(lines)
