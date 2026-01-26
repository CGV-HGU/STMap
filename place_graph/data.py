import time
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque


@dataclass
class DCRecord:
    """
    Discovered Context Record.
    Stores VLM planner's intention for future navigation.
    """
    idx: int  # 1..N (1=oldest, N=newest; re-assigned on each push)
    goal_flag: bool  # True if target visible in current observation
    goal_scene_type: str  # VLM output string (e.g., "kitchen", "bedroom")
    why: str  # One-sentence reason
    avoid_hint: str = ""  # Rule-based hint (e.g., "pattern:STUCK")

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "goal_flag": self.goal_flag,
            "goal_scene_type": self.goal_scene_type,
            "why": self.why,
            "avoid_hint": self.avoid_hint,
        }

    def __repr__(self):
        return f"<DC[{self.idx}] flag={self.goal_flag} scene={self.goal_scene_type}>"


class DCQueue:
    """
    FIFO Queue for Discovered Context records.
    Max size: N (default 5).
    Automatically re-indexes on push/pop.
    """
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._queue: deque = deque(maxlen=max_size)

    def push(self, goal_flag: bool, goal_scene_type: str, why: str, avoid_hint: str = ""):
        """Add new DCRecord. If full, oldest is popped automatically."""
        # Create record with temporary idx (will be re-indexed)
        record = DCRecord(
            idx=0,
            goal_flag=goal_flag,
            goal_scene_type=goal_scene_type,
            why=why,
            avoid_hint=avoid_hint,
        )
        self._queue.append(record)
        self._reindex()

    def _reindex(self):
        """Re-assign idx from 1..N (1=oldest, N=newest)."""
        for i, record in enumerate(self._queue):
            record.idx = i + 1

    def get_all(self) -> List[DCRecord]:
        """Return all records in order (oldest first)."""
        return list(self._queue)

    def to_list(self) -> List[dict]:
        """Serialize all records to list of dicts."""
        return [r.to_dict() for r in self._queue]

    def to_prompt_string(self) -> str:
        """Format DC Queue for VLM prompt injection."""
        if not self._queue:
            return "<No previous discovered context>"
        lines = []
        for r in self._queue:
            hint_part = f" [{r.avoid_hint}]" if r.avoid_hint else ""
            lines.append(f"[{r.idx}] goal_flag={r.goal_flag}, scene={r.goal_scene_type}, why=\"{r.why}\"{hint_part}")
        return "\n".join(lines)

    def clear(self):
        """Reset the queue."""
        self._queue.clear()

    def __len__(self):
        return len(self._queue)

    def __repr__(self):
        return f"<DCQueue size={len(self._queue)}/{self.max_size}>"

class Place:
    """
    Layer 2: Semantic Place (The Concept)
    Represents a semantic region (e.g., "Living Room", "Kitchen").
    Holds semantic attributes but no metric data.
    """
    def __init__(self, place_id, place_type="unknown"):
        self.place_id = place_id
        self.place_type = place_type
        
        # Semantic Attributes
        self.semantic_description = ""  # VLM Description
        self.objects_list = set()  # {label} - Used for context, not mapping
        
        # Topology Links
        self.anchors = [] # List of Anchor IDs that belong to this Place
        self.visit_count = 1
        self.last_visit_time = time.time()
        
        # Performance: Object set cache for Re-ID
        self._cached_object_set = None
        
    def add_object_context(self, label):
        """
        Add an object to the semantic signature.
        We don't care about position, just presence.
        """
        self.objects_list.add(label)
        self._cached_object_set = None  # Invalidate cache
    
    @property
    def object_set(self):
        """Cached object set for fast Re-ID comparison."""
        if self._cached_object_set is None:
            self._cached_object_set = set(self.objects_list)
        return self._cached_object_set

    @property
    def caption(self):
        return self.semantic_description

    @caption.setter
    def caption(self, value):
        self.semantic_description = value or ""

    @property
    def object_signature(self):
        return {label: 1 for label in self.objects_list}

    def __repr__(self):
        return f"<Place {self.place_id}: {self.place_type} ({len(self.objects_list)} objs)>"


class Anchor:
    """
    Layer 1: Topological Anchor (The Waypoint)
    Represents a navigable point in space (Breadcrumbs).
    Holds topological connectivity.
    """
    def __init__(self, anchor_id, pose, place_id):
        self.anchor_id = anchor_id
        self.pose = pose # (x, y, yaw)
        self.place_id = place_id # Parent Place (L2)
        
        self.neighbors = [] # List of connected Anchor IDs (Edges)
        self.timestamp = time.time()
        
        # Viz Flag
        self.is_door = False
        
    def add_neighbor(self, neighbor_id):
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)

    def __repr__(self):
        return f"<Anchor {self.anchor_id} in {self.place_id}>"
