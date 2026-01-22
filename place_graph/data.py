import time

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
