
# Fixed Vocabulary for Metric-Free Semantic Memory
# This list defines the ONLY allowed tokens for the memory system.
# Scene VLM outputs must be normalized to these tokens.

# 1. Target Objects (Habitat ObjectNav Categories)
TARGET_OBJECTS = {
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv_monitor",
    "sofa",
    "table",  # maps from dining table, coffee table, side table
    "counter",
    "sink",
    "cabinet", # wardrobe, cupboard
    "cushion",
    "fireplace",
    "gym_equipment",
    "seating",
    "towel",
    "clothes",
}

# 2. Structural Objects (Critical for Navigation / Place Recognition)
# 2. Structural Objects (Critical for Navigation / Place Recognition)
STRUCTURAL_OBJECTS = {
    "exit",       # Primary logic token (replaces door)
    "closed_door", # Visual landmark, but NOT an exit
    "window",
    "stairs",
    "elevator",
    "picture",    # Good for re-identification (visual landmarks)
    "mirror",
    "curtain",
    "refrigerator",
    "shower",
    "bathtub",
    "stove",
    "shelf",      # Bookshelf, rack
}

# 3. Full Allowed Vocabulary
ALLOWED_VOCAB = TARGET_OBJECTS | STRUCTURAL_OBJECTS

# 4. Normalization Mapping (Raw VLM Output -> Canonical Token)
# If a word is not here and not in ALLOWED_VOCAB, it will be dropped.
SYNONYM_MAP = {
    # Exits (Doors, Archways, Corridors)
    "door": "exit", # Default to exit if unspecified
    "doorway": "exit",
    "entrance": "exit",
    "gate": "exit",
    "sliding door": "exit",
    "archway": "exit",
    "open door": "exit",
    
    # Closed Doors (Obstacles)
    "closed door": "closed_door",
    "locked door": "closed_door",
    
    "door frame": "exit",
    "hallway": "exit", # Treat hallway opening as an exit
    "corridor": "exit", # Treat corridor opening as an exit
    "opening": "exit",
    "passage": "exit",
    
    # Furniture
    "couch": "sofa",
    "loveseat": "sofa",
    "dining table": "table",
    "coffee table": "table",
    "side table": "table",
    "desk": "table",
    "nightstand": "table",
    "wardrobe": "cabinet",
    "cupboard": "cabinet",
    "dresser": "cabinet",
    "bookshelf": "shelf",
    "rack": "shelf",
    "chest of drawers": "cabinet",
    "drawer": "cabinet",
    
    # Electronics
    "tv": "tv_monitor",
    "monitor": "tv_monitor",
    "screen": "tv_monitor",
    "television": "tv_monitor",
    
    # Appliances/Fixtures
    "fridge": "refrigerator",
    "oven": "stove",
    "washbasin": "sink",
    "vanity": "cabinet",
    "rug": "cushion", 
    "carpet": "cushion", # Treat floor textiles as cushions/rugs for now or ignore? Let's map to cushion for texture match or ignore. Actually rug is distinct. 
    # Let's remove rug/carpet mapping to cushion to avoid confusion, or keep 'rug' if we add it to vocab.
    # For now, let's map 'rug' to 'cushion' if it helps matching, or just drop it. 
    # Let's drop rug for now as it's on the floor and might be confusing with 'cushion' on sofa.
}

def normalize_token(raw_token: str) -> str:
    """
    Normalize a raw token string to a canonical vocabulary token.
    Returns None if the token is not allowed.
    """
    if not isinstance(raw_token, str):
        return None
        
    token = raw_token.lower().replace("_", " ").strip()
    
    # Direct match check
    if token in ALLOWED_VOCAB:
        return token
    
    # Synonym check
    if token in SYNONYM_MAP:
        return SYNONYM_MAP[token]
    
    # Substring check for robustness (e.g. "wooden chair" -> "chair")
    # We prioritize longer matches or specific rules if needed.
    # Reverse sort ALLOWED_VOCAB by length to match "dining table" (if in allowed) before "table"
    # But "dining table" is in synonym map.
    
    # Check if any allowed token appears in the raw token
    # e.g. "small white chair" -> "chair"
    for allowed in ALLOWED_VOCAB:
        # Pad with spaces to avoid partial word matches (e.g. "table" in "vegetable" - unlikely but good practice)
        # But for simplicity in this constrained domain:
        if allowed in token:
             return allowed
             
    # Check synonyms similarly
    for syn, target in SYNONYM_MAP.items():
        if syn in token:
            return target
            
    return None
