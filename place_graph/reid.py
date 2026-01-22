class PlaceReIdentifier:
    """
    Module 3: Place Re-identification.
    Determines if a new scene corresponds to a previously visited place.
    """
    def __init__(self):
        pass

    def identify(self, candidate_place, all_places):
        """
        Compare candidate_place against all existing places using Semantic Anchors.
        Returns: matching_place_id or None
        """
        import difflib
        
        # 1. Broad Phase: Filter by Type
        candidates = [p for p in all_places.values() if p.place_type == candidate_place.place_type]
        
        best_match = None
        max_score = 0.0
        
        # Weights
        W_CAPTION = 0.7
        W_OBJECT = 0.3
        THRESHOLD = 0.6
        
        cand_objs = candidate_place.object_set  # Use cached set
        cand_caption = candidate_place.semantic_description.lower()
        
        for place in candidates:
            # Skip self
            if place.place_id == candidate_place.place_id:
                continue
            
            # A. Caption Similarity (Levenshtein)
            place_caption = place.semantic_description.lower()
            caption_score = 0.0
            if cand_caption and place_caption:
                caption_score = difflib.SequenceMatcher(None, cand_caption, place_caption).ratio()
            
            # B. Object IoU
            place_objs = place.object_set  # Use cached set
            obj_score = 0.0
            intersection = len(cand_objs.intersection(place_objs))
            union = len(cand_objs.union(place_objs))
            if union > 0:
                obj_score = intersection / union
                
            # C. Fused Score
            # If no caption available, fallback entirely to object score
            if not cand_caption or not place_caption:
                final_score = obj_score 
            else:
                final_score = (caption_score * W_CAPTION) + (obj_score * W_OBJECT)
            
            # Debug
            # print(f"Checking {place.place_id}: Cap={caption_score:.2f}, Obj={obj_score:.2f} -> Tot={final_score:.2f}")

            if final_score > THRESHOLD:
                if final_score > max_score:
                    max_score = final_score
                    best_match = place.place_id
                    
                    # Quick Win: Early Termination for very high confidence
                    if max_score > 0.9:
                        print(f"[ReID] High confidence match! {candidate_place.place_id} -> {best_match} (Score: {max_score:.2f})")
                        return best_match
                    
        if best_match:
             print(f"[ReID] Match Found! {candidate_place.place_id} -> {best_match} (Score: {max_score:.2f})")
             
        return best_match
