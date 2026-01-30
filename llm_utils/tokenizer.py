
import cv2
import numpy as np
import time
import json
import re
from .gpt_request import _image_to_bytes, _MODEL
from vertexai.preview.generative_models import Part
from cv_utils.vocab import normalize_token, ALLOWED_VOCAB

# System prompt for the Tokenizer VLM
TOKENIZER_SYSTEM_PROMPT = """
You are a robotic vision system. Your job is to analyze a partial view (slice) of a room and identify both common objects and unique landmarks.

Input: An image (a 60-degree slice of the panoramic view).
Output: A strict JSON object.

Allowed Vocabulary (for "tokens"):
{allowed_vocab}

Rules:
1. **Tokens**: List common objects using ONLY the allowed vocabulary.
2. **Exits (CRITICAL)**: Any path you can walk through (open door, corridor opening, archway) MUST be labeled as "exit".
3. **Closed Doors (IMPORTANT)**: If a door is CLOSED or blocked, label it as "closed_door". Do NOT label it as "exit".
4. **Landmarks**: List 1-3 UNIQUE visual features that distinguish this specific area (e.g., "red_striped_sofa", "circular_mirror", "tall_blue_vase"). Be specific about colors, materials, or shapes.
5. **Hierarchy**: If something is a landmark, it can ALSO be in the tokens (e.g., tokens: ["sofa"], landmarks: ["red_striped_sofa"]).
6. **Description**: Provide a brief one-sentence visual summary highlighting environmental conditions (lighting, clutter).
5. Do NOT hallucinate. Only list what is clearly visible.

JSON Format:
{{
  "tokens": ["object1", "object2", ...],
  "landmarks": ["unique_feature1", "unique_feature2", ...],
  "description": "Short visual description."
}}
"""

def extract_tokens_and_description(image, retry=2):
    """
    Calls Gemini to extract tokens and description from a single image slice.
    Args:
        image: numpy array (BGR) or file path
        retry: number of retries
    Returns:
        dict: {"tokens": [str], "description": str}
        Returns fallback if failed.
    """
    
    vocab_str = ", ".join(sorted(list(ALLOWED_VOCAB)))
    full_prompt = TOKENIZER_SYSTEM_PROMPT.format(allowed_vocab=vocab_str)
    
    user_msg = "Analyze this image slice. List objects and describe."
    
    for attempt in range(retry + 1):
        try:
            image_bytes = _image_to_bytes(image)
            
            prompt = [full_prompt, user_msg, Part.from_data(data=image_bytes, mime_type="image/jpeg")]
            
            resp = _MODEL.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 512,
                    "temperature": 0.2, 
                    "response_mime_type": "application/json"
                },
            )
            
            raw_text = getattr(resp, "text", "").strip()
            
            # Clean up markdown code blocks if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            
            data = json.loads(raw_text)
            
            # Post-processing / Validation
            tokens = data.get("tokens", [])
            landmarks = data.get("landmarks", [])
            description = data.get("description", "")
            
            # Normalize and filter tokens
            valid_tokens = []
            for t in tokens:
                norm = normalize_token(t)
                if norm:
                    valid_tokens.append(norm)
            
            valid_tokens = sorted(list(set(valid_tokens)))
            
            return {
                "tokens": valid_tokens,
                "landmarks": [str(l).lower().strip() for l in landmarks],
                "description": description
            }
            
        except Exception as e:
            # print(f"Tokenizer Error (Attempt {attempt}): {e}")
            if attempt < retry:
                time.sleep(0.5)
            
    # Fallback
    return {"tokens": [], "description": "unknown"}
