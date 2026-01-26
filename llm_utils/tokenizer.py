
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
You are a robotic vision system. Your job is to analyze a partial view (slice) of a room and list the visible objects from a fixed vocabulary.
You must also provide a very brief one-sentence visual description of the scene slice.

Input: An image (a 30-degree slice of the panoramic view).
Output: A strict JSON object.

Allowed Vocabulary:
{allowed_vocab}

Rules:
1. Detect ONLY objects visible in the image.
2. Use ONLY the exact words from the allowed vocabulary. If an object is a synonym (e.g., 'couch'), map it mentally to the vocabulary ('sofa') but output the vocabulary word.
3. If 'door' or 'doorway' is visible, it is CRITICAL to list 'door'.
4. Do NOT hallucinate objects not present.
5. 'description' should describe lighting, colors, or unique visual features (e.g., "dark corner with a red sofa").

JSON Format:
{{
  "tokens": ["object1", "object2", ...],
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
            description = data.get("description", "")
            
            # Normalize and filter
            valid_tokens = []
            for t in tokens:
                norm = normalize_token(t)
                if norm:
                    valid_tokens.append(norm)
            
            # Remove duplicates while preserving order? No, set is fine for tokens.
            valid_tokens = sorted(list(set(valid_tokens)))
            
            return {
                "tokens": valid_tokens,
                "description": description
            }
            
        except Exception as e:
            # print(f"Tokenizer Error (Attempt {attempt}): {e}")
            if attempt < retry:
                time.sleep(0.5)
            
    # Fallback
    return {"tokens": [], "description": "unknown"}
