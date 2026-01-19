import cv2
import numpy as np
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

PROJECT_ID = "dogwood-method-480911-p3"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash-lite"

vertexai.init(project=PROJECT_ID, location=LOCATION)
_MODEL = GenerativeModel(MODEL_NAME)

def _image_to_bytes(image):
    """
    image: 파일 경로(str) 또는 numpy.ndarray(BGR, OpenCV)
    return: bytes (JPEG)
    """
    if isinstance(image, str):
        with open(image, "rb") as f:
            return f.read()
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("이미지 인코딩 실패")
        return buf.tobytes()
    raise TypeError("image must be a file path (str) or numpy.ndarray")

# -----------------------------
# 텍스트 전용 응답
# -----------------------------
def gpt_response(text_prompt, system_prompt=""):
    """
    Gemini로 텍스트 응답 생성. 기존 함수명 유지.
    """
    prompt = [text_prompt]
    if system_prompt:
        prompt = [system_prompt, text_prompt]
    resp = _MODEL.generate_content(
        prompt,
        generation_config={"max_output_tokens": 1000},
    )
    return getattr(resp, "text", "").strip()

# -----------------------------
# 텍스트 + 이미지(멀티모달) 응답
# -----------------------------
def gptv_response(text_prompt, image_prompt, system_prompt=""):
    """
    Gemini로 멀티모달 응답 생성. 기존 함수명 유지.
    image_prompt: 이미지 경로(str) 또는 OpenCV의 np.ndarray
    """
    image_bytes = _image_to_bytes(image_prompt)
    prompt = [text_prompt, Part.from_data(data=image_bytes, mime_type="image/jpeg")]
    if system_prompt:
        prompt.insert(0, system_prompt)
    resp = _MODEL.generate_content(
        prompt,
        generation_config={"max_output_tokens": 1000},
    )
    return getattr(resp, "text", "").strip()
