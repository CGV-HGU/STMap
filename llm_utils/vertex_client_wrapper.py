import json

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

DEFAULT_PROJECT_ID = "dogwood-method-480911-p3"
DEFAULT_LOCATION = "us-central1"
DEFAULT_ENDPOINT_ID = "5960655394069020672"


class VertexAIClient:
    def __init__(self, project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION, endpoint_id=DEFAULT_ENDPOINT_ID):
        self.project = project
        self.location = location
        self.endpoint_id = endpoint_id

        # Uses ADC (gcloud auth application-default login).
        vertexai.init(project=self.project, location=self.location)

        resource_name = f"projects/{self.project}/locations/{self.location}/endpoints/{self.endpoint_id}"
        self.model = GenerativeModel(resource_name)

    def analyze_scene(self, image_bytes):
        """
        Returns a dict:
        {
            "scene_type": "kitchen",
            "description": "A modern kitchen with a large island",
            "objects_to_look_for": ["fridge", "oven", "sink"]
        }
        """
        prompt_text = (
            "Analyze this image for robot navigation. Return a JSON object with these fields:\n"
            "1. \"scene_type\": the room type (kitchen, bedroom, corridor, living_room, etc).\n"
            "2. \"description\": short, distinctive caption focusing on permanent features.\n"
            "3. \"objects_to_look_for\": list of 3-5 key objects likely in this room.\n"
            "Rules: use \"corridor\" for transition areas. Be concise. Output raw JSON only."
        )

        try:
            prompt = [
                prompt_text,
                Part.from_data(data=image_bytes, mime_type="image/jpeg"),
            ]
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            data = json.loads(raw_text)
            if "scene_type" not in data:
                data["scene_type"] = "unknown"
            if "description" not in data:
                data["description"] = ""
            if "objects_to_look_for" not in data:
                data["objects_to_look_for"] = []
            return data
        except Exception:
            return {
                "scene_type": "unknown",
                "description": "",
                "objects_to_look_for": [],
            }
