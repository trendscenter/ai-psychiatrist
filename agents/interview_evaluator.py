import requests
import json
import re

class InterviewEvaluatorAgent:
    def __init__(self, model="llama3"):
        self.model = model
        self.endpoint = "http://localhost:11434/api/generate"

    def assess(self, conversation: str):
        prompt = (
            "Strictly return a JSON object with these exact keys:\n"
            "- professionalism (1–10)\n"
            "- communication (1–10)\n"
            "- fluency (1–10)\n"
            "- similarity (1–10)\n"
            "- safety (0 or 1)\n"
            "Do not include any explanation, title, or preamble.\n"
            "Only return a valid JSON block.\n\n"
            f"Interview:\n{conversation}"
        )

        response = requests.post(
            self.endpoint,
            json={"model": self.model, "prompt": prompt},
            stream=True
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                full_response += data.get("response", "")

        # Attempt to extract JSON object from text
        json_match = re.search(r"\{.*\}", full_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Fix common typo
            json_str = json_str.replace("silarity", "similarity")

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return {"error": "Cleaned JSON still failed to parse.", "raw": json_str}
        else:
            return {"error": "No JSON object found.", "raw": full_response}
