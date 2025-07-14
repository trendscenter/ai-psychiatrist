import requests
import json

class InterviewSimulator:
    def __init__(self, model="llama3"):
        self.model = model

    def simulate(self, topic: str):
        prompt = (
            f"You are a psychiatrist interviewing a patient about {topic}.\n"
            f"Generate a realistic, clear, 3-turn conversation:\n"
            f"Psychiatrist: ...\nPatient: ...\nPsychiatrist: ...\nPatient: ...\n"
            f"Include natural follow-up questions."
        )

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt
            },
            stream=True   # ✅ IMPORTANT: Ollama streams multiple JSON lines
        )

        full_response = ""

        for line in response.iter_lines():
            if line:
                chunk = line.decode("utf-8")
                data = json.loads(chunk)  # ✅ SAFE parse
                piece = data.get("response", "")
                full_response += piece

        return full_response
