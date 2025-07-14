import requests
import json

class QualitativeAssessor:
    def __init__(self, model="llama3"):
        self.model = model

    def assess(self, interview_text: str):
        prompt = f"""
Please analyze the following interview transcript and assess the patient's mental health status, focusing on signs of depression, from social and biological aspects. 

Please provide your assessment in a structured format, including the following sections:
1. Overall Impression in <overall> tags: A brief summary of the patient's mental health status.
2. Social Aspects in <social> tags: Observations related to the patient's interpersonal relationships, family situation, and any relevant social interactions that may influence their mental health.
3. Biological Aspects in <biological> tags: Observations related to the patient's physical health, including sleep quality, appetite, physical activity, stress level, and any other biological factors that may influence their mental health.
4. Additional Notes in <notes> tags: Any other relevant observations or comments that do not fit into the above categories.
5. Potential Risk Factors in <risks> tags: Any identified risk factors for depression or other mental health issues.

Here is the interview transcript:
{interview_text}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt.strip()
            },
            stream=True
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = line.decode("utf-8")
                data = json.loads(chunk)
                piece = data.get("response", "")
                full_response += piece

        return full_response
