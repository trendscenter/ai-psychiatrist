import requests
import json
import re

class QualitativeEvaluatorAgent:
    def __init__(self, model="llama3"):
        self.model = model
        self.endpoint = "http://localhost:11434/api/generate"

    def assess(self, qualitative_output: str):
        prompt = (
            "Evaluate the following qualitative assessment output using the following metrics.\n\n"
            "Content Quality Metrics:\n"
            "1. coherence (1–10): Is the response logically consistent?\n"
            "2. completeness (1–10): Does the assessment cover all relevant symptoms, severities, duration/frequency?\n"
            "3. specificity (1–10): Does it avoid vague/generic statements like 'the patient seems depressed'?\n\n"
            "Clinical Validity Metrics:\n"
            "4. accuracy (1–10): Are the signs/symptoms aligned with DSM-5 or PHQ-8? Are there factual inconsistencies?\n\n"
            "Strictly return a valid JSON object with the keys: coherence, completeness, specificity, accuracy.\n"
            "Do not include any explanations, just the JSON.\n\n"
            f"Qualitative Output:\n{qualitative_output}"
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

        json_match = re.search(r"\{.*\}", full_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return {"error": "Malformed JSON", "raw": json_match.group(0)}
        else:
            return {"error": "No JSON object found", "raw": full_response}
