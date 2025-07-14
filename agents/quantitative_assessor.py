import requests
import json

class QuantitativeAssessor:
    def __init__(self, model="llama3"):
        self.model = model

    def assess(self, interview_text: str):
        prompt = f"""
Please analyze the following interview transcript, assess the patient's signs of depression, and predict the eight-item Patient Health Questionnaire depression scale (PHQ-8) scores. The PHQ-8 counts the number of days in the past 2 weeks the respondent had experienced a particular depressive symptom, and assigns points (0 to 3) to each category: (1) 0 point: 0 to 1 day ("not at all"); (2) 1 point: 2 to 6 days ("several days"); (3) 2 points: 7 to 11 days ("more than half the days"); (4) 3 points: 12 to 14 days ("nearly every day").

Provide your assessment in a structured format, including the following assessments:
1. A score in <nointerest> tags when the patient reported little interest or pleasure in doing things.
2. A score in <depressed> tags when the patient reported feeling down, depressed, or hopeless.
3. A score in <sleep> tags when the patient reported trouble falling or staying asleep, or sleeping too much.
4. A score in <tired> tags when the patient reported feeling tired or having little energy.
5. A score in <appetite> tags when the patient reported poor appetite or overeating.
6. A score in <failure> tags when the patient reported feeling bad about themselves, or that they are a failure or have let themselves or their family down.
7. A score in <concentrating> tags when the patient reported trouble concentrating on things, such as reading the newspaper or watching television.
8. A score in <moving> tags when the patient reported moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that they have been moving around a lot more than usual.

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
