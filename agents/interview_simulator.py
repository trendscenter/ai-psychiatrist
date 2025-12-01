import requests
import json

class InterviewSimulator:
    def __init__(self, model="llama3"):
        self.model = model

    def simulate(self, topic: str = "depression", mode: int = 1):
        if mode == 1:
            # Few-shot example using real transcript sample
            example = (
                "Psychiatrist: Hello, Alison Wells, do you want to come and have a seat? Hi I’m Dr Taylor, one of the GPs here. What would you like me to call you?\n"
                "Patient: Alison will be fine.\n"
                "Psychiatrist: Ok, so what’s brought you here today, Alison?\n"
                "Patient: My sister's noticed I'm just a bit fed up really. She said I should come.\n"
                "Psychiatrist: Has this been going on for some time?\n"
                "Patient: Yeah, a few months really.\n"
                "Psychiatrist: Do you want to tell me a bit more about what’s been going on?\n"
                "Patient: Just things piling up. I just don’t seem to be coping. The kids and things.\n"
                "Psychiatrist: Would it be OK for me to ask you a few more detailed questions about how you’ve been feeling?\n"
                "Patient: Uhuh.\n"
                "Psychiatrist: How have you been feeling in yourself?\n"
                "Patient: A bit fed up. I get up in the morning and everything seems very black.\n"
                "Psychiatrist: Do you feel very miserable?\n"
                "Patient: Fed up, miserable.\n"
                "Psychiatrist: What about tearfulness?\n"
                "Patient: I dropped sugar the other day and burst into tears.\n"
                "Psychiatrist: And how is your energy?\n"
                "Patient: I used to do a lot with the kids, but now I just spend the day on the sofa.\n"
                "Psychiatrist: Ok.\n\n"
            )
        else:
            example = ""

        prompt = (
            f"{example}"
            f"Now simulate an interview related to depression based on the topic: {topic}.\n"
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
