import requests
import json

class QuantitativeAssessor:
    def __init__(self, model="llama3"):
        self.model = model

    def assess(self, interview_text: str):
        prompt = f"""You are a AI psychiatrist assessment tool specialized in analyzing interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a validated depression screening instrument that assesses eight of the nine DSM-IV criteria for depressive disorders over the past 2 weeks.

        PHQ-8 Scoring System:
        - 0 = Not at all (0-1 days in the past 2 weeks)
        - 1 = Several days (2-6 days)
        - 2 = More than half the days (7-11 days)
        - 3 = Nearly every day (12-14 days)

        Key Principles:
        1. Base all assessments strictly on evidence from the transcript
        2. When no relevant evidence exists for a symptom, output "N/A" rather than assuming absence (score of 0)
        3. Consider both explicit statements and behavioral indicators in the interview
        4. Be conservative - only assign scores when there is clear evidence of frequency
        5. Remember that absence of mention does not necessarily mean absence of symptom
        6. When reference examples are provided, use them for cross-analysis to inform scoring decisions
        7. Weight reference examples based on their similarity to the current evidence - highly similar references should be considered more heavily
        8. Follow the format specified exactly
        
        Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

        Interview transcript:
        <transcript>
        {interview_text}
        </transcript>

        Analyze each symptom using the following approach in <thinking> tags:
        1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
        2. When reference examples are provided, compare the current evidence with similar reference cases
        3. Evaluate the frequency/severity based on available evidence and reference comparisons
        4. Consider how similar the reference examples are to the current evidence - if highly similar, give more weight to the reference scores; if less similar, rely more on direct analysis
        5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
        6. Only assign numeric scores (0-3) when evidence clearly indicates frequency

        After your analysis, provide your final assessment in <answer> tags as a JSON object.

        For each symptom, provide:
        1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
        2. "reason": explanation of scoring decision, including cross-reference analysis when applicable and why N/A if applicable
        3. "score": integer 0-3 based on evidence, or "N/A" if no relevant evidence

        Return ONLY a JSON object in <answer> tags with these exact keys:
        - "PHQ8_NoInterest": {{evidence, reason, score}} for little interest or pleasure in doing things (anhedonia)
        - "PHQ8_Depressed": {{evidence, reason, score}} for feeling down, depressed, or hopeless (depressed mood)
        - "PHQ8_Sleep": {{evidence, reason, score}} for trouble falling or staying asleep, or sleeping too much (sleep problems)
        - "PHQ8_Tired": {{evidence, reason, score}} for feeling tired or having little energy (fatigue)
        - "PHQ8_Appetite": {{evidence, reason, score}} for poor appetite or overeating (appetite/weight changes)
        - "PHQ8_Failure": {{evidence, reason, score}} for feeling bad about yourself or that you are a failure (negative self-perception)
        - "PHQ8_Concentrating": {{evidence, reason, score}} for trouble concentrating on things like reading or watching TV (concentration problems)
        - "PHQ8_Moving": {{evidence, reason, score}} for moving or speaking slowly, or being fidgety/restless (psychomotor changes)"""


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
