import requests
import json

class QualitativeAssessor:
    def __init__(self, model="llama3"):
        self.model = model

    def assess(self, interview_text: str):
        prompt = f"""
            You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible. Be specific and avoid vague generalities. Use exact quotes from the transcript to support your assessment for each tag.
            
            The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
            {interview_text}
            
               Please:
              1. Provide an overall qualitative assessment of the subject's mental health.
              2. Summarize PHQ-8 symptoms if available (if not, state 'not available'), as well as frequency and severity if available.
              3. Summarize social aspects that may influence the subject's mental health. (for example, familial relationships, frienship dynamics, work environment, etc. that are relevant to the subjects mental health)
              4. Summarize biological aspects that may influence the subject's mental health. (for example, famillial history of mental health issues, previous or pre-existing mental health issues, stress levels, etc. that are relevant to the subjects mental health)
              5. Identify potential risk factors the subject may be experiencing.
              6. Use exact quotes from the transcript to support your assessment for each tag.
            
              Output should be formatted as bullet points with headings for each section using stars. Example: **Tiredness** <explanation of tiredness>. Do not include any additional text outside the bullet points
              Please answer in this XML format with each tag on a new line, properly indented. Use straight quotes instead of curly quotes, and do not include any additional text outside the XML tags:
            
              <assessment>
                <!-- Summary of participant's overall mental health -->
               <exact_quotes>
                <!-- Quotes from the transcript that support the assessment -->
                </exact_quotes>
              </assessment>
            
              <PHQ8_symptoms>
                <!-- Summary of PHQ-8 symptoms mentioned in the trancript:
                - Little interest or pleasure in doing things
                - Feeling down, depressed, or hopeless
                - Trouble falling or staying asleep, or sleeping too much
                - Feeling tired or having little energy
                - Poor appetite or overeating
                - Feeling bad about yourself — or that you are a failure or have let yourself or your family down
                - Trouble concentrating on things, such as reading the newspaper or watching television
                - Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual
            
                 For each symptom present, note:
                - Frequency (not at all, several days, more than half the days, nearly every day)
                - Duration (how long experienced)
                - Severity/impact on functioning
            
               If symptoms are not discussed, state "not assessed in interview" -->
            
               <little_interest_or_pleasure>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
               </little_interest or pleasure>
            
                <feeling_down_depressed_hopeless>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </feeling_down_depressed_hopeless>
            
                <trouble_sleeping>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </trouble_sleeping>
            
                <feeling_tired_little_energy>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </feeling_tired_little_energy>
            
                <poor_appetite_overeating>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </poor_appetite_overeating>
            
                <feeling_bad_about_self>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </feeling_bad_about_self>
            
                <trouble_concentrating>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </trouble_concentrating>
            
                <moving_speaking_slowly_or_fidgety>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </moving_speaking_slowly_or_fidgety>
            
            
               <exact_quotes>
                <!-- Quotes from the transcript that support the assessment -->
                </exact_quotes>
              </PHQ8_symptoms>
            
              <social_factors>
                <!-- Summary of social influences on patient's health -->
                <exact_quotes>
              </social_factors>
            
              <biological_factors>
                <!-- Summary of biological influences on patient's health -->
               <exact_quotes>
                <!-- Quotes from the transcript that support the assessment -->
                </exact_quotes>
              </biological_factors>
            
              <risk_factors>
                <!-- Summary of potential risk factors -->
                 <exact_quotes>
               <!-- Quotes from the transcript that support the assessment -->
               </exact_quotes>
              </risk_factors>
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
