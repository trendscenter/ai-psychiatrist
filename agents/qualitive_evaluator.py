import requests
import json
import re
import time

def parse_score_and_explanation(response_text):
    """
    Extract score and explanation from model response
    """
    score_patterns = [
        r'score[:\s]*(\d+)',
        r'(\d+)[/\s]*(?:out of\s*)?10',
        r'(\d+)[/\s]*10',
        r'rating[:\s]*(\d+)',
        r'^(\d+)',  # Number at start of line
    ]

    score = None
    for pattern in score_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            potential_score = int(match.group(1))
            if 1 <= potential_score <= 10:
                score = potential_score
                break

    return score, response_text.strip()


class QualitativeEvaluatorAgent:
    def __init__(self, model="llama3", endpoint="http://localhost:11434/api/chat"):
        self.model = model
        self.endpoint = endpoint

    def assess(self, transcript, qualitative_output: str):
        result = {}

        # metric prompts
        coherence_prompt = f"""Evaluate the following qualitative assessment output for COHERENCE only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.
            
            Coherence (1–5): Is the response logically consistent? Mistakes are logically inconsistent statements or contradictions within the assessment.
            
            Score from 1-5 solely based on the following criteria:
            
             - score of 5 : 0 mistakes
             - score of 4 : 1-2 mistakes
             - score of 3 : 3-4 mistakes
             - score of 2 : 5-6 mistakes
             - score of 1 : 7 or more mistakes
            
            Format your response as:
            Explanation: [Your rating, as text]
            Score: [your score, as a number between 1 and 5]
            
            ---
            Here is the transcript:
            {transcript}
            
            Here is the assessment based on the transcript:
            {qualitative_output}
            ---"""
        completeness_prompt = f"""Evaluate the following qualitative assessment output for COMPLETENESS only. Compare qualitative_assessment to the provided transcript. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

        Completeness (1–5): Does the assessment cover all relevant symptoms, severities, duration/frequency? Mistakes are missed PHQ-8 symptoms, or duration/frequency details.

        Score from 1-5 solely based on the following criteria: 
         - score of 5 : 0 mistakes
         - score of 4 : 1-2 mistakes
         - score of 3 : 3-4 mistakes
         - score of 2 : 5-6 mistakes
         - score of 1 : 7 or more mistakes

        Format your response as:
        Explanation: [Your rating, as text]
        Score: [your score, as a number between 1 and 5]

        Here is the transcript: 
        {transcript}

        Here is the assessment based on the transcript: 
        {qualitative_output}
        ---"""
        specificity_prompt = f"""Evaluate the following qualitative assessment output for SPECIFICITY only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

        Specificity (1–5): Is the assessment specific? Mistakes include using vague/generic statements like 'the patient seems depressed'.

        Score from 1-5 solely based on the following criteria: 
         - score of 5 : 0 mistakes
         - score of 4 : 1-2 mistakes
         - score of 3 : 3-4 mistakes
         - score of 2 : 5-6 mistakes
         - score of 1 : 7 or more mistakes

        Format your response as:
        Explanation: [Your rating, as text]
        Score: [your score, as a number between 1 and 5]

        ---
        Here is the transcript: 
        {transcript}

        Here is the assessment based on the transcript: 
        {qualitative_output}
        ---"""

        accuracy_prompt = f"""Evaluate the following qualitative assessment output for ACCURACY only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

        Accuracy (1–5): Are the signs/symptoms aligned with DSM-5 or PHQ-8? Mistakes are incorrect symptoms or incorrect duration/frequecy. 

        Score from 1-5 solely based on the following criteria: 
         - score of 5 : 0 mistakes
         - score of 4 : 1-2 mistakes
         - score of 3 : 3-4 mistakes
         - score of 2 : 5-6 mistakes
         - score of 1 : 7 or more mistakes

        Format your response as:
        Explanation: [Your rating, as text]
        Score: [your score, as a number between 1 and 5]

        ---
        Here is the transcript: 
        {transcript}

        Here is the assessment based on the transcript: 
        {qualitative_output}
        ---"""

        labels = ['coherence', 'completeness', 'accuracy', 'specificity']
        prompts = {
            "specificity": specificity_prompt,
            "accuracy": accuracy_prompt,
            "coherence": coherence_prompt,
            "completeness": completeness_prompt
        }

        # Build request
        reqs = {}
        print('creating requests....')
        for label in labels:
            print(prompts[label])
            request = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompts[label]}],
                "stream": False,
                "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
            }
            reqs[label] =  request

        responses = []
        contents = []
        result = {}
        for label, request in reqs.items():
            print(f"  Getting {label} response...")
            try:
                    print(request)
                    response = requests.post(self.endpoint, json=request, timeout=60)
                    if response.status_code == 200:
                        responses.append(response)
                        content = response.json()['message']['content']
                        print('content: ')
                        print(content)
                        contents.append(content)
                        score, explanation = parse_score_and_explanation(content)
                        result[label] = score
                        print(f"  {label} score: {score}")
                    else:
                        result[label] = None
                        print(f"  {label} request failed with status:", response.status_code)
            except Exception as e:
                    print(f"  Error during {label} request:", e)
                    result[label] = None


        return result
