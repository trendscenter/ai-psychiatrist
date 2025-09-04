from ollama import Client

class MetaReviewerAgent:
    def __init__(self):
        self.client = Client(host="http://localhost:11434")

    def review(self, interview: str, qualitative: str, quantitative: str) -> str:
        prompt = f'''You are an AI psychiatrist assistant specializing in depression. Your task is to analyze the interview transcript and related qualitative and quantitative assessments, and then predict the diagnosis label and severity level.
        
                Please review a participant's interview transcript, qualitative assessment, and quantitative assessment below.  

                Here is the interview transcript in <transcript> tags:
                <transcript>
                {interview}
                </transcript>

                Here are the overall assessment, social and biological risk factors that may influence the participant's mental health in <qualitative_assessment> tags:
                <qualitative_assessment>
                {qualitative}
                </qualitative_assessment>

                Here are the predicted PHQ-8 scores and explanations based on the available information in the interview transcript in <quantitative_assessment> tags:
                <quantitative_assessment>\n{quantitative}</quantitative_assessment>

                According to the Patient Health Questionnaire eight-item depression scale (PHQ-8), a total PHQ-8 score of 0 to 4 represents no significant depressive symptoms; a total score of 5 to 9 represents mild depressive symptoms; 10 to 14, moderate; 15 to 19, moderately severe; and 20 to 24, severe. \
                Note that the interview might not discuss all eight PHQ-8 items, so please try your best to predict the participant's condition based on the available information. \
                Given the interview transcript and assessments above, please predict a binary diagnosis label in <diagnosis> tags where 0 means that the participant has no depressive symptoms and 1 means that the participant has depressive symptoms. \
                Please also predict their severity level in <severity> tags, where 0 means no significant depressive symptoms, 1 means mild symptoms, 2 means moderate symptoms, 3 means moderately severe symptoms, and 4 means severe symptoms. \
                Please explain your predictions in <explanation> tags. Please provide answers in the XML format with each tag on a new line.
                '''
        response = self.client.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']
