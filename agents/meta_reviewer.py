from ollama import Client

class MetaReviewerAgent:
    def __init__(self):
        self.client = Client(host="http://localhost:11434")

    def review(self, interview: str, qualitative: str, quantitative: str) -> str:
        prompt = f"""
You are an AI meta-reviewer psychiatrist.

Please integrate the information below and produce a final diagnostic suggestion.

<interview>
{interview}
</interview>

<qualitative>
{qualitative}
</qualitative>

<quantitative>
{quantitative}
</quantitative>

Provide a concise diagnostic recommendation in <diagnosis> tags, plus any additional considerations in <considerations> tags.
"""
        response = self.client.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']
