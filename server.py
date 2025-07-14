from fastapi import FastAPI
from pydantic import BaseModel
from agents.interview_simulator import InterviewSimulator
from agents.qualitative_assessor import QualitativeAssessor
from agents.quantitative_assessor import QuantitativeAssessor
from agents.meta_reviewer import MetaReviewerAgent

app = FastAPI()

interview_agent = InterviewSimulator()
qualitative_assessor = QualitativeAssessor()
quantitative_assessor = QuantitativeAssessor()
meta_reviewer = MetaReviewerAgent()

class InterviewRequest(BaseModel):
    topic: str

@app.post("/full_pipeline")
def run_full_pipeline(request: InterviewRequest):
    # 1. Simulate interview
    conversation = interview_agent.simulate(request.topic)

    # 2. Run qualitative assessment
    qualitative_result = qualitative_assessor.assess(conversation)

    # 3. Run quantitative assessment
    quantitative_result = quantitative_assessor.assess(conversation)

    # 4. Run meta-review
    final_review = meta_reviewer.review(
        interview=conversation,
        qualitative=qualitative_result,
        quantitative=quantitative_result
    )

    return {
        "conversation": conversation,
        "qualitative": qualitative_result,
        "quantitative": quantitative_result,
        "meta_review": final_review
    }
