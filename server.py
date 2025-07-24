from fastapi import FastAPI
from pydantic import BaseModel
from agents.interview_simulator import InterviewSimulator
from agents.interview_evaluator import InterviewEvaluatorAgent
from agents.quantitative_assessor import QuantitativeAssessor
from agents.qualitative_assessor import QualitativeAssessor
from agents.meta_reviewer import MetaReviewerAgent
from agents.qualitive_evaluator import QualitativeEvaluatorAgent

app = FastAPI()

# Initialize all agents
interview_agent = InterviewSimulator()
interview_evaluator = InterviewEvaluatorAgent()
quantitative_assessor = QuantitativeAssessor()
meta_reviewer = MetaReviewerAgent()
qualitative_assessor = QualitativeAssessor()
qualitative_evaluator = QualitativeEvaluatorAgent()

# Request schema (mode=0 for zero-shot, mode=1 for few-shot)
class InterviewRequest(BaseModel):
    mode: int = 0

@app.post("/full_pipeline")
def run_full_pipeline(request: InterviewRequest):
    # 1. Simulate interview on depression
    conversation = interview_agent.simulate(mode=request.mode)

    # 2. Evaluate conversation qualitatively
    interview_evaluation = interview_evaluator.assess(conversation)

    # 3. Quantitative PHQ scoring
    quantitative_result = quantitative_assessor.assess(conversation)

    # 3. qualitative scoring
    qualitative_result = qualitative_assessor.assess(conversation)

    qualitative_evaluation = qualitative_evaluator.assess(qualitative_result)

    # 4. Meta-review aggregates all agent outputs
    final_review = meta_reviewer.review(
        interview=conversation,
        quantitative=quantitative_result,
        qualitative=qualitative_result

    )

    return {
        "conversation": conversation,
        "interview_evaluation": interview_evaluation,
        'qualitative_result': qualitative_result,
        "quantitative_score": quantitative_result,
        "quantitative_evaluation": qualitative_evaluation,
        "meta_review": final_review
    }
