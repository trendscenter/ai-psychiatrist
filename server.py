# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.interview_simulator import InterviewSimulator
from agents.interview_evaluator import InterviewEvaluatorAgent
from agents.quantitative_assessor_f import QuantitativeAssessor as QuantitativeAssessorF
from agents.qualitative_assessor_f import QualitativeAssessor
from agents.meta_reviewer import MetaReviewerAgent
from agents.qualitive_evaluator import QualitativeEvaluatorAgent
from agents.quantitative_assessor_z import QuantitativeAssessorZ

app = FastAPI(title="AI Psychiatrist Pipeline", version="1.2.0")

# --- Shared agents ---
interview_loader = InterviewSimulator()  # reads fixed file only
interview_evaluator = InterviewEvaluatorAgent()
qualitative_assessor = QualitativeAssessor()
qualitative_evaluator = QualitativeEvaluatorAgent()
meta_reviewer = MetaReviewerAgent()

# --- Quantitative variants ---
quantitative_assessor_F = QuantitativeAssessorF()  # few-shot
quantitative_assessor_Z = QuantitativeAssessorZ()  # zero-shot

class InterviewRequest(BaseModel):
    mode: int = Field(0, ge=0, le=1, description="0=zero-shot (Z), 1=few-shot (F) for quantitative assessor ONLY")

@app.post("/full_pipeline")
def run_full_pipeline(request: InterviewRequest):
    # 0) Always load the fixed transcript (no user-provided text/path)
    try:
        conversation = interview_loader.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcript load error: {e}")

    # 1) Pick quantitative assessor by mode
    if request.mode == 0:
        quantitative_assessor = quantitative_assessor_Z
        quantitative_variant = "Z"
    else:
        quantitative_assessor = quantitative_assessor_F
        quantitative_variant = "F"

    # 2) Quantitative PHQ scoring
    quantitative_result = quantitative_assessor.assess(conversation)

    # 3) Qualitative scoring
    qualitative_result = qualitative_assessor.assess(conversation)

    # 4) Qualitative evaluation
    qualitative_evaluation = qualitative_evaluator.assess(conversation, qualitative_result)

    # 5) Meta review
    final_review = meta_reviewer.review(
        interview=conversation,
        quantitative=quantitative_result,
        qualitative=qualitative_result
    )

    return {
        "mode": request.mode,
        "quantitative_variant": quantitative_variant,
        "conversation": conversation,
        "qualitative_result": qualitative_result,
        "quantitative_evaluation": qualitative_evaluation,
        "quantitative_score": quantitative_result,
        "meta_review": final_review
    }
