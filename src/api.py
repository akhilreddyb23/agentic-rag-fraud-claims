"""FastAPI demo endpoint for the agentic RAG fraud & claims pipeline."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agents import run_pipeline

app = FastAPI(
    title="Agentic RAG: Insurance Fraud & Claims Intelligence (demo)",
    description=(
        "Portfolio demo. Runs a 3-agent LangGraph pipeline (retriever, "
        "fraud scorer, summarizer) over a synthetic claim corpus. "
        "All data is fake."
    ),
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    claim_id: str | None = Field(
        default=None, description="Corpus claim id, e.g. 'CLM-0007'"
    )
    claim_text: str | None = Field(
        default=None, description="Free-text claim description to analyze"
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Similar claims to return")


class SimilarClaim(BaseModel):
    claim_id: str
    claim_type: str
    amount: float
    similarity: float
    description: str


class FraudReason(BaseModel):
    rule: str
    points: float
    detail: str


class FraudResult(BaseModel):
    score: float
    band: str
    predicted_label: str
    threshold: float
    amount_zscore: float
    reasons: list[FraudReason]


class AnalyzeResponse(BaseModel):
    claim_id: str
    similar_claims: list[SimilarClaim]
    fraud: FraudResult
    summary: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    if not request.claim_id and not request.claim_text:
        raise HTTPException(
            status_code=422, detail="Provide claim_id or claim_text."
        )
    result = run_pipeline(
        claim_id=request.claim_id, claim_text=request.claim_text, top_k=request.top_k
    )
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])

    claim = result["claim"]
    similar = [
        SimilarClaim(
            claim_id=e["claim"]["claim_id"],
            claim_type=e["claim"]["claim_type"],
            amount=e["claim"]["amount"],
            similarity=round(e["score"], 4),
            description=e["claim"]["description"],
        )
        for e in result.get("similar_claims", [])
    ]
    fraud = FraudResult(**result["fraud"])
    return AnalyzeResponse(
        claim_id=claim.get("claim_id", "QUERY"),
        similar_claims=similar,
        fraud=fraud,
        summary=result["summary"],
    )
