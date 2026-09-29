"""Multi-agent pipeline built with LangGraph.

Three agents run in sequence, each as a graph node:

    retriever_agent  ->  fraud_scorer_agent  ->  summarizer_agent

The retriever agent finds similar historical claims (RAG context), the
fraud scorer agent evaluates deterministic rules + anomaly signals, and
the summarizer agent produces the final case summary.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .corpus import claim_text, get_claim, load_claims
from .fraud_rules import score_claim
from .retrieval import ClaimRetriever
from .summarizer import format_summary

_corpus: list[dict] | None = None
_retriever: ClaimRetriever | None = None


def get_corpus() -> list[dict]:
    global _corpus
    if _corpus is None:
        _corpus = load_claims()
    return _corpus


def get_retriever() -> ClaimRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ClaimRetriever(get_corpus())
    return _retriever


class PipelineState(TypedDict, total=False):
    claim_id: str | None
    claim_text: str | None
    claim: dict | None
    similar_claims: list[dict]
    fraud: dict | None
    summary: str | None
    error: str | None


def retriever_agent(state: PipelineState) -> PipelineState:
    """Agent 1: retrieve similar claims to ground the analysis (RAG)."""
    retriever = get_retriever()
    if state.get("claim_id"):
        similar = retriever.similar_to_claim(state["claim_id"], top_k=5)
    else:
        similar = retriever.search(state.get("claim_text") or "", top_k=5)
    return {"similar_claims": similar}


def fraud_scorer_agent(state: PipelineState) -> PipelineState:
    """Agent 2: score the claim with deterministic rules + anomaly model."""
    fraud = score_claim(state["claim"], get_corpus())
    return {"fraud": fraud}


def summarizer_agent(state: PipelineState) -> PipelineState:
    """Agent 3: write the final case summary for the investigator."""
    summary = format_summary(
        state["claim"], state["fraud"], state.get("similar_claims", [])
    )
    return {"summary": summary}


def resolve_claim(state: PipelineState) -> PipelineState:
    """Look up the corpus claim, or build an ad-hoc claim from free text."""
    if state.get("claim_id"):
        claim = get_claim(state["claim_id"], get_corpus())
        if claim is None:
            return {"error": f"Unknown claim id: {state['claim_id']}"}
        return {"claim": claim}
    text = state.get("claim_text") or ""
    claim = {
        "claim_id": "QUERY",
        "claimant_name": "Unknown",
        "claim_type": "unknown",
        "amount": 0.0,
        "claim_date": "2024-01-01",
        "policy_start": "2023-01-01",
        "policy_id": "n/a",
        "description": text,
        "line_items": [],
    }
    return {"claim": claim}


def _route_after_resolve(state: PipelineState) -> str:
    return "error" if state.get("error") else "retriever"


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("resolve", resolve_claim)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("fraud_scorer", fraud_scorer_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", _route_after_resolve, {"retriever": "retriever", "error": END})
    graph.add_edge("retriever", "fraud_scorer")
    graph.add_edge("fraud_scorer", "summarizer")
    graph.add_edge("summarizer", END)
    return graph.compile()


_pipeline = None


def run_pipeline(
    claim_id: str | None = None, claim_text: str | None = None, top_k: int = 5
) -> dict:
    """Run the full agent pipeline for a claim id or free-text claim.

    Returns a dict with claim, similar_claims, fraud and summary keys
    (or an 'error' key when the claim id is unknown).
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    if not claim_id and not claim_text:
        return {"error": "Provide claim_id or claim_text."}
    state: PipelineState = {"claim_id": claim_id, "claim_text": claim_text}
    result = _pipeline.invoke(state)
    if top_k != 5 and result.get("similar_claims"):
        result["similar_claims"] = result["similar_claims"][:top_k]
    return result
