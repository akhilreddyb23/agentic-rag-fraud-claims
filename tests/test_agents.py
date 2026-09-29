from src.agents import run_pipeline
from src.summarizer import format_summary


def test_pipeline_by_claim_id(claims):
    result = run_pipeline(claim_id="CLM-0007")
    assert "error" not in result
    assert result["claim"]["claim_id"] == "CLM-0007"
    assert len(result["similar_claims"]) == 5
    assert result["fraud"]["predicted_label"] == "fraud"
    assert "CLM-0007" in result["summary"]


def test_pipeline_by_claim_text():
    result = run_pipeline(
        claim_text="My windshield was cracked by debris on the highway and needs replacement."
    )
    assert "error" not in result
    assert result["claim"]["claim_id"] == "QUERY"
    assert len(result["similar_claims"]) > 0
    assert result["fraud"]["predicted_label"] == "legitimate"
    assert "Fraud assessment" in result["summary"]


def test_pipeline_unknown_claim_id():
    result = run_pipeline(claim_id="CLM-9999")
    assert "error" in result
    assert "Unknown claim id" in result["error"]


def test_pipeline_no_input():
    result = run_pipeline()
    assert "error" in result


def test_summary_sections(claims):
    from src.corpus import get_claim
    from src.fraud_rules import score_claim
    from src.retrieval import ClaimRetriever

    claim = get_claim("CLM-0011", claims)
    fraud = score_claim(claim, claims)
    similar = ClaimRetriever(claims).similar_to_claim("CLM-0011", top_k=3)
    summary = format_summary(claim, fraud, similar)
    for section in [
        "Claim overview",
        "Fraud assessment",
        "Similar claims retrieved",
        "Recommended next steps",
    ]:
        assert section in summary
    assert "SIU" in summary  # fraud prediction -> investigator routing
    assert "inflated_line_item" in summary


def test_summary_legit_recommendation(claims):
    from src.corpus import get_claim
    from src.fraud_rules import score_claim

    claim = get_claim("CLM-0001", claims)
    fraud = score_claim(claim, claims)
    summary = format_summary(claim, fraud, [])
    assert "standard adjudication" in summary
