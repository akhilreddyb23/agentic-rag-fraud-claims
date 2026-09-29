from src.corpus import get_claim
from src.fraud_rules import (
    FRAUD_THRESHOLD,
    amount_zscore,
    check_duplicate_amount,
    check_early_claim,
    check_high_frequency,
    check_inflated_items,
    check_round_amount,
    score_claim,
)


def test_duplicate_amount_rule_fires(claims):
    claim = get_claim("CLM-0007", claims)
    hit, detail = check_duplicate_amount(claim, claims)
    assert hit
    assert "CLM-0006" in detail


def test_duplicate_amount_rule_quiet_for_legit(claims):
    claim = get_claim("CLM-0001", claims)
    hit, _ = check_duplicate_amount(claim, claims)
    assert not hit


def test_early_claim_rule_fires(claims):
    claim = get_claim("CLM-0011", claims)
    hit, detail = check_early_claim(claim)
    assert hit
    assert "5 days" in detail


def test_inflated_item_rule_fires(claims):
    claim = get_claim("CLM-0018", claims)
    hits = check_inflated_items(claim)
    assert len(hits) == 1
    item, cost, ref = hits[0]
    assert item == "office visit"
    assert cost == 1850.0 and ref == 180.0


def test_inflated_item_rule_quiet_for_legit(claims):
    claim = get_claim("CLM-0009", claims)
    assert check_inflated_items(claim) == []


def test_high_frequency_rule_fires(claims):
    claim = get_claim("CLM-0012", claims)
    hit, detail = check_high_frequency(claim, claims)
    assert hit
    assert "3 claims" in detail


def test_high_frequency_rule_quiet_for_legit(claims):
    claim = get_claim("CLM-0001", claims)
    hit, _ = check_high_frequency(claim, claims)
    assert not hit


def test_round_amount_rule(claims):
    hit, _ = check_round_amount(get_claim("CLM-0007", claims))
    assert hit
    hit, _ = check_round_amount(get_claim("CLM-0001", claims))
    assert not hit


def test_fraud_claim_scores_above_threshold(claims):
    for cid in ["CLM-0006", "CLM-0007", "CLM-0011", "CLM-0012", "CLM-0013",
                "CLM-0014", "CLM-0017", "CLM-0018", "CLM-0022"]:
        result = score_claim(get_claim(cid, claims), claims)
        assert result["score"] >= FRAUD_THRESHOLD, f"{cid} scored {result['score']}"
        assert result["predicted_label"] == "fraud"
        assert len(result["reasons"]) > 0


def test_legit_claim_scores_below_threshold(claims):
    for cid in ["CLM-0001", "CLM-0005", "CLM-0009", "CLM-0015", "CLM-0023"]:
        result = score_claim(get_claim(cid, claims), claims)
        assert result["score"] < FRAUD_THRESHOLD, f"{cid} scored {result['score']}"
        assert result["predicted_label"] == "legitimate"


def test_score_is_capped_and_explained(claims):
    claim = get_claim("CLM-0014", claims)
    result = score_claim(claim, claims)
    assert 0 <= result["score"] <= 100
    rules = {r["rule"] for r in result["reasons"]}
    assert {"inflated_line_item", "high_frequency", "round_amount"} <= rules
    assert result["band"] in {"low", "elevated", "high"}


def test_anomaly_zscore_direction(claims):
    # The $7,500 travel claim is far above its peers -> large positive z.
    z = amount_zscore(get_claim("CLM-0022", claims), claims)
    assert z > 1.5
    # A typical travel claim sits near the peer mean.
    z2 = amount_zscore(get_claim("CLM-0024", claims), claims)
    assert abs(z2) < 1.5
