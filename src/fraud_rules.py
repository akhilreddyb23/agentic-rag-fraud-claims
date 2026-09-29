"""Deterministic fraud rules + statistical anomaly scoring for claims.

The score is a weighted sum of rule hits plus an amount-anomaly bonus,
capped at 100. A claim with score >= FRAUD_THRESHOLD is flagged as
likely fraud. Every point is explainable via the returned reasons list.
"""

import statistics

from .corpus import parse_date

FRAUD_THRESHOLD = 40

# Deterministic rule weights (points).
W_DUPLICATE_AMOUNT = 35
W_EARLY_CLAIM = 30
W_INFLATED_ITEM = 30
W_HIGH_FREQUENCY = 25
W_ROUND_AMOUNT = 10
W_ANOMALY_STRONG = 15  # |z| >= 2.0
W_ANOMALY_MILD = 10    # 1.5 <= |z| < 2.0

EARLY_CLAIM_DAYS = 14
DUPLICATE_WINDOW_DAYS = 90
FREQUENCY_WINDOW_DAYS = 60
FREQUENCY_MIN_CLAIMS = 3
INFLATED_MULTIPLE = 3.0

# Typical (reference) costs for common line items, in USD.
# A line item whose cost exceeds INFLATED_MULTIPLE x reference is flagged.
REFERENCE_ITEM_COSTS = {
    "windshield replacement": 350.0,
    "window glass replacement": 400.0,
    "bumper repair": 800.0,
    "paint job": 600.0,
    "stereo replacement": 700.0,
    "water extraction": 1800.0,
    "drywall repair": 1200.0,
    "plumbing repair": 900.0,
    "water heater replacement": 1400.0,
    "roof shingle repair": 2500.0,
    "tree removal": 1100.0,
    "mold remediation": 3500.0,
    "cabinet replacement": 7500.0,
    "appliance replacement": 8000.0,
    "laptop replacement": 1400.0,
    "camera replacement": 850.0,
    "office visit": 180.0,
    "specialist consultation": 350.0,
    "lab panel": 400.0,
    "urgent care visit": 350.0,
    "physical therapy session": 150.0,
    "mri scan": 900.0,
    "dental crown": 1100.0,
    "prescription": 60.0,
    "hotel stay": 220.0,
    "flight rebooking": 400.0,
    "towing": 180.0,
}


def _norm(name: str) -> str:
    return name.strip().lower()


def check_duplicate_amount(claim: dict, corpus: list[dict]) -> tuple[bool, str]:
    """Same claimant filed another claim for the exact same amount recently."""
    claimant = claim.get("claimant_name")
    amount = claim.get("amount")
    claim_date = parse_date(claim["claim_date"])
    for other in corpus:
        if other["claim_id"] == claim.get("claim_id"):
            continue
        if other.get("claimant_name") != claimant:
            continue
        if other.get("amount") != amount:
            continue
        delta = abs((parse_date(other["claim_date"]) - claim_date).days)
        if delta <= DUPLICATE_WINDOW_DAYS:
            return True, (
                f"Duplicate amount ${amount:,.0f} also claimed by {claimant} "
                f"in {other['claim_id']} ({delta} days apart)"
            )
    return False, ""


def check_early_claim(claim: dict) -> tuple[bool, str]:
    """Claim filed very shortly after the policy started."""
    delta = (parse_date(claim["claim_date"]) - parse_date(claim["policy_start"])).days
    if delta <= EARLY_CLAIM_DAYS:
        return True, (
            f"Claim filed {delta} days after policy start "
            f"({claim['policy_start']} -> {claim['claim_date']})"
        )
    return False, ""


def check_inflated_items(claim: dict) -> list[tuple[str, float, float]]:
    """Line items costing far more than the reference cost.

    Returns a list of (item_name, cost, reference_cost).
    """
    hits = []
    for li in claim.get("line_items", []):
        ref = REFERENCE_ITEM_COSTS.get(_norm(li["item"]))
        if ref is None:
            continue
        if li["cost"] > INFLATED_MULTIPLE * ref:
            hits.append((li["item"], li["cost"], ref))
    return hits


def check_high_frequency(claim: dict, corpus: list[dict]) -> tuple[bool, str]:
    """Claimant filed >= FREQUENCY_MIN_CLAIMS claims in a 60-day window."""
    claimant = claim.get("claimant_name")
    claim_date = parse_date(claim["claim_date"])
    count = 0
    for other in corpus:
        if other.get("claimant_name") != claimant:
            continue
        delta = abs((parse_date(other["claim_date"]) - claim_date).days)
        if delta <= FREQUENCY_WINDOW_DAYS:
            count += 1
    if count >= FREQUENCY_MIN_CLAIMS:
        return True, (
            f"{claimant} filed {count} claims within a "
            f"{FREQUENCY_WINDOW_DAYS}-day window"
        )
    return False, ""


def check_round_amount(claim: dict) -> tuple[bool, str]:
    """Suspiciously round claim amount (>= $1,000, exact multiple of $100)."""
    amount = claim.get("amount", 0)
    if amount >= 1000 and amount % 100 == 0:
        return True, f"Round claim amount ${amount:,.0f}"
    return False, ""


def amount_zscore(claim: dict, corpus: list[dict]) -> float:
    """Z-score of the claim amount vs. other claims of the same type."""
    peers = [
        c["amount"]
        for c in corpus
        if c.get("claim_type") == claim.get("claim_type")
        and c["claim_id"] != claim.get("claim_id")
    ]
    if len(peers) < 2:
        return 0.0
    mean = statistics.fmean(peers)
    stdev = statistics.pstdev(peers)
    if stdev == 0:
        return 0.0
    return (claim.get("amount", 0) - mean) / stdev


def score_claim(claim: dict, corpus: list[dict]) -> dict:
    """Score a claim for fraud risk.

    Returns a dict with score (0-100), band, predicted label ('fraud' /
    'legitimate'), and a list of reasons explaining the score.
    """
    reasons: list[dict] = []
    score = 0.0

    hit, detail = check_duplicate_amount(claim, corpus)
    if hit:
        score += W_DUPLICATE_AMOUNT
        reasons.append({"rule": "duplicate_amount", "points": W_DUPLICATE_AMOUNT, "detail": detail})

    hit, detail = check_early_claim(claim)
    if hit:
        score += W_EARLY_CLAIM
        reasons.append({"rule": "early_claim", "points": W_EARLY_CLAIM, "detail": detail})

    for item, cost, ref in check_inflated_items(claim):
        score += W_INFLATED_ITEM
        reasons.append(
            {
                "rule": "inflated_line_item",
                "points": W_INFLATED_ITEM,
                "detail": (
                    f"Line item '{item}' billed at ${cost:,.0f}, "
                    f"{cost / ref:.1f}x the typical ${ref:,.0f}"
                ),
            }
        )

    hit, detail = check_high_frequency(claim, corpus)
    if hit:
        score += W_HIGH_FREQUENCY
        reasons.append({"rule": "high_frequency", "points": W_HIGH_FREQUENCY, "detail": detail})

    hit, detail = check_round_amount(claim)
    if hit:
        score += W_ROUND_AMOUNT
        reasons.append({"rule": "round_amount", "points": W_ROUND_AMOUNT, "detail": detail})

    z = amount_zscore(claim, corpus)
    if abs(z) >= 2.0:
        score += W_ANOMALY_STRONG
        reasons.append(
            {
                "rule": "amount_anomaly",
                "points": W_ANOMALY_STRONG,
                "detail": f"Amount ${claim.get('amount', 0):,.0f} is {abs(z):.1f} std-devs from the {claim.get('claim_type')} mean",
            }
        )
    elif abs(z) >= 1.5:
        score += W_ANOMALY_MILD
        reasons.append(
            {
                "rule": "amount_anomaly",
                "points": W_ANOMALY_MILD,
                "detail": f"Amount ${claim.get('amount', 0):,.0f} is {abs(z):.1f} std-devs from the {claim.get('claim_type')} mean",
            }
        )

    score = min(100.0, score)
    if score >= 70:
        band = "high"
    elif score >= FRAUD_THRESHOLD:
        band = "elevated"
    else:
        band = "low"

    return {
        "claim_id": claim.get("claim_id"),
        "score": round(score, 1),
        "band": band,
        "predicted_label": "fraud" if score >= FRAUD_THRESHOLD else "legitimate",
        "threshold": FRAUD_THRESHOLD,
        "amount_zscore": round(z, 2),
        "reasons": reasons,
    }
