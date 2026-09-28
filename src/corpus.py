"""Corpus loading for the synthetic insurance-claim dataset.

All data is synthetic and generated for demonstration purposes only.
"""

import json
from datetime import date
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "claims.json"


def load_claims(path: str | Path = DATA_PATH) -> list[dict]:
    """Load the synthetic claim corpus from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["claims"]


def get_claim(claim_id: str, claims: list[dict]) -> dict | None:
    """Return the claim with the given id, or None."""
    for claim in claims:
        if claim["claim_id"] == claim_id:
            return claim
    return None


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def claim_text(claim: dict) -> str:
    """Text representation of a claim used for retrieval indexing.

    The claim type is repeated to give same-type claims a relevance boost,
    since type is a strong similarity signal in this corpus.
    """
    items = " ".join(li["item"] for li in claim.get("line_items", []))
    ctype = claim.get("claim_type", "")
    return (
        f"{ctype} {ctype} {ctype} claim. {claim.get('description', '')} "
        f"Line items: {items}."
    )
