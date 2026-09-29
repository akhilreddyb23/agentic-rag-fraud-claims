"""Template-based summarizer agent output.

Produces a structured, human-readable case summary from the claim,
its fraud assessment, and the retrieved similar claims. Deterministic
(no LLM) so the demo runs on any CPU box.
"""


def format_summary(claim: dict, fraud: dict, similar: list[dict]) -> str:
    """Build the agent case summary text."""
    lines = []
    lines.append(f"Case summary for claim {claim.get('claim_id', 'QUERY')}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Claim overview")
    lines.append("-" * 60)
    lines.append(f"  Claimant   : {claim.get('claimant_name', 'Unknown')}")
    lines.append(f"  Type       : {claim.get('claim_type', 'unknown')}")
    lines.append(f"  Amount     : ${claim.get('amount', 0):,.2f}")
    lines.append(f"  Filed      : {claim.get('claim_date', 'unknown')}")
    lines.append(f"  Policy     : {claim.get('policy_id', 'unknown')} "
                 f"(started {claim.get('policy_start', 'unknown')})")
    lines.append(f"  Description: {claim.get('description', 'n/a')}")
    lines.append("")
    lines.append("Fraud assessment")
    lines.append("-" * 60)
    lines.append(
        f"  Risk score : {fraud['score']}/100 "
        f"({fraud['band']} risk; threshold {fraud['threshold']})"
    )
    lines.append(f"  Prediction : {fraud['predicted_label'].upper()}")
    if fraud["reasons"]:
        lines.append("  Reasons    :")
        for r in fraud["reasons"]:
            lines.append(f"    - [{r['rule']}] +{r['points']} pts: {r['detail']}")
    else:
        lines.append("  Reasons    : no fraud indicators triggered")
    lines.append("")
    lines.append("Similar claims retrieved")
    lines.append("-" * 60)
    if similar:
        for entry in similar:
            c = entry["claim"]
            lines.append(
                f"  - {c['claim_id']} ({c['claim_type']}, "
                f"${c['amount']:,.0f}, similarity {entry['score']:.2f}): "
                f"{c['description'][:90]}..."
            )
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Recommended next steps")
    lines.append("-" * 60)
    if fraud["predicted_label"] == "fraud":
        lines.append("  1. Route to the SIU (special investigations unit) queue.")
        lines.append("  2. Request supporting invoices and verify vendor details.")
        lines.append("  3. Cross-check the similar claims above for linked activity.")
    else:
        lines.append("  1. Continue standard adjudication.")
        lines.append("  2. No additional fraud review required at this time.")
    lines.append("")
    lines.append("Note: generated from synthetic demo data; not a real fraud determination.")
    return "\n".join(lines)
