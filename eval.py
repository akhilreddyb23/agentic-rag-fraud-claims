"""Evaluation script: retrieval recall@k + fraud-scoring accuracy.

Run from the repo root:

    python eval.py
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.corpus import claim_text, load_claims  # noqa: E402
from src.fraud_rules import FRAUD_THRESHOLD, score_claim  # noqa: E402
from src.retrieval import ClaimRetriever  # noqa: E402


def eval_retrieval(claims, ks=(1, 3, 5)):
    retriever = ClaimRetriever(claims)
    # Ground truth: claims sharing the same scenario tag.
    by_scenario = defaultdict(list)
    for c in claims:
        by_scenario[c["scenario"]].append(c["claim_id"])
    evaluated = 0
    hits = {k: 0 for k in ks}
    total_relevant = 0
    for claim in claims:
        relevant = [cid for cid in by_scenario[claim["scenario"]] if cid != claim["claim_id"]]
        if not relevant:
            continue
        evaluated += 1
        total_relevant += len(relevant)
        results = retriever.search(claim_text(claim), top_k=max(ks), exclude_id=claim["claim_id"])
        retrieved_ids = [e["claim"]["claim_id"] for e in results]
        for k in ks:
            hits[k] += len(set(retrieved_ids[:k]) & set(relevant))
    print("Retrieval (TF-IDF) — recall@k over synthetic scenarios")
    print(f"  evaluated claims: {evaluated}, total relevant pairs: {total_relevant}")
    for k in ks:
        recall = hits[k] / total_relevant if total_relevant else 0.0
        print(f"  recall@{k}: {recall:.3f}")
    return {k: hits[k] / total_relevant for k in ks}


def eval_fraud(claims):
    tp = fp = tn = fn = 0
    for claim in claims:
        result = score_claim(claim, claims)
        predicted = 1 if result["score"] >= FRAUD_THRESHOLD else 0
        actual = claim["fraud_label"]
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("\nFraud scoring vs. synthetic labels")
    print(f"  threshold: {FRAUD_THRESHOLD}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  precision: {precision:.3f}")
    print(f"  recall   : {recall:.3f}")
    print(f"  F1       : {f1:.3f}")
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    claims = load_claims()
    print(f"Corpus: {len(claims)} synthetic claims "
          f"({sum(c['fraud_label'] for c in claims)} labeled fraud)")
    eval_retrieval(claims)
    eval_fraud(claims)


if __name__ == "__main__":
    main()
