from src.corpus import claim_text, get_claim


def test_search_returns_top_k(retriever):
    results = retriever.search("windshield cracked on highway", top_k=3)
    assert len(results) == 3
    for entry in results:
        assert "claim" in entry and "score" in entry
        assert 0.0 <= entry["score"] <= 1.0


def test_similar_to_claim_excludes_self(retriever):
    results = retriever.similar_to_claim("CLM-0007", top_k=5)
    ids = [e["claim"]["claim_id"] for e in results]
    assert "CLM-0007" not in ids
    assert len(ids) == 5


def test_duplicate_claim_retrieved(retriever):
    # CLM-0006 and CLM-0007 describe the same staged accident; each
    # should surface the other near the top.
    results = retriever.similar_to_claim("CLM-0006", top_k=3)
    ids = [e["claim"]["claim_id"] for e in results]
    assert "CLM-0007" in ids


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("   ", top_k=5) == []


def test_unknown_claim_id_raises(retriever):
    import pytest

    with pytest.raises(KeyError):
        retriever.similar_to_claim("CLM-9999")


def test_recall_at_5_high(claims, retriever):
    # Ground truth: same-scenario claims. TF-IDF should recover them.
    from collections import defaultdict

    by_scenario = defaultdict(list)
    for c in claims:
        by_scenario[c["scenario"]].append(c["claim_id"])
    hits = total = 0
    for claim in claims:
        relevant = {cid for cid in by_scenario[claim["scenario"]] if cid != claim["claim_id"]}
        if not relevant:
            continue
        total += len(relevant)
        retrieved = {
            e["claim"]["claim_id"]
            for e in retriever.search(claim_text(claim), top_k=5, exclude_id=claim["claim_id"])
        }
        hits += len(relevant & retrieved)
    recall = hits / total
    assert recall >= 0.8, f"recall@5 too low: {recall:.3f}"
