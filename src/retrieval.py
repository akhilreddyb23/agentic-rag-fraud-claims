"""TF-IDF vector retrieval over the synthetic claim corpus (CPU-only)."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import claim_text, get_claim


class ClaimRetriever:
    """Retrieves similar claims using TF-IDF cosine similarity."""

    def __init__(self, claims: list[dict]):
        self.claims = claims
        self._ids = [c["claim_id"] for c in claims]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([claim_text(c) for c in claims])

    def search(
        self, query_text: str, top_k: int = 5, exclude_id: str | None = None
    ) -> list[dict]:
        """Return the top_k most similar claims as [{'claim':..., 'score':...}]."""
        if not query_text or not query_text.strip():
            return []
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(self.claims)), key=lambda i: sims[i], reverse=True)
        results = []
        for i in ranked:
            claim = self.claims[i]
            if exclude_id is not None and claim["claim_id"] == exclude_id:
                continue
            results.append({"claim": claim, "score": float(sims[i])})
            if len(results) >= top_k:
                break
        return results

    def similar_to_claim(self, claim_id: str, top_k: int = 5) -> list[dict]:
        """Retrieve claims similar to the corpus claim with the given id."""
        claim = get_claim(claim_id, self.claims)
        if claim is None:
            raise KeyError(f"Unknown claim id: {claim_id}")
        return self.search(claim_text(claim), top_k=top_k, exclude_id=claim_id)
