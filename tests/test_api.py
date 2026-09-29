def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_by_claim_id(client):
    response = client.post("/analyze", json={"claim_id": "CLM-0007"})
    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == "CLM-0007"
    assert len(body["similar_claims"]) == 5
    first = body["similar_claims"][0]
    assert {"claim_id", "claim_type", "amount", "similarity", "description"} <= set(first)
    fraud = body["fraud"]
    assert fraud["predicted_label"] == "fraud"
    assert fraud["score"] >= fraud["threshold"]
    assert len(fraud["reasons"]) > 0
    assert "Fraud assessment" in body["summary"]


def test_analyze_by_claim_text(client):
    response = client.post(
        "/analyze",
        json={"claim_text": "Basement flooded after a pipe burst, need water extraction and drywall repair."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == "QUERY"
    assert len(body["similar_claims"]) > 0
    assert body["fraud"]["predicted_label"] in {"fraud", "legitimate"}


def test_analyze_unknown_claim_id_404(client):
    response = client.post("/analyze", json={"claim_id": "CLM-9999"})
    assert response.status_code == 404


def test_analyze_missing_input_422(client):
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyze_top_k_respected(client):
    response = client.post("/analyze", json={"claim_id": "CLM-0001", "top_k": 2})
    assert response.status_code == 200
    assert len(response.json()["similar_claims"]) == 2
