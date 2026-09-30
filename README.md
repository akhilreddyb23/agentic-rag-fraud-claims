# Agentic RAG: Insurance Fraud & Claims Intelligence
[![CI](https://github.com/akhilreddyb23/agentic-rag-fraud-claims/actions/workflows/ci.yml/badge.svg)](https://github.com/akhilreddyb23/agentic-rag-fraud-claims/actions)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


A portfolio demonstration of a **multi-agent RAG pipeline** for insurance
claims triage and fraud detection. It runs entirely on CPU with light,
open-source dependencies — no GPU, no embedding models, no API keys.

> **All claim data in this repo is synthetic and fictional.** It was
> generated to demonstrate the technique. No real customers, employers,
> or production deployments are involved or implied.

## Architecture

```
                    ┌─────────────────────────┐
  POST /analyze     │      LangGraph          │
  {claim_id}  ───▶  │  pipeline               │
  {claim_text}      │                         │
                    │  ┌───────────────────┐  │
                    │  │ 1. retriever_agent│──┼──▶ TF-IDF vector search over
                    │  └────────┬──────────┘  │    the synthetic claim corpus
                    │           ▼             │    (scikit-learn, CPU-only)
                    │  ┌───────────────────┐  │
                    │  │ 2. fraud_scorer   │──┼──▶ deterministic rules
                    │  │    _agent         │  │    (duplicate amounts, early
                    │  └────────┬──────────┘  │    claims, inflated line items,
                    │           ▼             │    filing frequency, round
                    │  ┌───────────────────┐  │    amounts) + z-score anomaly
                    │  │ 3. summarizer     │──┼──▶ structured case summary
                    │  │    _agent         │  │    + investigator next steps
                    │  └───────────────────┘  │
                    └─────────────────────────┘
```

* **Retriever** (`src/retrieval.py`): TF-IDF (1–2 grams) over claim
  descriptions, types, and line items; cosine similarity, top-k.
* **Fraud scoring** (`src/fraud_rules.py`): weighted deterministic rules
  plus an amount z-score vs. same-type peers. Score 0–100, threshold 40;
  every point is explained in the `reasons` list.
* **Summarizer** (`src/summarizer.py`): deterministic template that turns
  the claim + fraud result + retrieved neighbors into an investigator
  case summary.
* **API** (`src/api.py`): FastAPI `POST /analyze` accepting a `claim_id`
  or free-text `claim_text`, returning similar claims, the fraud
  assessment, and the summary.

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the demo API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Then:

```bash
# Analyze a known (synthetic) claim id
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"claim_id": "CLM-0007"}'

# Or analyze free-text claim narrative
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"claim_text": "Basement flooded after a pipe burst, need water extraction and drywall repair."}'

# Health check
curl http://127.0.0.1:8000/health
```

Try `CLM-0007` (staged duplicate auto claim), `CLM-0011` (claim filed
5 days after policy start with an inflated invoice), or `CLM-0018`
(duplicate medical billing with upcoded line items). `CLM-0001` is a
clean claim for contrast.

## Run the eval

```bash
python eval.py
```

Reports:

* **Retrieval** — recall@1/3/5, where ground-truth relevance = claims
  sharing the same synthetic `scenario` tag.
* **Fraud scoring** — precision / recall / F1 of the rule-based
  predictions against the synthetic `fraud_label`s.

Latest run on this corpus (27 synthetic claims, 9 labeled fraud):

* recall@1 = 0.293, recall@3 = 0.759, recall@5 = 0.983
* fraud precision = 1.00, recall = 1.00, F1 = 1.00

(The synthetic fraud patterns were designed to trigger the rules, so the
perfect fraud score measures internal consistency of the demo — not
real-world performance.)

## Run the tests

```bash
pytest -q
```

Covers: TF-IDF retrieval behavior and recall, each fraud rule firing /
staying quiet, score thresholds and bands, the LangGraph pipeline for
claim-id / free-text / error paths, summary formatting, and the FastAPI
endpoint (via `TestClient`).

## Layout

```
data/claims.json      28 synthetic claims (clearly fake names/dates)
src/corpus.py         corpus loading helpers
src/retrieval.py      TF-IDF ClaimRetriever
src/fraud_rules.py    deterministic rules + anomaly scoring
src/summarizer.py     case-summary template
src/agents.py         LangGraph 3-agent pipeline
src/api.py            FastAPI app
eval.py               retrieval + fraud eval script
tests/                pytest suite
```

## Limitations

* Retrieval uses TF-IDF keyword overlap, not semantic embeddings — it
  misses paraphrases.
* Fraud rules are hand-written heuristics tuned to the synthetic data;
  a production system would learn thresholds from real labeled data and
  add graph/network features.
* The summarizer is template-based, not a generative LLM, so summaries
  are structured but not free-form.
