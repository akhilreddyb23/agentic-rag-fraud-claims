import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.corpus import get_claim, load_claims
from src.retrieval import ClaimRetriever


@pytest.fixture(scope="session")
def claims():
    return load_claims()


@pytest.fixture(scope="session")
def retriever(claims):
    return ClaimRetriever(claims)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from src.api import app

    return TestClient(app)
