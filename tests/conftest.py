import os

# Tests never upload traces: .env turns LangSmith on for the API, and the fake model's turns would
# bury real ones. Set before anything imports LangChain; the tracing test collects runs locally.
os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"] = "false"

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from app.agent.graph import build_graph
from app.extensions import Extensions
from app.store import Store
from tests.helpers import new_id


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; run `docker compose up -d` and use `uv run --env-file .env pytest`")
    return url


@pytest.fixture(scope="session")
def store(database_url):
    store = Store.connect(database_url)
    store.init_schema()
    yield store
    store.close()


@pytest.fixture
def household(store):
    """A fresh synthetic household per test, so tests never share rows."""
    household_id = new_id("h")
    leo, asha = f"{household_id}-leo", f"{household_id}-asha"
    store.upsert_members(
        household_id,
        [
            {"id": leo, "name": "Leo", "aliases": ["my son"], "age_years": 2},
            {"id": asha, "name": "Asha", "aliases": ["mum"], "age_years": 72, "conditions": ["hypertension"]},
        ],
    )
    return household_id, leo, asha


@pytest.fixture
def api_client(store, tmp_path):
    def client_for(model, handoff_dir=tmp_path / "handoffs", extensions=None):
        app = create_app()
        # No lifespan: use the test-owned pool and synthetic households, never real model settings.
        app.state.store = store
        app.state.handoff_dir = handoff_dir
        app.state.agent = build_graph(store, model, handoff_dir=handoff_dir, extensions=extensions or Extensions())
        app.state.consolidate = Mock(side_effect=AssertionError("Unexpected consolidation"))
        return TestClient(app)
    return client_for
