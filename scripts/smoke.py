"""One synthetic API conversation using the configured model and database.

Run: uv run --env-file .env python -m scripts.smoke
"""

import json
from uuid import uuid4

from fastapi.testclient import TestClient

from api.main import create_app


def main() -> None:
    app = create_app()
    with TestClient(app) as client:
        household = f"smoke-{uuid4().hex[:12]}"
        app.state.store.upsert_members(household, [
            {"id": f"{household}-member", "name": "Demo", "aliases": ["me"], "age_years": 30},
        ])
        path = f"/households/{household}/sessions/{household}-session/messages"
        payload = {"text": "Why is sleep important?", "turn_id": "first"}
        first = client.post(path, json=payload)
        if first.status_code != 200:
            raise RuntimeError(f"Smoke request failed ({first.status_code}): {first.json()}")
        assert first.json()["decision"] == "answer"
        assert first.json()["reply"].strip()
        replay = client.post(path, json=payload)
        assert replay.status_code == 200 and replay.json() == first.json()
        context = app.state.store.load_context(household, f"{household}-session")
        assert len(context.recent_messages) == 2
        print(json.dumps({"user": payload["text"], **first.json(), "replay": "matched",
                          "persisted_messages": len(context.recent_messages)}, indent=2))


if __name__ == "__main__":
    main()
