import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from app.agents.intake.observe import observe_payload
from app.models.records import Message
from app.triage.protocol import load_protocols
from tests.test_graph import make_context, make_episode, make_member


@pytest.mark.parametrize("has_episode", [True, False])
def test_observation_keeps_each_message_once_with_its_order_and_provenance(has_episode):
    now = datetime.now(UTC)
    messages = tuple(
        Message(id=i, role="user" if i % 2 else "assistant", content="same words",
                member_id="leo", episode_id="e1", decision=None,
                created_at=now + timedelta(seconds=i), session_id="old" if i == 1 else "s1")
        for i in range(1, 5)
    )
    episode = make_episode("e1", "leo") if has_episode else None
    context = replace(make_context(make_member("leo", "Leo")), recent_messages=messages[1:])
    store = Mock(episode_history=Mock(return_value=messages[:3]))
    payload = json.loads(observe_payload(
        {"context": context, "member_id": "leo", "selected_episode": episode,
         "household_id": "h1", "text": "an update"}, store, load_protocols(),
    ))

    assert [m["id"] for m in payload["episode_history"]] == ([1] if has_episode else [])
    assert [m["id"] for m in payload["recent_messages"]] == [2, 3, 4]
    combined = payload["episode_history"] + payload["recent_messages"]
    assert len({m["id"] for m in combined}) == len(combined)
    for rendered in combined:
        original = messages[rendered["id"] - 1]
        assert rendered["role"] == original.role
        assert rendered["member_id"] == original.member_id
        assert rendered["episode_id"] == original.episode_id
        assert rendered["session_id"] == original.session_id
        assert rendered["created_at"].startswith(original.created_at.isoformat())
    if has_episode:
        store.episode_history.assert_called_once_with("h1", "e1")
    else:
        store.episode_history.assert_not_called()
