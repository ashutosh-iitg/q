import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from app.agent.tools import episode_tools, member_tools, to_json
from tests.test_graph import make_context, make_episode, make_member

CONTEXT = make_context(make_member("leo", "Leo"), make_member("asha", "Asha"),
                       episodes=(make_episode("e1", "leo"), make_episode("e2", "asha")))


def tools_by_name(tools):
    return {t.name: t for t in tools}


def test_every_time_an_agent_reads_says_how_long_ago_it_was():
    now = datetime.now(UTC)
    rendered = json.loads(to_json({"said": now, "reading": now - timedelta(days=1),
                                   "summary": {"closed_at": now - timedelta(days=3)}}))
    assert rendered["said"] == f"{now.isoformat()} (today)"
    assert rendered["reading"].endswith("(yesterday)")
    assert rendered["summary"]["closed_at"].endswith("(3 days ago)")


def test_list_episodes_filters_by_member():
    tools = tools_by_name(episode_tools(CONTEXT, Mock(), "h1"))
    assert [e["id"] for e in json.loads(tools["list_episodes"].invoke({"member_id": "asha"}))] == ["e2"]
    assert len(json.loads(tools["list_episodes"].invoke({"member_id": None}))) == 2


def test_episode_history_reads_only_this_households_listed_episodes():
    store = Mock(episode_history=Mock(return_value=()))
    tools = tools_by_name(episode_tools(CONTEXT, store, "h1"))
    tools["episode_history"].invoke({"episode_id": "e1"})
    store.episode_history.assert_called_once_with("h1", "e1")
    with pytest.raises(ValueError, match="Unknown episode_id"):
        tools["episode_history"].invoke({"episode_id": "another-familys-episode"})
    assert store.episode_history.call_count == 1


def test_member_tools_reject_unknown_members():
    with pytest.raises(ValueError, match="Unknown member_id"):
        tools_by_name(member_tools(CONTEXT))["member_facts"].invoke({"member_id": "stranger"})
