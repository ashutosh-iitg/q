from datetime import UTC, datetime
from pathlib import Path

import pytest
from unittest.mock import Mock

from app.agent.graph import PARALLEL_BRANCHES, build_graph, route_after_decide, route_after_load
from app.agent.nodes import ModelOutputError, fixed_reply, reconcile
from app.agent.state import Episode, Member, SessionContext
from app.policy import FIXED_REPLIES, Decision
from tests.fakes import episode, member, screen


def make_member(member_id: str, name: str) -> Member:
    return Member(id=member_id, name=name, aliases=(), age_years=None, conditions=())


def make_episode(episode_id: str, member_id: str) -> Episode:
    return Episode(id=episode_id, member_id=member_id, complaint="fever", status="triaging", slots={},
                   questions_asked=0, check_back_after=None, updated_at=datetime.now(UTC))


def make_context(*members: Member, episodes: tuple[Episode, ...] = ()) -> SessionContext:
    return SessionContext(
        session_id="s1",
        is_new_session=False,
        members=members,
        facts=(),
        episodes=episodes,
        recent_messages=(),
        prior_summaries=(),
        member_hint_id=None,
        pending_episode_id=None,
    )


def test_graph_compiles_with_turn_nodes():
    nodes = set(build_graph(Mock(), Mock(), handoff_dir=Path("unused")).get_graph().nodes)

    assert {"load_context", *PARALLEL_BRANCHES, "resolve_episode", "reconcile", "observe", "decide", "handoff", "fixed_reply",
            "respond", "save_turn"} <= nodes


def test_new_turns_fan_out_and_replays_stop():
    assert route_after_load({"replayed": False}) == PARALLEL_BRANCHES
    assert route_after_load({"replayed": True}) == "__end__"


def test_emergencies_reply_at_once_and_doctor_decisions_hand_over_first():
    assert route_after_decide({"decision": Decision.EMERGENCY}) == "fixed_reply"
    assert route_after_decide({"decision": Decision.DOCTOR}) == "handoff"


@pytest.mark.parametrize("decision", [Decision.ASK, Decision.ANSWER, Decision.HOME_MONITORING,
                                      Decision.ASK_MEMBER, Decision.ASK_EPISODE])
def test_generated_decisions_route_to_respond(decision):
    assert route_after_decide({"decision": decision}) == "respond"


def test_emergency_reply_is_the_fixed_message():
    state = {"decision": Decision.EMERGENCY, "context": make_context()}

    assert fixed_reply(state) == {"reply": FIXED_REPLIES[Decision.EMERGENCY]}


LEO, ASHA = make_member("leo", "Leo"), make_member("asha", "Asha")
LEO_FEVER = make_episode("e1", "leo")


def joined(member_r, episode_r):
    state = {"context": make_context(LEO, ASHA, episodes=(LEO_FEVER,)), "screen": screen(),
             "member_resolution": member_r, "episode_resolution": episode_r}
    return reconcile(state)


def test_agreeing_agents_select_the_episode():
    result = joined(member("leo"), episode("e1"))
    assert (result["clarification"], result["member_id"], result["selected_episode"]) == (None, "leo", LEO_FEVER)


def test_disagreeing_agents_ask_which_member():
    result = joined(member("asha"), episode("e1"))
    assert result["clarification"].kind == "member"
    assert "Asha" in result["clarification"].needed_info and "Leo" in result["clarification"].needed_info
    assert result["extraction"].member_id == "asha" and result["episode_id"] is None


def test_unresolved_member_asks_before_the_episode():
    result = joined(member(status="unresolved", needed_info="who", responder_instructions="name them"),
                    episode(status="unresolved", needed_info="which illness"))
    assert (result["clarification"].kind, result["clarification"].responder_instructions) == ("member", "name them")


def test_unresolved_episode_asks_which_illness():
    result = joined(member("leo"), episode(status="unresolved", needed_info="same fever?"))
    assert (result["clarification"].kind, result["clarification"].needed_info) == ("episode", "same fever?")


def test_message_about_no_member_never_touches_an_episode():
    result = joined(member(), episode("e1"))
    assert (result["clarification"], result["selected_episode"]) == (None, None)


@pytest.mark.parametrize(("member_r", "episode_r"), [
    (member("stranger"), episode()),
    (member("leo"), episode("not-listed")),
])
def test_unknown_ids_from_the_agents_are_rejected(member_r, episode_r):
    with pytest.raises(ModelOutputError):
        joined(member_r, episode_r)


def test_episode_resolution_receives_the_member_and_cannot_lookup_another_members_episode():
    import json
    from dataclasses import replace
    from langchain_core.messages import ToolMessage
    from app.agent.resolvers import resolve_episode
    from app.agent.state import EpisodeResolution
    from tests.fakes import FakeModel, agent_payload, call

    asha_episode = make_episode("asha-fever", "asha")
    state = {"household_id": "h", "text": "Leo is still ill", "member_resolution": member("leo"),
             "context": replace(make_context(LEO, ASHA, episodes=(LEO_FEVER, asha_episode)),
                                pending_episode_id="asha-fever")}

    def after_list(messages):
        assert agent_payload(messages)["member"]["id"] == "leo"
        assert agent_payload(messages)["pending_episode_id"] is None
        rows = json.loads(messages[-1].content)
        assert [row["id"] for row in rows] == ["e1"]
        return call("episode_history", episode_id="asha-fever")

    def after_foreign_lookup(messages):
        assert isinstance(messages[-1], ToolMessage) and messages[-1].status == "error"
        return episode("e1")

    model = FakeModel(episodes=[call("list_episodes", member_id=None), after_list, after_foreign_lookup])
    store = Mock()
    assert resolve_episode(state, model=model, store=store)["episode_resolution"].episode_id == "e1"
    store.episode_history.assert_not_called()
    assert len(model.inputs[EpisodeResolution]) == 3


@pytest.mark.parametrize("resolution", [member(), member(status="unresolved")])
def test_no_episode_agent_call_without_a_resolved_member(resolution):
    from app.agent.resolvers import resolve_episode
    state = {"member_resolution": resolution, "context": make_context(LEO, ASHA)}
    model = Mock()
    assert resolve_episode(state, model=model, store=Mock())["episode_resolution"].status == "none"
    model.bind_tools.assert_not_called()


def test_graph_orders_episode_resolution_after_member_resolution():
    graph = build_graph(Mock(), Mock(), handoff_dir=Path("unused")).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("resolve_member", "resolve_episode") in edges
    assert ("load_context", "resolve_episode") not in edges
