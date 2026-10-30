"""Real graph + real Postgres; only the model boundary is faked."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.intake.questions import IntakeQuestions
from app.agents.loop import MAX_AGENT_STEPS, MAX_LOOKUP_ROUNDS
from app.memory.writes import EpisodeWrite
from app.models.outputs import EpisodeResolution, MemberResolution, Observation, Screen
from app.triage.policy import FIXED_REPLIES, Decision
from tests.fakes import HANDOFF, FakeModel, agent_payload, answered, call, episode, member, observation, screen
from tests.helpers import fever_episode, make_turn, new_id, post

FEVER_RULED_OUT = dict(stiff_neck="absent", confused="absent", hard_to_wake="absent", rash_not_fading="absent",
                       drinking_less="absent", less_urine="absent")


@pytest.mark.integration
def test_api_answers_persists_and_replays_without_model_calls(store, household, api_client):
    household_id, _, _ = household
    session = new_id("s")
    model = FakeModel(replies=["Sleep supports your health."])
    client = api_client(model)
    first = post(client, household_id, session, "Why does sleep matter?", turn_id="first")
    assert first.status_code == 200
    assert first.json()["decision"] == "answer"
    assert first.json()["reply"] == "Sleep supports your health."
    assert post(client, household_id, session, "Why does sleep matter?", turn_id="first").json() == first.json()
    assert {role: len(calls) for role, calls in model.inputs.items()} == {
        Screen: 1, MemberResolution: 1, EpisodeResolution: 0, Observation: 1, IntakeQuestions: 0,
        HANDOFF: 0}
    assert model.reply_calls == 1
    assert {names[-1] for names, _ in model.bindings} == {"MemberResolution", "Observation"}
    # One prompt cache per role, so requests sharing a role's fixed prompt reach the same cache.
    assert all(kwargs["prompt_cache_key"] == f"q:{names[-1]}" for names, kwargs in model.bindings)
    assert len(store.load_context(household_id, session).recent_messages) == 2
    assert post(client, household_id, session, "Different question", turn_id="first").status_code == 409


@pytest.mark.integration
def test_profile_is_passed_to_the_agents_and_rejects_other_households(store, household, api_client):
    household_id, leo, _ = household
    model = FakeModel(members=[member(leo)], observations=[observation(intent="other")], replies=["Hello Leo."])
    client = api_client(model)
    session = new_id("s")
    assert post(client, household_id, session, "I feel fine", profile_id="another-profile").status_code == 404
    assert model.inputs[MemberResolution] == []
    assert post(client, household_id, session, "I feel fine", profile_id=leo).status_code == 200
    for role in (MemberResolution, EpisodeResolution):
        payload = agent_payload(model.inputs[role][0])
        assert (payload["profile_id"], payload["text"]) == (leo, "I feel fine")


@pytest.mark.integration
def test_demo_accounts_seed_idempotently(store, api_client):
    from app.demo import demo_accounts
    for _ in range(2):
        for account in demo_accounts():
            store.upsert_members(account["household_id"], account["members"])
    client = api_client(FakeModel())
    accounts = client.get("/demo/accounts").json()
    assert len(accounts) == 10
    for account in accounts:
        context = store.load_context(account["household_id"], new_id("s"))
        assert {m.id for m in context.members} == {m["id"] for m in account["members"]}


@pytest.mark.integration
def test_episode_continues_across_sessions_with_its_history(store, household, api_client):
    household_id, leo, _ = household
    first_session, second_session = new_id("s"), new_id("s")

    def continue_open_episode(messages):
        # The new session has no pending question, so the agent finds the episode with its tools.
        assert agent_payload(messages)["pending_episode_id"] is None
        return call("list_episodes", member_id=leo)

    def pick_listed_episode(messages):
        listed, = [m for m in messages if isinstance(m, ToolMessage)]
        assert '"complaint": "fever"' in listed.content
        episode_id = listed.content.split('"id": "')[1].split('"')[0]
        return episode(episode_id)

    reply_text = "12 hours, 38 C, mild, no stiff neck, not confused, wakes easily, no rash, drinking and peeing fine"

    def followup(messages):
        payload = agent_payload(messages)
        assert payload["episode"]["complaint"] == "fever"
        assert payload["protocol"]["id"] == "fever"
        assert [m["content"] for m in payload["episode_history"]][0] == "Leo feels hot"
        assert payload["recent_messages"] == []
        return observation(intent="update", complaint="fever", answers=answered(reply_text, **FEVER_RULED_OUT),
                           slots={"duration_hours": 12, "severity": "mild", "temperature_c": 38})

    model = FakeModel(
        members=[member(leo), member(leo)],
        episodes=[episode(status="new"), continue_open_episode, pick_listed_episode],
        observations=[observation(intent="symptom_report", complaint="fever", facts=[
            {"kind": "symptom", "subject": "fever", "text": "Leo feels hot"},
            {"kind": "condition", "subject": "invented", "text": "has diabetes"},
        ]), followup],
        replies=["Rest and fluids."],
    )
    client = api_client(model)
    assert post(client, household_id, first_session, "Leo feels hot").json()["decision"] == "ask"
    asked = agent_payload(model.inputs[IntakeQuestions][0])["next_question"]["ask"]
    assert asked == "Is there a stiff neck?"  # one question, the most urgent first
    result = post(client, household_id, second_session, reply_text)
    assert result.status_code == 200
    assert result.json()["decision"] == "home_monitoring"
    reply = result.json()["reply"]
    assert "Precautions:\n- Offer plenty of fluids" in reply  # from the fever protocol, not the model
    assert "Check back with Q in 24 hours" in reply
    context = store.load_context(household_id, second_session)
    assert len(context.episodes) == 1
    assert context.episodes[0].status == "monitoring"
    assert context.episodes[0].questions_asked == 1
    assert context.episodes[0].check_back_after is not None
    assert [(f.member_id, f.text) for f in context.facts] == [(leo, "Leo feels hot")]
    assert len(store.episode_history(household_id, context.episodes[0].id)) == 4


@pytest.mark.integration
@pytest.mark.parametrize("decision", [Decision.EMERGENCY, Decision.DOCTOR])
def test_escalation_does_not_wait_for_an_unresolved_member(store, household, api_client, decision):
    household_id, _, _ = household
    text = "can't breathe" if decision == Decision.EMERGENCY else "I want a doctor"
    found = screen(red_flags=[{"system": "breathing", "sign": "struggling to breathe", "quote": text}]) \
        if decision == Decision.EMERGENCY else screen(doctor_requested=True)
    model = FakeModel(screens=[found], members=[member(status="unresolved", needed_info="who")])
    result = post(api_client(model), household_id, new_id("s"), text)
    assert result.status_code == 200
    assert result.json()["reply"].startswith(FIXED_REPLIES[decision])
    assert model.reply_calls == 0
    assert model.inputs[Observation] == []


@pytest.mark.integration
def test_unresolved_member_passes_the_agents_instructions_to_the_responder(store, household, api_client):
    household_id, _, _ = household
    unresolved = member(status="unresolved", needed_info="Whether 'the little one' is Leo",
                        responder_instructions="Ask if they mean Leo.")
    model = FakeModel(members=[unresolved], replies=["Do you mean Leo?"])
    session = new_id("s")
    result = post(api_client(model), household_id, session, "the little one is coughing")
    assert result.json() == {**result.json(), "decision": "ask_member", "reply": "Do you mean Leo?"}
    clarification = model.reply_inputs[0]["clarification"]
    assert clarification == {"kind": "member", "needed_info": "Whether 'the little one' is Leo",
                             "responder_instructions": "Ask if they mean Leo."}
    assert model.inputs[Observation] == []
    assert not store.load_context(household_id, session).episodes


@pytest.mark.integration
def test_unresolved_episode_asks_which_illness(store, household, api_client):
    household_id, leo, _ = household
    unsure = episode(status="unresolved", needed_info="Whether this is last week's fever",
                     responder_instructions="Ask if it is the same fever.")
    model = FakeModel(members=[member(leo)], episodes=[unsure], replies=["Is this the same fever?"])
    result = post(api_client(model), household_id, new_id("s"), "Leo is hot again")
    assert result.json()["decision"] == "ask_episode"
    assert model.reply_inputs[0]["clarification"]["kind"] == "episode"


@pytest.mark.integration
def test_agents_that_disagree_ask_which_member(store, household, api_client):
    household_id, leo, asha = household
    session, leo_fever = new_id("s"), new_id("e")
    store.load_context(household_id, session)
    store.save_turn(make_turn(household_id, session, episode=fever_episode(leo_fever, leo, {})))
    model = FakeModel(members=[member(asha)], episodes=[episode(leo_fever)], replies=["Asha or Leo?"])
    result = post(api_client(model), household_id, session, "Mum is better")
    assert result.json()["decision"] == "ask_member"
    assert "Asha" in model.reply_inputs[0]["clarification"]["needed_info"]
    assert "Leo" in model.reply_inputs[0]["clarification"]["needed_info"]


@pytest.mark.integration
def test_unknown_episode_reference_is_not_saved(store, household, api_client):
    household_id, leo, _ = household
    session = new_id("s")
    model = FakeModel(members=[member(leo)], episodes=[episode("not-an-episode")])
    result = post(api_client(model), household_id, session, "Leo is better")
    assert result.status_code == 502 and "emergency number" in result.json()["detail"]
    assert not store.load_context(household_id, session).recent_messages


@pytest.mark.integration
def test_tool_errors_go_back_to_the_agent(store, household, api_client):
    household_id, leo, _ = household

    def corrected(messages):
        error, = [m for m in messages if isinstance(m, ToolMessage)]
        assert error.status == "error" and "Unknown member_id" in error.content
        return member(leo)

    model = FakeModel(members=[call("member_facts", member_id="someone-else"), corrected],
                      observations=[observation(intent="other")], replies=["Noted."])
    assert post(api_client(model), household_id, new_id("s"), "Leo is fine").status_code == 200
    assert len(model.inputs[MemberResolution]) == 2


@pytest.mark.integration
def test_parallel_lookups_all_return_before_the_answer(store, household, api_client):
    household_id, leo, _ = household
    both = AIMessage(content="", tool_calls=[
        {"name": "list_members", "args": {}, "id": new_id("call")},
        {"name": "member_facts", "args": {"member_id": leo}, "id": new_id("call")},
    ])

    def answer_after_both(messages):
        assert sorted(m.name for m in messages if isinstance(m, ToolMessage)) == ["list_members", "member_facts"]
        return member(leo)

    model = FakeModel(members=[both, answer_after_both], observations=[observation(intent="other")],
                      replies=["Noted."])
    assert post(api_client(model), household_id, new_id("s"), "Leo is fine").status_code == 200


@pytest.mark.integration
def test_the_episode_agent_reads_a_past_episode_it_found(store, household, api_client):
    household_id, leo, _ = household
    session, fever = new_id("s"), new_id("e")
    store.load_context(household_id, session)
    store.save_turn(make_turn(household_id, session, member_id=leo, text="Leo has a fever", episode=EpisodeWrite(
        id=fever, member_id=leo, complaint="fever", status="resolved_reported", slots={},
        questions_asked_delta=0, check_back_after=None)))

    def read_found_episode(messages):
        listed = [m for m in messages if isinstance(m, ToolMessage)][-1]
        assert fever in listed.content
        return call("episode_history", episode_id=fever)

    def decide_after_history(messages):
        history = [m for m in messages if isinstance(m, ToolMessage)][-1]
        assert history.name == "episode_history" and "Leo has a fever" in history.content
        return episode(status="new")

    model = FakeModel(members=[member(leo)], episodes=[call("list_episodes", member_id=leo), read_found_episode,
                                                       decide_after_history],
                      observations=[observation(intent="other")], replies=["Noted."])
    assert post(api_client(model), household_id, new_id("s"), "Leo's fever is back").status_code == 200
    assert len(model.inputs[EpisodeResolution]) == 3  # list, then history, then the answer


@pytest.mark.integration
def test_an_answer_sent_with_a_lookup_is_dropped_until_the_result_is_read(store, household, api_client):
    household_id, leo, _ = household
    blind = AIMessage(content="", tool_calls=[
        {"name": "list_members", "args": {}, "id": new_id("call")},
        {"name": "MemberResolution", "args": member(status="unresolved").model_dump(), "id": new_id("call")},
    ])

    def answer_after_reading(messages):
        assert [c["name"] for c in messages[-2].tool_calls] == ["list_members"]  # the blind answer is gone
        return member(leo)

    model = FakeModel(members=[blind, answer_after_reading], observations=[observation(intent="other")],
                      replies=["Noted."])
    assert post(api_client(model), household_id, new_id("s"), "Leo is fine").json()["decision"] == "answer"


@pytest.mark.integration
def test_after_the_lookup_rounds_the_agent_must_answer(store, household, api_client):
    household_id, leo, _ = household
    model = FakeModel(members=[call("list_members")] * MAX_LOOKUP_ROUNDS + [member(leo)],
                      observations=[observation(intent="other")], replies=["Noted."])
    assert post(api_client(model), household_id, new_id("s"), "Leo is fine").status_code == 200
    assert len(model.inputs[MemberResolution]) == MAX_AGENT_STEPS


@pytest.mark.integration
def test_agent_that_never_answers_falls_back_to_asking(store, household, api_client):
    household_id, _, _ = household
    model = FakeModel(members=[call("list_members")] * MAX_AGENT_STEPS, replies=["Who is this about?"])
    result = post(api_client(model), household_id, new_id("s"), "he's hot")
    assert result.json()["decision"] == "ask_member"
    assert len(model.inputs[MemberResolution]) == MAX_AGENT_STEPS


@pytest.mark.integration
def test_model_failure_is_retryable_without_partial_turn(store, household, api_client):
    import httpx
    from openai import APITimeoutError

    household_id, _, _ = household
    session = new_id("s")

    def fail(payload):
        raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    client = api_client(FakeModel(screens=[fail, screen()], replies=["A reply after retry."]))
    payload = {"turn_id": "retry-me"}
    assert post(client, household_id, session, "Why sleep?", **payload).status_code == 502
    assert not store.load_context(household_id, session).recent_messages
    assert post(client, household_id, session, "Why sleep?", **payload).status_code == 200
    assert len(store.load_context(household_id, session).recent_messages) == 2


@pytest.mark.integration
@pytest.mark.parametrize("tool_fails", [False, True])
def test_turn_trace_nests_agent_tools_and_groups_replay(store, household, api_client, tool_fails, monkeypatch):
    from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    household_id, leo, _ = household
    looked_up = "unknown-member" if tool_fails else leo
    model = FakeModel(members=[call("member_facts", member_id=looked_up), member()], replies=["Sleep matters."])
    client = api_client(model)
    collector = RunCollectorCallbackHandler()
    client.app.state.agent = client.app.state.agent.with_config(callbacks=[collector])
    session = new_id("s")
    payload = {"turn_id": "traced"}
    assert post(client, household_id, session, "Why sleep?", **payload).status_code == 200
    assert post(client, household_id, session, "Why sleep?", **payload).status_code == 200

    first, replay = collector.traced_runs

    def descendants(run):
        for child in run.child_runs:
            yield child
            yield from descendants(child)

    assert first.name == replay.name == "q.turn"
    runs = list(descendants(first))
    tool_run, = [run for run in runs if run.run_type == "tool"]
    assert tool_run.name == "member_facts"
    assert tool_run.inputs == {"member_id": looked_up}
    assert tool_run.parent_run_id in {run.id for run in runs if run.name == "resolve_member"}
    assert bool(tool_run.error) == tool_fails
    for run in (first, replay, tool_run):
        assert run.metadata["session_id"] == session
        assert run.metadata["household_id"] == household_id
        assert run.metadata["turn_id"] == "traced"
    assert replay.outputs["replayed"] is True
    assert not any(run.run_type == "tool" for run in descendants(replay))
    # Readable: each model call is named for its agent and call, and the turn's output is a summary.
    names = {run.name for run in runs}
    assert {"MemberResolution · call 1", "MemberResolution · call 2", "Screen"} <= names
    assert "context" not in first.outputs
    assert first.outputs["decision"] == "answer" and first.outputs["reply"] == "Sleep matters."


def test_a_dropped_answer_leaves_no_call_in_the_message_content():
    from app.agents.loop import _only

    lookup = {"name": "list_members", "args": {}, "id": "call_lookup"}
    reply = AIMessage(content=[{"type": "reasoning", "id": "rs_1"},
                               {"type": "function_call", "call_id": "call_lookup", "name": "list_members"},
                               {"type": "function_call", "call_id": "call_answer", "name": "MemberResolution"}],
                      tool_calls=[lookup, {"name": "MemberResolution", "args": {}, "id": "call_answer"}])
    kept = _only(reply, [lookup])
    assert [c["id"] for c in kept.tool_calls] == ["call_lookup"]
    assert [b.get("call_id") for b in kept.content] == [None, "call_lookup"]  # reasoning kept, answer gone


@pytest.mark.integration
@pytest.mark.parametrize("failure", [
    {"episodes": [episode("not-an-episode")]},
    {"observations": [call("get_protocol", protocol_id="fever")] * MAX_AGENT_STEPS},  # never answers
], ids=["unknown_episode", "observe_never_answers"])
def test_an_emergency_is_answered_even_when_another_step_fails(store, household, api_client, failure):
    household_id, leo, _ = household
    text = "Leo can't breathe"
    found = screen(red_flags=[{"system": "breathing", "sign": "struggling to breathe", "quote": text}])
    model = FakeModel(screens=[found], members=[member(leo)], **failure)
    result = post(api_client(model), household_id, new_id("s"), text)
    assert result.status_code == 200
    assert result.json()["reply"] == FIXED_REPLIES[Decision.EMERGENCY]


@pytest.mark.integration
def test_a_new_complaint_during_an_open_episode_opens_its_own_episode(store, household, api_client):
    household_id, leo, _ = household
    session, fever = new_id("s"), new_id("e")
    store.load_context(household_id, session)
    store.save_turn(make_turn(household_id, session, member_id=leo, episode=fever_episode(fever, leo, {})))
    model = FakeModel(members=[member(leo)], episodes=[episode(fever)],
                      observations=[observation(intent="symptom_report", complaint="cough")])
    assert post(api_client(model), household_id, session, "Fever's gone, but now Leo has a cough").status_code == 200
    complaints = {e.complaint: e.id for e in store.load_context(household_id, session).episodes}
    assert set(complaints) == {"fever", "cough"} and complaints["fever"] == fever
