"""The deep engine: code runs both ends of the turn, the supervisor and subagents the middle."""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.intake.questions import IntakeQuestions
from app.deep.engine import FALLBACK_REPLY, DeepEngine
from app.triage.policy import FIXED_REPLIES, Decision
from tests.deep_fakes import ScriptedChat, task, turn_data
from tests.fakes import call, episode, member, observation, screen, submit
from tests.helpers import new_id

RED_FLAG = {"system": "breathing", "sign": "struggling to breathe", "quote": "can't breathe"}


def run(store, model, household_id, text, tmp_path, *, turn_id=None, session_id=None, profile_id=None):
    engine = DeepEngine(store, model, handoff_dir=tmp_path / "handoffs")
    return engine.invoke({"household_id": household_id, "session_id": session_id or new_id("s"),
                          "turn_id": turn_id or new_id("t"), "text": text, "profile_id": profile_id})


def decide(decision, reason="protocol", urgency=None):
    return call("set_decision", decision=decision, reason=reason, urgency=urgency)


def send(reply=None):
    return call("submit_turn", reply=reply)


def ask_next(step):
    question = turn_data(step.system)["next_question"]
    return submit(IntakeQuestions(message=question["ask"], new_sign=None))


def test_the_screen_runs_before_any_agent_and_an_emergency_needs_no_subagent(store, household, tmp_path):
    household_id, _, _ = household
    reply = FIXED_REPLIES[Decision.EMERGENCY]
    model = ScriptedChat(screens=[screen(red_flags=[RED_FLAG])],
                         scripts={"supervisor": [decide("emergency", "red_flag"), send(reply)]})
    result = run(store, model, household_id, "Leo can't breathe", tmp_path)
    assert model.order == ["screen", "supervisor", "supervisor"]
    assert (result["decision"], result["reply"]) == (Decision.EMERGENCY, reply)
    assert turn_data(model.by_role("supervisor")[0].system)["screen"]["red_flags"][0]["sign"] == "struggling to breathe"


def test_a_saved_turn_is_replayed_without_calling_any_agent(store, household, tmp_path):
    household_id, _, _ = household
    session, turn = new_id("s"), new_id("t")
    model = ScriptedChat(screens=[screen(red_flags=[RED_FLAG])],
                         scripts={"supervisor": [decide("emergency", "red_flag"), send("Call now.")]})
    first = run(store, model, household_id, "can't breathe", tmp_path, turn_id=turn, session_id=session)
    calls = len(model.calls)
    again = run(store, model, household_id, "can't breathe", tmp_path, turn_id=turn, session_id=session)
    assert again["replayed"] and again["reply"] == first["reply"] and len(model.calls) == calls


def test_an_intake_turn_runs_context_intake_and_questions_each_with_its_own_view(store, household, tmp_path):
    household_id, leo, _ = household
    text = "Leo has had a fever since last night"
    model = ScriptedChat(scripts={
        "supervisor": [task("context"), task("intake"), decide("ask", "intake_incomplete"), task("questions"), send()],
        "context:member": [submit(member(leo))],
        "context:episode": [submit(episode(status="new"))],
        "intake": [submit(observation(intent="symptom_report", complaint="fever"))],
        "questions": [ask_next],
    })
    result = run(store, model, household_id, text, tmp_path)

    assert result["decision"] == Decision.ASK and result["reply"].endswith("?")
    member_view, episode_view = model.by_role("context:member")[0], model.by_role("context:episode")[0]
    assert set(member_view.tools) == {"list_members", "member_facts", "recent_messages", "MemberResolution"}
    assert turn_data(member_view.system)["text"] == text  # code gives the message, not the supervisor's brief
    assert turn_data(episode_view.system)["member"]["id"] == leo  # the episode step is scoped to the member
    assert turn_data(model.by_role("intake")[0].system)["member"]["id"] == leo
    later = turn_data(model.by_role("supervisor")[-1].system)["this_turn"]
    assert later["triage"]["sufficient"] is False and later["draft_reply"] == result["reply"]
    saved = store.load_context(household_id, new_id("s"))
    assert [(e.member_id, e.complaint, e.questions_asked) for e in saved.episodes] == [(leo, "fever", 1)]


def test_the_intake_report_gives_the_supervisor_the_policy_suggestion(store, household, tmp_path):
    household_id, leo, _ = household
    model = ScriptedChat(scripts={
        "supervisor": [task("context"), task("intake"), decide("ask"), task("questions"), send()],
        "context:member": [submit(member(leo))], "context:episode": [submit(episode(status="new"))],
        "intake": [submit(observation(intent="symptom_report", complaint="fever"))], "questions": [ask_next],
    })
    run(store, model, household_id, "Leo has a fever", tmp_path)
    report = [m for m in model.by_role("supervisor")[2].messages if isinstance(m, ToolMessage)][-1]
    assert json.loads(report.content)["policy_suggestion"] == {"decision": "ask", "reason": "intake_incomplete",
                                                               "urgency": None}


def test_a_tool_error_goes_back_to_the_supervisor_to_correct(store, household, tmp_path):
    household_id, leo, _ = household
    model = ScriptedChat(screens=[screen(doctor_requested=True)], scripts={
        "supervisor": [task("context"), task("intake"), decide("doctor"), decide("doctor", urgency="doctor_now"),
                       task("handoff"), send(FIXED_REPLIES[Decision.DOCTOR])],
        "context:member": [submit(member(leo))], "context:episode": [submit(episode(status="new"))],
        "intake": [submit(observation(intent="symptom_report", complaint="fever"))],
        "handoff": [call("escalate_to_doctor", chief_complaint="Fever", history=[], symptom_timeline=[],
                         relevant_history=None)],
    })
    result = run(store, model, household_id, "I want a doctor to see Leo", tmp_path)
    errors = [m for m in model.by_role("supervisor")[3].messages if isinstance(m, ToolMessage) and m.status == "error"]
    assert "needs an urgency" in errors[0].content
    assert result["decision"] == Decision.DOCTOR and result["urgency"] == "doctor_now"
    assert result["handoff_path"] and (tmp_path / "handoffs").exists()


def test_submit_refuses_a_doctor_reply_before_the_note_is_saved(store, household, tmp_path):
    household_id, leo, _ = household
    model = ScriptedChat(scripts={
        "supervisor": [task("context"), task("intake"), decide("doctor", urgency="doctor_24h"),
                       send("Transferring you now."), AIMessage(content="I give up")],
        "context:member": [submit(member(leo))], "context:episode": [submit(episode(status="new"))],
        "intake": [submit(observation(intent="symptom_report", complaint="fever"))],
    })
    result = run(store, model, household_id, "Leo has a fever", tmp_path)
    refused = [m for m in model.by_role("supervisor")[-1].messages
               if isinstance(m, ToolMessage) and m.status == "error"]
    assert "before the doctor's note is saved" in refused[0].content
    # Code keeps the promise the decision made: a doctor reply and a saved minimal note.
    assert result["decision"] == Decision.DOCTOR and result["reply"].startswith(FIXED_REPLIES[Decision.DOCTOR])
    assert result["handoff_path"]


@pytest.mark.parametrize("screens, expected", [
    ([screen()], (Decision.ANSWER, FALLBACK_REPLY)),
    ([screen(red_flags=[RED_FLAG])], (Decision.EMERGENCY, FIXED_REPLIES[Decision.EMERGENCY])),
])
def test_a_supervisor_that_never_submits_gets_a_safe_reply(store, household, tmp_path, screens, expected):
    household_id, _, _ = household
    model = ScriptedChat(screens=screens, scripts={"supervisor": [AIMessage(content="Hmm.")]})
    result = run(store, model, household_id, "hello, I can't breathe", tmp_path)
    assert (result["decision"], result["reply"]) == expected


def test_a_subagent_called_too_early_is_told_why_and_records_nothing(store, household, tmp_path):
    household_id, leo, _ = household
    model = ScriptedChat(scripts={
        "supervisor": [task("intake"), task("context"), task("intake"), decide("ask"), task("questions"), send()],
        "context:member": [submit(member(leo))], "context:episode": [submit(episode(status="new"))],
        "intake": [submit(observation(intent="symptom_report", complaint="fever"))], "questions": [ask_next],
    })
    result = run(store, model, household_id, "Leo has a fever", tmp_path)
    early = next(m for m in model.by_role("supervisor")[1].messages if isinstance(m, ToolMessage))
    assert "context subagent has not recorded" in early.content
    assert result["decision"] == Decision.ASK


def test_every_agent_call_is_offered_only_its_roles_tools(store, household, tmp_path):
    household_id, leo, _ = household
    model = ScriptedChat(scripts={
        "supervisor": [task("context"), task("intake"), decide("ask"), task("questions"), send()],
        "context:member": [submit(member(leo))], "context:episode": [submit(episode(status="new"))],
        "intake": [submit(observation(intent="symptom_report", complaint="fever"))], "questions": [ask_next],
    })
    run(store, model, household_id, "Leo has a fever", tmp_path)
    offered = {c.role: set(c.tools) for c in model.calls}
    assert offered["supervisor"] == {"task", "set_decision", "submit_turn"}
    assert offered["intake"] == {"get_protocol", "Observation"}
    assert "IntakeQuestions" in offered["questions"] and "write_file" not in set().union(*offered.values())
    assert all(c.settings.get("parallel_tool_calls") is False for c in model.by_role("supervisor"))
