"""Multi-turn symptom intake and doctor handoff: real graph and Postgres, scripted model."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.agent.handoff import NO_NARRATIVE
from app.agent.loop import MAX_AGENT_STEPS
from app.agent.intake_agent import IntakeQuestions
from app.agent.state import Observation
from app.policy import FIXED_REPLIES, URGENCY_ADVICE, Decision
from app.store import EpisodeWrite
from tests.fakes import HANDOFF, FakeModel, agent_payload, answered, call, draft, episode, member, observation, screen
from tests.helpers import make_turn, new_id, post

FIRST = "Mum has had chest pain on and off since yesterday"
SECOND = ("No, it's not happening right now and never lasts more than a minute. No breathlessness, "
          "sweating or nausea, she hasn't felt faint, and it doesn't spread")
CHEST_EMERGENCY_NO = dict(pain_now="absent", lasted_over_5_min="absent", breathless="absent", sweating="absent", nausea="absent",
                          faint="absent", pain_spreads="absent")


def notes(directory):
    return sorted(directory.rglob("*.json")) if directory.exists() else []


PAIN_NOW = "Is the chest pain happening right now?"


def asked(model, turn=0):
    """The one question the intake agent was told to ask on its nth turn."""
    return agent_payload(model.inputs[IntakeQuestions][turn])["next_question"]["ask"]


def continue_pending(messages):
    return episode(agent_payload(messages)["pending_episode_id"])


@pytest.mark.integration
def test_chest_pain_in_a_high_risk_member_is_handed_to_a_doctor_with_an_intake_note(
        store, household, api_client, tmp_path):
    household_id, _, asha = household  # Asha: 72, hypertension
    handoffs = tmp_path / "handoffs"

    def write_note(messages):
        evidence = agent_payload(messages)["evidence"]
        first = next(e for e in evidence if e["content"] == FIRST)
        return draft(chief_complaint="Intermittent chest pain", symptom_timeline=[
            {"message_id": "current", "onset": None, "finding": "No pain now; no breathlessness or sweating",
             "quote": "No breathlessness"},
            {"message_id": first["message_id"], "onset": "yesterday", "finding": "Chest pain, intermittent",
             "quote": "chest pain on and off"},
            {"message_id": first["message_id"], "onset": None, "finding": "Invented", "quote": "crushing pain"},
        ])

    model = FakeModel(
        members=[member(asha), member(asha)],
        episodes=[episode(status="new"), continue_pending],
        observations=[
            observation(intent="symptom_report", complaint="chest_pain"),
            observation(intent="update", complaint="chest_pain", answers=answered(SECOND, **CHEST_EMERGENCY_NO)),
        ],
        handoffs=[write_note],
    )
    client = api_client(model)
    session = new_id("s")

    first = post(client, household_id, session, FIRST)
    assert first.json()["decision"] == "ask"
    assert asked(model) == PAIN_NOW  # one question at a time, the most urgent first
    assert notes(handoffs) == []

    second = post(client, household_id, session, SECOND)
    assert second.json()["decision"] == "doctor"
    assert second.json()["reply"] == f"{FIXED_REPLIES[Decision.DOCTOR]} {URGENCY_ADVICE['doctor_now']}"

    note_path, = notes(handoffs)
    note = json.loads(note_path.read_text())
    assert note["patient"]["name"] == "Asha" and note["patient"]["known_conditions"] == ["hypertension"]
    assert note["escalation"] == {"urgency": "doctor_now", "reason": "protocol", "doctor_requested": False,
                                  "red_flags": [], "doctor_flags": []}
    assert note["triage"]["protocol"]["id"] == "chest_pain"
    assert [d["question_id"] for d in note["triage"]["deciding"]] == ["cardiac_risk_member"]
    states = {f["question_id"]: f["state"] for f in note["triage"]["findings"]}
    assert states["pain_spreads"] == "ruled_out" and "exertion_trigger" not in states  # only what was discussed
    negatives = {n["finding"]: n for n in note["pertinent_negatives"]}
    assert negatives["Pain spreading to the arm, jaw, neck or back"]["reported_in"]["conversation"] == "this conversation"
    # Discovery order, not the order the model listed them; the unsupported entry is dropped.
    assert [t["finding"] for t in note["symptom_timeline"]] == [
        "Chest pain, intermittent", "No pain now; no breathlessness or sweating"]
    assert [m["content"] for m in note["transcript"]][::2] == [FIRST, SECOND]
    assert note["chief_complaint"] == "Intermittent chest pain"

    saved, = store.load_context(household_id, session).episodes
    assert (saved.status, saved.protocol_id) == ("doctor", "chest_pain")
    assert saved.findings["pain_spreads"]["value"] == "absent"


@pytest.mark.integration
def test_a_positive_emergency_answer_is_an_emergency_without_waiting_for_a_note(
        store, household, api_client, tmp_path):
    household_id, _, asha = household
    text = "Yes, it's spreading to her left arm"
    model = FakeModel(members=[member(asha)], episodes=[episode(status="new")], observations=[
        observation(intent="symptom_report", complaint="chest_pain",
                    answers=answered(text, pain_spreads="present")),
    ])
    result = post(api_client(model), household_id, new_id("s"), text)
    assert result.json()["reply"] == FIXED_REPLIES[Decision.EMERGENCY]
    assert model.reply_calls == 0 and model.inputs[HANDOFF] == []
    assert notes(tmp_path / "handoffs") == []


@pytest.mark.integration
def test_answers_without_supporting_words_do_not_count(store, household, api_client):
    household_id, _, asha = household
    invented = [{"question_id": "pain_spreads", "value": "absent", "quote": "it does not spread"},
                {"question_id": "made_up", "value": "absent", "quote": "chest pain"}]
    model = FakeModel(members=[member(asha)], episodes=[episode(status="new")], observations=[
        observation(intent="symptom_report", complaint="chest_pain", answers=invented),
    ])
    session = new_id("s")
    assert post(api_client(model), household_id, session, "She has chest pain").json()["decision"] == "ask"
    assert asked(model) == PAIN_NOW
    assert dict(store.load_context(household_id, session).episodes[0].findings) == {}


@pytest.mark.integration
def test_a_check_back_asks_the_emergency_questions_again(store, household, api_client):
    household_id, leo, _ = household
    session, fever = new_id("s"), new_id("e")
    store.load_context(household_id, session)
    earlier = {q: {"value": "absent", "quote": "no", "at": "2026-10-27T09:00:00+00:00"}
               for q in ("stiff_neck", "confused", "hard_to_wake", "rash_not_fading", "drinking_less", "less_urine")}
    store.save_turn(make_turn(household_id, session, member_id=leo, episode=EpisodeWrite(
        id=fever, member_id=leo, complaint="fever", status="monitoring",
        slots={"duration_hours": 12, "severity": "mild", "temperature_c": 38}, questions_asked_delta=1,
        check_back_after=datetime.now(UTC) + timedelta(hours=12), protocol_id="fever", findings=earlier,
    )))
    model = FakeModel(members=[member(leo)], episodes=[episode(fever)],
                      observations=[observation(intent="update", complaint="fever")])
    assert post(api_client(model), household_id, session, "Leo is about the same").json()["decision"] == "ask"
    assert asked(model) == "Is there a stiff neck?"
    saved, = store.load_context(household_id, session).episodes
    assert set(saved.findings) == {"drinking_less", "less_urine"}  # emergency answers cleared until asked again


@pytest.mark.integration
def test_the_note_is_still_saved_when_the_agent_never_finishes(store, household, api_client, tmp_path):
    household_id, _, _ = household
    model = FakeModel(screens=[screen(doctor_requested=True)],
                      handoffs=[call("past_session_summaries")] * MAX_AGENT_STEPS)
    result = post(api_client(model), household_id, new_id("s"), "I want a doctor")
    assert result.json()["decision"] == "doctor"
    note_path, = notes(tmp_path / "handoffs")
    note = json.loads(note_path.read_text())
    assert note["history_of_present_illness"] == NO_NARRATIVE
    assert note["escalation"]["doctor_requested"] is True and note["patient"] is None
    assert note_path.parent.name == "no-episode"


@pytest.mark.integration
def test_an_unsaved_note_fails_the_turn_so_q_never_claims_a_handoff(store, household, api_client, tmp_path):
    household_id, _, _ = household
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    session = new_id("s")
    model = FakeModel(screens=[screen(doctor_requested=True)])
    result = post(api_client(model, handoff_dir=blocked), household_id, session, "I want a doctor")
    assert result.status_code == 503
    assert "call your local emergency number" in result.json()["detail"]
    assert not store.load_context(household_id, session).recent_messages


@pytest.mark.integration
def test_unfinished_intake_is_handed_over_at_the_turn_limit(store, household, api_client, tmp_path):
    from app.policy import MAX_INTAKE_TURNS

    household_id, _, asha = household
    turns = MAX_INTAKE_TURNS + 1
    model = FakeModel(members=[member(asha)] * turns,
                      episodes=[episode(status="new"), *[continue_pending] * (turns - 1)],
                      observations=[observation(intent="symptom_report", complaint="chest_pain"),
                                    *[observation(intent="update", complaint="chest_pain")] * (turns - 1)])
    client, session = api_client(model), new_id("s")
    decisions = [post(client, household_id, session, "not sure").json()["decision"] for _ in range(turns)]
    assert decisions == ["ask"] * MAX_INTAKE_TURNS + ["doctor"]
    note, = notes(tmp_path / "handoffs")
    assert json.loads(note.read_text())["escalation"]["reason"] == "intake_turn_limit"
    assert len(model.inputs[Observation]) == turns


@pytest.mark.integration
def test_the_saved_note_can_be_read_back_by_turn(store, household, api_client):
    household_id, _, _ = household
    client = api_client(FakeModel(screens=[screen(doctor_requested=True)]))
    turn = post(client, household_id, new_id("s"), "I want a doctor").json()["turn_id"]
    note = client.get(f"/households/{household_id}/handoffs/{turn}")
    assert note.status_code == 200 and note.json()["escalation"]["doctor_requested"] is True
    assert client.get(f"/households/{household_id}/handoffs/{new_id('t')}").status_code == 404
    assert client.get(f"/households/another-household/handoffs/{turn}").status_code == 404
