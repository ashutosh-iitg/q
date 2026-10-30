"""Symptoms that came with a complaint and have since gone, and questions asked back during intake."""

import json
from types import SimpleNamespace

import pytest

from app.agents.intake.merge import merge_associated
from app.agents.intake.questions import IntakeQuestions
from app.models.outputs import AssociatedSymptom
from app.triage.engine import profile, triage
from app.triage.policy import Decision, rule
from app.triage.protocol import load_protocols
from tests.fakes import FakeModel, agent_payload, answered, draft, episode, extraction, member, observation, screen
from tests.helpers import new_id, post

COUGH = load_protocols()["cough"]
ADULT = profile(28, [])
FIRST = ("I have a dry cough for last two weeks. I had fever initially which subsided after taking "
         "paracetamol but the cough is not going away")
FEVER_GONE = "I had fever initially which subsided"


def cough_states(**values):
    findings = {q: {"value": v, "quote": v} for q, v in values.items()}
    return triage(COUGH, findings, {"duration_hours": 336}, ADULT).states


def test_a_resolved_finding_is_history_not_a_current_sign_below_urgent_levels():
    assert cough_states(fever_with_cough="resolved")["fever_with_cough"] is False


def test_an_urgent_sign_that_came_and_went_still_counts():
    assert cough_states(coughing_blood="resolved")["coughing_blood"] is True
    assert cough_states(blue_lips="resolved")["blue_lips"] is True


def test_related_symptoms_need_a_quote_and_the_latest_status_wins():
    earlier = {"fever": {"status": "active", "quote": "a fever", "at": "2026-10-28T10:00:00+00:00"}}
    reported = [AssociatedSymptom(symptom="Fever", status="resolved", quote=FEVER_GONE),
                AssociatedSymptom(symptom="sore_throat", status="active", quote="invented words")]
    merged = merge_associated(earlier, reported, FIRST)
    assert set(merged) == {"fever"}
    assert merged["fever"]["status"] == "resolved" and merged["fever"]["quote"] == FEVER_GONE


def test_a_question_asked_back_mid_intake_keeps_the_intake_going():
    found = triage(COUGH, {}, {}, ADULT)
    context = SimpleNamespace(members=[SimpleNamespace(id="riya")], episodes=[])
    clarifying = extraction("riya", intent="clarification", complaint="cough")
    assert rule(clarifying, context, None, found) == (Decision.ASK, "clarification_requested")


@pytest.mark.integration
def test_a_resolved_fever_is_stored_on_the_cough_episode_and_shown_to_the_intake_agent(
        store, household, api_client):
    household_id, _, asha = household
    session = new_id("s")
    model = FakeModel(
        members=[member(asha)], episodes=[episode(status="new")],
        observations=[observation(intent="symptom_report", complaint="cough", slots={"duration_hours": 336},
                                  answers=answered(FEVER_GONE, fever_with_cough="resolved"),
                                  associated=[{"symptom": "fever", "status": "resolved", "quote": FEVER_GONE}])])
    assert post(api_client(model), household_id, session, FIRST).json()["decision"] == "ask"
    saved, = store.load_context(household_id, session).episodes
    assert saved.associated["fever"]["status"] == "resolved"
    assert saved.findings["fever_with_cough"]["value"] == "resolved"
    payload = agent_payload(model.inputs[IntakeQuestions][0])
    assert payload["associated_symptoms"]["fever"]["status"] == "resolved"
    assert not any(q["id"] == "fever_with_cough" and q["positive"] for q in payload["already_known"])


@pytest.mark.integration
def test_who_do_you_mean_is_answered_by_the_intake_agent_not_the_general_reply(store, household, api_client):
    household_id, _, asha = household
    session = new_id("s")
    model = FakeModel(
        members=[member(asha), member(asha)],
        episodes=[episode(status="new"), lambda messages: episode(agent_payload(messages)["pending_episode_id"])],
        observations=[observation(intent="symptom_report", complaint="cough"),
                      observation(intent="clarification", complaint="cough")])
    client = api_client(model)
    post(client, household_id, session, "I have a cough")
    result = post(client, household_id, session, "Who is they?")
    assert result.json()["decision"] == "ask" and model.reply_calls == 0
    payload = agent_payload(model.inputs[IntakeQuestions][1])
    assert payload["user_asked_back"] is True and payload["unanswered_last_time"] is True


@pytest.mark.integration
def test_the_doctor_note_separates_active_and_resolved_related_symptoms(store, household, api_client, tmp_path):
    household_id, _, asha = household  # Asha: 72, hypertension
    handoffs = tmp_path / "handoffs"
    text = f"{FEVER_GONE} but now the cough gives me severe chest pain"
    model = FakeModel(
        screens=[screen(doctor_flags=[{"flag": "severe_pain", "quote": "severe chest pain"}])], members=[member(asha)], episodes=[episode(status="new")],
        observations=[observation(intent="symptom_report", complaint="cough",
                                  associated=[{"symptom": "fever", "status": "resolved", "quote": FEVER_GONE}])],
        handoffs=[draft()])
    assert post(api_client(model, handoff_dir=handoffs), household_id, new_id("s"), text).json()["decision"] == "doctor"
    note = json.loads(next(handoffs.rglob("*.json")).read_text())
    fever, = note["associated_symptoms"]
    assert (fever["symptom"], fever["status"], fever["quote"]) == ("fever", "resolved", FEVER_GONE)
