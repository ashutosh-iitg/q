from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.agent.state import Clarification, Episode
from app.policy import MAX_INTAKE_TURNS, Decision, decide, doctor_urgency, rule
from app.triage.engine import profile, triage
from app.triage.protocol import load_protocols
from tests.fakes import extraction
from tests.test_graph import make_context, make_member

FEVER = load_protocols()["fever"]
CONTEXT = make_context(make_member("leo", "Leo"))
RULED_OUT = {q: {"value": "absent", "quote": "no"} for q in (
    "stiff_neck", "confused", "hard_to_wake", "rash_not_fading", "drinking_less", "less_urine")}


def symptomatic(**changes):
    return extraction("leo", intent="symptom_report", complaint="fever", slots={
        "duration_hours": 12, "severity": "mild", "temperature_c": 38, "trend": None,
        "systolic_bp": None, "diastolic_bp": None,
    }, **changes)


def fever_triage(findings=RULED_OUT, age=2, **slots):
    reported = {**symptomatic().slots.model_dump(), **slots}
    return triage(FEVER, findings, reported, profile(age, []))


def with_episode(status, questions_asked=0, check_back_after=None):
    episode = Episode("episode", "leo", "fever", status, {}, questions_asked, check_back_after, datetime.now(UTC))
    return replace(CONTEXT, episodes=(episode,))


@pytest.mark.parametrize(("changes", "expected"), [
    ({"member_id": None, "red_flags": [{"system": "breathing", "sign": "struggling to breathe", "quote": "can't breathe"}]}, Decision.EMERGENCY),
    ({"member_id": None, "intent": "general_question"}, Decision.ANSWER),
    ({"member_id": None, "intent": "doctor_request"}, Decision.DOCTOR),
    ({"member_id": None}, Decision.ASK_MEMBER),
    ({"resolved": True}, Decision.ANSWER),
])
def test_routing_priority(changes, expected):
    assert decide(symptomatic().model_copy(update=changes), CONTEXT, triage=fever_triage()) == expected


@pytest.mark.parametrize(("found", "expected"), [
    (fever_triage(), (Decision.HOME_MONITORING, "protocol")),
    (fever_triage(temperature_c=40.5), (Decision.DOCTOR, "protocol")),
    (fever_triage(age=0.1), (Decision.EMERGENCY, "protocol")),
    (fever_triage(findings={}), (Decision.ASK, "intake_incomplete")),
])
def test_protocol_sufficiency_decides_intake(found, expected):
    assert rule(symptomatic(), CONTEXT, triage=found) == expected


def test_a_protocol_emergency_beats_an_earlier_doctor_status():
    found = fever_triage(findings={**RULED_OUT, "rash_not_fading": {"value": "present", "quote": "yes"}})
    assert decide(symptomatic(episode_id="episode"), with_episode("doctor"), triage=found) == Decision.EMERGENCY


def test_unfinished_intake_goes_to_a_doctor_after_the_turn_limit():
    context = with_episode("triaging", questions_asked=MAX_INTAKE_TURNS)
    decision, reason = rule(symptomatic(episode_id="episode"), context, triage=fever_triage(findings={}))
    assert (decision, reason) == (Decision.DOCTOR, "intake_turn_limit")
    assert doctor_urgency(reason, None) == "doctor_now"


def test_urgency_comes_from_the_protocol_level_that_decided():
    found = fever_triage(duration_hours=80)
    decision, reason = rule(symptomatic(), CONTEXT, triage=found)
    assert (decision, doctor_urgency(reason, found)) == (Decision.DOCTOR, "doctor_24h")


def test_overdue_unchanged_monitoring_routes_to_doctor():
    output = symptomatic(episode_id="episode")
    output.slots.trend = "same"
    context = with_episode("monitoring", 1, datetime.now(UTC) - timedelta(hours=1))
    assert rule(output, context, triage=fever_triage()) == (Decision.DOCTOR, "not_improving")


@pytest.mark.parametrize(("changes", "kind", "expected"), [
    ({"red_flags": [{"system": "breathing", "sign": "struggling to breathe", "quote": "can't breathe"}]}, "member", Decision.EMERGENCY),
    ({"doctor_flags": [{"flag": "severe_pain", "quote": "agony"}]}, "episode", Decision.DOCTOR),
    ({}, "member", Decision.ASK_MEMBER),
    ({}, "episode", Decision.ASK_EPISODE),
])
def test_clarification_comes_after_safety_screening(changes, kind, expected):
    output = symptomatic().model_copy(update=changes)
    assert decide(output, CONTEXT, Clarification(kind, "needed", None)) == expected
