"""Protocol loading and the sufficiency rule. Pure code: no model, no database."""

import pytest

from app.triage.engine import NEW_SIGN, evaluate, profile, triage
from app.triage.protocol import GENERIC, load_protocols, protocol_for, with_signs

PROTOCOLS = load_protocols()
CHEST = PROTOCOLS["chest_pain"]
MUM = profile(72, ["hypertension"])  # the example: intermittent chest pain since yesterday
YOUNG = profile(25, [])


def answers(**values):
    return {question: {"value": value, "quote": value} for question, value in values.items()}


EMERGENCY_NO = answers(pain_now="absent", lasted_over_5_min="absent", breathless="absent", sweating="absent", nausea="absent",
                       faint="absent", pain_spreads="absent")


def test_every_shipped_protocol_is_marked_as_a_placeholder():
    assert {p.source for p in PROTOCOLS.values()} == {"synthetic-placeholder"}


def test_complaints_map_to_protocols_with_a_generic_fallback():
    assert protocol_for("chest_tightness", PROTOCOLS).id == "chest_pain"
    assert protocol_for("earache", PROTOCOLS).id == GENERIC
    assert protocol_for(None, PROTOCOLS).id == GENERIC


def test_intake_asks_one_question_at_a_time_most_urgent_first():
    result = triage(CHEST, {}, {}, MUM)
    assert not result.sufficient
    assert [q.id for q in result.pending] == ["pain_now"]
    assert result.floor == "doctor_now"  # already known from her age and hypertension


def test_one_positive_emergency_answer_is_sufficient_on_its_own():
    result = triage(CHEST, answers(pain_spreads="present"), {}, MUM)
    assert (result.disposition, result.deciding) == ("emergency", ("pain_spreads",))


def test_an_unknown_answer_cannot_rule_out_an_emergency():
    assert triage(CHEST, answers(breathless="unknown"), {}, YOUNG).disposition == "emergency"


def test_answered_signs_are_skipped_and_the_next_open_one_is_asked():
    result = triage(CHEST, answers(pain_now="absent", breathless="absent"), {}, YOUNG)
    assert [q.id for q in result.pending] == ["lasted_over_5_min"]


def test_a_high_risk_member_is_escalated_once_emergency_is_ruled_out():
    result = triage(CHEST, EMERGENCY_NO, {}, MUM)
    assert (result.disposition, result.deciding) == ("doctor_now", ("cardiac_risk_member",))


def test_home_care_needs_every_question_ruled_out():
    partial = triage(CHEST, {**EMERGENCY_NO, **answers(exertion_trigger="absent")}, {}, YOUNG)
    assert not partial.sufficient
    assert [q.id for q in partial.pending] == ["fever"]
    everything = {**EMERGENCY_NO, **answers(exertion_trigger="absent", fever="absent", cough="absent", similar_before="absent")}
    assert triage(CHEST, everything, {"duration_hours": 30}, YOUNG).disposition == "home_care"


def test_reported_values_answer_rule_questions():
    fever = PROTOCOLS["fever"]
    ruled_out = answers(stiff_neck="absent", confused="absent", hard_to_wake="absent", rash_not_fading="absent",
                        drinking_less="absent", less_urine="absent")
    child = profile(9, [])
    assert triage(fever, ruled_out, {"temperature_c": 40.2}, child).disposition == "doctor_now"
    waiting = triage(fever, ruled_out, {}, child)
    assert [q.id for q in waiting.pending] == ["very_high_temperature"]


def test_existing_conditions_lower_the_threshold():
    bp = PROTOCOLS["blood_pressure"]
    reading = {"systolic_bp": 152, "diastolic_bp": 96}
    no_danger = dict(bp_chest_pain="absent", severe_headache="absent", confusion="absent", one_sided_weakness="absent",
                     vision_changes="absent")
    calm = answers(**no_danger, dizziness="absent", headache="absent")
    assert triage(bp, calm, reading, MUM).disposition == "doctor_routine"
    dizzy = answers(**no_danger, dizziness="present", headache="absent")
    assert triage(bp, dizzy, reading, MUM).disposition == "doctor_24h"


@pytest.mark.parametrize(("rule", "slots", "member", "expected"), [
    ({"condition_in": ["diabetes"]}, {}, profile(50, ["Type 2 diabetes"]), True),
    ({"condition_in": ["diabetes"]}, {}, profile(50, ["asthma"]), False),
    ({"age_at_least": 40}, {}, profile(None, []), True),  # unknown age can't rule out a risk
    ({"any": [{"slot": "systolic_bp", "at_least": 180}, {"slot": "diastolic_bp", "at_least": 120}]},
     {"systolic_bp": 150}, YOUNG, None),  # the other number is still needed
    ({"any": [{"slot": "systolic_bp", "at_least": 180}, {"slot": "diastolic_bp", "at_least": 120}]},
     {"systolic_bp": 190}, YOUNG, True),
    ({"all": [{"slot": "severity", "equals": "moderate"}, {"has_conditions": True}]},
     {"severity": "mild"}, MUM, False),
])
def test_rules_are_true_false_or_waiting_for_a_value(rule, slots, member, expected):
    assert evaluate(rule, slots, member) is expected


@pytest.mark.parametrize("bad", [
    "questions: {urgent: [{id: a, ask: x}]}",
    "questions: {emergency: [{id: a}]}",
    "questions: {emergency: [{id: a, ask: x}, {id: a, ask: y}]}",
    "questions: {doctor_now: [{id: a, rule: {slot: temperature_c, at_least: 40}}]}",
    "questions: {doctor_now: [{id: a, ask: x, rule: {weight_over: 3}}]}",
    "questions: {emergency: [{id: a, ask: x}]}",  # no sign to record the answer against
])
def test_invalid_protocols_are_rejected(tmp_path, bad):
    base = ("id: x\ntitle: X\nsource: test\nversion: 1\nhome_care: []\nwatch_for: []\n"
            "check_back_hours: 1\n")
    (tmp_path / "x.yaml").write_text(base + bad)
    with pytest.raises(ValueError, match="Invalid protocol x.yaml"):
        load_protocols(tmp_path)


THROAT_SIGNS = {f"sign_{i}": {"sign": sign, "level": "emergency", "ask": f"{sign}?"}
                for i, sign in enumerate(["Drooling or unable to swallow saliva", "Trouble breathing",
                                          "Swelling of the neck"], 1)}


def test_the_generic_guideline_asks_the_agent_for_danger_signs_first():
    result = triage(PROTOCOLS["generic"], {}, {}, YOUNG)
    assert [(q.id, q.level) for q in result.pending] == [(NEW_SIGN, "emergency")]


def test_signs_the_agent_chose_are_asked_and_count_towards_the_minimum():
    protocol = with_signs(PROTOCOLS["generic"], THROAT_SIGNS)
    assert [q.id for q in triage(protocol, THROAT_SIGNS, {}, YOUNG).pending] == ["sign_1"]
    two = {k: {**v, "value": "absent", "quote": "no"} for k, v in list(THROAT_SIGNS.items())[:2]}
    assert [q.id for q in triage(with_signs(PROTOCOLS["generic"], two), two, {}, YOUNG).pending] == [NEW_SIGN]
    ruled_out = {k: {**v, "value": "absent", "quote": "no"} for k, v in THROAT_SIGNS.items()}
    assert [q.id for q in triage(protocol, ruled_out, {}, YOUNG).pending] == ["severe"]  # then severity
    present = {**ruled_out, "sign_2": {**THROAT_SIGNS["sign_2"], "value": "present", "quote": "yes"}}
    assert triage(protocol, present, {}, YOUNG).disposition == "emergency"


def test_an_added_sign_extends_a_fixed_protocol_after_its_own_questions():
    added = {"sign_1": {"sign": "Coughing up pink frothy sputum", "level": "emergency", "ask": "?"}}
    protocol = with_signs(CHEST, added)
    assert [q.id for q in protocol.at("emergency")][-1] == "sign_1"
    assert [q.id for q in triage(protocol, {**EMERGENCY_NO, **added}, {}, YOUNG).pending] == ["sign_1"]


def test_negative_generated_signs_never_authorize_home_monitoring():
    findings = {k: {**v, "value": "absent", "quote": "no"} for k, v in THROAT_SIGNS.items()}
    result = triage(with_signs(PROTOCOLS["generic"], findings), findings,
                    {"severity": "mild", "duration_hours": 12}, YOUNG)
    assert result.disposition == "doctor_24h"


def test_generic_protocol_cannot_opt_into_home_care(tmp_path):
    source = ("id: generic\ntitle: Generic\nsource: test\nversion: 1\nquestions: {}\n"
              "home_care: []\nwatch_for: []\ncheck_back_hours: 24\nallow_home_care: true\n")
    (tmp_path / "generic.yaml").write_text(source)
    with pytest.raises(ValueError, match="false for the generic"):
        load_protocols(tmp_path)
