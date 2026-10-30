"""The intake agent words each question; code chooses what it asks and checks the wording."""

import pytest
from langchain_core.messages import ToolMessage

from app.agents.intake.questions import (
    FALLBACK_LEAD,
    MAX_ADDED_SIGNS,
    MAX_MESSAGE_CHARS,
    IntakeQuestions,
    NewSign,
    checked_reply,
    sign_question,
    with_new_sign,
)
from app.agents.loop import MAX_AGENT_STEPS, ModelOutputError
from app.memory.writes import EpisodeWrite
from app.models.outputs import Observation
from app.triage.engine import profile, triage
from app.triage.protocol import load_protocols, with_signs
from tests.fakes import FakeModel, agent_payload, answered, ask_pending, call, episode, member, observation
from tests.helpers import fever_episode, make_turn, new_id, post

FEVER = load_protocols()["fever"]
SLOTS = {"duration_hours": 12, "severity": "mild", "temperature_c": 38}
EMERGENCY_RULED_OUT = {q: {"value": "absent", "quote": "no"} for q in ("stiff_neck", "confused", "hard_to_wake",
                                                                    "rash_not_fading")}


def fever_triage(findings):
    return triage(FEVER, findings, SLOTS, profile(2, []))


GENERIC = load_protocols()["generic"]
DROOLING = NewSign(sign="Drooling or unable to swallow saliva")
CHOSEN_RULED_OUT = {f"sign_{i}": {"sign": sign, "level": "emergency", "ask": f"{sign}?", "value": "absent",
                                  "quote": "no"} for i, sign in enumerate(["Trouble breathing", "Drooling", "Stiff neck"], 1)}


def generic_triage(findings):
    return triage(with_signs(GENERIC, findings), findings, {}, profile(2, []))


def test_the_agents_own_wording_is_sent():
    message = "You mentioned the cough has lasted two weeks. First, a routine check: is your neck stiff?"
    assert checked_reply(IntakeQuestions(message=message, new_sign=None), fever_triage({})) == message


@pytest.mark.parametrize("message", ["  ", "Stiff neck? Confused?", "Tell me about his neck.",
                                     "Is it stiff? Let me know.", "?" * 2, "x" * MAX_MESSAGE_CHARS + "?"])
def test_wording_that_is_not_one_short_question_falls_back_to_the_protocol(message):
    assert checked_reply(IntakeQuestions(message=message, new_sign=None), fever_triage({})) == "Is there a stiff neck?"


def test_an_answered_question_moves_the_fallback_to_the_next_open_one():
    found = fever_triage({**EMERGENCY_RULED_OUT, "drinking_less": {"value": "absent", "quote": "no"}})
    assert checked_reply(IntakeQuestions(message="", new_sign=None), found) == (
        "Are they passing much less urine than usual?")


@pytest.mark.integration
def test_the_payload_tells_the_agent_who_it_is_talking_to(store, household, api_client):
    household_id, leo, _ = household
    model = FakeModel(members=[member(leo)], episodes=[episode(status="new")],
                      observations=[observation(intent="symptom_report", complaint="fever", slots=SLOTS)])
    post(api_client(model), household_id, new_id("s"), "Leo has a fever")
    payload = agent_payload(model.inputs[IntakeQuestions][0])
    assert payload["writer_is_patient"] is False and payload["questions_asked_before"] == 0
    assert "Is there a stiff neck?" not in payload["still_to_check"]  # the current question is not repeated there
    assert payload["still_to_check"]


@pytest.mark.integration
def test_the_agent_can_look_up_history_and_word_the_question(store, household, api_client):
    household_id, _, asha = household  # Asha: 72, hypertension
    message = "Sorry Asha's unwell, especially with her blood pressure. Is the chest pain happening right now?"

    def write_questions(messages):
        assert isinstance(messages[-1], ToolMessage) and messages[-1].name == "member_facts"
        return IntakeQuestions(message=message, new_sign=None)

    model = FakeModel(members=[member(asha)], episodes=[episode(status="new")],
                      observations=[observation(intent="symptom_report", complaint="chest_pain")],
                      intakes=[call("member_facts", member_id=asha), write_questions])
    result = post(api_client(model), household_id, new_id("s"), "Mum has chest pain on and off")
    assert result.json()["decision"] == "ask"
    assert result.json()["reply"] == message
    payload = agent_payload(model.inputs[IntakeQuestions][0])
    assert payload["next_question"]["id"] == "pain_now"  # the most urgent sign, alone
    assert payload["already_positive_level"] == "doctor_now"  # her age and hypertension, already known
    assert model.reply_calls == 0


@pytest.mark.integration
def test_the_questions_are_still_asked_when_the_agent_never_finishes(store, household, api_client):
    household_id, leo, _ = household
    model = FakeModel(members=[member(leo)], episodes=[episode(status="new")],
                      observations=[observation(intent="symptom_report", complaint="fever", slots=SLOTS)],
                      intakes=[call("past_session_summaries")] * MAX_AGENT_STEPS)
    result = post(api_client(model), household_id, new_id("s"), "Leo has a fever")
    assert result.json()["decision"] == "ask"
    assert result.json()["reply"] == f"{FALLBACK_LEAD} Is there a stiff neck?"


@pytest.mark.integration
def test_a_pending_question_is_not_hidden_by_a_ruled_out_copy_of_its_wording(store, household, api_client):
    household_id, leo, _ = household  # Leo: 2, no conditions, so the "vulnerable and moderate" copy is ruled out
    session, runny = new_id("s"), new_id("e")
    store.load_context(household_id, session)
    store.save_turn(make_turn(household_id, session, member_id=leo, pending=runny, episode=EpisodeWrite(
        id=runny, member_id=leo, complaint="runny_nose", status="triaging", slots={}, questions_asked_delta=3,
        check_back_after=None, protocol_id="generic",
        findings={k: {**v, "session_id": session} for k, v in CHOSEN_RULED_OUT.items()})))  # answered in this session
    model = FakeModel(members=[member(leo)], episodes=[episode(runny)],
                      observations=[observation(intent="update", complaint="runny_nose")])
    assert post(api_client(model), household_id, session, "Still runny").json()["decision"] == "ask"
    payload = agent_payload(model.inputs[IntakeQuestions][0])
    severity = "How severe is it - mild, moderate or severe?"
    assert payload["next_question"]["ask"] == severity
    assert severity not in [q["ask"] for q in payload["already_known"]]


def test_on_a_choose_sign_turn_the_agent_must_name_its_sign():
    found = generic_triage({})
    with pytest.raises(ModelOutputError):
        checked_reply(IntakeQuestions(message="Is she drooling?", new_sign=None), found)
    chosen = IntakeQuestions(message="Is she drooling?", new_sign=DROOLING)
    assert checked_reply(chosen, found) == "Is she drooling?"
    assert with_new_sign(chosen, found, {}) == {
        "sign_1": {"sign": "Drooling or unable to swallow saliva", "level": "emergency",
                   "ask": sign_question(DROOLING.sign)}}


def test_a_sign_added_to_a_fixed_protocol_is_kept_for_later_and_capped():
    extra = IntakeQuestions(message="Is there a stiff neck?", new_sign=NewSign(sign="Had a seizure"))
    found = fever_triage({})
    assert checked_reply(extra, found) == "Is there a stiff neck?"
    assert with_new_sign(extra, found, {})["sign_1"] == {"sign": "Had a seizure", "level": "emergency",
                                                       "ask": sign_question("Had a seizure")}
    full = {f"sign_{i}": {"sign": str(i), "level": "emergency", "ask": "?"} for i in range(1, MAX_ADDED_SIGNS + 1)}
    assert with_new_sign(extra, found, full) == full


@pytest.mark.integration
def test_signs_chosen_for_a_complaint_without_a_protocol_are_saved_and_answered(store, household, api_client):
    household_id, leo, _ = household
    session, asked = new_id("s"), "Is Leo drooling, or struggling to swallow his own saliva?"
    reply = "No, he's swallowing fine"
    model = FakeModel(
        members=[member(leo), member(leo)],
        episodes=[episode(status="new"), lambda messages: episode(agent_payload(messages)["pending_episode_id"])],
        observations=[observation(intent="symptom_report", complaint="sore_throat"),
                      observation(intent="update", complaint="sore_throat", answers=answered(reply, sign_1="absent"))],
        intakes=[IntakeQuestions(message=asked, new_sign=DROOLING), ask_pending])
    client = api_client(model)
    assert post(client, household_id, session, "Leo has a sore throat").json()["reply"] == asked
    first, = store.load_context(household_id, session).episodes
    assert first.findings["sign_1"] == {"sign": "Drooling or unable to swallow saliva", "level": "emergency",
                                        "ask": sign_question(DROOLING.sign)}
    assert post(client, household_id, session, reply).json()["decision"] == "ask"
    assert agent_payload(model.inputs[IntakeQuestions][1])["next_question"]["choose_sign"] is True
    saved, = store.load_context(household_id, session).episodes
    assert saved.findings["sign_1"]["value"] == "absent" and saved.findings["sign_1"]["sign"] == DROOLING.sign
    assert "sign_2" in saved.findings  # the next sign the agent chose


@pytest.mark.integration
def test_a_fallback_sign_is_asked_when_the_agent_cannot_choose_one(store, household, api_client):
    household_id, leo, _ = household
    session = new_id("s")
    model = FakeModel(members=[member(leo)], episodes=[episode(status="new")],
                      observations=[observation(intent="symptom_report", complaint="sore_throat")],
                      intakes=[call("past_session_summaries")] * MAX_AGENT_STEPS)
    assert post(api_client(model), household_id, session, "Leo has a sore throat").json()["reply"] == (
        f"{FALLBACK_LEAD} Is it getting rapidly worse?")  # the guideline's reviewed wording
    saved, = store.load_context(household_id, session).episodes
    assert saved.findings["sign_1"]["sign"] == "Getting rapidly worse"


@pytest.mark.integration
def test_another_unfinished_intake_is_passed_on_so_it_is_not_dropped(store, household, api_client):
    household_id, leo, asha = household
    session, leo_fever = new_id("s"), new_id("e")
    store.load_context(household_id, session)
    store.save_turn(make_turn(household_id, session, member_id=leo, episode=fever_episode(leo_fever, leo, {})))
    model = FakeModel(members=[member(asha)], episodes=[episode(status="new")],
                      observations=[observation(intent="symptom_report", complaint="chest_pain")])
    assert post(api_client(model), household_id, session, "Also mum has chest pain").json()["decision"] == "ask"
    payload = agent_payload(model.inputs[IntakeQuestions][0])
    assert payload["other_open_intakes"] == [{"member": "Leo", "complaint": "fever"}]
    assert payload["unanswered_last_time"] is False


@pytest.mark.integration
def test_a_question_left_unanswered_is_flagged_so_it_is_rephrased(store, household, api_client):
    household_id, leo, _ = household
    session, leo_fever = new_id("s"), new_id("e")
    store.load_context(household_id, session)
    store.save_turn(make_turn(household_id, session, member_id=leo, episode=fever_episode(leo_fever, leo, {}),
                              pending=leo_fever))
    model = FakeModel(members=[member(leo)], episodes=[episode(leo_fever)],
                      observations=[observation(intent="update", complaint="fever")])
    assert post(api_client(model), household_id, session, "He's still hot").json()["decision"] == "ask"
    payload = agent_payload(model.inputs[IntakeQuestions][0])
    assert payload["next_question"]["id"] == "stiff_neck" and payload["unanswered_last_time"] is True
    assert agent_payload(model.inputs[Observation][0])["last_question"] == {
        "id": "stiff_neck", "finding": "Stiff neck"}  # observe is told the finding Q asked about
    assert payload["other_open_intakes"] == []
