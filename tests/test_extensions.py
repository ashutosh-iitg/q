"""Extension points: each out-of-scope capability plugs in through Extensions, and is absent by default."""

import pytest
from langchain_core.messages import ToolMessage

from app.agent.intake_agent import IntakeQuestions
from app.extensions import Extensions
from tests.fakes import FakeModel, call, draft, episode, member, observation, screen
from tests.helpers import new_id, post

LAB = [{"test": "Troponin", "value": "0.01 ng/mL", "range": "< 0.04", "date": "2026-10-20"}]


@pytest.mark.integration
def test_knowledge_base_passages_reach_the_reply_for_a_general_question(store, household, api_client):
    household_id, _, _ = household
    asked = []
    knowledge = Extensions(knowledge=lambda query: asked.append(query) or ["Sleep guidance: 9-12 hours at age 6-12."])
    model = FakeModel(replies=["Children of that age need 9-12 hours."])
    post(api_client(model, extensions=knowledge), household_id, new_id("s"), "How much sleep do kids need?")
    assert asked == ["How much sleep do kids need?"]
    assert model.reply_inputs[0]["reference"] == ["Sleep guidance: 9-12 hours at age 6-12."]


@pytest.mark.integration
def test_lab_results_are_a_lookup_only_when_an_analyser_is_plugged_in(store, household, api_client):
    household_id, _, asha = household

    def ask_after_reading_labs(messages):
        result = messages[-1]
        assert isinstance(result, ToolMessage) and result.name == "lab_results" and "Troponin" in result.content
        return IntakeQuestions(message="Is the chest pain happening right now?", new_sign=None)

    def model_for(intakes=None):
        return FakeModel(members=[member(asha)], episodes=[episode(status="new")], intakes=intakes,
                         observations=[observation(intent="symptom_report", complaint="chest_pain")])

    plugged = model_for([call("lab_results", member_id=asha), ask_after_reading_labs])
    labs = Extensions(lab_results=lambda household, member_id: LAB if member_id == asha else [])
    assert post(api_client(plugged, extensions=labs), household_id, new_id("s"), "Mum has chest pain").status_code == 200
    default = model_for()
    assert post(api_client(default), household_id, new_id("s"), "Mum has chest pain").status_code == 200
    assert not any("lab_results" in names for names, _ in default.bindings)


@pytest.mark.integration
@pytest.mark.parametrize("delivery_fails", [False, True])
def test_a_saved_note_is_delivered_and_a_failed_delivery_fails_the_turn(store, household, api_client, delivery_fails):
    household_id, leo, _ = household
    delivered = []

    def deliver(note):
        if delivery_fails:
            raise ConnectionError("telemedicine service unavailable")
        delivered.append(note)

    model = FakeModel(screens=[screen(doctor_requested=True)], members=[member(leo)], handoffs=[draft()])
    result = post(api_client(model, extensions=Extensions(deliver_handoff=deliver)), household_id, new_id("s"),
                  "I want a doctor to see Leo")
    assert result.status_code == (503 if delivery_fails else 200)
    assert [n["patient"]["member_id"] for n in delivered] == ([] if delivery_fails else [leo])
