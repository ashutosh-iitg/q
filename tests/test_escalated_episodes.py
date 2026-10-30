"""An episode already handed to a doctor keeps its note and urgency: Q never hands it over twice."""

from app.memory.writes import EpisodeWrite
from app.triage.policy import ALREADY_WITH_DOCTOR, URGENCY_ADVICE, Decision
from tests.fakes import HANDOFF, FakeModel, episode, member, observation, screen
from tests.helpers import make_turn, new_id, post


def handed_over(store, household_id, member_id, urgency="doctor_24h"):
    """An episode Q already handed to a doctor, with the urgency and note it was handed over with."""
    session, cough = new_id("s"), new_id("e")
    store.load_context(household_id, session)
    store.save_turn(make_turn(household_id, session, member_id=member_id, episode=EpisodeWrite(
        id=cough, member_id=member_id, complaint="cough", status="doctor", slots={}, questions_asked_delta=1,
        check_back_after=None, protocol_id="cough", urgency=urgency, handoff_id="first-note",
    )))
    return session, cough


def test_the_store_keeps_an_episodes_urgency_and_note_across_later_updates(store, household):
    household_id, leo, _ = household
    session, cough = handed_over(store, household_id, leo)
    store.save_turn(make_turn(household_id, session, member_id=leo, episode=EpisodeWrite(
        id=cough, member_id=leo, complaint="cough", status="doctor", slots={}, questions_asked_delta=0,
        check_back_after=None)))
    saved, = store.load_context(household_id, session).episodes
    assert (saved.urgency, saved.handoff_id) == ("doctor_24h", "first-note")


def test_a_follow_up_on_a_handed_over_episode_reminds_without_a_second_note(store, household, api_client):
    household_id, leo, _ = household
    session, cough = handed_over(store, household_id, leo)
    model = FakeModel(members=[member(leo)], episodes=[episode(cough)],
                      observations=[observation(intent="update", complaint="cough")])
    client = api_client(model)
    turn = new_id("t")
    result = post(client, household_id, session, "Leo is still coughing", turn_id=turn).json()

    assert result["decision"] == Decision.DOCTOR
    assert result["reply"] == f"{ALREADY_WITH_DOCTOR} {URGENCY_ADVICE['doctor_24h']}"  # the original urgency
    assert model.inputs[HANDOFF] == []  # no note agent ran
    assert client.get(f"/households/{household_id}/handoffs/{turn}").status_code == 404


def test_a_new_doctor_sign_on_a_handed_over_episode_still_writes_a_new_note(store, household, api_client):
    household_id, leo, _ = household
    session, cough = handed_over(store, household_id, leo)
    sign = {"flag": "severe_pain", "quote": "terrible pain"}
    model = FakeModel(screens=[screen(doctor_flags=[sign])], members=[member(leo)], episodes=[episode(cough)],
                      observations=[observation(intent="update", complaint="cough")])
    result = post(api_client(model), household_id, session, "Leo is in terrible pain now").json()
    assert result["decision"] == Decision.DOCTOR and len(model.inputs[HANDOFF]) == 1
