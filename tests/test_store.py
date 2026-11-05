"""Integration tests against a real Postgres. Skipped when DATABASE_URL is unset."""

from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from app.memory.store import Store, normalize_subject
from app.memory.writes import EpisodeOwnershipError, FactWrite, SessionOwnershipError, TurnConflictError
from tests.helpers import fact, fever_episode, make_turn, new_id

pytestmark = pytest.mark.integration

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "household.json"


def test_first_load_opens_a_new_session(store, household):
    household_id, _, _ = household
    session_id = new_id("s")

    first = store.load_context(household_id, session_id)
    second = store.load_context(household_id, session_id)

    assert first.is_new_session and not second.is_new_session
    assert [m.name for m in first.members] == ["Asha", "Leo"]
    assert first.members[0].conditions == ("hypertension",)


def test_session_from_another_household_is_rejected(store, household):
    household_id, _, _ = household
    session_id = new_id("s")
    store.load_context(household_id, session_id)

    with pytest.raises(SessionOwnershipError):
        store.load_context(new_id("other"), session_id)


def test_saved_turn_survives_a_restart(store, household, database_url):
    household_id, leo, _ = household
    session_id, episode_id = new_id("s"), new_id("e")
    store.load_context(household_id, session_id)
    turn = make_turn(
        household_id,
        session_id,
        member_id=leo,
        episode=fever_episode(episode_id, leo, {"duration_hours": 12}),
        facts=[FactWrite(member_id=leo, episode_id=episode_id, kind="symptom", subject="fever", text="felt hot overnight")],
        pending=episode_id,
    )

    assert store.save_turn(turn)
    restarted = Store.connect(database_url)
    try:
        context = restarted.load_context(household_id, session_id)
    finally:
        restarted.close()

    assert [m.role for m in context.recent_messages] == ["user", "assistant"]
    assert context.member_hint_id == leo and context.pending_episode_id == episode_id
    [episode] = context.episodes
    assert episode.slots == {"duration_hours": 12} and episode.questions_asked == 1
    [fact] = context.facts
    assert (fact.text, fact.status, fact.source) == ("felt hot overnight", "provisional", "user_report")
    assert (fact.kind, fact.subject, fact.session_id, fact.episode_id) == ("symptom", "fever", session_id, episode_id)


def test_reprocessing_a_turn_does_not_duplicate_it(store, household):
    household_id, leo, _ = household
    session_id, episode_id = new_id("s"), new_id("e")
    store.load_context(household_id, session_id)
    turn = make_turn(household_id, session_id, member_id=leo, episode=fever_episode(episode_id, leo, {}))

    assert store.save_turn(turn)
    assert not store.save_turn(turn)

    context = store.load_context(household_id, session_id)
    assert len(context.recent_messages) == 2
    assert context.episodes[0].questions_asked == 1


def test_slots_merge_across_turns(store, household):
    household_id, leo, _ = household
    session_id, episode_id = new_id("s"), new_id("e")
    store.load_context(household_id, session_id)

    store.save_turn(make_turn(household_id, session_id, episode=fever_episode(episode_id, leo, {"duration_hours": 12})))
    store.save_turn(make_turn(household_id, session_id, episode=fever_episode(episode_id, leo, {"temperature_c": 38.9})))

    [episode] = store.load_context(household_id, session_id).episodes
    assert episode.slots == {"duration_hours": 12, "temperature_c": 38.9}
    assert episode.questions_asked == 2


def test_turn_keys_are_scoped_to_session_and_replay_checks_text(store, household):
    household_id, _, _ = household
    sessions = [new_id("s"), new_id("s")]
    for session_id in sessions:
        store.load_context(household_id, session_id)
        assert store.save_turn(make_turn(household_id, session_id, turn_id="first", text="hello"))
        assert store.saved_turn(household_id, session_id, "first", "hello")["reply"] == "How high is his temperature?"
        with pytest.raises(TurnConflictError):
            store.saved_turn(household_id, session_id, "first", "changed")
    assert store.saved_turn("another-household", sessions[0], "first", "hello") is None


def test_episode_of_another_member_is_rejected_and_the_turn_rolled_back(store, household):
    household_id, leo, asha = household
    session_id, episode_id = new_id("s"), new_id("e")
    store.load_context(household_id, session_id)
    store.save_turn(make_turn(household_id, session_id, episode=fever_episode(episode_id, leo, {})))

    with pytest.raises(EpisodeOwnershipError):
        store.save_turn(make_turn(household_id, session_id, episode=fever_episode(episode_id, asha, {})))

    assert len(store.load_context(household_id, session_id).recent_messages) == 2


def test_resolved_episodes_drop_out_after_the_lookback(store, household, database_url):
    household_id, leo, _ = household
    session_id, old, recent = new_id("s"), new_id("e"), new_id("e")
    store.load_context(household_id, session_id)
    for episode_id in (old, recent):
        store.save_turn(
            make_turn(household_id, session_id, episode=fever_episode(episode_id, leo, {}, "resolved_reported"))
        )
    with psycopg.connect(database_url) as conn:
        conn.execute("UPDATE episodes SET updated_at = now() - interval '15 days' WHERE id = %s", (old,))

    assert [e.id for e in store.load_context(household_id, session_id).episodes] == [recent]


def test_recent_messages_are_capped_and_oldest_first(store, household):
    household_id, _, _ = household
    session_id = new_id("s")
    store.load_context(household_id, session_id)
    turn_ids = [new_id("t") for _ in range(6)]
    for turn_id in turn_ids:
        store.save_turn(make_turn(household_id, session_id, turn_id=turn_id))

    messages = store.load_context(household_id, session_id).recent_messages

    assert len(messages) == 10
    assert [m.id for m in messages] == sorted(m.id for m in messages)
    assert messages[-1].role == "assistant"


def test_a_user_report_can_never_be_verified(store, household, database_url):
    household_id, leo, _ = household
    session_id = new_id("s")
    store.load_context(household_id, session_id)
    store.save_turn(make_turn(household_id, session_id, facts=[fact(leo, "has asthma", kind="condition", subject="asthma")]))
    [reported] = store.load_context(household_id, session_id).facts

    with psycopg.connect(database_url) as conn, pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE member_facts SET status = 'verified' WHERE id = %s", (reported.id,))


def test_readings_share_a_normalised_subject_and_keep_their_measured_time(store, household):
    household_id, _, asha = household
    session_id = new_id("s")
    measured = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)
    store.load_context(household_id, session_id)
    readings = [
        fact(asha, "BP 148/94 this morning", subject="Blood Pressure", effective_at=measured),
        fact(asha, "BP 152/96 after lunch", subject="blood-pressure"),
    ]

    store.save_turn(make_turn(household_id, session_id, facts=readings))

    facts = store.load_context(household_id, session_id).facts
    assert {f.subject for f in facts} == {"blood_pressure"}
    assert [f.effective_at for f in facts] == [measured, None]


def test_unknown_fact_kind_is_rejected(store, household):
    household_id, leo, _ = household
    session_id = new_id("s")
    store.load_context(household_id, session_id)

    with pytest.raises(psycopg.errors.CheckViolation):
        store.save_turn(make_turn(household_id, session_id, facts=[fact(leo, "likes dinosaurs", kind="hobby")]))


@pytest.mark.parametrize(
    ("raw", "normalised"),
    [("Blood Pressure", "blood_pressure"), ("blood-pressure", "blood_pressure"), ("  Fever!  ", "fever")],
)
def test_normalize_subject(raw, normalised):
    assert normalize_subject(raw) == normalised


def test_fixture_household_loads(store):
    store.load_fixture(FIXTURE_PATH)

    context = store.load_context("demo", new_id("s"))

    assert {m.name for m in context.members} == {"Ravi", "Asha", "Maya", "Ashwin"}
