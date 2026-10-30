"""Consolidation against a real Postgres, with the LLM replaced by deterministic fakes."""

import json
from datetime import UTC, datetime

import psycopg
import pytest

from app.consolidation import (
    SESSION_IDLE_MINUTES,
    ConsolidationResult,
    FactDecision,
    MissedFact,
    consolidate_idle_sessions,
    consolidate_session,
    render_input,
)
from tests.helpers import fact, make_turn, new_id

pytestmark = pytest.mark.integration


def keep_all(summary="Discussed Leo's fever; home monitoring advised."):
    def consolidate(source):
        decisions = [
            FactDecision(fact_id=f.id, action="keep", other_fact_id=None, subject=f.subject)
            for f in source.session_facts
        ]
        return ConsolidationResult(summary=summary, fact_decisions=decisions, missed_facts=[])

    return consolidate


def returning(**result):
    return lambda source: ConsolidationResult(**{"summary": "s", "fact_decisions": [], "missed_facts": [], **result})


def closed_session(store, household_id, member_id, facts):
    session_id = new_id("s")
    store.load_context(household_id, session_id)
    for item in facts:
        store.save_turn(make_turn(household_id, session_id, member_id=member_id, text=item.text, facts=[item]))
    assert store.close_session(household_id, session_id)
    return session_id


def age_session(database_url, session_id, minutes):
    with psycopg.connect(database_url) as conn:
        conn.execute(
            "UPDATE sessions SET last_active_at = now() - make_interval(mins => %s) WHERE id = %s", (minutes, session_id)
        )


def statuses(database_url, *session_ids):
    with psycopg.connect(database_url) as conn:
        rows = conn.execute("SELECT id, status FROM sessions WHERE id = ANY(%s)", (list(session_ids),)).fetchall()
    return dict(rows)


def test_only_idle_sessions_other_than_the_current_one_are_closed(store, household, database_url):
    household_id, _, _ = household
    idle, recent, current = new_id("s"), new_id("s"), new_id("s")
    for session_id in (idle, recent, current):
        store.load_context(household_id, session_id)
    age_session(database_url, idle, SESSION_IDLE_MINUTES + 1)
    age_session(database_url, current, SESSION_IDLE_MINUTES + 1)

    closed = store.close_idle_sessions(household_id, current, SESSION_IDLE_MINUTES)

    assert closed == [idle]
    assert statuses(database_url, idle, recent, current) == {idle: "closed", recent: "active", current: "active"}


def test_llm_input_holds_the_whole_session_and_long_term_memory(store, household):
    household_id, leo, _ = household
    session_id = closed_session(
        store, household_id, leo, [fact(leo, "felt hot overnight", kind="symptom"), fact(leo, "38.9 C at 9pm")]
    )

    source = store.load_consolidation_input(session_id)
    payload = json.loads(render_input(source))

    assert [m["role"] for m in payload["transcript"]] == ["user", "assistant", "user", "assistant"]
    assert payload["transcript"][1]["decision"] == "ask"
    assert payload["session"]["status"] == "closed"
    assert len(payload["members"]) == 2 and len(payload["facts"]) == 2
    assert payload["session_fact_ids"] == [f.id for f in source.session_facts]


def test_consolidation_writes_the_summary_and_promotes_facts(store, household):
    household_id, leo, _ = household
    session_id = closed_session(store, household_id, leo, [fact(leo, "felt hot overnight", kind="symptom")])

    assert consolidate_session(store, keep_all(), session_id)

    later = store.load_context(household_id, new_id("s"))
    assert [s.summary for s in later.prior_summaries] == ["Discussed Leo's fever; home monitoring advised."]
    assert [(f.text, f.status) for f in later.facts] == [("felt hot overnight", "consolidated")]


def test_duplicates_and_corrections_supersede_earlier_facts(store, household):
    household_id, leo, _ = household
    first = closed_session(
        store,
        household_id,
        leo,
        [
            fact(leo, "allergic to penicillin", kind="allergy", subject="penicillin"),
            fact(leo, "takes ibuprofen daily", kind="medication", subject="ibuprofen"),
        ],
    )
    consolidate_session(store, keep_all(), first)
    earlier = {f.text: f for f in store.load_context(household_id, new_id("s")).facts}
    second = closed_session(
        store,
        household_id,
        leo,
        [
            fact(leo, "penicillin allergy", kind="allergy", subject="penicillin"),
            fact(leo, "stopped ibuprofen last week", kind="medication", subject="ibuprofen"),
        ],
    )
    repeat, stopped = store.load_consolidation_input(second).session_facts
    decisions = [
        FactDecision(
            fact_id=repeat.id,
            action="duplicate_of",
            other_fact_id=earlier["allergic to penicillin"].id,
            subject="penicillin",
        ),
        FactDecision(
            fact_id=stopped.id, action="replaces", other_fact_id=earlier["takes ibuprofen daily"].id, subject="ibuprofen"
        ),
    ]

    assert consolidate_session(store, returning(fact_decisions=decisions), second)

    texts = {f.text for f in store.load_context(household_id, new_id("s")).facts}
    assert texts == {"allergic to penicillin", "stopped ibuprofen last week"}


def test_invalid_llm_references_are_ignored(store, household):
    household_id, leo, _ = household
    session_id = closed_session(store, household_id, leo, [fact(leo, "felt hot", kind="symptom"), fact(leo, "38.9 C")])
    hot, reading = store.load_consolidation_input(session_id).session_facts
    bad_message = MissedFact(
        member_id=leo, source_message_id=999_999_999, kind="note", subject="x", text="bad message", effective_at=None
    )
    bad_member = MissedFact(
        member_id="someone-else", source_message_id=1, kind="note", subject="x", text="bad member", effective_at=None
    )
    decisions = [
        FactDecision(fact_id=999_999_999, action="keep", other_fact_id=None, subject="x"),
        # A duplicate must point at an earlier session's fact, not one from this session.
        FactDecision(fact_id=hot.id, action="duplicate_of", other_fact_id=reading.id, subject="fever"),
    ]

    consolidate_session(store, returning(fact_decisions=decisions, missed_facts=[bad_message, bad_member]), session_id)

    facts = store.load_context(household_id, new_id("s")).facts
    assert sorted((f.text, f.status) for f in facts) == [("38.9 C", "consolidated"), ("felt hot", "consolidated")]


def test_missed_facts_are_added_with_their_source_message(store, household):
    household_id, leo, _ = household
    session_id = new_id("s")
    store.load_context(household_id, session_id)
    store.save_turn(make_turn(
        household_id, session_id, member_id=leo, text="felt hot, 38.9 C, slept badly",
        facts=[fact(leo, "felt hot", kind="symptom")],
    ))
    store.close_session(household_id, session_id)
    [user_message] = [m for m in store.load_consolidation_input(session_id).transcript if m.role == "user"]
    missed = [
        MissedFact(
            member_id=leo,
            source_message_id=user_message.id,
            kind="observation",
            subject="Temperature",
            text="38.9 C",
            effective_at="2026-10-25T21:00:00+00:00",
        ),
        MissedFact(
            member_id=leo,
            source_message_id=user_message.id,
            kind="note",
            subject="sleep",
            text="slept badly",
            effective_at="2026-10-25T23:00:00",
        ),
    ]

    consolidate_session(store, returning(missed_facts=missed), session_id)

    added = {f.text: f for f in store.load_context(household_id, new_id("s")).facts if f.text != "felt hot"}
    reading = added["38.9 C"]
    assert (reading.subject, reading.status, reading.session_id) == ("temperature", "consolidated", session_id)
    assert reading.source_message_id == user_message.id
    assert reading.effective_at == datetime(2026, 10, 25, 21, tzinfo=UTC)
    assert added["slept badly"].effective_at is None  # a naive time is ambiguous


def test_missed_facts_require_quoted_evidence_and_the_recorded_member(store, household):
    household_id, leo, asha = household
    session_id = closed_session(store, household_id, leo, [fact(leo, "felt hot", kind="symptom")])
    message = store.load_consolidation_input(session_id).transcript[0]
    missed = [
        MissedFact(member_id=asha, source_message_id=message.id, kind="symptom", subject="fever",
                   text="felt hot", effective_at=None),
        MissedFact(member_id=leo, source_message_id=message.id, kind="observation", subject="temperature",
                   text="39.9 C", effective_at=None),
    ]
    assert consolidate_session(store, returning(missed_facts=missed), session_id)
    facts = store.load_context(household_id, new_id("s")).facts
    assert [(f.member_id, f.text) for f in facts] == [(leo, "felt hot")]


def test_a_session_is_consolidated_only_once(store, household):
    household_id, leo, _ = household
    session_id = closed_session(store, household_id, leo, [fact(leo, "felt hot", kind="symptom")])
    calls = []

    def counting(source):
        calls.append(source.session.id)
        return keep_all()(source)

    assert consolidate_session(store, counting, session_id)
    assert not consolidate_session(store, counting, session_id)
    assert calls == [session_id]


def test_a_session_reopened_during_the_llm_call_is_left_alone(store, household, database_url):
    household_id, leo, _ = household
    session_id = closed_session(store, household_id, leo, [fact(leo, "felt hot", kind="symptom")])

    def reopening(source):
        store.load_context(household_id, session_id)  # the family sends another message meanwhile
        return keep_all()(source)

    assert not consolidate_session(store, reopening, session_id)
    assert statuses(database_url, session_id) == {session_id: "active"}
    assert [f.status for f in store.load_context(household_id, session_id).facts] == ["provisional"]


def test_a_failed_llm_call_leaves_the_session_closed_and_it_is_retried(store, household):
    household_id, leo, _ = household
    session_id = closed_session(store, household_id, leo, [fact(leo, "felt hot", kind="symptom")])

    def failing(source):
        raise RuntimeError("model unavailable")

    assert not consolidate_session(store, failing, session_id)
    assert consolidate_idle_sessions(store, keep_all(), household_id, new_id("s")) == [session_id]


def test_idle_sessions_are_closed_and_consolidated_together(store, household, database_url):
    household_id, leo, _ = household
    idle = new_id("s")
    store.load_context(household_id, idle)
    store.save_turn(make_turn(household_id, idle, member_id=leo, facts=[fact(leo, "felt hot", kind="symptom")]))
    age_session(database_url, idle, SESSION_IDLE_MINUTES + 1)

    assert consolidate_idle_sessions(store, keep_all(), household_id, new_id("s")) == [idle]
    assert statuses(database_url, idle) == {idle: "consolidated"}


@pytest.mark.parametrize("fails", [False, True])
def test_consolidation_has_its_own_session_trace(store, household, fails):
    from langsmith import get_current_run_tree, trace, tracing_context

    household_id, leo, _ = household
    session = closed_session(store, household_id, leo, [fact(leo, "felt hot", kind="symptom")])
    captured = []

    def traced_consolidator(source):
        captured.append(get_current_run_tree())
        if fails:
            raise RuntimeError("synthetic model failure")
        return keep_all()(source)

    # Local traces exercise real SDK context propagation without exporting health data.
    with tracing_context(enabled="local"), trace("caller"):
        assert consolidate_session(store, traced_consolidator, session) == (not fails)
    run, = captured
    assert run.name == "q.consolidation" and run.parent_run_id is None
    assert run.metadata["session_id"] == session
    assert run.metadata["household_id"] == household_id
    assert bool(run.error) == fails
    if not fails:
        assert run.outputs == {"applied": True}
