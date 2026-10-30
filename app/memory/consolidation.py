"""Session consolidation: one structured LLM call turns a closed session into long-term memory.

Runs in the background, never on the response path. The LLM sees the whole session and the household's
long-term memory, but code validates every ID it returns before anything is written. It can only
summarise, promote, deduplicate or supersede this session's facts, and add facts the user stated.
It can never verify a fact, change an episode or edit procedure.
"""

import json
import logging
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Literal

from langchain_openai import ChatOpenAI
from langsmith import trace
from pydantic import BaseModel

from app.memory.serialize import json_default, plain
from app.memory.store import Store, normalize_subject
from app.memory.writes import ConsolidationWrite, FactWrite, MissedFactWrite
from app.models.records import ConsolidationInput, FactKind

logger = logging.getLogger(__name__)


SESSION_IDLE_MINUTES = 30


MAX_SUMMARY_CHARS = 1500


class FactDecision(BaseModel):
    fact_id: int
    action: Literal["keep", "duplicate_of", "replaces"]
    other_fact_id: int | None
    subject: str


class MissedFact(BaseModel):
    member_id: str
    source_message_id: int
    kind: FactKind
    subject: str
    text: str
    effective_at: str | None


class ConsolidationResult(BaseModel):
    summary: str
    fact_decisions: list[FactDecision]
    missed_facts: list[MissedFact]


Consolidate = Callable[[ConsolidationInput], ConsolidationResult]


INSTRUCTIONS = """\
You consolidate one finished conversation of Q, a family health assistant, into the household's long-term memory.

The input is JSON with:
- session: the session's state.
- transcript: every message, oldest first. User messages are what the family said. Assistant messages are Q's \
replies, with Q's decision (emergency, doctor, ask, answer, home_monitoring, ask_member, ask_episode).
- members: the household.
- facts: long-term facts already stored, with kind, subject, status and source. session_fact_ids lists the ones \
recorded during this session.
- episodes: health concerns being tracked.
- prior_summaries: summaries of earlier sessions.

Return three things.

summary: 2 to 5 plain sentences for Q to read in a later session. Name the members discussed, their concerns, \
what Q decided, and anything left open (unanswered questions, monitoring check-backs). Use only what the \
transcript says. No diagnoses, advice, speculation or reassurance. It is read on later days, so date it: open \
with the session's date from the transcript's created_at ("On 28 Oct 2026, ..."), and write when things \
happened as dates or times of that day ("150/95 that morning"), never a bare today, yesterday or now.

fact_decisions: exactly one decision for each id in session_fact_ids.
- keep: a distinct fact. Set subject to a lowercase snake_case concept label, reusing an existing subject for \
the same concept (blood_pressure, not bp).
- duplicate_of: it says the same thing as an earlier fact about the same member. other_fact_id is that fact.
- replaces: the user corrected or updated an earlier fact about the same member. other_fact_id is the outdated \
fact. A new measurement is not a replacement: keep every reading.
other_fact_id must be a fact from an earlier session (not in session_fact_ids), and null for keep.

missed_facts: facts the user explicitly stated in this session that no fact records yet. Cite the user message \
id the fact comes from. text must be an exact quote from that message. member_id must equal the message's \
recorded member_id; skip unassigned messages and facts about a different member in mixed-member messages. \
Only include what the user said about a clearly identified member. Never include Q's \
statements, inferences or diagnoses. effective_at is an ISO 8601 timestamp with a UTC offset when the user said \
when it happened or was measured; otherwise null.
"""


def llm_consolidator(model: ChatOpenAI) -> Consolidate:
    """Bind the mini-tier model to the consolidation schema with strict structured output."""
    structured = model.with_structured_output(ConsolidationResult, method="json_schema", strict=True,
                                              prompt_cache_key="q:consolidation")

    def consolidate(source: ConsolidationInput) -> ConsolidationResult:
        return structured.invoke([("system", INSTRUCTIONS), ("human", render_input(source))])

    return consolidate


def consolidate_idle_sessions(
    store: Store, consolidate: Consolidate, household_id: str, active_session_id: str
) -> list[str]:
    """Close the household's idle sessions and consolidate each. Returns the IDs consolidated."""
    session_ids = store.close_idle_sessions(household_id, active_session_id, SESSION_IDLE_MINUTES)
    return [session_id for session_id in session_ids if consolidate_session(store, consolidate, session_id)]


def consolidate_session(store: Store, consolidate: Consolidate, session_id: str) -> bool:
    source = store.load_consolidation_input(session_id)
    if source is None:
        return False  # reopened, already consolidated, or unknown
    # Background work is a separate root, even if the caller has an active trace.
    with trace(
        "q.consolidation", parent="ignore", tags=["consolidation"],
        inputs={"session_id": session_id},
        metadata={"session_id": session_id, "household_id": source.session.household_id},
    ) as run:
        try:
            result = consolidate(source)
        except Exception as exc:
            run.end(error=f"{type(exc).__name__}: {exc}")
            logger.exception("consolidation LLM call failed for session %s; it stays closed and is retried", session_id)
            return False
        applied = store.apply_consolidation(plan_from(source, result))
        run.end(outputs={"applied": applied})
        logger.info("session %s consolidation %s", session_id, "applied" if applied else "skipped: session changed")
        return applied


def render_input(source: ConsolidationInput) -> str:
    payload = {
        "session": plain(source.session),
        "session_fact_ids": [f.id for f in source.session_facts],
        "transcript": [plain(m) for m in source.transcript],
        "members": [plain(m) for m in source.members],
        "facts": [plain(f) for f in source.facts],
        "episodes": [plain(e) for e in source.episodes],
        "prior_summaries": [plain(s) for s in source.prior_summaries],
    }
    return json.dumps(payload, default=json_default)


def plan_from(source: ConsolidationInput, result: ConsolidationResult) -> ConsolidationWrite:
    """Validate the LLM output against the session's own data. Invalid items are logged and skipped."""
    session_facts = {f.id: f for f in source.session_facts}
    earlier_facts = {f.id: f for f in source.facts if f.id not in session_facts}
    promotions, duplicates, replacements, decided = [], [], [], set()
    for decision in result.fact_decisions:
        fact = session_facts.get(decision.fact_id)
        if fact is None or fact.id in decided:
            logger.warning("consolidation decision for unknown or repeated fact %s skipped", decision.fact_id)
            continue
        decided.add(fact.id)
        other = earlier_facts.get(decision.other_fact_id)
        target_ok = other is not None and other.member_id == fact.member_id
        if decision.action == "duplicate_of" and target_ok:
            duplicates.append((fact.id, other.id))
            continue
        if decision.action == "replaces" and target_ok:
            replacements.append((fact.id, other.id))
        elif decision.action != "keep":
            logger.warning("fact %s: invalid %s target %s; kept", fact.id, decision.action, decision.other_fact_id)
        promotions.append((fact.id, normalize_subject(decision.subject) or fact.subject))
    promotions.extend((f.id, f.subject) for f in session_facts.values() if f.id not in decided)
    return ConsolidationWrite(
        session_id=source.session.id,
        summary=result.summary.strip()[:MAX_SUMMARY_CHARS] or None,
        promotions=tuple(promotions),
        duplicates=tuple(duplicates),
        replacements=tuple(replacements),
        missed_facts=tuple(_missed_facts(source, result)),
    )


def _missed_facts(source: ConsolidationInput, result: ConsolidationResult) -> Iterator[MissedFactWrite]:
    member_ids = {m.id for m in source.members}
    user_messages = {m.id: m for m in source.transcript if m.role == "user"}
    for missed in result.missed_facts:
        message = user_messages.get(missed.source_message_id)
        subject = normalize_subject(missed.subject)
        if (
            missed.member_id not in member_ids
            or message is None
            or message.member_id != missed.member_id
            or not subject
            or not missed.text.strip()
            or missed.text.strip() not in message.content
        ):
            logger.warning("missed fact skipped: unsupported evidence or member (message %s)", missed.source_message_id)
            continue
        fact = FactWrite(
            member_id=missed.member_id,
            episode_id=message.episode_id if message.member_id == missed.member_id else None,
            kind=missed.kind,
            subject=subject,
            text=missed.text.strip(),
            effective_at=_parse_time(missed.effective_at),
        )
        yield MissedFactWrite(fact=fact, source_message_id=message.id)


def _parse_time(value: str | None) -> datetime | None:
    """Accept only timezone-aware ISO 8601; a naive time is ambiguous, so it is dropped."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        logger.warning("unparseable effective_at %r dropped", value)
        return None
    return parsed if parsed.tzinfo else None
