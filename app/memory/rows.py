"""Map database rows to records."""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any

from app.models.records import (
    ConsolidationInput,
    Episode,
    Member,
    MemberFact,
    Message,
    SessionContext,
    SessionRecord,
    SessionSummary,
)


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _member(m: Mapping[str, Any]) -> Member:
    return Member(
        id=m["id"],
        name=m["name"],
        aliases=tuple(m["aliases"]),
        age_years=float(m["age_years"]) if m["age_years"] is not None else None,
        conditions=tuple(m["conditions"]),
    )


def _fact(f: Mapping[str, Any]) -> MemberFact:
    return MemberFact(
        id=f["id"],
        member_id=f["member_id"],
        session_id=f["session_id"],
        episode_id=f["episode_id"],
        kind=f["kind"],
        subject=f["subject"],
        text=f["text"],
        effective_at=_ts(f["effective_at"]),
        status=f["status"],
        source=f["source"],
        source_message_id=f["source_message_id"],
        created_at=_ts(f["created_at"]),
    )


def _episode(e: Mapping[str, Any]) -> Episode:
    return Episode(
        id=e["id"],
        member_id=e["member_id"],
        complaint=e["complaint"],
        status=e["status"],
        slots=MappingProxyType(e["slots"]),
        questions_asked=e["questions_asked"],
        check_back_after=_ts(e["check_back_after"]),
        updated_at=_ts(e["updated_at"]),
        protocol_id=e.get("protocol_id"),
        findings=MappingProxyType(e.get("findings") or {}),
        associated=MappingProxyType(e.get("associated") or {}),
    )


def to_message(msg: Mapping[str, Any]) -> Message:
    return Message(
        id=msg["id"],
        role=msg["role"],
        content=msg["content"],
        member_id=msg["member_id"],
        episode_id=msg["episode_id"],
        decision=msg["decision"],
        created_at=_ts(msg["created_at"]),
        session_id=msg.get("session_id"),
    )


def _summary(s: Mapping[str, Any]) -> SessionSummary:
    return SessionSummary(session_id=s["session_id"], summary=s["summary"], closed_at=_ts(s["closed_at"]))


def _session(s: Mapping[str, Any]) -> SessionRecord:
    return SessionRecord(
        id=s["id"],
        household_id=s["household_id"],
        status=s["status"],
        member_hint_id=s["member_hint_id"],
        pending_episode_id=s["pending_episode_id"],
        started_at=_ts(s["started_at"]),
        last_active_at=_ts(s["last_active_at"]),
        closed_at=_ts(s["closed_at"]),
    )


def to_context(row: Mapping[str, Any]) -> SessionContext:
    session = row["session"]
    return SessionContext(
        session_id=session["id"],
        is_new_session=session["is_new"],
        members=tuple(_member(m) for m in row["members"]),
        facts=tuple(_fact(f) for f in row["facts"]),
        episodes=tuple(_episode(e) for e in row["episodes"]),
        recent_messages=tuple(to_message(msg) for msg in row["messages"]),
        prior_summaries=tuple(_summary(s) for s in row["summaries"]),
        member_hint_id=session["member_hint_id"],
        pending_episode_id=session["pending_episode_id"],
    )


def to_consolidation_input(row: Mapping[str, Any]) -> ConsolidationInput:
    return ConsolidationInput(
        session=_session(row["session"]),
        transcript=tuple(to_message(msg) for msg in row["messages"]),
        members=tuple(_member(m) for m in row["members"]),
        facts=tuple(_fact(f) for f in row["facts"]),
        episodes=tuple(_episode(e) for e in row["episodes"]),
        prior_summaries=tuple(_summary(s) for s in row["summaries"]),
    )
