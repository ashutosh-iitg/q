"""Postgres persistence: one read round trip and one write transaction per turn."""

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.agent.state import (
    ConsolidationInput,
    Episode,
    FactKind,
    FactStatus,
    Member,
    MemberFact,
    Message,
    SessionContext,
    SessionRecord,
    SessionSummary,
)
from app.policy import Decision

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
RECENT_MESSAGES = 10
PRIOR_SUMMARIES = 3
RESOLVED_LOOKBACK_DAYS = 14
EPISODE_HISTORY_MESSAGES = 20


class SessionOwnershipError(LookupError):
    """The session exists but belongs to another household."""


class EpisodeOwnershipError(LookupError):
    """The episode exists but belongs to another member."""


class TurnConflictError(ValueError):
    """An idempotency key was reused with different message text."""


@dataclass(frozen=True)
class EpisodeWrite:
    id: str  # caller generates the ID for a new episode
    member_id: str
    complaint: str
    status: str
    slots: Mapping[str, Any]  # merged into the stored slots
    questions_asked_delta: int
    check_back_after: datetime | None
    related_episode_id: str | None = None
    protocol_id: str | None = None
    findings: Mapping[str, Any] | None = None  # replaces the stored findings when given
    associated: Mapping[str, Any] | None = None  # replaces the stored associated symptoms when given


@dataclass(frozen=True)
class FactWrite:
    member_id: str
    episode_id: str | None
    kind: FactKind
    subject: str  # normalised on write
    text: str
    effective_at: datetime | None = None
    supersedes_id: int | None = None


@dataclass(frozen=True)
class TurnWrite:
    household_id: str
    session_id: str
    turn_id: str
    user_text: str
    reply: str
    decision: Decision
    member_id: str | None
    episode: EpisodeWrite | None
    facts: tuple[FactWrite, ...]
    pending_episode_id: str | None


@dataclass(frozen=True)
class MissedFactWrite:
    """A fact the transcript states but the turn missed, found at consolidation."""

    fact: FactWrite
    source_message_id: int


@dataclass(frozen=True)
class ConsolidationWrite:
    """Validated consolidation output. Every ID has been checked against the session's own data."""

    session_id: str
    summary: str | None
    promotions: tuple[tuple[int, str], ...]  # (fact_id, canonical subject)
    duplicates: tuple[tuple[int, int], ...]  # (fact_id, earlier fact it repeats)
    replacements: tuple[tuple[int, int], ...]  # (fact_id, earlier fact it replaces)
    missed_facts: tuple[MissedFactWrite, ...]


LOAD_CONTEXT_SQL = """
WITH s AS (
  INSERT INTO sessions (id, household_id) VALUES (%(session_id)s, %(household_id)s)
  ON CONFLICT (id) DO UPDATE SET last_active_at = now(), status = 'active'
    WHERE sessions.household_id = EXCLUDED.household_id
  RETURNING id, member_hint_id, pending_episode_id, (xmax = 0) AS is_new
),
m AS (SELECT * FROM members WHERE household_id = %(household_id)s)
SELECT
  (SELECT row_to_json(s) FROM s) AS session,
  (SELECT coalesce(json_agg(m ORDER BY m.name), '[]') FROM m) AS members,
  (SELECT coalesce(json_agg(f ORDER BY f.id), '[]')
     FROM member_facts f JOIN m ON m.id = f.member_id
    WHERE f.superseded_by IS NULL) AS facts,
  (SELECT coalesce(json_agg(e ORDER BY e.updated_at DESC), '[]')
     FROM episodes e JOIN m ON m.id = e.member_id
    WHERE e.status <> 'resolved_reported'
       OR e.updated_at > now() - make_interval(days => %(resolved_days)s)) AS episodes,
  (SELECT coalesce(json_agg(x ORDER BY x.id), '[]')
     FROM (SELECT * FROM messages WHERE session_id = %(session_id)s
           ORDER BY id DESC LIMIT %(recent)s) x) AS messages,
  (SELECT coalesce(json_agg(y ORDER BY y.closed_at), '[]')
     FROM (SELECT id AS session_id, summary, closed_at FROM sessions
            WHERE household_id = %(household_id)s AND status = 'consolidated' AND summary IS NOT NULL
            ORDER BY closed_at DESC LIMIT %(summaries)s) y) AS summaries
"""

UPSERT_MEMBER_SQL = """
INSERT INTO members (id, household_id, name, aliases, age_years, conditions)
VALUES (%(id)s, %(household_id)s, %(name)s, %(aliases)s, %(age_years)s, %(conditions)s)
ON CONFLICT (id) DO UPDATE SET household_id = EXCLUDED.household_id, name = EXCLUDED.name,
  aliases = EXCLUDED.aliases, age_years = EXCLUDED.age_years, conditions = EXCLUDED.conditions
"""

INSERT_MESSAGE_SQL = """
INSERT INTO messages (session_id, turn_id, role, content, member_id, episode_id, decision)
VALUES (%(session_id)s, %(turn_id)s, %(role)s, %(content)s, %(member_id)s, %(episode_id)s, %(decision)s)
ON CONFLICT (session_id, turn_id, role) DO NOTHING
RETURNING id
"""

UPSERT_EPISODE_SQL = """
INSERT INTO episodes (id, member_id, complaint, status, slots, questions_asked,
                      check_back_after, related_episode_id, protocol_id, findings, associated)
VALUES (%(id)s, %(member_id)s, %(complaint)s, %(status)s, %(slots)s, %(questions_asked_delta)s,
        %(check_back_after)s, %(related_episode_id)s, %(protocol_id)s, coalesce(%(findings)s::jsonb, '{}'),
        coalesce(%(associated)s::jsonb, '{}'))
ON CONFLICT (id) DO UPDATE SET
  status = EXCLUDED.status,
  slots = episodes.slots || EXCLUDED.slots,
  protocol_id = coalesce(EXCLUDED.protocol_id, episodes.protocol_id),
  findings = coalesce(%(findings)s::jsonb, episodes.findings),
  associated = coalesce(%(associated)s::jsonb, episodes.associated),
  questions_asked = episodes.questions_asked + EXCLUDED.questions_asked,
  check_back_after = EXCLUDED.check_back_after,
  updated_at = now()
  WHERE episodes.member_id = EXCLUDED.member_id
RETURNING id
"""

INSERT_FACT_SQL = """
INSERT INTO member_facts (member_id, session_id, episode_id, kind, subject, text, effective_at,
                          status, source, source_message_id)
VALUES (%(member_id)s, %(session_id)s, %(episode_id)s, %(kind)s, %(subject)s, %(text)s, %(effective_at)s,
        %(status)s, 'user_report', %(source_message_id)s)
RETURNING id
"""

CLOSE_IDLE_SESSIONS_SQL = """
WITH closed AS (
  UPDATE sessions SET status = 'closed', closed_at = now()
  WHERE household_id = %(household_id)s AND id <> %(active_session_id)s AND status = 'active'
    AND last_active_at < now() - make_interval(mins => %(idle_minutes)s)
  RETURNING id
)
SELECT id FROM closed
UNION  -- also retry sessions whose earlier consolidation failed
SELECT id FROM sessions
WHERE household_id = %(household_id)s AND id <> %(active_session_id)s AND status = 'closed'
"""

CLOSE_SESSION_SQL = """
UPDATE sessions SET status = 'closed', closed_at = now()
WHERE id = %(session_id)s AND household_id = %(household_id)s AND status = 'active'
"""

LOAD_CONSOLIDATION_SQL = """
WITH s AS (SELECT * FROM sessions WHERE id = %(session_id)s AND status = 'closed'),
m AS (SELECT members.* FROM members JOIN s USING (household_id))
SELECT
  (SELECT row_to_json(s) FROM s) AS session,
  (SELECT coalesce(json_agg(m ORDER BY m.name), '[]') FROM m) AS members,
  (SELECT coalesce(json_agg(f ORDER BY f.id), '[]')
     FROM member_facts f JOIN m ON m.id = f.member_id
    WHERE f.superseded_by IS NULL) AS facts,
  (SELECT coalesce(json_agg(e ORDER BY e.updated_at DESC), '[]')
     FROM episodes e JOIN m ON m.id = e.member_id
    WHERE e.status <> 'resolved_reported'
       OR e.updated_at > now() - make_interval(days => %(resolved_days)s)
       OR e.id IN (SELECT episode_id FROM messages WHERE session_id = %(session_id)s)) AS episodes,
  (SELECT coalesce(json_agg(x ORDER BY x.id), '[]')
     FROM messages x WHERE x.session_id = %(session_id)s) AS messages,
  (SELECT coalesce(json_agg(y ORDER BY y.closed_at), '[]')
     FROM (SELECT p.id AS session_id, p.summary, p.closed_at FROM sessions p JOIN s USING (household_id)
            WHERE p.status = 'consolidated' AND p.summary IS NOT NULL AND p.id <> s.id
            ORDER BY p.closed_at DESC LIMIT %(summaries)s) y) AS summaries
"""

MARK_CONSOLIDATED_SQL = """
UPDATE sessions SET status = 'consolidated', summary = %(summary)s
WHERE id = %(session_id)s AND status = 'closed'
"""

PROMOTE_FACT_SQL = """
UPDATE member_facts SET status = 'consolidated', subject = %(subject)s
WHERE id = %(fact_id)s AND session_id = %(session_id)s AND status = 'provisional' AND superseded_by IS NULL
"""

MARK_DUPLICATE_SQL = """
UPDATE member_facts f SET superseded_by = kept.id
FROM member_facts kept
WHERE f.id = %(fact_id)s AND f.session_id = %(session_id)s AND f.superseded_by IS NULL
  AND kept.id = %(kept_id)s AND kept.member_id = f.member_id AND kept.superseded_by IS NULL
"""

REPLACE_FACT_SQL = """
UPDATE member_facts old SET superseded_by = new.id
FROM member_facts new
WHERE new.id = %(fact_id)s AND new.session_id = %(session_id)s
  AND old.id = %(replaced_id)s AND old.member_id = new.member_id AND old.superseded_by IS NULL
"""

SUPERSEDE_FACT_SQL = """
UPDATE member_facts SET superseded_by = %(new_id)s
WHERE id = %(old_id)s AND member_id = %(member_id)s AND superseded_by IS NULL
"""

UPDATE_SESSION_SQL = """
UPDATE sessions SET member_hint_id = coalesce(%(member_id)s, member_hint_id),
  pending_episode_id = %(pending_episode_id)s, last_active_at = now()
WHERE id = %(session_id)s AND household_id = %(household_id)s
"""


class Store:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    @classmethod
    def connect(cls, database_url: str) -> "Store":
        return cls(ConnectionPool(database_url, open=True, kwargs={"row_factory": dict_row}))

    def close(self) -> None:
        self._pool.close()

    def init_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(SCHEMA_PATH.read_text())

    def load_fixture(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.upsert_members(data["household_id"], data["members"])

    def upsert_members(self, household_id: str, members: Iterable[Mapping[str, Any]]) -> None:
        rows = [
            {
                "id": m["id"],
                "household_id": household_id,
                "name": m["name"],
                "aliases": Jsonb(list(m.get("aliases", []))),
                "age_years": m.get("age_years"),
                "conditions": Jsonb(list(m.get("conditions", []))),
            }
            for m in members
        ]
        with self._pool.connection() as conn:
            conn.cursor().executemany(UPSERT_MEMBER_SQL, rows)

    def load_context(self, household_id: str, session_id: str) -> SessionContext:
        """Open or touch the session and read everything a turn may need, in one statement."""
        params = {
            "household_id": household_id,
            "session_id": session_id,
            "recent": RECENT_MESSAGES,
            "summaries": PRIOR_SUMMARIES,
            "resolved_days": RESOLVED_LOOKBACK_DAYS,
        }
        with self._pool.connection() as conn:
            row = conn.execute(LOAD_CONTEXT_SQL, params).fetchone()
        if row["session"] is None:
            raise SessionOwnershipError(f"session {session_id} does not belong to household {household_id}")
        return _to_context(row)

    def episode_history(self, household_id: str, episode_id: str) -> tuple[Message, ...]:
        """An episode's latest messages across all of the household's sessions, oldest first."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT to_jsonb(m) AS message FROM messages m JOIN sessions s ON s.id = m.session_id
                   WHERE s.household_id = %s AND m.episode_id = %s
                   ORDER BY m.id DESC LIMIT %s""",
                (household_id, episode_id, EPISODE_HISTORY_MESSAGES),
            ).fetchall()
        return tuple(_message(row["message"]) for row in reversed(rows))  # JSON rows, like load_context

    def has_member(self, household_id: str, member_id: str) -> bool:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT 1 FROM members WHERE household_id = %s AND id = %s", (household_id, member_id)
            ).fetchone() is not None

    def save_turn(self, turn: TurnWrite) -> bool:
        """Persist a turn atomically. Returns False when this turn was already saved."""
        with self._pool.connection() as conn:
            user_message = conn.execute(
                INSERT_MESSAGE_SQL, _message_params(turn, "user", turn.user_text, decision=None)
            ).fetchone()
            if user_message is None:
                return False
            if turn.episode is not None:
                _upsert_episode(conn, turn.episode)
            conn.execute(INSERT_MESSAGE_SQL, _message_params(turn, "assistant", turn.reply, turn.decision))
            for fact in turn.facts:
                _insert_fact(conn, fact, turn.session_id, user_message["id"])
            updated = conn.execute(UPDATE_SESSION_SQL, _session_params(turn)).rowcount
            if updated != 1:
                raise SessionOwnershipError(f"session {turn.session_id} not found for {turn.household_id}")
        return True

    def saved_turn(self, household_id: str, session_id: str, turn_id: str, text: str) -> dict[str, Any] | None:
        """Read a committed response, including turns older than the context window."""
        with self._pool.connection() as conn:
            row = conn.execute(
                """SELECT u.content AS text, a.content AS reply, a.decision
                   FROM messages u JOIN messages a USING (session_id, turn_id)
                   JOIN sessions s ON s.id = u.session_id
                   WHERE s.household_id = %s AND u.session_id = %s AND u.turn_id = %s
                     AND u.role = 'user' AND a.role = 'assistant'""",
                (household_id, session_id, turn_id),
            ).fetchone()
        if row is None:
            return None
        if row["text"] != text:
            raise TurnConflictError("turn_id already used with different text in this session")
        return {"reply": row["reply"], "decision": Decision(row["decision"])}

    def close_idle_sessions(self, household_id: str, active_session_id: str, idle_minutes: int) -> list[str]:
        """Close the household's idle sessions; returns every closed session awaiting consolidation."""
        params = {"household_id": household_id, "active_session_id": active_session_id, "idle_minutes": idle_minutes}
        with self._pool.connection() as conn:
            return [row["id"] for row in conn.execute(CLOSE_IDLE_SESSIONS_SQL, params).fetchall()]

    def close_session(self, household_id: str, session_id: str) -> bool:
        """Explicitly close an active session. False if it is unknown, another household's, or not active."""
        with self._pool.connection() as conn:
            params = {"household_id": household_id, "session_id": session_id}
            return conn.execute(CLOSE_SESSION_SQL, params).rowcount == 1

    def load_consolidation_input(self, session_id: str) -> ConsolidationInput | None:
        """Read a closed session with its household's long-term memory. None if the session isn't closed."""
        params = {"session_id": session_id, "resolved_days": RESOLVED_LOOKBACK_DAYS, "summaries": PRIOR_SUMMARIES}
        with self._pool.connection() as conn:
            row = conn.execute(LOAD_CONSOLIDATION_SQL, params).fetchone()
        return _to_consolidation_input(row) if row["session"] else None

    def apply_consolidation(self, write: ConsolidationWrite) -> bool:
        """Apply consolidation atomically. False if the session was reopened or already consolidated."""
        session_id = write.session_id
        with self._pool.connection() as conn:
            marked = conn.execute(MARK_CONSOLIDATED_SQL, {"session_id": session_id, "summary": write.summary})
            if marked.rowcount != 1:
                return False
            for fact_id, subject in write.promotions:
                conn.execute(PROMOTE_FACT_SQL, {"fact_id": fact_id, "subject": subject, "session_id": session_id})
            for fact_id, kept_id in write.duplicates:
                conn.execute(MARK_DUPLICATE_SQL, {"fact_id": fact_id, "kept_id": kept_id, "session_id": session_id})
            for fact_id, replaced_id in write.replacements:
                params = {"fact_id": fact_id, "replaced_id": replaced_id, "session_id": session_id}
                conn.execute(REPLACE_FACT_SQL, params)
            for missed in write.missed_facts:
                _insert_fact(conn, missed.fact, session_id, missed.source_message_id, status="consolidated")
        return True


def _message_params(turn: TurnWrite, role: str, content: str, decision: Decision | None) -> dict[str, Any]:
    return {
        "session_id": turn.session_id,
        "turn_id": turn.turn_id,
        "role": role,
        "content": content,
        "member_id": turn.member_id,
        "episode_id": turn.episode.id if turn.episode else None,
        "decision": decision,
    }


def _session_params(turn: TurnWrite) -> dict[str, Any]:
    return {
        "session_id": turn.session_id,
        "household_id": turn.household_id,
        "member_id": turn.member_id,
        "pending_episode_id": turn.pending_episode_id,
    }


def _upsert_episode(conn: Connection, episode: EpisodeWrite) -> None:
    params = {
        "id": episode.id,
        "member_id": episode.member_id,
        "complaint": episode.complaint,
        "status": episode.status,
        "slots": Jsonb(dict(episode.slots)),
        "questions_asked_delta": episode.questions_asked_delta,
        "check_back_after": episode.check_back_after,
        "related_episode_id": episode.related_episode_id,
        "protocol_id": episode.protocol_id,
        "findings": None if episode.findings is None else Jsonb(dict(episode.findings)),
        "associated": None if episode.associated is None else Jsonb(dict(episode.associated)),
    }
    if conn.execute(UPSERT_EPISODE_SQL, params).fetchone() is None:
        raise EpisodeOwnershipError(f"episode {episode.id} does not belong to member {episode.member_id}")


def normalize_subject(subject: str) -> str:
    """'Blood Pressure' and 'blood-pressure' both become 'blood_pressure'."""
    return re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")


def _insert_fact(
    conn: Connection, fact: FactWrite, session_id: str, source_message_id: int, status: FactStatus = "provisional"
) -> None:
    params = {
        "status": status,
        "member_id": fact.member_id,
        "session_id": session_id,
        "episode_id": fact.episode_id,
        "kind": fact.kind,
        "subject": normalize_subject(fact.subject),
        "text": fact.text,
        "effective_at": fact.effective_at,
        "source_message_id": source_message_id,
    }
    new_id = conn.execute(INSERT_FACT_SQL, params).fetchone()["id"]
    if fact.supersedes_id is None:
        return
    superseded = conn.execute(
        SUPERSEDE_FACT_SQL, {"new_id": new_id, "old_id": fact.supersedes_id, "member_id": fact.member_id}
    ).rowcount
    if superseded != 1:
        logger.warning("fact %s not superseded: missing, another member's, or already superseded", fact.supersedes_id)


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


def _message(msg: Mapping[str, Any]) -> Message:
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


def _to_context(row: Mapping[str, Any]) -> SessionContext:
    session = row["session"]
    return SessionContext(
        session_id=session["id"],
        is_new_session=session["is_new"],
        members=tuple(_member(m) for m in row["members"]),
        facts=tuple(_fact(f) for f in row["facts"]),
        episodes=tuple(_episode(e) for e in row["episodes"]),
        recent_messages=tuple(_message(msg) for msg in row["messages"]),
        prior_summaries=tuple(_summary(s) for s in row["summaries"]),
        member_hint_id=session["member_hint_id"],
        pending_episode_id=session["pending_episode_id"],
    )


def _to_consolidation_input(row: Mapping[str, Any]) -> ConsolidationInput:
    return ConsolidationInput(
        session=_session(row["session"]),
        transcript=tuple(_message(msg) for msg in row["messages"]),
        members=tuple(_member(m) for m in row["members"]),
        facts=tuple(_fact(f) for f in row["facts"]),
        episodes=tuple(_episode(e) for e in row["episodes"]),
        prior_summaries=tuple(_summary(s) for s in row["summaries"]),
    )
