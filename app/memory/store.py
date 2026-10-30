"""Postgres persistence: one read round trip and one write transaction per turn."""

import json
import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.memory.rows import to_consolidation_input, to_context, to_message
from app.memory.sql import (
    CLOSE_IDLE_SESSIONS_SQL,
    CLOSE_SESSION_SQL,
    INSERT_FACT_SQL,
    INSERT_MESSAGE_SQL,
    LOAD_CONSOLIDATION_SQL,
    LOAD_CONTEXT_SQL,
    MARK_CONSOLIDATED_SQL,
    MARK_DUPLICATE_SQL,
    PROMOTE_FACT_SQL,
    REPLACE_FACT_SQL,
    SUPERSEDE_FACT_SQL,
    UPDATE_SESSION_SQL,
    UPSERT_EPISODE_SQL,
    UPSERT_MEMBER_SQL,
)
from app.memory.writes import (
    ConsolidationWrite,
    EpisodeOwnershipError,
    EpisodeWrite,
    FactWrite,
    SessionOwnershipError,
    TurnConflictError,
    TurnWrite,
)
from app.models.records import ConsolidationInput, FactStatus, Message, SessionContext
from app.triage.policy import Decision

logger = logging.getLogger(__name__)


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


RECENT_MESSAGES = 10


PRIOR_SUMMARIES = 3


RESOLVED_LOOKBACK_DAYS = 14


EPISODE_HISTORY_MESSAGES = 20


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
        return to_context(row)

    def episode_history(self, household_id: str, episode_id: str) -> tuple[Message, ...]:
        """An episode's latest messages across all of the household's sessions, oldest first."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT to_jsonb(m) AS message FROM messages m JOIN sessions s ON s.id = m.session_id
                   WHERE s.household_id = %s AND m.episode_id = %s
                   ORDER BY m.id DESC LIMIT %s""",
                (household_id, episode_id, EPISODE_HISTORY_MESSAGES),
            ).fetchall()
        return tuple(to_message(row["message"]) for row in reversed(rows))  # JSON rows, like load_context

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
        return to_consolidation_input(row) if row["session"] else None

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
        "urgency": episode.urgency,
        "handoff_id": episode.handoff_id,
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
