"""Merge quoted answers and related symptoms into an episode's findings."""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from app.agents.evidence import is_quoted
from app.memory.store import normalize_subject
from app.models.outputs import Answer, AssociatedSymptom
from app.models.records import Episode
from app.triage.protocol import Protocol

logger = logging.getLogger(__name__)


def merge_associated(previous: Mapping[str, Any], reported: list[AssociatedSymptom], text: str) -> dict[str, Any]:
    """The episode's other symptoms with this message's quoted reports applied; the latest status wins."""
    now = datetime.now(UTC).isoformat()
    supported = {subject: {"status": s.status, "quote": s.quote, "at": now}
                 for s in reported if (subject := normalize_subject(s.symptom)) and is_quoted(s.quote, text)}
    if len(supported) < len(reported):
        logger.warning("Dropped %d related symptoms not supported by the message", len(reported) - len(supported))
    return {**previous, **supported}


def merge_findings(protocol: Protocol, episode: Episode | None, answers: list[Answer], text: str, *,
                   session_id: str | None = None, last_asked: tuple[str, str] | None = None) -> dict[str, Any]:
    """Earlier findings plus this message's supported answers.

    Each answer records the session it was given in and, when it answers the question Q asked last
    turn, Q's actual words (last_asked: question ID, message), so a note can show what was asked.
    """
    previous = {k: kept for k, v in (episode.findings if episode else {}).items()
                if (kept := _unanswered(v) if _ask_again(protocol, k, v, episode, session_id) else dict(v))}
    now = datetime.now(UTC).isoformat()
    # A chosen sign's entry also holds its definition (sign, level, ask), so answers are merged into it.
    supported = {a.question_id: {**previous.get(a.question_id, {}), "value": a.value, "quote": a.quote, "at": now,
                                 "session_id": session_id,
                                 "asked": last_asked[1] if last_asked and last_asked[0] == a.question_id else None}
                 for a in answers if protocol.question(a.question_id) and is_quoted(a.quote, text)}
    if len(supported) < len(answers):
        logger.warning("Dropped %d answers not supported by the message or protocol", len(answers) - len(supported))
    return {**previous, **supported}


def _ask_again(protocol: Protocol, question_id: str, finding: Mapping[str, Any], episode: Episode | None,
               session_id: str | None) -> bool:
    """An emergency answer is asked again at a check-back, or when it was given in an earlier session."""
    question = protocol.question(question_id)
    level = question.level if question else finding.get("level")
    if level != "emergency" or "value" not in finding:
        return False
    return bool(episode and episode.status == "monitoring") or finding.get("session_id") != session_id


def _unanswered(finding: Mapping[str, Any]) -> dict[str, Any]:
    """The finding without its answer; a chosen sign keeps its definition so it can be asked again."""
    return {k: finding[k] for k in ("sign", "level", "ask") if k in finding}
