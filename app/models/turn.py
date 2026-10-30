"""Scratch state for one turn's graph run, and what the run returns."""

from collections.abc import Mapping
from typing import Any, TypedDict

from app.models.outputs import EpisodeResolution, Extraction, MemberResolution, Screen
from app.models.records import Clarification, Episode, SessionContext
from app.triage.engine import Triage
from app.triage.policy import Decision


class TurnState(TypedDict, total=False):
    """Scratch state for one graph run. Durable memory lives in Postgres, not in graph state."""

    household_id: str
    session_id: str
    turn_id: str
    text: str
    profile_id: str | None
    context: SessionContext
    screen: Screen
    member_resolution: MemberResolution
    episode_resolution: EpisodeResolution
    clarification: Clarification | None
    member_id: str | None  # resolved member, before observation
    extraction: Extraction
    selected_episode: Episode | None
    episode_id: str | None
    related_episode_id: str | None
    protocol_id: str | None
    findings: Mapping[str, Any]  # merged; written back whole
    associated: Mapping[str, Any]  # the episode's other symptoms, merged; written back whole
    triage: Triage | None
    replayed: bool
    decision: Decision
    decision_reason: str
    urgency: str | None  # doctor decisions only: doctor_now, doctor_24h or doctor_routine
    handoff_path: str | None
    reply: str
    is_new_session: bool
    unanswered_last_time: bool  # the intake's next question was asked last turn and not answered


class TurnOutput(TypedDict, total=False):
    """What a graph run returns, and so what its trace shows as output: the outcome, not the scratch state."""

    member_id: str | None
    episode_id: str | None
    protocol_id: str | None
    decision: Decision
    decision_reason: str
    urgency: str | None
    handoff_path: str | None
    reply: str
    replayed: bool
    is_new_session: bool
