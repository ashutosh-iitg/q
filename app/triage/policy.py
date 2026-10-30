"""Deterministic decision policy. The models observe; this module decides.

Per-complaint thresholds live in triage protocols (app/triage); this module orders the checks that
apply to every complaint: safety screening, clarification, follow-up rules, then the protocol's
sufficiency.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.outputs import Extraction
    from app.models.records import Clarification, Episode, SessionContext
    from app.triage.engine import Triage


class Decision(StrEnum):
    EMERGENCY = "emergency"
    ASK_MEMBER = "ask_member"
    ASK_EPISODE = "ask_episode"
    DOCTOR = "doctor"
    ANSWER = "answer"
    ASK = "ask"
    HOME_MONITORING = "home_monitoring"


# Placeholder wording pending clinical review. Decisions listed here never reach the LLM.
# Clarifying questions (ask_member, ask_episode) are generated from the resolver agents' instructions.
FIXED_REPLIES: dict[Decision, str] = {
    Decision.EMERGENCY: (
        "This may be a medical emergency. Call your local emergency number now "
        "or go to the nearest emergency department. Don't wait for a reply from Q."
    ),
    Decision.DOCTOR: (
        "Based on what you've shared, this needs a doctor's review. I'm transferring "
        "this case to a human doctor now. If symptoms get worse before then, call "
        "your local emergency number."
    ),
}
URGENCY_ADVICE = {
    "doctor_now": "Please arrange to be seen by a clinician within the next few hours.",
    "doctor_24h": "Please arrange to be seen by a clinician within 24 hours.",
    "doctor_routine": "Please book a routine appointment with a clinician in the next few days.",
}
# A doctor decision on an episode whose note is already saved: remind, don't hand it over again.
ALREADY_WITH_DOCTOR = (
    "This is already with a doctor for review, so I haven't sent it again. If symptoms get worse before "
    "then, call your local emergency number."
)
DEFAULT_URGENCY = "doctor_now"  # when a rule outside the protocol escalates: err towards sooner

MAX_INTAKE_TURNS = 10  # one-question turns before an unfinished intake goes to a doctor; tune by evaluation
GENERAL_WATCH_FOR = (
    "Call your local emergency number for trouble breathing, chest pain that won't go away, "
    "confusion, a seizure, a severe allergic reaction or severe bleeding."
)


def decide(
    extraction: "Extraction", context: "SessionContext", clarification: "Clarification | None" = None,
    triage: "Triage | None" = None,
) -> Decision:
    return rule(extraction, context, clarification, triage)[0]


def rule(
    extraction: "Extraction", context: "SessionContext", clarification: "Clarification | None" = None,
    triage: "Triage | None" = None,
) -> tuple[Decision, str]:
    """The decision and the reason code that produced it; first match wins.
    Safety screening comes first, so an unresolved member or episode never delays an escalation."""
    member = next((m for m in context.members if m.id == extraction.member_id), None)
    episode = next((e for e in context.episodes if e.id == extraction.episode_id), None)
    if extraction.red_flags:
        return Decision.EMERGENCY, "red_flag"
    if extraction.intent == "doctor_request":
        return Decision.DOCTOR, "doctor_requested"
    if extraction.doctor_flags:
        return Decision.DOCTOR, "doctor_flag"
    if clarification:
        return (Decision.ASK_MEMBER if clarification.kind == "member" else Decision.ASK_EPISODE), "clarification"
    if extraction.intent == "general_question":
        return Decision.ANSWER, "general_question"
    if member is None:
        return Decision.ASK_MEMBER, "no_member"
    if triage and triage.disposition == "emergency":
        return Decision.EMERGENCY, "protocol"
    if extraction.intent == "other" or extraction.resolved:
        return Decision.ANSWER, "no_active_symptom"
    if episode and episode.status in ("emergency", "doctor"):
        return Decision(episode.status), "already_escalated"
    recurrence = any(
        e.member_id == member.id and e.complaint == extraction.complaint
        and e.status == "resolved_reported" and e.id != extraction.episode_id
        for e in context.episodes
    )
    if recurrence:
        return Decision.DOCTOR, "recurrence"
    if episode and episode.status == "monitoring":
        trend = extraction.slots.trend
        overdue = episode.check_back_after and datetime.now(UTC) >= episode.check_back_after
        if trend == "worse" or (trend == "same" and overdue):
            return Decision.DOCTOR, "not_improving"
    if extraction.intent == "clarification" and triage and not triage.sufficient:
        return Decision.ASK, "clarification_requested"  # they asked about Q's question instead of answering
    if triage and triage.sufficient:
        if triage.disposition == "home_care":
            return Decision.HOME_MONITORING, "protocol"
        return Decision.DOCTOR, "protocol"
    if episode and episode.questions_asked >= MAX_INTAKE_TURNS:
        return Decision.DOCTOR, "intake_turn_limit"
    return Decision.ASK, "intake_incomplete"


def doctor_urgency(reason: str, triage: "Triage | None") -> str:
    """How soon a doctor decision needs a clinician; the protocol's level when it decided."""
    return triage.disposition if reason == "protocol" and triage else DEFAULT_URGENCY


def note_on_file(extraction: "Extraction", context: "SessionContext",
                 fallback_episode_id: str | None = None) -> "Episode | None":
    """The episode, when its intake note is already with a doctor and this message adds no new doctor
    sign: a doctor decision then reminds instead of handing the same case over again. The fallback is
    the episode the message was resolved to, for a message that drops it ("let's go back to Ayaan")."""
    episode_id = extraction.episode_id or fallback_episode_id
    episode = next((e for e in context.episodes if e.id == episode_id), None)
    if episode is None or episode.status != Decision.DOCTOR or extraction.doctor_flags:
        return None
    return episode
