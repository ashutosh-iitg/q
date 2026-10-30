"""Structured outputs the models return, validated by code before use."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.records import FactKind


class Slots(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_hours: float | None
    severity: Literal["mild", "moderate", "severe"] | None
    trend: Literal["better", "same", "worse"] | None
    temperature_c: float | None
    systolic_bp: int | None
    diastolic_bp: int | None


class QuotedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: FactKind
    subject: str
    text: str  # exact quote from the current message, for the selected member


class RedFlag(BaseModel):
    """A danger sign that is an emergency whatever the complaint, grouped by body system."""

    model_config = ConfigDict(extra="forbid")

    system: Literal["breathing", "circulation", "neurological", "allergic_reaction", "bleeding"]
    sign: str  # short label, e.g. "struggling to breathe"
    quote: str


class DoctorFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: Literal["severe_pain", "dehydration", "rash_with_fever", "chronic_change"]
    quote: str


class Screen(BaseModel):
    """Safety screen of the current message; needs no member or episode, so it never waits on them."""

    model_config = ConfigDict(extra="forbid")

    red_flags: list[RedFlag]
    doctor_flags: list[DoctorFlag]
    doctor_requested: bool


class MemberResolution(BaseModel):
    """Member agent's final answer, submitted as a tool call."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["resolved", "none", "unresolved"]  # none: not about a family member's health
    member_id: str | None
    reason: str
    needed_info: str | None  # unresolved only: what would settle it
    responder_instructions: str | None  # unresolved only: how to ask


class EpisodeResolution(BaseModel):
    """Episode agent's final answer, submitted as a tool call."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["existing", "new", "none", "unresolved"]  # none: no personal health episode
    episode_id: str | None  # existing only
    reason: str
    needed_info: str | None
    responder_instructions: str | None


class Answer(BaseModel):
    """The user's answer to one protocol question, with the words that give it."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    value: Literal["present", "absent", "resolved", "unknown"] = Field(description=(
        "Whether the question's finding is there now, whatever wording Q used to ask: present, absent, "
        "resolved when it was there earlier in this illness and has gone, or unknown when the user doesn't know"))
    quote: str  # exact words from the current message


class AssociatedSymptom(BaseModel):
    """Another symptom of the same illness, e.g. the fever that came with a cough, and whether it is still there."""

    model_config = ConfigDict(extra="forbid")

    symptom: str  # short snake_case label, e.g. fever
    status: Literal["active", "resolved"]
    quote: str  # exact words from the current message


class Observation(BaseModel):
    """What the message says about the resolved member and episode."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["symptom_report", "update", "clarification", "general_question", "other"]
    complaint: str | None
    slots: Slots
    answers: list[Answer]
    associated: list[AssociatedSymptom]
    resolved: bool
    facts: list[QuotedFact]


class Extraction(BaseModel):
    """Merged turn findings that policy and persistence read; assembled in code, not by one model call."""

    model_config = ConfigDict(extra="forbid")

    member_id: str | None
    episode_id: str | None  # existing ID; null for a new episode or general question
    intent: Literal["symptom_report", "update", "clarification", "general_question", "doctor_request", "other"]
    complaint: str | None
    slots: Slots
    red_flags: list[RedFlag]
    doctor_flags: list[DoctorFlag]
    resolved: bool
    facts: list[QuotedFact]
