"""The intake-note draft the handoff agent writes, with a source for every statement."""


from pydantic import BaseModel, ConfigDict

CURRENT = "current"  # evidence ID of the message being handled, which isn't saved yet


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str  # from evidence; "current" for this message
    onset: str | None  # when it started or was noticed, in the user's terms, e.g. "yesterday evening"
    finding: str
    quote: str  # exact words from that message


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str  # a user message from evidence; "current" for this message
    quote: str  # the user's exact words in that message that say this


class Statement(BaseModel):
    """One sentence of the history, with the user's words that support it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    sources: list[Source]


class IntakeNoteDraft(BaseModel):
    """The parts of the intake note the model writes."""

    model_config = ConfigDict(extra="forbid")

    chief_complaint: str
    history: list[Statement]  # the history of present illness, one supported sentence at a time
    symptom_timeline: list[TimelineEntry]  # in the order the findings were reported
    relevant_history: str | None
