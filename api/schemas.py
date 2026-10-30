from typing import Annotated

from fastapi import Path
from pydantic import BaseModel, Field, StringConstraints

from app.policy import Decision

MAX_MESSAGE_CHARS = 2000
ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

PathId = Annotated[str, Path(pattern=ID_PATTERN)]


class MessageIn(BaseModel):
    profile_id: str | None = Field(default=None, pattern=ID_PATTERN, description="Who is writing; the subject of first-person reports")
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)]
    turn_id: str | None = Field(
        default=None, pattern=ID_PATTERN, description="Idempotency key; generated when omitted"
    )


class MessageOut(BaseModel):
    turn_id: str
    decision: Decision
    reply: str
