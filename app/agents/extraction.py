"""Combine the screen and an observation into the extraction that policy reads."""

from app.models.outputs import Extraction, Observation, Slots
from app.models.turn import TurnState

# What a turn that stops at a clarifying question records: nothing observed yet.
NO_OBSERVATION = Observation(
    intent="other", complaint=None, slots=Slots.model_validate(dict.fromkeys(Slots.model_fields)),
    answers=[], associated=[], resolved=False, facts=[],
)


def extraction_from(state: TurnState, member_id: str | None, observation: Observation) -> Extraction:
    found = state["screen"]
    return Extraction(
        member_id=member_id, episode_id=None,
        intent="doctor_request" if found.doctor_requested else observation.intent,
        complaint=observation.complaint, slots=observation.slots, red_flags=found.red_flags,
        doctor_flags=found.doctor_flags, resolved=observation.resolved, facts=observation.facts,
    )
