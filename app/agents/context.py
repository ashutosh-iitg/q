"""The turn context every agent sees, rendered as JSON."""

from datetime import UTC, datetime, timedelta

from app.agents.tools import to_json
from app.models.turn import TurnState

OPEN_INTAKE_WINDOW = timedelta(days=1)  # an intake left unfinished longer ago is no longer "in progress"


def render_context(state: TurnState, **extra) -> str:
    # Stable fields first and this message last, so a prompt cache can reuse more of the prefix.
    return to_json({"context": state["context"], "profile_id": state.get("profile_id"), **extra,
                    "text": state["text"], "now": datetime.now(UTC)})


def other_open_intakes(state: TurnState) -> list[dict[str, str]]:
    """Other recent intakes still unfinished, so Q comes back to them instead of dropping them."""
    context, cutoff = state["context"], datetime.now(UTC) - OPEN_INTAKE_WINDOW
    names = {m.id: m.name for m in context.members}
    return [{"member": names.get(e.member_id, e.member_id), "complaint": e.complaint} for e in context.episodes
            if e.status == "triaging" and e.id != state.get("episode_id") and e.updated_at >= cutoff]
