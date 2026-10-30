"""Where the turn graph goes after each step."""

from langgraph.graph import END

from app.models.turn import TurnState
from app.triage.policy import FIXED_REPLIES, Decision


def route_after_reconcile(state: TurnState) -> str:
    return "decide" if state["clarification"] else "observe"


def route_after_decide(state: TurnState) -> str:
    """Doctor decisions hand the case over before replying; emergencies reply at once; the intake
    agent asks the protocol's next questions."""
    if state["decision"] == Decision.DOCTOR:
        return "handoff"
    if state["decision"] == Decision.ASK and state.get("triage"):
        return "intake_agent"
    return "fixed_reply" if state["decision"] in FIXED_REPLIES else "respond"


PARALLEL_BRANCHES = ["screen", "resolve_member"]


def route_after_load(state: TurnState) -> str | list[str]:
    return END if state["replayed"] else PARALLEL_BRANCHES
