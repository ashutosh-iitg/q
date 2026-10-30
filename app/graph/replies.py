"""Reply nodes: fixed safety text, or a generated answer or home-care reply."""

from langchain_core.runnables import Runnable

from app.agents.context import other_open_intakes, render_context
from app.agents.loop import ModelOutputError
from app.extensions import Knowledge
from app.models.turn import TurnState
from app.prompts.respond import RESPOND_PROMPT
from app.triage.policy import FIXED_REPLIES, GENERAL_WATCH_FOR, URGENCY_ADVICE, Decision


def fixed_reply(state: TurnState) -> TurnState:
    reply = FIXED_REPLIES[state["decision"]]
    if state.get("urgency"):
        reply += " " + URGENCY_ADVICE[state["urgency"]]
    return {"reply": reply}


def respond(state: TurnState, *, model: Runnable, knowledge: Knowledge | None = None) -> TurnState:
    # A plugged-in knowledge base grounds general answers (see app/extensions.py).
    response = model.invoke([("system", RESPOND_PROMPT), ("human", reply_payload(state, knowledge))],
                            config={"run_name": "reply"}, prompt_cache_key="q:reply")
    return {"reply": finish_reply(state, response.text)}


def reply_payload(state: TurnState, knowledge: Knowledge | None = None) -> str:
    """What the reply writer sees: the decision, what was observed, and any reference passages."""
    reference = knowledge(state["text"]) if knowledge and state["decision"] == Decision.ANSWER else None
    return render_context(state, decision=state["decision"], extraction=state["extraction"].model_dump(),
                          clarification=state.get("clarification"), associated_symptoms=state.get("associated"),
                          other_open_intakes=other_open_intakes(state), reference=reference)


def finish_reply(state: TurnState, text: str) -> str:
    """The written reply, plus the protocol's fixed home-care text when Q is monitoring at home."""
    reply = text.strip()
    if not reply:
        raise ModelOutputError("Empty model response")
    if state["decision"] == Decision.HOME_MONITORING:
        reply += "\n\n" + home_care_advice(state)
    return reply


def home_care_advice(state: TurnState) -> str:
    """Precautions, warning signs and check-back, from the protocol: fixed text, not generated."""
    protocol = state["triage"].protocol
    lines = ["Precautions:", *(f"- {line}" for line in protocol.home_care),
             "Get help sooner if:", *(f"- {line}" for line in protocol.watch_for),
             GENERAL_WATCH_FOR,
             f"Check back with Q in {protocol.check_back_hours} hours with an update."]
    return "\n".join(lines)
