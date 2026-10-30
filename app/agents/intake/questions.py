"""The intake agent asks the next screening question in its own words; code chooses what to ask.

Code picks the finding to settle next (most urgent first) and later decides the outcome. The protocol's
wording is guidance: the agent phrases one question for this person and conversation, e.g. "you" when the
writer is the patient, or answering "who do you mean?" before asking again. Code accepts the wording only
if it is a single short question, and otherwise asks from the protocol text. Generic screening can
escalate but cannot authorize home monitoring.
"""

import logging

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.context import other_open_intakes
from app.agents.loop import ModelOutputError, run_agent
from app.agents.tools import episode_tools, lab_tools, member_tools, to_json
from app.extensions import LabResults
from app.memory.store import Store
from app.models.turn import TurnState
from app.prompts.intake import INTAKE_AGENT_PROMPT
from app.triage.engine import NEW_SIGN, Triage

logger = logging.getLogger(__name__)

LOOKUPS = {"member_facts", "list_episodes", "episode_history", "past_session_summaries"}
FALLBACK_LEAD = "To work out what's needed, first:"
MAX_ADDED_SIGNS = 2  # signs the agent may add to an episode beyond its guideline's minimum
MAX_MESSAGE_CHARS = 400  # a short acknowledgement or clarification, then the question


class IntakeQuestions(BaseModel):
    """The question to send, written by the agent for next_question, and an optional new sign."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(description="The message sent to the user: one question, in your own words, that "
                                     "finds out next_question (or new_sign on a choose_sign turn)")
    new_sign: "NewSign | None" = Field(description=(
        "On a choose_sign turn, the sign this message asks about; otherwise null, or an emergency sign "
        "the protocol misses, to ask in a later turn"))


class NewSign(BaseModel):
    """A danger sign the agent chose for this complaint. Answers are recorded against sign."""

    model_config = ConfigDict(extra="forbid")

    sign: str = Field(description="One finding, stated as present, e.g. 'Drooling or unable to swallow saliva'")


IntakeQuestions.model_rebuild()


def intake_agent(
    state: TurnState, *, model: BaseChatModel, store: Store, lab_results: LabResults | None = None,
) -> TurnState:
    context = state["context"]
    lookups = [t for t in [*member_tools(context), *episode_tools(context, store, state["household_id"])]
               if t.name in LOOKUPS] + lab_tools(context, state["household_id"], lab_results)
    try:
        result = run_agent(model, system=INTAKE_AGENT_PROMPT, payload=question_payload(state), tools=lookups,
                           answer=IntakeQuestions)
        return apply_questions(state, result)
    except (ModelOutputError, ValidationError) as exc:
        logger.warning("Intake agent did not finish (%s); asking from the protocol text", exc)
    return fallback_question(state)


def apply_questions(state: TurnState, result: IntakeQuestions) -> TurnState:
    """The checked question as the reply, and any sign the agent added to the findings."""
    triage = state["triage"]
    return {"reply": _with_context(checked_reply(result, triage), state),
            "findings": with_new_sign(result, triage, dict(state.get("findings") or {}))}


def fallback_question(state: TurnState) -> TurnState:
    """The protocol's own wording, when the agent couldn't write the question."""
    triage, findings = state["triage"], dict(state.get("findings") or {})
    question = triage.pending[0]
    if question.id != NEW_SIGN:
        fallback = IntakeQuestions(message="", new_sign=None)
        return {"reply": _with_context(f"{FALLBACK_LEAD} {protocol_question(fallback, triage)}", state)}
    chosen = {v["sign"] for v in findings.values() if "sign" in v}
    sign = next(s for s in triage.protocol.fallback_signs if s["sign"] not in chosen)
    fallback = IntakeQuestions(message=sign["ask"], new_sign=NewSign(sign=sign["sign"]))
    return {"reply": _with_context(f"{FALLBACK_LEAD} {fallback.message}", state),
            "findings": with_new_sign(fallback, triage, findings)}


def _with_context(question: str, state: TurnState) -> str:
    others = other_open_intakes(state)
    if others and not any(m.role == "assistant" and "come back to" in m.content
                          for m in state["context"].recent_messages):
        names = ", ".join(f"{e['member']}'s {e['complaint'].replace('_', ' ')}" for e in others)
        return f"I'll come back to {names} after this. {question}"
    return question


def checked_reply(result: IntakeQuestions, triage: Triage) -> str:
    """The agent's wording when it is one short question; otherwise the protocol's own question."""
    if triage.pending[0].id == NEW_SIGN and result.new_sign is None:
        raise ModelOutputError("Intake agent chose no sign on a choose-a-sign turn")
    message = result.message.strip()
    if is_one_question(message):
        return message
    logger.warning("Intake agent's message is not a single question (%r); asking from the protocol", message)
    return protocol_question(result, triage)


def is_one_question(message: str) -> bool:
    return 0 < len(message) <= MAX_MESSAGE_CHARS and message.count("?") == 1 and message.endswith("?")


def protocol_question(result: IntakeQuestions, triage: Triage) -> str:
    question = triage.pending[0]
    if question.id == NEW_SIGN:
        return sign_question(result.new_sign.sign)
    return sign_question(question.finding) if question.authored else question.ask


def sign_question(sign: str) -> str:
    sign = sign.strip().rstrip(".?")
    if not sign or any(c in sign for c in "?\n\r"):
        raise ModelOutputError("Invalid danger sign")
    return f"Is this happening: {sign}?"


def with_new_sign(result: IntakeQuestions, triage: Triage, findings: dict) -> dict:
    """The findings with the agent's sign added: asked now on a choose-a-sign turn, else later.
    Code sets the level; an added sign on a fixed protocol is an emergency sign, capped per episode."""
    if result.new_sign is None:
        return findings
    question, chosen = triage.pending[0], sum("sign" in v for v in findings.values())
    choosing = question.id == NEW_SIGN
    if not choosing and chosen >= triage.protocol.agent_signs.get("emergency", 0) + MAX_ADDED_SIGNS:
        logger.warning("Intake agent added more signs than allowed; ignoring %r", result.new_sign.sign)
        return findings
    level = question.level if choosing else "emergency"
    return {**findings, f"sign_{chosen + 1}": {"sign": result.new_sign.sign, "level": level,
                                               "ask": sign_question(result.new_sign.sign)}}


def question_payload(state: TurnState) -> str:
    triage, context, extraction = state["triage"], state["context"], state["extraction"]
    member = next((m for m in context.members if m.id == state["member_id"]), None)
    writer = next((m for m in context.members if m.id == state.get("profile_id")), None)
    episode = state.get("selected_episode")
    question = triage.pending[0]
    # Protocols can reuse a question's wording at several levels (severity, duration). A copy ruled
    # out by the member's profile must not make the pending one look already answered.
    return to_json({  # from most to least stable, so a prompt cache can reuse more of the prefix
        "profile_id": state.get("profile_id"), "member": member,
        "writer": writer.name if writer else None,
        "writer_is_patient": writer is not None and member is not None and writer.id == member.id,
        "protocol": {"id": triage.protocol.id, "title": triage.protocol.title},
        "episode_id": episode.id if episode else None, "complaint": extraction.complaint,
        "questions_asked_before": episode.questions_asked if episode else 0,
        "recent_messages": context.recent_messages,
        "reported_values": extraction.slots.model_dump(exclude_none=True),
        "associated_symptoms": state.get("associated") or {},
        "already_known": [{"id": q.id, "ask": q.ask, "positive": triage.states[q.id]}
                          for q in triage.protocol.questions
                          if q.ask and triage.states[q.id] is not None and q.ask != question.ask],
        "still_to_check": sorted({q.ask for q in triage.protocol.questions
                                  if q.ask and triage.states[q.id] is None and q.ask != question.ask}),
        "already_positive_level": triage.floor,
        "other_open_intakes": other_open_intakes(state),
        "unanswered_last_time": state.get("unanswered_last_time", False),
        "user_asked_back": extraction.intent == "clarification",
        "next_question": ({"id": NEW_SIGN, "level": question.level, "choose_sign": True,
                           "chosen_signs": [q.finding for q in triage.protocol.questions if q.authored]}
                          if question.id == NEW_SIGN else
                          {"id": question.id, "level": question.level, "ask": question.ask,
                           "finding": question.finding}),
        "text": state["text"],
    })
