"""Graph nodes. Agents live in resolvers.py, intake.py and handoff.py; prompts in prompts.py."""

from datetime import UTC, datetime, timedelta

from langchain_core.runnables import Runnable

from app.agent.evidence import is_quoted
from app.agent.loop import ModelOutputError
from app.agent.prompts import RESPOND_PROMPT, SCREEN_PROMPT
from app.agent.state import (
    Clarification, EpisodeResolution, Extraction, Member, MemberResolution, Observation, Screen, Slots, TurnState,
)
from app.agent.tools import to_json
from app.extensions import Knowledge
from app.policy import FIXED_REPLIES, GENERAL_WATCH_FOR, URGENCY_ADVICE, Decision, doctor_urgency, rule
from app.store import EpisodeWrite, FactWrite, Store, TurnWrite, normalize_subject

DEFAULT_CHECK_BACK_HOURS = 24
OPEN_INTAKE_WINDOW = timedelta(days=1)  # an intake left unfinished longer ago is no longer "in progress"

# What a turn that stops at a clarifying question records: nothing observed yet.
NO_OBSERVATION = Observation(
    intent="other", complaint=None, slots=Slots.model_validate(dict.fromkeys(Slots.model_fields)),
    answers=[], associated=[], resolved=False, facts=[],
)


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


def load_context(state: TurnState, *, store: Store) -> TurnState:
    context = store.load_context(state["household_id"], state["session_id"])
    saved = store.saved_turn(state["household_id"], state["session_id"], state["turn_id"], state["text"])
    return {"context": context, "is_new_session": context.is_new_session, "replayed": saved is not None,
            **(saved or {})}


def screen(state: TurnState, *, screener: Runnable) -> TurnState:
    """Safety flags from the current message alone, in parallel with the resolver agents."""
    recent = [{"role": m.role, "content": m.content} for m in state["context"].recent_messages]
    result = Screen.model_validate(screener.invoke([
        ("system", SCREEN_PROMPT), ("human", to_json({"text": state["text"], "recent_messages": recent})),
    ], config={"run_name": "Screen"}))
    for flag in [*result.red_flags, *result.doctor_flags]:
        if not is_quoted(flag.quote, state["text"]):
            raise ModelOutputError("Flag is missing supporting evidence")
    return {"screen": result}


def reconcile(state: TurnState) -> TurnState:
    """Join the parallel branches: check the agents' IDs and whether their answers agree.
    With a red flag, a bad ID is dropped instead of failing the turn, so the emergency reply goes out."""
    context = state["context"]
    member_r, episode_r = state["member_resolution"], state["episode_resolution"]
    members = {m.id: m for m in context.members}
    emergency = bool(state["screen"].red_flags)
    if member_r.status == "resolved" and member_r.member_id not in members:
        if not emergency:
            raise ModelOutputError("Unknown member reference")
        member_r = member_r.model_copy(update={"status": "none", "member_id": None})
    episode = None
    if episode_r.status == "existing":
        episode = next((e for e in context.episodes if e.id == episode_r.episode_id), None)
        if episode is None and not emergency:
            raise ModelOutputError("Unknown episode reference")
    member_id = member_r.member_id if member_r.status == "resolved" else None
    clarification = _clarification(member_r, episode_r, episode, members)
    if clarification:
        return {"clarification": clarification, "member_id": member_id, "selected_episode": None,
                "extraction": extraction_from(state, member_id, NO_OBSERVATION), "episode_id": None,
                "related_episode_id": None, "triage": None}
    # Not about a member, so it must not touch anyone's episode.
    return {"clarification": None, "member_id": member_id, "selected_episode": episode if member_id else None}


def _clarification(
    member_r: MemberResolution, episode_r: EpisodeResolution, episode, members: dict[str, Member],
) -> Clarification | None:
    if member_r.status == "unresolved":
        return Clarification("member", member_r.needed_info or "Which family member this message is about",
                             member_r.responder_instructions)
    if member_r.status == "resolved" and episode and episode.member_id != member_r.member_id:
        named, owner = members[member_r.member_id].name, members[episode.member_id].name
        return Clarification(
            "member", f"Whether this is about {named}, or about {owner}'s {episode.complaint}",
            f"Ask which person this message is about, naming {named} and {owner}.",
        )
    if member_r.status == "resolved" and episode_r.status == "unresolved":
        return Clarification("episode", episode_r.needed_info or "Whether this continues an earlier illness",
                             episode_r.responder_instructions)
    return None


def extraction_from(state: TurnState, member_id: str | None, observation: Observation) -> Extraction:
    found = state["screen"]
    return Extraction(
        member_id=member_id, episode_id=None,
        intent="doctor_request" if found.doctor_requested else observation.intent,
        complaint=observation.complaint, slots=observation.slots, red_flags=found.red_flags,
        doctor_flags=found.doctor_flags, resolved=observation.resolved, facts=observation.facts,
    )


def route_after_reconcile(state: TurnState) -> str:
    return "decide" if state["clarification"] else "observe"


def decide(state: TurnState) -> TurnState:
    triage = state.get("triage")
    decision, reason = rule(state["extraction"], state["context"], state.get("clarification"), triage)
    urgency = doctor_urgency(reason, triage) if decision == Decision.DOCTOR else None
    return {"decision": decision, "decision_reason": reason, "urgency": urgency}


def route_after_decide(state: TurnState) -> str:
    """Doctor decisions hand the case over before replying; emergencies reply at once; the intake
    agent asks the protocol's next questions."""
    if state["decision"] == Decision.DOCTOR:
        return "handoff"
    if state["decision"] == Decision.ASK and state.get("triage"):
        return "intake_agent"
    return "fixed_reply" if state["decision"] in FIXED_REPLIES else "respond"


def fixed_reply(state: TurnState) -> TurnState:
    reply = FIXED_REPLIES[state["decision"]]
    if state.get("urgency"):
        reply += " " + URGENCY_ADVICE[state["urgency"]]
    return {"reply": reply}


def respond(state: TurnState, *, model: Runnable, knowledge: Knowledge | None = None) -> TurnState:
    # A plugged-in knowledge base grounds general answers (see app/extensions.py).
    reference = knowledge(state["text"]) if knowledge and state["decision"] == Decision.ANSWER else None
    response = model.invoke([
        ("system", RESPOND_PROMPT),
        ("human", render_context(state, decision=state["decision"],
                                 extraction=state["extraction"].model_dump(),
                                 clarification=state.get("clarification"),
                                 associated_symptoms=state.get("associated"),
                                 other_open_intakes=other_open_intakes(state), reference=reference)),
    ], config={"run_name": "reply"}, prompt_cache_key="q:reply")
    reply = response.text.strip()
    if not reply:
        raise ModelOutputError("Empty model response")
    if state["decision"] == Decision.HOME_MONITORING:
        reply += "\n\n" + home_care_advice(state)
    return {"reply": reply}


def home_care_advice(state: TurnState) -> str:
    """Precautions, warning signs and check-back, from the protocol: fixed text, not generated."""
    protocol = state["triage"].protocol
    lines = ["Precautions:", *(f"- {line}" for line in protocol.home_care),
             "Get help sooner if:", *(f"- {line}" for line in protocol.watch_for),
             GENERAL_WATCH_FOR,
             f"Check back with Q in {protocol.check_back_hours} hours with an update."]
    return "\n".join(lines)


def save_turn(state: TurnState, *, store: Store) -> TurnState:
    extraction, decision = state["extraction"], state["decision"]
    episode = None
    if state["episode_id"]:
        previous = state["selected_episode"]
        status = {
            Decision.EMERGENCY: "emergency", Decision.DOCTOR: "doctor",
            Decision.HOME_MONITORING: "monitoring",
        }.get(decision, "resolved_reported" if extraction.resolved else previous.status if previous else "triaging")
        triage = state.get("triage")
        check_back = previous.check_back_after if previous and status == "monitoring" else None
        if status == "monitoring" and check_back is None:
            hours = triage.protocol.check_back_hours if triage else DEFAULT_CHECK_BACK_HOURS
            check_back = datetime.now(UTC) + timedelta(hours=hours)
        episode = EpisodeWrite(
            id=state["episode_id"], member_id=extraction.member_id,
            complaint=extraction.complaint or "unspecified", status=status,
            slots=extraction.slots.model_dump(), questions_asked_delta=int(decision == Decision.ASK),
            check_back_after=check_back, related_episode_id=state["related_episode_id"],
            protocol_id=state.get("protocol_id"), findings=state.get("findings"),
            associated=state.get("associated"),
        )
    facts = tuple(
        FactWrite(member_id=extraction.member_id, episode_id=state["episode_id"], kind=f.kind,
                  subject=normalize_subject(f.subject), text=f.text)
        for f in extraction.facts
        if extraction.member_id and f.text.strip() and f.text in state["text"] and normalize_subject(f.subject)
    )
    saved = store.save_turn(TurnWrite(
        household_id=state["household_id"], session_id=state["session_id"], turn_id=state["turn_id"],
        user_text=state["text"], reply=state["reply"], decision=decision, member_id=extraction.member_id,
        episode=episode, facts=facts,
        pending_episode_id=state["episode_id"] if episode and episode.status != "resolved_reported" else None,
    ))
    if not saved:  # another request committed this same turn while the model was running
        return store.saved_turn(state["household_id"], state["session_id"], state["turn_id"], state["text"])
    return {}
