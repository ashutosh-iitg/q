"""Graph nodes around the agents: load context, screen, reconcile, decide, save."""

from datetime import UTC, datetime, timedelta

from langchain_core.runnables import Runnable

from app.agents.evidence import is_quoted
from app.agents.extraction import NO_OBSERVATION, extraction_from
from app.agents.loop import ModelOutputError
from app.agents.tools import to_json
from app.memory.store import Store, normalize_subject
from app.memory.writes import EpisodeWrite, FactWrite, TurnWrite
from app.models.outputs import EpisodeResolution, MemberResolution, Screen
from app.models.records import Clarification, Member
from app.models.turn import TurnState
from app.prompts.screen import SCREEN_PROMPT
from app.triage.policy import DEFAULT_URGENCY, Decision, doctor_urgency, note_on_file, rule

DEFAULT_CHECK_BACK_HOURS = 24


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


def decide(state: TurnState) -> TurnState:
    triage = state.get("triage")
    decision, reason = rule(state["extraction"], state["context"], state.get("clarification"), triage)
    if decision != Decision.DOCTOR:
        return {"decision": decision, "decision_reason": reason, "urgency": None, "note_on_file": False}
    handed_over = note_on_file(state["extraction"], state["context"])
    urgency = (handed_over.urgency or DEFAULT_URGENCY) if handed_over else doctor_urgency(reason, triage)
    return {"decision": decision, "decision_reason": reason, "urgency": urgency, "note_on_file": bool(handed_over)}


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
            urgency=state.get("urgency") if decision == Decision.DOCTOR else None,
            handoff_id=state["turn_id"] if state.get("handoff_path") else None,
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
