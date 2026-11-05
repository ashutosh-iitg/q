"""Symptom intake: observe the message against the episode's triage protocol.

The observe agent maps the user's words to the protocol's questions, quoting them, and reads
reported values into slots. For a new complaint it picks the protocol and looks its questions up
with get_protocol. Code keeps only answers to that protocol's questions whose quote is in the
message, merges them into the episode's findings, and runs the sufficiency check. Each answer keeps
what Q actually asked and the session it was given in. At a check-back, or when an episode continues in
a new session, emergency answers are cleared so they're asked again: an earlier "no" doesn't rule one
out now.

Other symptoms of the same illness (a fever that came with the cough) are kept on the episode with
whether each is still there, so a fever that has gone is never treated, asked about or handed over as
current. A question asked back about Q's last question ("who do you mean?") keeps the intake going.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from app.agents.extraction import NO_OBSERVATION, extraction_from
from app.agents.intake.merge import merge_associated, merge_findings
from app.agents.loop import ModelOutputError, run_agent
from app.agents.tools import to_json
from app.memory.store import Store, normalize_subject
from app.models.outputs import Observation, Slots
from app.models.records import Episode, Member, MemberFact
from app.models.turn import TurnState
from app.prompts.intake import OBSERVE_PROMPT
from app.triage.engine import NEW_SIGN, Profile, profile, triage
from app.triage.protocol import Protocol, Question, protocol_for, with_signs


def protocol_tools(protocols: Mapping[str, Protocol]) -> list[BaseTool]:
    @tool
    def get_protocol(protocol_id: str) -> str:
        """A triage protocol's questions, most urgent first: id, level and the question to answer."""
        protocol = protocols.get(protocol_id)
        if protocol is None:
            raise ValueError(f"Unknown protocol_id {protocol_id!r}; use an ID from protocol_catalog")
        return to_json(_view(protocol))

    return [get_protocol]


def observe(state: TurnState, *, model: BaseChatModel, store: Store, protocols: Mapping[str, Protocol]) -> TurnState:
    try:
        observation = run_agent(model, system=OBSERVE_PROMPT, payload=observe_payload(state, store, protocols),
                                tools=protocol_tools(protocols), answer=Observation)
    except (ModelOutputError, ValidationError):
        if not state["screen"].red_flags:
            raise
        observation = NO_OBSERVATION  # an emergency is answered even when observation fails
    return apply_observation(state, observation, protocols)


def last_question(state: TurnState, protocols: Mapping[str, Protocol]) -> Question | None:
    """What Q asked last turn: the next question then. The reply most likely answers it."""
    context, episode = state["context"], state["selected_episode"]
    if not (episode and episode.questions_asked):
        return None
    member = next((m for m in context.members if m.id == state["member_id"]), None)
    known = with_signs(_protocol_of(episode, protocols), episode.findings)
    before = triage(known, episode.findings, episode.slots, _profile(member, context.facts))
    return before.pending[0] if before.pending and before.pending[0].id != NEW_SIGN else None


def observe_payload(state: TurnState, store: Store, protocols: Mapping[str, Protocol]) -> str:
    """What the observe agent sees: the member, the episode across sessions, and Q's last question."""
    context, member_id, episode = state["context"], state["member_id"], state["selected_episode"]
    member = next((m for m in context.members if m.id == member_id), None)
    known = with_signs(_protocol_of(episode, protocols), episode.findings) if episode else None
    last = last_question(state, protocols)
    history = store.episode_history(state["household_id"], episode.id) if episode else ()
    recent_ids = {m.id for m in context.recent_messages}
    return to_json({  # from most to least stable, so a prompt cache can reuse more of the prefix
        "protocol_catalog": [{"id": p.id, "complaints": list(p.complaints)} for p in protocols.values()],
        "profile_id": state.get("profile_id"),
        "member": member, "member_facts": [f for f in context.facts if f.member_id == member_id],
        "episode": episode, "protocol": known and _view(known),
        "episode_history": [m for m in history if m.id not in recent_ids],
        "recent_messages": context.recent_messages,
        "last_question": last and {"id": last.id, "finding": last.finding},
        "text": state["text"], "now": datetime.now(UTC),
    })


def apply_observation(state: TurnState, observation: Observation, protocols: Mapping[str, Protocol]) -> TurnState:
    """Code's part of observing: keep quoted answers, pick or open the episode, merge, run triage."""
    context, member_id, episode = state["context"], state["member_id"], state["selected_episode"]
    member = next((m for m in context.members if m.id == member_id), None)
    member_profile = _profile(member, context.facts)
    last = last_question(state, protocols)
    if observation.intent == "clarification" and not (episode and last):
        observation = observation.model_copy(update={"intent": "other"})  # no question of Q's to clarify
    if episode and observation.complaint and normalize_subject(observation.complaint) != episode.complaint:
        episode = None  # a new condition opens its own episode
    if observation.intent in ("general_question", "other") and not observation.resolved:
        episode = None  # education and small talk must not mutate an active episode
    if episode and episode.status == "resolved_reported" and not observation.resolved:
        episode = None  # a recurrence opens a distinct episode
    if episode:
        previous = {k: v for k, v in episode.slots.items() if k in Slots.model_fields}
        slots = Slots.model_validate({**observation.slots.model_dump(), **previous,
                                     **observation.slots.model_dump(exclude_none=True)})
        observation = observation.model_copy(update={"slots": slots, "complaint": episode.complaint})
    observation = observation.model_copy(update={"complaint": normalize_subject(observation.complaint or "") or None})
    extraction = extraction_from(state, member_id, observation).model_copy(
        update={"episode_id": episode.id if episode else None},
    )
    has_episode = bool(member_id and (episode or extraction.intent in ("symptom_report", "update")))
    related = next((e.id for e in context.episodes if e.member_id == member_id
                    and e.complaint == extraction.complaint and e.status == "resolved_reported"), None)
    result = {
        "extraction": extraction, "selected_episode": episode,
        "episode_id": (episode.id if episode else uuid4().hex) if has_episode else None,
        "related_episode_id": related if episode is None else None,
        "protocol_id": None, "findings": None, "triage": None, "associated": None,
    }
    if not has_episode:
        return result
    associated = merge_associated(episode.associated if episode else {}, observation.associated, state["text"])
    protocol = (with_signs(_protocol_of(episode, protocols), episode.findings) if episode
                else protocol_for(extraction.complaint, protocols))
    asked = next((m.content for m in reversed(context.recent_messages) if m.role == "assistant"), None)
    findings = merge_findings(protocol, episode, observation.answers, state["text"],
                              session_id=state["session_id"], last_asked=(last.id, asked) if last and asked else None)
    found = triage(protocol, findings, extraction.slots.model_dump(), member_profile)
    # The question asked last turn is still the next one: the reply didn't answer it.
    unanswered = bool(episode and last and found.pending and found.pending[0].id == last.id)
    return {**result, "protocol_id": protocol.id, "findings": findings, "triage": found,
            "associated": associated, "unanswered_last_time": unanswered}


def _protocol_of(episode: Episode, protocols: Mapping[str, Protocol]) -> Protocol:
    return protocols.get(episode.protocol_id) or protocol_for(episode.complaint, protocols)


def _profile(member: Member | None, facts: tuple[MemberFact, ...]) -> Profile:
    """Seeded conditions plus conditions remembered from earlier conversations."""
    if member is None:
        return profile(None, [])
    remembered = [f.subject for f in facts if f.member_id == member.id and f.kind == "condition"]
    return profile(member.age_years, [*member.conditions, *remembered])


def _view(protocol: Protocol) -> dict[str, Any]:
    """Each question as the finding an answer is recorded against, so Q's wording can't flip it."""
    return {"id": protocol.id, "complaints": list(protocol.complaints),
            "questions": [{"id": q.id, "level": q.level, "finding": q.finding} for q in protocol.questions if q.ask]}
