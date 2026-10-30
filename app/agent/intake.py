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

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from app.agent.evidence import is_quoted
from app.agent.loop import ModelOutputError, run_agent
from pydantic import ValidationError

from app.agent.nodes import NO_OBSERVATION, extraction_from
from app.agent.prompts import OBSERVE_PROMPT
from app.agent.state import Answer, AssociatedSymptom, Episode, Member, MemberFact, Observation, Slots, TurnState
from app.agent.tools import to_json
from app.store import Store, normalize_subject
from app.triage.engine import NEW_SIGN, Profile, profile, triage
from app.triage.protocol import Protocol, protocol_for, with_signs

logger = logging.getLogger(__name__)


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
    context, member_id, episode = state["context"], state["member_id"], state["selected_episode"]
    member = next((m for m in context.members if m.id == member_id), None)
    known = with_signs(_protocol_of(episode, protocols), episode.findings) if episode else None
    member_profile = _profile(member, context.facts)
    # What Q asked last turn: the next question then. The reply most likely answers it.
    before = triage(known, episode.findings, episode.slots, member_profile) if episode and episode.questions_asked else None
    last = before.pending[0] if before and before.pending and before.pending[0].id != NEW_SIGN else None
    payload = to_json({  # from most to least stable, so a prompt cache can reuse more of the prefix
        "protocol_catalog": [{"id": p.id, "complaints": list(p.complaints)} for p in protocols.values()],
        "profile_id": state.get("profile_id"),
        "member": member, "member_facts": [f for f in context.facts if f.member_id == member_id],
        "episode": episode, "protocol": known and _view(known),
        "episode_history": store.episode_history(state["household_id"], episode.id) if episode else (),
        "recent_messages": context.recent_messages,
        "last_question": last and {"id": last.id, "finding": last.finding},
        "text": state["text"], "now": datetime.now(UTC),
    })
    try:
        observation = run_agent(model, system=OBSERVE_PROMPT, payload=payload, tools=protocol_tools(protocols),
                                answer=Observation)
    except (ModelOutputError, ValidationError):
        if not state["screen"].red_flags:
            raise
        observation = NO_OBSERVATION  # an emergency is answered even when observation fails
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


def merge_associated(previous: Mapping[str, Any], reported: list[AssociatedSymptom], text: str) -> dict[str, Any]:
    """The episode's other symptoms with this message's quoted reports applied; the latest status wins."""
    now = datetime.now(UTC).isoformat()
    supported = {subject: {"status": s.status, "quote": s.quote, "at": now}
                 for s in reported if (subject := normalize_subject(s.symptom)) and is_quoted(s.quote, text)}
    if len(supported) < len(reported):
        logger.warning("Dropped %d related symptoms not supported by the message", len(reported) - len(supported))
    return {**previous, **supported}


def merge_findings(protocol: Protocol, episode: Episode | None, answers: list[Answer], text: str, *,
                   session_id: str | None = None, last_asked: tuple[str, str] | None = None) -> dict[str, Any]:
    """Earlier findings plus this message's supported answers.

    Each answer records the session it was given in and, when it answers the question Q asked last
    turn, Q's actual words (last_asked: question ID, message), so a note can show what was asked.
    """
    previous = {k: kept for k, v in (episode.findings if episode else {}).items()
                if (kept := _unanswered(v) if _ask_again(protocol, k, v, episode, session_id) else dict(v))}
    now = datetime.now(UTC).isoformat()
    # A chosen sign's entry also holds its definition (sign, level, ask), so answers are merged into it.
    supported = {a.question_id: {**previous.get(a.question_id, {}), "value": a.value, "quote": a.quote, "at": now,
                                 "session_id": session_id,
                                 "asked": last_asked[1] if last_asked and last_asked[0] == a.question_id else None}
                 for a in answers if protocol.question(a.question_id) and is_quoted(a.quote, text)}
    if len(supported) < len(answers):
        logger.warning("Dropped %d answers not supported by the message or protocol", len(answers) - len(supported))
    return {**previous, **supported}


def _ask_again(protocol: Protocol, question_id: str, finding: Mapping[str, Any], episode: Episode | None,
               session_id: str | None) -> bool:
    """An emergency answer is asked again at a check-back, or when it was given in an earlier session."""
    question = protocol.question(question_id)
    level = question.level if question else finding.get("level")
    if level != "emergency" or "value" not in finding:
        return False
    return bool(episode and episode.status == "monitoring") or finding.get("session_id") != session_id


def _unanswered(finding: Mapping[str, Any]) -> dict[str, Any]:
    """The finding without its answer; a chosen sign keeps its definition so it can be asked again."""
    return {k: finding[k] for k in ("sign", "level", "ask") if k in finding}


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
