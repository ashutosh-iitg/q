"""Resolve the member first, then resolve an episode within that member's context."""

import logging
from dataclasses import replace
from datetime import UTC, datetime

from langchain_core.language_models import BaseChatModel

from app.agents.loop import ModelOutputError, run_agent
from app.agents.tools import episode_tools, member_tools, to_json
from app.memory.store import Store
from app.models.outputs import EpisodeResolution, MemberResolution
from app.models.records import Member, SessionContext
from app.models.turn import TurnState
from app.prompts.resolvers import EPISODE_AGENT_PROMPT, MEMBER_AGENT_PROMPT

logger = logging.getLogger(__name__)

def resolver_payload(state: TurnState, **extra) -> str:
    context = state["context"]
    return to_json({  # stable fields first, so a prompt cache can reuse more of the prefix
        "profile_id": state.get("profile_id"), "member_hint_id": context.member_hint_id,
        "pending_episode_id": context.pending_episode_id, **extra,
        "text": state["text"], "now": datetime.now(UTC),
    })


def member_scope(context: SessionContext, member: Member) -> tuple[SessionContext, str | None]:
    """The context narrowed to one member's episodes, and the pending episode if it is theirs."""
    scoped = replace(context, members=(member,),
                     episodes=tuple(e for e in context.episodes if e.member_id == member.id))
    pending = context.pending_episode_id if any(e.id == context.pending_episode_id for e in scoped.episodes) else None
    return scoped, pending


def resolve_member(state: TurnState, *, model: BaseChatModel) -> TurnState:
    try:
        resolution = run_agent(model, system=MEMBER_AGENT_PROMPT, payload=resolver_payload(state),
                               tools=member_tools(state["context"]), answer=MemberResolution)
    except ModelOutputError as exc:
        logger.warning("Member agent gave no answer: %s", exc)
        resolution = MemberResolution(status="unresolved", member_id=None, reason=str(exc),
                                      needed_info="Which family member this message is about",
                                      responder_instructions=None)
    return {"member_resolution": resolution}


def resolve_episode(state: TurnState, *, model: BaseChatModel, store: Store) -> TurnState:
    context, resolved = state["context"], state["member_resolution"]
    member = next((m for m in context.members if m.id == resolved.member_id), None)
    if resolved.status != "resolved" or member is None:
        # Clarification or a general question needs no episode lookup. Reconcile still validates IDs.
        return {"episode_resolution": EpisodeResolution(
            status="none", episode_id=None, reason="No resolved member", needed_info=None,
            responder_instructions=None)}
    scoped, pending = member_scope(context, member)
    tools = episode_tools(scoped, store, state["household_id"])
    try:
        resolution = run_agent(model, system=EPISODE_AGENT_PROMPT,
                               payload=resolver_payload(state, member=member, pending_episode_id=pending),
                               tools=tools, answer=EpisodeResolution)
    except ModelOutputError as exc:
        logger.warning("Episode agent gave no answer: %s", exc)
        resolution = EpisodeResolution(status="unresolved", episode_id=None, reason=str(exc),
                                       needed_info="Whether this continues an earlier illness or is a new one",
                                       responder_instructions=None)
    return {"episode_resolution": resolution}
