"""Read-only lookups the resolver agents call.

Each set is built for one turn and bound to its household in code, so the model never passes a
household ID and can't reach another family's data. Invalid IDs raise ValueError, which the agent
loop returns to the model as a tool error so it can correct itself.
"""

import json
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.agent.state import SessionContext
from app.consolidation import _json_default, _plain
from app.extensions import LabResults
from app.store import Store


def to_json(payload: Any) -> str:
    """JSON for agents. Every time says how long ago it was, so "this morning" in yesterday's
    message reads as yesterday morning."""
    def default(value: Any) -> Any:
        if isinstance(value, datetime):
            return f"{value.isoformat()} ({_days_from_today(value)})"
        return _plain(value) if hasattr(value, "__dataclass_fields__") else _json_default(value)
    return json.dumps(payload, default=default)


def _days_from_today(value: datetime) -> str:
    # UTC calendar days: households have no time zone yet.
    days = (datetime.now(UTC).date() - value.astimezone(UTC).date()).days
    if days in (0, 1, -1):
        return {0: "today", 1: "yesterday", -1: "tomorrow"}[days]
    return f"{days} days ago" if days > 0 else f"in {-days} days"


def _require_member(context: SessionContext, member_id: str) -> None:
    if member_id not in {m.id for m in context.members}:
        raise ValueError(f"Unknown member_id {member_id!r}; use an ID from list_members")


def _recent_messages(context: SessionContext) -> BaseTool:
    @tool
    def recent_messages() -> str:
        """This session's latest messages, oldest first, with the member and episode each was about."""
        return to_json(context.recent_messages)

    return recent_messages


def member_tools(context: SessionContext) -> list[BaseTool]:
    @tool
    def list_members() -> str:
        """The household's members: id, name, aliases, age in years and known conditions."""
        return to_json(context.members)

    @tool
    def member_facts(member_id: str) -> str:
        """Remembered health facts about one member, such as conditions, medications or past readings."""
        _require_member(context, member_id)
        return to_json([f for f in context.facts if f.member_id == member_id])

    return [list_members, member_facts, _recent_messages(context)]


def lab_tools(context: SessionContext, household_id: str, results: LabResults | None) -> list[BaseTool]:
    """The lab_results lookup, only when a lab-report analyser is plugged in (see app/extensions.py)."""
    if results is None:
        return []

    @tool
    def lab_results(member_id: str) -> str:
        """Analysed lab reports for one member: test, value, reference range and date. Empty if none."""
        _require_member(context, member_id)
        return to_json(results(household_id, member_id))

    return [lab_results]


def episode_tools(context: SessionContext, store: Store, household_id: str) -> list[BaseTool]:
    @tool
    def list_episodes(member_id: str | None) -> str:
        """Open and recently resolved health episodes: id, member_id, complaint, status, slots and
        last update. Pass a member_id to filter, or null for every member."""
        if member_id is not None:
            _require_member(context, member_id)
        return to_json([e for e in context.episodes if member_id in (None, e.member_id)])

    @tool
    def episode_history(episode_id: str) -> str:
        """An episode's past conversation across sessions, oldest first."""
        if episode_id not in {e.id for e in context.episodes}:
            raise ValueError(f"Unknown episode_id {episode_id!r}; use an ID from list_episodes")
        return to_json(store.episode_history(household_id, episode_id))

    @tool
    def past_session_summaries() -> str:
        """Summaries of the household's recent earlier sessions."""
        return to_json(context.prior_summaries)

    return [list_episodes, episode_history, past_session_summaries, _recent_messages(context)]
