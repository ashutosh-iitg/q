"""Middleware that decides, on every model call, what each deep agent sees and may do.

A role's view is a function of the turn ledger, so it moves on as the turn does: the context agent sees
the member resolver's prompt and tools until a member is recorded, then the episode resolver's, scoped
to that member. A role whose inputs aren't ready (intake before context) is told so and offered no
tools, instead of guessing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError


@dataclass(frozen=True)
class Phase:
    key: str  # names the prompt cache and the trace, e.g. "context:member"
    prompt: str
    data: str | None  # the role's turn data, JSON
    tools: frozenset[str]  # the only tools the model is offered
    parallel_tools: bool = True


def blocked(role: str, why: str) -> Phase:
    return Phase(f"{role}:blocked", f"You can't do this yet: {why}. Reply with exactly that sentence.", None,
                 frozenset())


class RoleView(AgentMiddleware):
    """`view` gives the role's phase for the next model call; `done` says its result is recorded, which
    ends the role's run. A final tool that returns an error doesn't end it, so the model can fix it."""

    def __init__(self, role: str, view: Callable[[], Phase], done: Callable[[], bool]) -> None:
        super().__init__()
        self.role = role
        self.view = view
        self.done = done

    @property
    def name(self) -> str:
        return f"RoleView[{self.role}]"

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return {"jump_to": "end"} if self.done() else None

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        phase = self.view()
        prompt = phase.prompt if phase.data is None else f"{phase.prompt}\n\nturn_data:\n{phase.data}"
        tools = [t for t in request.tools if _name(t) in phase.tools]
        settings = {**request.model_settings, "prompt_cache_key": f"q:deep:{phase.key}"}
        if tools and not phase.parallel_tools:
            settings["parallel_tool_calls"] = False
        return handler(request.override(system_message=SystemMessage(prompt), tools=tools,
                                        tool_choice="any" if tools else None, model_settings=settings))


class ToolErrors(AgentMiddleware):
    """A tool's ValueError (an unknown ID, an unquoted answer) goes back to the model to correct, as in
    the graph's agent loop. Anything else, such as a failed doctor handoff, fails the turn."""

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role

    @property
    def name(self) -> str:
        return f"ToolErrors[{self.role}]"

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        try:
            return handler(request)
        except (ValueError, ValidationError) as exc:
            call = request.tool_call
            return ToolMessage(f"Error: {exc}", tool_call_id=call["id"], name=call["name"], status="error")


class ReportBack(AgentMiddleware):
    """A subagent that ends on its final tool hands back that tool's report. deepagents returns a
    subagent's last AI text to the supervisor, and a run that ends on a tool has none."""

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role

    @property
    def name(self) -> str:
        return f"ReportBack[{self.role}]"

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, ToolMessage):
            return {"messages": [AIMessage(content=last.text)]}
        if isinstance(last, HumanMessage):  # called again after its result was recorded
            return {"messages": [AIMessage(content="Nothing to do: it's already recorded; see this_turn.")]}
        return None


def _name(tool: Any) -> str | None:
    return getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None)
