"""A scripted chat model for the deep engine: one script per role, the role read from the system prompt."""

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from tests.fakes import call, screen

ROLES = {  # the first words of each role's system prompt
    "You are Q's turn supervisor": "supervisor",
    "You are Q's member resolver": "context:member",
    "You are Q's episode resolver": "context:episode",
    "You extract observations": "intake",
    "You are Q's symptom intake": "questions",
    "You are Q, a calm": "respond",
    "You write the intake note": "handoff",
    "Your result is recorded": "done",
    "You can't do this yet": "blocked",
}


@dataclass(frozen=True)
class Call:
    role: str
    system: str
    tools: tuple[str, ...]
    messages: tuple[BaseMessage, ...]
    settings: dict[str, Any]


def role_of(system: str) -> str:
    return next((role for start, role in ROLES.items() if system.startswith(start)), "unknown")


def turn_data(system: str) -> dict:
    return json.loads(system.split("turn_data:\n", 1)[1])


def task(subagent: str) -> AIMessage:
    return call("task", description=f"Do the {subagent} step.", subagent_type=subagent)


class ScriptedChat(BaseChatModel):
    """Each script is a list consumed one model call at a time; an item is an AIMessage or a callable
    taking the Call. The done and blocked phases answer for themselves."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    scripts: dict[str, list] = Field(default_factory=dict)
    screens: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)
    order: list = Field(default_factory=list)  # "screen" and role names, in call order

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        system = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        role = role_of(system)
        tools = tuple(t["function"]["name"] for t in kwargs.get("tools") or [])
        settings = {k: v for k, v in kwargs.items() if k != "tools"}
        step = Call(role, system, tools, tuple(messages[1:]), settings)
        self.calls.append(step)
        self.order.append(role)
        script = self.scripts.get(role)
        if role == "done":
            item = AIMessage(content="recorded")
        elif role == "blocked":
            item = AIMessage(content=system.split(": ", 1)[1].split(". Reply", 1)[0])
        elif not script:
            raise AssertionError(f"No script left for {role}")
        else:
            item = script.pop(0)
        message = item(step) if callable(item) else item
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        def run(_: Any) -> Any:
            self.order.append("screen")
            return self.screens.pop(0) if self.screens else screen()
        return RunnableLambda(run)

    def by_role(self, role: str) -> list[Call]:
        return [c for c in self.calls if c.role == role]
