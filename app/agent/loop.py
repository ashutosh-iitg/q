"""How every Q agent runs: lookups in rounds, then the answer.

Each call offers the model its read-only lookups and its answer (strict schemas, tool_choice
"required"). Lookups only: code runs them all and calls again, so the model can follow up on
what it read (list episodes, then read one's history). The answer only: done. Both at once: the
answer was written before its lookups' results, so it is discarded and the lookups run. After
MAX_LOOKUP_ROUNDS a last call offers only the answer and forces it. No call is spent on "ready",
and an answer is never written before the lookups it asked for. Each call is traced under its
answer's name and number ("EpisodeResolution · call 1"), so a trace reads without opening calls.
"""

from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class ModelOutputError(ValueError):
    """The model returned unusable references, evidence, or an empty response."""


MAX_LOOKUP_ROUNDS = 3  # rounds of lookups per agent per turn; a round may run several
MAX_AGENT_STEPS = MAX_LOOKUP_ROUNDS + 1  # model calls per agent per turn, at most

Answer = TypeVar("Answer", bound=BaseModel)


def run_agent(
    model: BaseChatModel, *, system: str, payload: str, tools: list[BaseTool], answer: type[Answer] | BaseTool,
) -> Answer | ToolMessage:
    """Look things up with `tools` for up to MAX_LOOKUP_ROUNDS rounds, then call `answer`.

    `answer` is either a schema the result is validated against, or a final tool that is run with
    the model's arguments (its ToolMessage is returned).
    """
    final = answer.name if isinstance(answer, BaseTool) else answer.__name__
    by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(payload)]
    cache = f"q:{final}"  # one prompt cache per role: its instructions and tools are the shared prefix
    if tools:
        bound = model.bind_tools([*tools, answer], tool_choice="required", strict=True, prompt_cache_key=cache)
        for step in range(1, MAX_LOOKUP_ROUNDS + 1):
            reply = bound.invoke(messages, config=_named(final, f"call {step}"))
            lookups = [c for c in reply.tool_calls if c["name"] != final]
            if not lookups:
                return _submit(answer, final, reply)
            messages = [*messages, _only(reply, lookups), *(_run_tool(by_name, call) for call in lookups)]
    forced = model.bind_tools([answer], tool_choice=final, strict=True, prompt_cache_key=cache)
    return _submit(answer, final, forced.invoke(messages, config=_named(final, "answer")))


def _named(final: str, step: str) -> RunnableConfig:
    """Trace name for one model call, e.g. "MemberResolution · call 2"; tagged with the agent's answer."""
    return {"run_name": f"{final} · {step}", "tags": [final]}


def _only(reply: AIMessage, lookups: list[ToolCall]) -> AIMessage:
    """The reply with just these calls: an answer sent alongside lookups is dropped.

    The Responses API also keeps each call as a function_call content block, and rejects a call
    sent back without a result, so the dropped answer's block goes too.
    """
    kept = {call["id"] for call in lookups}
    content = reply.content if isinstance(reply.content, str) else [
        block for block in reply.content
        if not (isinstance(block, dict) and block.get("type") == "function_call" and block.get("call_id") not in kept)
    ]
    return reply.model_copy(update={"tool_calls": lookups, "content": content})


def _submit(answer: type[Answer] | BaseTool, final: str, reply: AIMessage) -> Answer | ToolMessage:
    submitted = next((c for c in reply.tool_calls if c["name"] == final), None)
    if submitted is None:
        raise ModelOutputError(f"{final} agent did not answer")
    return answer.invoke(submitted) if isinstance(answer, BaseTool) else answer.model_validate(submitted["args"])


def _run_tool(by_name: dict[str, BaseTool], call: ToolCall) -> ToolMessage:
    tool = by_name.get(call["name"])
    if tool is None:
        return ToolMessage(f"Error: no tool named {call['name']!r}", tool_call_id=call["id"], status="error")
    try:
        return tool.invoke(call)
    except ValueError as exc:  # bad arguments or IDs: let the model correct itself
        return ToolMessage(f"Error: {exc}", tool_call_id=call["id"], status="error")
