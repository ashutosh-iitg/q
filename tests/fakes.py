"""A scripted stand-in for ChatOpenAI and builders for each role's output."""

import json
import threading

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.agent.handoff import IntakeNoteDraft
from app.agent.intake_agent import IntakeQuestions, NewSign
from app.agent.state import EpisodeResolution, Extraction, MemberResolution, Observation, Screen, Slots
from tests.helpers import new_id

EMPTY_SLOTS = {name: None for name in Slots.model_fields}


def extraction(member_id=None, **changes):
    return Extraction.model_validate({
        "member_id": member_id, "episode_id": None, "intent": "general_question", "complaint": None,
        "slots": EMPTY_SLOTS, "red_flags": [], "doctor_flags": [], "resolved": False, "facts": [], **changes,
    })


def screen(**changes):
    return Screen.model_validate({"red_flags": [], "doctor_flags": [], "doctor_requested": False, **changes})


def member(member_id=None, status=None, **changes):
    return MemberResolution.model_validate({
        "status": status or ("resolved" if member_id else "none"), "member_id": member_id, "reason": "test",
        "needed_info": None, "responder_instructions": None, **changes,
    })


def episode(episode_id=None, status=None, **changes):
    return EpisodeResolution.model_validate({
        "status": status or ("existing" if episode_id else "none"), "episode_id": episode_id, "reason": "test",
        "needed_info": None, "responder_instructions": None, **changes,
    })


def observation(**changes):
    slots = {**EMPTY_SLOTS, **changes.pop("slots", {})}
    return Observation.model_validate({
        "intent": "general_question", "complaint": None, "slots": slots, "answers": [], "associated": [],
        "resolved": False, "facts": [], **changes,
    })


def answered(text, **values):
    """Protocol answers quoting the whole message, e.g. answered("no to all", pain_spreads="absent")."""
    return [{"question_id": question, "value": value, "quote": text} for question, value in values.items()]


def draft(**changes):
    return IntakeNoteDraft.model_validate({
        "chief_complaint": "chest pain", "history": [], "symptom_timeline": [], "relevant_history": None,
        **changes,
    })


def call(name, **args):
    """An agent step that calls one lookup tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": new_id("call")}])


def submit(answer: BaseModel, name=None):
    """An agent step that submits its final answer, or calls its final tool."""
    return call(name or type(answer).__name__, **answer.model_dump())


def ask_pending(messages):
    """The intake agent asking the next question, as the default script does."""
    question = agent_payload(messages)["next_question"]
    if question.get("choose_sign"):  # a guideline turn: the agent names a sign and asks it
        sign = NewSign(sign=f"Danger sign {len(messages)}")
        return IntakeQuestions(message=f"Any danger sign {len(messages)}?", new_sign=sign)
    return IntakeQuestions(message=question["ask"], new_sign=None)


HANDOFF = "handoff"  # the intake-note agent's role; its final step is the escalate_to_doctor tool


class FakeModel:
    """Each role has its own script, because the parallel branches call the model concurrently.

    A role without a script returns its default on every call. A script is a list consumed one model
    call at a time; an item is an output, or a callable taking that call's input and returning one.
    Agents may also return AIMessages with tool calls (lookups, see app.agent.loop); a bare answer
    is submitted for them.
    """

    DEFAULTS = {Screen: screen(), MemberResolution: member(), EpisodeResolution: episode(),
                Observation: observation(), IntakeQuestions: ask_pending, HANDOFF: draft()}

    def __init__(self, *, screens=None, members=None, episodes=None, observations=None, intakes=None,
                 handoffs=None, replies=()):
        self.scripts = {Screen: screens, MemberResolution: members, EpisodeResolution: episodes,
                        Observation: observations,
                        IntakeQuestions: intakes, HANDOFF: handoffs}
        self.inputs = {role: [] for role in self.scripts}  # payload dicts; full message lists for agents
        self.replies = list(replies)
        self.reply_inputs = []
        self.bindings = []  # (tool names, bind_tools keyword arguments), one per bind
        self._lock = threading.Lock()

    @property
    def reply_calls(self):
        return len(self.reply_inputs)

    def _next(self, role, given):
        with self._lock:
            self.inputs[role].append(given)
            script = self.scripts[role]
            result = self.DEFAULTS[role] if script is None else script.pop(0)
        return result(given) if callable(result) else result

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda messages: self._next(schema, json.loads(messages[-1][1])))

    def bind_tools(self, tools, **kwargs):
        names = [t.name if isinstance(t, BaseTool) else t.__name__ for t in tools]
        self.bindings.append((names, kwargs))
        final = tools[-1]  # the answer schema or final tool is bound last
        role = HANDOFF if isinstance(final, BaseTool) else final

        def invoke(messages):
            step = self._next(role, messages)
            return step if isinstance(step, AIMessage) else submit(step, names[-1])
        return RunnableLambda(invoke)

    def invoke(self, messages, config=None, **kwargs):
        with self._lock:
            self.reply_inputs.append(json.loads(messages[-1][1]))
            return AIMessage(content=self.replies.pop(0))


def agent_payload(messages):
    """The JSON input an agent was given, from its message list."""
    return json.loads(messages[1].content)
