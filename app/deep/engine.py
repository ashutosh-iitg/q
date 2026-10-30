"""The deep engine: the same turn as the graph, with a deepagents supervisor in the middle.

Code runs both ends, as in the graph: it loads the household's memory, replays a turn already saved,
and screens the message before any agent runs; afterwards it fills what the agents didn't (a safe
reply, a minimal doctor note) and saves the turn atomically. Between them the supervisor decides
the order of work and the outcome, calling subagents through deepagents' task tool.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.errors import GraphRecursionError

from app.agents.extraction import NO_OBSERVATION, extraction_from
from app.agents.handoff.note import handoff_inputs, save_minimal_note
from app.agents.intake.questions import fallback_question
from app.deep.agents import build_turn_agent
from app.deep.ledger import TurnLedger
from app.deep.tools import Deps
from app.extensions import Extensions
from app.graph.nodes import load_context, save_turn, screen
from app.graph.replies import fixed_reply
from app.memory.store import Store
from app.models.outputs import Screen
from app.models.turn import TurnOutput, TurnState
from app.triage.policy import DEFAULT_URGENCY, Decision
from app.triage.protocol import Protocol, load_protocols

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 120  # graph steps across the supervisor and its subagents; the call limits bind first
FALLBACK_REPLY = ("I couldn't work through that just now. Please send it again. If symptoms are severe or "
                  "getting worse, call your local emergency number now.")
OUTPUT_KEYS = tuple(TurnOutput.__annotations__)
START = "Handle this turn. Everything you need is in turn_data."


class DeepEngine:
    def __init__(self, store: Store, model: BaseChatModel, *, handoff_dir: Path,
                 protocols: Mapping[str, Protocol] | None = None, extensions: Extensions = Extensions()) -> None:
        self.model = model
        self.deps = Deps(store, dict(protocols or load_protocols()), handoff_dir, extensions)
        self.screener = model.with_structured_output(Screen, method="json_schema", strict=True,
                                                     prompt_cache_key="q:Screen")
        self._turn = RunnableLambda(self._run, name="q.turn")

    def invoke(self, payload: TurnState, config: RunnableConfig | None = None) -> TurnOutput:
        return self._turn.invoke(payload, config)

    def _run(self, payload: TurnState, config: RunnableConfig) -> TurnOutput:
        state: TurnState = {**payload, **load_context(payload, store=self.deps.store)}
        if state["replayed"]:
            return _output(state)
        ledger = TurnLedger({**state, **screen(state, screener=self.screener)})
        agent = build_turn_agent(self.model, ledger, self.deps)
        try:
            agent.invoke({"messages": [HumanMessage(START)]},
                         {**config, "recursion_limit": RECURSION_LIMIT, "run_name": "supervisor"})
        except GraphRecursionError:
            logger.warning("Deep engine hit its step limit on turn %s", state["turn_id"])
        ledger.apply(complete(ledger.state, self.deps))
        return _output({**ledger.state, **save_turn(ledger.state, store=self.deps.store)})


def complete(state: TurnState, deps: Deps) -> TurnState:
    """What the agents left undone, done by code, so every turn saves a reply and keeps its promises."""
    update: dict[str, Any] = {}
    if "extraction" not in state:  # the supervisor answered without intake (an emergency) or gave up
        update |= {"extraction": extraction_from(state, state.get("member_id"), NO_OBSERVATION),
                   "episode_id": None, "selected_episode": state.get("selected_episode"),
                   "related_episode_id": None}
    if not state.get("reply"):
        update |= _fallback_reply({**state, **update})
    state = {**state, **update}
    if state["decision"] == Decision.DOCTOR and not (state.get("handoff_path") or state.get("note_on_file")):
        state = {**state, "urgency": state.get("urgency") or DEFAULT_URGENCY}
        note = handoff_inputs(state, deps.store, deps.handoff_dir, deps.extensions.lab_results,
                              deps.extensions.deliver_handoff)
        save_minimal_note(state, note)
        update |= {"urgency": state["urgency"], "handoff_path": str(note.path)}
    return update


def _fallback_reply(state: TurnState) -> TurnState:
    """The agents ended without a reply. A red flag still gets the emergency text, a doctor decision the
    doctor text, an intake its next question; anything else gets a safe retry message."""
    logger.warning("Deep engine ended turn %s without a reply (decision=%s)", state["turn_id"], state.get("decision"))
    if state["screen"].red_flags or state.get("decision") == Decision.EMERGENCY:
        reply = fixed_reply({**state, "decision": Decision.EMERGENCY, "urgency": None})["reply"]
        return {"decision": Decision.EMERGENCY, "decision_reason": "red_flag", "urgency": None, "reply": reply}
    if state.get("decision") == Decision.DOCTOR:
        urgency = state.get("urgency") or DEFAULT_URGENCY
        return {"urgency": urgency, "reply": fixed_reply({**state, "urgency": urgency})["reply"]}
    found = state.get("triage")
    if state.get("decision") == Decision.ASK and found and found.pending:
        return {**fallback_question(state), "decision_reason": state.get("decision_reason") or "intake_incomplete"}
    return {"decision": Decision.ANSWER, "decision_reason": "deep_fallback", "urgency": None,
            "reply": FALLBACK_REPLY}


def _output(state: TurnState) -> TurnOutput:
    return {key: state.get(key) for key in OUTPUT_KEYS if key in state or key in ("urgency", "handoff_path")}
