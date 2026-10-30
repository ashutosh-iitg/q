"""One graph run per user turn.

load -> [screen | member agent -> episode agent] -> reconcile -> observe -> decide -> reply -> save.
Screening runs alongside resolution; episode resolution consumes the member result.
A doctor decision runs the handoff agent, which saves an intake note, before the reply.
An ask decision runs the intake agent, which writes the protocol's next questions.
"""

from collections.abc import Mapping
from functools import partial
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.handoff.note import handoff
from app.agents.intake.observe import observe
from app.agents.intake.questions import intake_agent
from app.agents.resolvers import resolve_episode, resolve_member
from app.extensions import Extensions
from app.graph.nodes import decide, load_context, reconcile, save_turn, screen
from app.graph.replies import fixed_reply, respond
from app.graph.routing import PARALLEL_BRANCHES, route_after_decide, route_after_load, route_after_reconcile
from app.memory.store import Store
from app.models.outputs import Screen
from app.models.turn import TurnOutput, TurnState
from app.triage.protocol import Protocol, load_protocols


def build_graph(
    store: Store, model: ChatOpenAI, *, handoff_dir: Path, protocols: Mapping[str, Protocol] | None = None,
    extensions: Extensions = Extensions(),
) -> CompiledStateGraph:
    protocols = protocols or load_protocols()
    screener = model.with_structured_output(Screen, method="json_schema", strict=True, prompt_cache_key="q:Screen")
    graph = StateGraph(TurnState, output_schema=TurnOutput)
    graph.add_node("load_context", partial(load_context, store=store))
    graph.add_node("screen", partial(screen, screener=screener))
    graph.add_node("resolve_member", partial(resolve_member, model=model))
    graph.add_node("resolve_episode", partial(resolve_episode, model=model, store=store))
    graph.add_node(reconcile)
    graph.add_node("observe", partial(observe, model=model, store=store, protocols=protocols))
    graph.add_node(decide)
    graph.add_node("handoff", partial(handoff, model=model, store=store, directory=handoff_dir,
                                      lab_results=extensions.lab_results, deliver=extensions.deliver_handoff))
    graph.add_node(fixed_reply)
    graph.add_node("intake_agent", partial(intake_agent, model=model, store=store,
                                           lab_results=extensions.lab_results))
    graph.add_node("respond", partial(respond, model=model, knowledge=extensions.knowledge))
    graph.add_node("save_turn", partial(save_turn, store=store))

    graph.add_edge(START, "load_context")
    graph.add_conditional_edges("load_context", route_after_load, [END, *PARALLEL_BRANCHES])
    graph.add_edge("resolve_member", "resolve_episode")
    graph.add_edge(["screen", "resolve_episode"], "reconcile")
    graph.add_conditional_edges("reconcile", route_after_reconcile, ["observe", "decide"])
    graph.add_edge("observe", "decide")
    graph.add_conditional_edges("decide", route_after_decide,
                                ["handoff", "fixed_reply", "intake_agent", "respond"])
    graph.add_edge("handoff", "fixed_reply")
    graph.add_edge("fixed_reply", "save_turn")
    graph.add_edge("intake_agent", "save_turn")
    graph.add_edge("respond", "save_turn")
    graph.add_edge("save_turn", END)
    return graph.compile()
