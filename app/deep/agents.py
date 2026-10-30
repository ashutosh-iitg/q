"""The turn supervisor and its subagents, assembled with deepagents for one turn.

Who sees what (set by each RoleView on every model call):
- supervisor: the household, tracked episodes, this conversation tagged by member and episode, earlier
  conversations' summaries, the safety screen, and what this run has recorded so far.
- context: the member resolver's view, then the episode resolver's, scoped to the member it chose.
- intake: the observe agent's view: the member, the episode's findings and protocol, its history across
  sessions, and the question Q asked last turn.
- questions: the intake question writer's view: what's known, what's left, and who is writing.
- respond: the decision, what was observed, and any knowledge-base passages.
- handoff: everything code knows exactly for the doctor's note, and the evidence it may quote.
"""

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from app.agents.handoff.note import LOOKUPS as HANDOFF_LOOKUPS
from app.agents.handoff.note import handoff_inputs
from app.agents.intake.observe import observe_payload
from app.agents.intake.questions import LOOKUPS as QUESTION_LOOKUPS
from app.agents.intake.questions import question_payload
from app.agents.loop import MAX_AGENT_STEPS
from app.agents.resolvers import member_scope, resolver_payload
from app.agents.tools import episode_tools, lab_tools, member_tools
from app.deep.ledger import TurnLedger
from app.deep.roles import Phase, ReportBack, RoleView, ToolErrors, blocked
from app.deep.tools import (
    RESPOND_DECISIONS,
    Deps,
    context_tools,
    escalation_tool,
    intake_tools,
    question_tool,
    reply_tool,
    resolved_member,
    scoped_episode_tools,
    supervisor_tools,
)
from app.deep.views import supervisor_view
from app.graph.replies import reply_payload
from app.prompts.deep import FINISH_WITH_TOOL, RESPOND_FINISH, SUPERVISOR_PROMPT
from app.prompts.handoff import HANDOFF_PROMPT
from app.prompts.intake import INTAKE_AGENT_PROMPT, OBSERVE_PROMPT
from app.prompts.resolvers import EPISODE_AGENT_PROMPT, MEMBER_AGENT_PROMPT
from app.prompts.respond import RESPOND_PROMPT
from app.triage.policy import Decision

SUPERVISOR_CALLS = 12  # context, intake, decide, act, submit, with room to correct a tool error
CONTEXT_CALLS = 2 * MAX_AGENT_STEPS + 1  # member, then episode, then "recorded"
ROLE_CALLS = MAX_AGENT_STEPS + 1  # lookups, the answer, and one correction
RECORDED = Phase("done", "Your result is recorded. Reply with the single word: recorded.", None, frozenset())
FS_AND_PLANNING = frozenset({"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute",
                             "write_todos"})

# Q's agents use only Q's tools: no virtual filesystem, planning list or general-purpose subagent.
register_harness_profile("openai", HarnessProfile(
    excluded_tools=FS_AND_PLANNING, general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))


def build_turn_agent(model: BaseChatModel, ledger: TurnLedger, deps: Deps) -> CompiledStateGraph:
    state = ledger.state
    context, household = state["context"], state["household_id"]
    members = member_tools(context)
    lookups = {t.name: t for t in [*members, *episode_tools(context, deps.store, household)]}
    labs = lab_tools(context, household, deps.extensions.lab_results)
    question_lookups = [lookups[n] for n in sorted(QUESTION_LOOKUPS)] + labs
    handoff_lookups = [lookups[n] for n in sorted(HANDOFF_LOOKUPS)] + labs
    views = RoleViews(ledger, deps, members)

    def drafted() -> bool:
        return bool(ledger.get("draft_reply"))

    subagents = [
        _subagent("context", "Works out which family member and which of their illness episodes this message is "
                  "about, or what Q must ask to know.",
                  [*members, *scoped_episode_tools(ledger, deps.store), *context_tools(ledger)],
                  views.context, lambda: "episode_resolution" in ledger.state, CONTEXT_CALLS),
        _subagent("intake", "Reads the message against the episode's triage protocol, records the answers, and "
                  "reports the triage status with Q's policy suggestion.",
                  intake_tools(ledger, deps.protocols), views.intake, lambda: bool(ledger.get("observed")),
                  ROLE_CALLS),
        _subagent("questions", "Drafts the next intake question, after you decide to ask.",
                  [*question_lookups, question_tool(ledger)], lambda: views.questions(question_lookups), drafted,
                  ROLE_CALLS),
        _subagent("respond", "Drafts the reply for an answer, home monitoring or a clarifying question, after "
                  "you decide.", [reply_tool(ledger)], views.respond, drafted, ROLE_CALLS),
        _subagent("handoff", "Writes and saves the doctor's intake note, after you decide on a doctor.",
                  [*handoff_lookups, escalation_tool(ledger, deps)], lambda: views.handoff(handoff_lookups),
                  lambda: bool(ledger.get("handoff_path") or ledger.get("note_on_file")), ROLE_CALLS),
    ]
    supervisor = RoleView("supervisor", lambda: Phase(
        "supervisor", SUPERVISOR_PROMPT, supervisor_view(ledger.state),
        frozenset({"task", "set_decision", "submit_turn"}), parallel_tools=False), lambda: bool(ledger.get("reply")))
    return create_deep_agent(
        model=model, tools=supervisor_tools(ledger), system_prompt=SUPERVISOR_PROMPT, subagents=subagents,
        middleware=[supervisor, ToolErrors("supervisor"),
                    ModelCallLimitMiddleware(run_limit=SUPERVISOR_CALLS, exit_behavior="end")],
        name="q-supervisor",
    )


def _subagent(name: str, description: str, tools: list[BaseTool], view, done, calls: int) -> dict:
    return {"name": name, "description": description, "system_prompt": description, "tools": tools,
            "middleware": [RoleView(name, view, done), ToolErrors(name), ReportBack(name),
                           ModelCallLimitMiddleware(run_limit=calls, exit_behavior="end")]}


def _names(tools: list[BaseTool], *extra: str) -> frozenset[str]:
    return frozenset({*(t.name for t in tools), *extra})


class RoleViews:
    """Each subagent's view of the turn, read from the ledger on every model call."""

    def __init__(self, ledger: TurnLedger, deps: Deps, members: list[BaseTool]) -> None:
        self.ledger, self.deps, self.members = ledger, deps, members

    def context(self) -> Phase:
        state = self.ledger.state
        if "member_resolution" not in state:
            return Phase("context:member", MEMBER_AGENT_PROMPT + FINISH_WITH_TOOL.format(answer="MemberResolution"),
                         resolver_payload(state), _names(self.members, "MemberResolution"))
        if "episode_resolution" not in state:
            member = resolved_member(state)
            _, pending = member_scope(state["context"], member)
            return Phase("context:episode",
                         EPISODE_AGENT_PROMPT + FINISH_WITH_TOOL.format(answer="EpisodeResolution"),
                         resolver_payload(state, member=member, pending_episode_id=pending),
                         frozenset({"list_episodes", "episode_history", "past_session_summaries",
                                    "recent_messages", "EpisodeResolution"}))
        return RECORDED

    def intake(self) -> Phase:
        state = self.ledger.state
        if "episode_resolution" not in state:
            return blocked("intake", "the context subagent has not recorded the member and episode")
        if state.get("clarification"):
            return blocked("intake", "Q needs a clarification first, so there is nothing to observe yet")
        if state.get("observed"):
            return RECORDED
        return Phase("intake:observe", OBSERVE_PROMPT + FINISH_WITH_TOOL.format(answer="Observation"),
                     observe_payload(state, self.deps.store, self.deps.protocols),
                     frozenset({"get_protocol", "Observation"}))

    def questions(self, lookups: list[BaseTool]) -> Phase:
        state = self.ledger.state
        found = state.get("triage")
        if state.get("decision") != Decision.ASK or not (found and found.pending):
            return blocked("questions", "the decision is not ask, or the intake has no question left")
        return Phase("questions", INTAKE_AGENT_PROMPT + FINISH_WITH_TOOL.format(answer="IntakeQuestions"),
                     question_payload(state), _names(lookups, "IntakeQuestions"))

    def respond(self) -> Phase:
        state = self.ledger.state
        if state.get("decision") not in RESPOND_DECISIONS or "extraction" not in state:
            return blocked("respond", "set an answer, home_monitoring or clarification decision first")
        return Phase("respond", RESPOND_PROMPT + RESPOND_FINISH,
                     reply_payload(state, self.deps.extensions.knowledge), frozenset({"Reply"}))

    def handoff(self, lookups: list[BaseTool]) -> Phase:
        state = self.ledger.state
        if state.get("decision") != Decision.DOCTOR:
            return blocked("handoff", "set a doctor decision first")
        note = handoff_inputs(state, self.deps.store, self.deps.handoff_dir)
        return Phase("handoff", HANDOFF_PROMPT + FINISH_WITH_TOOL.format(answer="escalate_to_doctor"),
                     note.payload, _names(lookups, "escalate_to_doctor"))
