"""The deep engine's tools, built per turn over its ledger.

Every role finishes with a tool named like the graph's answer schema (MemberResolution, Observation,
IntakeQuestions, ...). The tool runs the same code the graph runs after that agent (reconcile,
apply_observation, checked_reply, escalate_to_doctor), so validation is shared; a ValueError goes back
to the model to fix. Nothing is recorded until the check passes.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.agents.handoff.draft import IntakeNoteDraft
from app.agents.handoff.note import handoff_inputs
from app.agents.intake.observe import apply_observation, protocol_tools
from app.agents.intake.questions import IntakeQuestions, apply_questions
from app.agents.resolvers import member_scope
from app.agents.tools import episode_tools
from app.deep.ledger import TurnLedger
from app.deep.views import context_report, intake_report
from app.extensions import Extensions
from app.graph.nodes import reconcile
from app.graph.replies import finish_reply, fixed_reply
from app.memory.store import Store
from app.models.outputs import EpisodeResolution, MemberResolution, Observation
from app.models.records import Member
from app.triage.policy import DEFAULT_URGENCY, Decision, note_on_file
from app.triage.protocol import Protocol

RESPOND_DECISIONS = frozenset({Decision.ANSWER, Decision.HOME_MONITORING, Decision.ASK_MEMBER, Decision.ASK_EPISODE})
NEXT_STEP = {
    Decision.ASK: "call the questions subagent, then submit_turn without a reply",
    Decision.DOCTOR: "call the handoff subagent to save the doctor's note, then submit_turn with your reply",
    Decision.EMERGENCY: "submit_turn with your reply",
}
NO_EPISODE = EpisodeResolution(status="none", episode_id=None, reason="No resolved member", needed_info=None,
                               responder_instructions=None)


@dataclass(frozen=True)
class Deps:
    store: Store
    protocols: dict[str, Protocol]
    handoff_dir: Path
    extensions: Extensions = field(default_factory=Extensions)


class DecisionInput(BaseModel):
    decision: Decision
    reason: str = Field(description="A short reason code, e.g. protocol, red_flag, doctor_requested")
    urgency: Literal["doctor_now", "doctor_24h", "doctor_routine"] | None = Field(
        description="Doctor decisions only")


class SubmitInput(BaseModel):
    reply: str | None = Field(description="Only when you wrote the reply yourself; empty sends the draft")


def resolved_member(state: dict[str, Any]) -> Member | None:
    resolution = state.get("member_resolution")
    if resolution is None or resolution.status != "resolved":
        return None
    return next((m for m in state["context"].members if m.id == resolution.member_id), None)


def context_tools(ledger: TurnLedger) -> list[BaseTool]:
    @tool("MemberResolution", args_schema=MemberResolution)
    def record_member(**answer: Any) -> str:
        """Record which family member this message is about. Call it once, after looking members up."""
        resolution = MemberResolution.model_validate(answer)
        state = {**ledger.state, "member_resolution": resolution}
        if resolution.status == "resolved" and resolved_member(state) is None:
            raise ValueError(f"Unknown member_id {resolution.member_id!r}; use an ID from list_members")
        member = resolved_member(state)
        if member is None:  # nobody to find an episode for: settle the context now, as the graph does
            state = {**state, "episode_resolution": NO_EPISODE}
            ledger.apply({"member_resolution": resolution, "episode_resolution": NO_EPISODE, **reconcile(state)})
            return context_report(ledger.state)
        ledger.apply({"member_resolution": resolution})
        return f"Recorded: {member.name}. Now decide the episode."

    @tool("EpisodeResolution", args_schema=EpisodeResolution)
    def record_episode(**answer: Any) -> str:
        """Record whether this continues one of the member's episodes or starts a new one. Call it once."""
        resolution = EpisodeResolution.model_validate(answer)
        state = {**ledger.state, "episode_resolution": resolution}
        ledger.apply({"episode_resolution": resolution, **reconcile(state)})  # reconcile checks the IDs
        return context_report(ledger.state)

    return [record_member, record_episode]


def scoped_episode_tools(ledger: TurnLedger, store: Store) -> list[BaseTool]:
    """The episode lookups, limited to the member the context agent recorded."""
    def scoped() -> dict[str, BaseTool]:
        state = ledger.state
        member = resolved_member(state)
        if member is None:
            raise ValueError("Record the member with MemberResolution first")
        context, _ = member_scope(state["context"], member)
        return {t.name: t for t in episode_tools(context, store, state["household_id"])}

    @tool
    def list_episodes(member_id: str | None) -> str:
        """Open and recently resolved health episodes: id, member_id, complaint, status, slots and
        last update. Pass a member_id to filter, or null for every member."""
        return scoped()["list_episodes"].invoke({"member_id": member_id})

    @tool
    def episode_history(episode_id: str) -> str:
        """An episode's past conversation across sessions, oldest first."""
        return scoped()["episode_history"].invoke({"episode_id": episode_id})

    @tool
    def past_session_summaries() -> str:
        """Summaries of the household's recent earlier sessions."""
        return scoped()["past_session_summaries"].invoke({})

    return [list_episodes, episode_history, past_session_summaries]


def intake_tools(ledger: TurnLedger, protocols: dict[str, Protocol]) -> list[BaseTool]:
    @tool("Observation", args_schema=Observation)
    def record_observation(**answer: Any) -> str:
        """Record what the message says, against the episode's protocol. Call it once."""
        update = apply_observation(ledger.state, Observation.model_validate(answer), protocols)
        ledger.apply({**update, "observed": True})
        return intake_report(ledger.state)

    return [*protocol_tools(protocols), record_observation]


def question_tool(ledger: TurnLedger) -> BaseTool:
    @tool("IntakeQuestions", args_schema=IntakeQuestions)
    def record_question(**answer: Any) -> str:
        """Submit the next intake question. Call it once."""
        update = apply_questions(ledger.state, IntakeQuestions.model_validate(answer))
        ledger.apply({"findings": update["findings"], "draft_reply": update["reply"]})
        return json.dumps({"draft_reply": update["reply"]})

    return record_question


def reply_tool(ledger: TurnLedger) -> BaseTool:
    @tool("Reply")
    def record_reply(text: str) -> str:
        """Submit the reply to the user. Call it once."""
        reply = finish_reply(ledger.state, text)
        ledger.apply({"draft_reply": reply})
        return json.dumps({"draft_reply": reply})

    return record_reply


def escalation_tool(ledger: TurnLedger, deps: Deps) -> BaseTool:
    @tool("escalate_to_doctor", args_schema=IntakeNoteDraft)
    def escalate_to_doctor(**draft: Any) -> str:
        """Hand this case to a human doctor with your intake note. Call it exactly once, when the
        note is complete."""
        note = handoff_inputs(ledger.state, deps.store, deps.handoff_dir, deps.extensions.lab_results,
                              deps.extensions.deliver_handoff)
        result = note.escalate.invoke(draft)
        ledger.apply({"handoff_path": str(note.path)})
        return result

    return escalate_to_doctor


def supervisor_tools(ledger: TurnLedger) -> list[BaseTool]:
    @tool("set_decision", args_schema=DecisionInput)
    def set_decision(decision: Decision, reason: str, urgency: str | None = None) -> str:
        """Record what Q does this turn. Call it before respond, questions or handoff."""
        state, decision = ledger.state, Decision(decision)
        handed_over = (note_on_file(state["extraction"], state["context"])
                       if decision == Decision.DOCTOR and "extraction" in state else None)
        if handed_over:  # one note per hand-over: the reminder keeps the urgency the case went with
            urgency = handed_over.urgency or urgency or DEFAULT_URGENCY
        problem = decision_problem(state, decision, urgency)
        if problem:
            raise ValueError(problem)
        update = {"decision": decision, "decision_reason": reason, "draft_reply": None,
                  "urgency": urgency if decision == Decision.DOCTOR else None, "note_on_file": bool(handed_over)}
        if handed_over:
            ledger.apply({**update, "draft_reply": fixed_reply({**state, **update})["reply"]})
            return (f"Recorded {decision}. This case's doctor note is already saved, so Q doesn't hand it over "
                    "again: don't call handoff; submit_turn without a reply sends the reminder.")
        ledger.apply(update)
        step = NEXT_STEP.get(decision, "call the respond subagent, then submit_turn without a reply")
        return f"Recorded {decision}. Next: {step}."

    @tool("submit_turn", args_schema=SubmitInput)
    def submit_turn(reply: str | None = None) -> str:
        """Send Q's reply and end the turn. Call it exactly once."""
        state = ledger.state
        decision = state.get("decision")
        if decision is None:
            raise ValueError("Call set_decision first")
        if decision == Decision.DOCTOR and not (state.get("handoff_path") or state.get("note_on_file")):
            raise ValueError("Call the handoff subagent first: Q never says a case is handed over before "
                             "the doctor's note is saved")
        written = (reply or "").strip()
        text = (finish_reply(state, written) if decision == Decision.HOME_MONITORING else written) if written \
            else state.get("draft_reply")
        if not text:
            raise ValueError("There is no reply yet: call the subagent that drafts it, or write it yourself")
        ledger.apply({"reply": text})
        return "Sent."

    return [set_decision, submit_turn]


def decision_problem(state: dict[str, Any], decision: Decision, urgency: str | None) -> str | None:
    """What makes a decision impossible to carry out; it says nothing about whether it is wise."""
    found, extraction = state.get("triage"), state.get("extraction")
    intake_open = bool(state.get("episode_id") and found and found.pending and extraction
                       and extraction.intent in ("symptom_report", "update") and not extraction.resolved)
    next_finding = found.pending[0].finding if intake_open else None
    checks: list[tuple[bool, str]] = [
        (decision != Decision.EMERGENCY and "extraction" not in state,
         "Call the context subagent, then the intake subagent, before deciding"),
        (decision == Decision.DOCTOR and urgency is None, "A doctor decision needs an urgency"),
        (decision == Decision.ASK and not (found and found.pending),
         "There is no intake question left to ask for this episode"),
        (decision == Decision.HOME_MONITORING and found is None, "Home monitoring needs a triaged episode"),
        (decision == Decision.ANSWER and intake_open,
         f"The intake for this episode is still open (next to check: {next_finding}): decide ask to "
         "continue it, or doctor or emergency"),
    ]
    return next((message for failed, message in checks if failed), None)
