"""Escalation to a doctor: the intake-note agent and its escalate_to_doctor tool.

Every doctor decision runs this agent. It looks up what it needs, writes the clinical narrative,
and calls escalate_to_doctor, which saves the note as JSON. Code owns everything that must be
exact (identity, triage, findings, transcript) and adds it to the note; the model writes only
the narrative and the symptom timeline. If the agent can't finish, code calls the same tool with a
minimal draft, so a doctor decision always produces a note, and Q never claims a handoff it didn't save.

Faithfulness: every history statement and timeline entry must quote the user's own words from a user
message. Q's words, and bare replies ("no", "nothing of that sort") that only make sense next to the
question, are not evidence; such claims are dropped and listed in the note. Denials come from code
(pertinent_negatives), each with the question Q actually asked. The note covers only what was
discussed, and each item says which conversation reported it, and when.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from app.agents.handoff.draft import CURRENT, IntakeNoteDraft
from app.agents.handoff.faithfulness import narrative, pertinent_negatives, reported_in
from app.agents.loop import ModelOutputError, run_agent
from app.agents.tools import clinical_tools, to_json
from app.extensions import DeliverHandoff, LabResults
from app.memory.store import Store
from app.models.turn import TurnState
from app.prompts.handoff import HANDOFF_PROMPT
from app.triage.engine import Triage

logger = logging.getLogger(__name__)


SAFE_PART = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


LOOKUPS = {"member_facts", "list_episodes", "past_session_summaries"}


class HandoffError(RuntimeError):
    """The note could not be saved, so the case was not handed over."""


def handoff(state: TurnState, *, model: BaseChatModel, store: Store, directory: Path,
            lab_results: LabResults | None = None, deliver: DeliverHandoff | None = None) -> TurnState:
    note = handoff_inputs(state, store, directory, lab_results, deliver)
    try:
        run_agent(model, system=HANDOFF_PROMPT, payload=note.payload, tools=note.lookups, answer=note.escalate)
    except (ModelOutputError, ValidationError) as exc:
        logger.warning("Intake-note agent did not finish (%s); saving a minimal note", exc)
        save_minimal_note(state, note)
    return {"handoff_path": str(note.path)}


@dataclass(frozen=True)
class HandoffInputs:
    """What the intake-note agent works with: its payload, lookups, and the tool that saves the note."""

    payload: str
    lookups: list[BaseTool]
    escalate: BaseTool
    path: Path


def handoff_inputs(state: TurnState, store: Store, directory: Path, lab_results: LabResults | None = None,
                   deliver: DeliverHandoff | None = None) -> HandoffInputs:
    evidence = _evidence(state, store)
    record = _record(state, evidence)
    path = _note_path(directory, state)
    context = state["context"]
    lookups = clinical_tools(context, store, state["household_id"], names=LOOKUPS, lab_results=lab_results)
    payload = to_json({key: record[key] for key in ("patient", "reported_by", "escalation", "triage",
                                                    "pertinent_negatives", "associated_symptoms", "reported_values")}
                      | {"evidence": evidence})
    return HandoffInputs(payload, lookups, escalation_tool(record, evidence, path, deliver), path)


def save_minimal_note(state: TurnState, note: HandoffInputs) -> None:
    """A doctor decision always produces a note, even when the agent didn't write one."""
    note.escalate.invoke(_minimal_draft(state).model_dump())


def escalation_tool(record: dict[str, Any], evidence: list[dict[str, Any]], path: Path,
                    deliver: DeliverHandoff | None = None) -> BaseTool:
    @tool("escalate_to_doctor", args_schema=IntakeNoteDraft)
    def escalate_to_doctor(**draft: Any) -> str:
        """Hand this case to a human doctor with your intake note. Call it exactly once, when the
        note is complete."""
        reporter, patient = record["reported_by"], record["patient"]
        named = reporter["name"] if reporter and patient and reporter["member_id"] != patient["member_id"] else None
        note = {**record, **narrative(IntakeNoteDraft.model_validate(draft), evidence, reporter=named,
                                       session_id=record["session_id"])}
        _write(path, note)
        if deliver:  # a telemedicine integration, plugged in through app/extensions.py
            try:
                deliver(json.loads(path.read_text()))
            except Exception as exc:  # any delivery failure means the case was not handed over
                raise HandoffError(f"Could not deliver handoff {record['handoff_id']}: {exc}") from exc
        return f"Case handed to a doctor as {record['handoff_id']}"

    return escalate_to_doctor


def _evidence(state: TurnState, store: Store) -> list[dict[str, Any]]:
    """What was said, oldest first: the episode across sessions, else this session, then this message."""
    episode = state.get("selected_episode")
    earlier = (store.episode_history(state["household_id"], episode.id) if episode
               else state["context"].recent_messages)
    current = {"message_id": CURRENT, "role": "user", "content": state["text"], "at": datetime.now(UTC),
               "session_id": state["session_id"]}
    return [*({"message_id": str(m.id), "role": m.role, "content": m.content, "at": m.created_at,
               "session_id": m.session_id or state["session_id"]} for m in earlier), current]


def _record(state: TurnState, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything in the note that code knows exactly."""
    context, extraction, found = state["context"], state["extraction"], state.get("triage")
    members = {m.id: m for m in context.members}
    patient = members.get(extraction.member_id)
    reporter = members.get(state.get("profile_id"))
    return {
        "handoff_id": state["turn_id"], "created_at": datetime.now(UTC).isoformat(),
        "household_id": state["household_id"], "session_id": state["session_id"],
        "episode_id": state.get("episode_id"),
        "patient": patient and {
            "member_id": patient.id, "name": patient.name, "age_years": patient.age_years,
            "known_conditions": list(patient.conditions),
            "remembered_facts": [{"kind": f.kind, "subject": f.subject, "text": f.text, "status": f.status,
                                  "source": f.source} for f in context.facts if f.member_id == patient.id],
        },
        "reported_by": reporter and {"member_id": reporter.id, "name": reporter.name},
        "escalation": {
            "urgency": state["urgency"], "reason": state["decision_reason"],
            "doctor_requested": extraction.intent == "doctor_request",
            "red_flags": [f.model_dump() for f in extraction.red_flags],
            "doctor_flags": [f.model_dump() for f in extraction.doctor_flags],
        },
        "triage": found and _triage_record(found, state.get("findings") or {}, state["session_id"]),
        "pertinent_negatives": found and pertinent_negatives(found, state.get("findings") or {}, state["session_id"]),
        # Other symptoms of this illness, each marked active or resolved, so a symptom that has gone
        # (a fever that settled while the cough goes on) never reads as current.
        "associated_symptoms": [{"symptom": symptom, **details}
                                for symptom, details in (state.get("associated") or {}).items()],
        "reported_values": {k: v for k, v in extraction.slots.model_dump().items() if v is not None},
        "transcript": evidence,
    }


def _triage_record(found: Triage, findings: dict[str, Any], session_id: str) -> dict[str, Any]:
    """The protocol's outcome, and only the findings that were actually discussed."""
    protocol = found.protocol
    label = {True: "positive", False: "ruled_out", None: "unanswered"}
    return {
        "protocol": {"id": protocol.id, "title": protocol.title, "version": protocol.version,
                     "source": protocol.source},
        "sufficient": found.sufficient, "disposition": found.disposition,
        "deciding": [{"question_id": q.id, "finding": q.finding} for q in protocol.questions if q.id in found.deciding],
        "findings": [{"question_id": q.id, "level": q.level, "finding": q.finding, "chosen_by_agent": q.authored,
                      "state": label[found.states[q.id]], **(f := dict(findings[q.id])),
                      "reported_in": reported_in(f.get("session_id"), f.get("at"), session_id)}
                     for q in protocol.questions if "value" in findings.get(q.id, {})],
    }


def _minimal_draft(state: TurnState) -> IntakeNoteDraft:
    return IntakeNoteDraft(
        chief_complaint=state["extraction"].complaint or "Not recorded",
        history=[], symptom_timeline=[], relevant_history=None,
    )


def read_note(directory: Path, household_id: str, turn_id: str) -> dict[str, Any] | None:
    """The intake note saved for this turn, or None; the episode folder isn't needed to find it."""
    if not (SAFE_PART.match(household_id) and SAFE_PART.match(turn_id)):
        return None
    found = next(directory.joinpath(household_id).glob(f"*/{turn_id}.json"), None)
    return json.loads(found.read_text()) if found else None


def _note_path(directory: Path, state: TurnState) -> Path:
    parts = (state["household_id"], state.get("episode_id") or "no-episode", state["turn_id"])
    if not all(SAFE_PART.match(part) for part in parts):
        raise HandoffError(f"Unsafe handoff path parts: {parts}")
    return directory.joinpath(parts[0], parts[1], f"{parts[2]}.json")


def _write(path: Path, note: dict[str, Any]) -> None:
    """Atomic, so a doctor never reads half a note; a retried turn overwrites its own note."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(note, indent=2,
                                      default=lambda v: v.isoformat() if isinstance(v, datetime) else str(v)))
        os.replace(partial, path)
    except OSError as exc:
        raise HandoffError(f"Could not save handoff {path}: {exc}") from exc
