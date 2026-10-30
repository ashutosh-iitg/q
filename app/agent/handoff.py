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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, ValidationError

from app.agent.evidence import is_quoted
from app.agent.loop import ModelOutputError, run_agent
from app.agent.prompts import HANDOFF_PROMPT
from app.agent.state import TurnState
from app.agent.tools import episode_tools, lab_tools, member_tools, to_json
from app.extensions import DeliverHandoff, LabResults
from app.store import Store
from app.triage.engine import ABSENT, Triage

logger = logging.getLogger(__name__)

SAFE_PART = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
WORD = re.compile(r"[a-z']+")
# Words of a reply that only answers the question it follows: "no", "not really", "nothing of that sort".
REPLY_ONLY_WORDS = frozenset(
    "no nope nah not none nothing never yes yeah yep yup sure ok okay fine of that this sort kind like really "
    "at all i i'm im it it's its is isn't was wasn't don't dont do did didn't think so there there's".split())
NO_NARRATIVE = "No verified narrative; see the findings and transcript."
CURRENT = "current"  # evidence ID of the message being handled, which isn't saved yet
LOOKUPS = {"member_facts", "list_episodes", "past_session_summaries"}


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str  # from evidence; "current" for this message
    onset: str | None  # when it started or was noticed, in the user's terms, e.g. "yesterday evening"
    finding: str
    quote: str  # exact words from that message


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str  # a user message from evidence; "current" for this message
    quote: str  # the user's exact words in that message that say this


class Statement(BaseModel):
    """One sentence of the history, with the user's words that support it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    sources: list[Source]


class IntakeNoteDraft(BaseModel):
    """The parts of the intake note the model writes."""

    model_config = ConfigDict(extra="forbid")

    chief_complaint: str
    history: list[Statement]  # the history of present illness, one supported sentence at a time
    symptom_timeline: list[TimelineEntry]  # in the order the findings were reported
    relevant_history: str | None


class HandoffError(RuntimeError):
    """The note could not be saved, so the case was not handed over."""


def handoff(state: TurnState, *, model: BaseChatModel, store: Store, directory: Path,
            lab_results: LabResults | None = None, deliver: DeliverHandoff | None = None) -> TurnState:
    evidence = _evidence(state, store)
    record = _record(state, evidence)
    path = _note_path(directory, state)
    escalate = escalation_tool(record, evidence, path, deliver)
    context = state["context"]
    lookups = [t for t in [*member_tools(context), *episode_tools(context, store, state["household_id"])]
               if t.name in LOOKUPS] + lab_tools(context, state["household_id"], lab_results)
    payload = to_json({key: record[key] for key in ("patient", "reported_by", "escalation", "triage",
                                                    "pertinent_negatives", "associated_symptoms", "reported_values")}
                      | {"evidence": evidence})
    try:
        run_agent(model, system=HANDOFF_PROMPT, payload=payload, tools=lookups, answer=escalate)
    except (ModelOutputError, ValidationError) as exc:
        logger.warning("Intake-note agent did not finish (%s); saving a minimal note", exc)
        escalate.invoke(_minimal_draft(state).model_dump())
    return {"handoff_path": str(path)}


def escalation_tool(record: dict[str, Any], evidence: list[dict[str, Any]], path: Path,
                    deliver: DeliverHandoff | None = None) -> BaseTool:
    @tool("escalate_to_doctor", args_schema=IntakeNoteDraft)
    def escalate_to_doctor(**draft: Any) -> str:
        """Hand this case to a human doctor with your intake note. Call it exactly once, when the
        note is complete."""
        reporter, patient = record["reported_by"], record["patient"]
        named = reporter["name"] if reporter and patient and reporter["member_id"] != patient["member_id"] else None
        note = {**record, **_narrative(IntakeNoteDraft.model_validate(draft), evidence, reporter=named,
                                       session_id=record["session_id"])}
        _write(path, note)
        if deliver:  # a telemedicine integration, plugged in through app/extensions.py
            try:
                deliver(json.loads(path.read_text()))
            except Exception as exc:  # any delivery failure means the case was not handed over
                raise HandoffError(f"Could not deliver handoff {record['handoff_id']}: {exc}") from exc
        return f"Case handed to a doctor as {record['handoff_id']}"

    return escalate_to_doctor


def _narrative(draft: IntakeNoteDraft, evidence: list[dict[str, Any]], *, reporter: str | None,
               session_id: str) -> dict[str, Any]:
    """The model's part, keeping only statements and timeline entries the user's own words support."""
    order = {e["message_id"]: i for i, e in enumerate(evidence)}
    by_id = {e["message_id"]: e for e in evidence}

    def where(message_id: str) -> dict[str, Any]:
        message = by_id[message_id]
        return reported_in(message.get("session_id"), message["at"], session_id)

    kept, dropped = [], []
    for statement in draft.history:
        problems = [p for s in statement.sources if (p := _source_problem(s.message_id, s.quote, by_id))]
        if problems or not statement.sources:
            dropped.append({"kind": "history", "text": statement.text, "why": problems[0] if problems else "no source"})
            continue
        kept.append({"text": statement.text,
                     "sources": [{**s.model_dump(), "reported_in": where(s.message_id)} for s in statement.sources]})
    timeline = []
    for entry in sorted(draft.symptom_timeline, key=lambda t: order.get(t.message_id, len(order))):
        if problem := _source_problem(entry.message_id, entry.quote, by_id):
            dropped.append({"kind": "timeline", "text": entry.finding, "why": problem})
            continue
        timeline.append({**entry.model_dump(), "reported_at": by_id[entry.message_id]["at"],
                         "reported_in": where(entry.message_id)})
    if dropped:
        logger.warning("Dropped %d unsupported note statements", len(dropped))
    lead = [f"Reported by {reporter}."] if reporter else []
    return {"chief_complaint": draft.chief_complaint,
            "history_of_present_illness": " ".join(lead + ([s["text"] for s in kept] or [NO_NARRATIVE])),
            "history_sources": kept, "symptom_timeline": timeline, "relevant_history": draft.relevant_history,
            "faithfulness": {"rule": "Each statement quotes the user's own words; bare replies and Q's words are "
                                     "not evidence.", "dropped": dropped}}


def _source_problem(message_id: str, quote: str, by_id: dict[str, dict[str, Any]]) -> str | None:
    """Why a quote can't support a statement, or None when it can."""
    message = by_id.get(message_id)
    if message is None:
        return "cites a message that is not in the conversation"
    if message["role"] != "user":
        return "cites Q's words, not the user's"
    if not is_quoted(quote, message["content"]):
        return "the quote is not in the message it cites"
    if refers_back(quote):
        return "the quote only answers Q's question, so it cannot show what was reported"
    return None


def refers_back(quote: str) -> bool:
    """True for a reply that only means something next to the question it answers."""
    return all(word in REPLY_ONLY_WORDS for word in WORD.findall(quote.lower().replace("\u2019", "'")))


def reported_in(session_id: str | None, at: Any, current_session: str) -> dict[str, Any]:
    """Which conversation reported something, and when, for the doctor's chronology."""
    conversation = "this conversation" if session_id == current_session else "an earlier conversation"
    moment = datetime.fromisoformat(at) if isinstance(at, str) else at
    if not isinstance(moment, datetime):
        return {"conversation": conversation, "session_id": session_id, "at": None, "label": conversation}
    moment = moment.astimezone(UTC)
    return {"conversation": conversation, "session_id": session_id, "at": moment.isoformat(),
            "label": f"{conversation}, {moment:%H:%M} UTC on {moment.day} {moment:%b %Y}"}


def pertinent_negatives(found: Triage, findings: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    """The warning signs the patient denied when asked, with Q's question and where it was answered."""
    return [{"finding": q.finding, "asked": f.get("asked"), "answer": f.get("quote"),
             "reported_in": reported_in(f.get("session_id"), f.get("at"), session_id)}
            for q in found.protocol.questions
            if (f := findings.get(q.id, {})).get("value") in ABSENT and found.states[q.id] is False]


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
