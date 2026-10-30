"""Doctor-note faithfulness: keep only statements the user's own words support, and say where each was reported."""

import logging
import re
from datetime import UTC, datetime
from typing import Any

from app.agents.evidence import is_quoted
from app.agents.handoff.draft import IntakeNoteDraft
from app.triage.engine import ABSENT, Triage

logger = logging.getLogger(__name__)


WORD = re.compile(r"[a-z']+")


# Words of a reply that only answers the question it follows: "no", "not really", "nothing of that sort".
REPLY_ONLY_WORDS = frozenset(
    "no nope nah not none nothing never yes yeah yep yup sure ok okay fine of that this sort kind like really "
    "at all i i'm im it it's its is isn't was wasn't don't dont do did didn't think so there there's".split())


NO_NARRATIVE = "No verified narrative; see the findings and transcript."


def narrative(draft: IntakeNoteDraft, evidence: list[dict[str, Any]], *, reporter: str | None,
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
