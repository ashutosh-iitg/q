"""What the supervisor sees each turn, and the reports subagents return to it.

Context across turns: no agent keeps a thread between turns. Every turn the supervisor starts fresh
from Postgres, so it sees the household, every tracked episode with its status, this conversation
tagged by member and episode, and summaries of earlier conversations, never a transcript replayed
from an earlier run. Subagents see their own role's payload (built by the graph's code), so what each
one knows is set by code, not by how the supervisor paraphrases the message.
"""

from datetime import UTC, datetime
from typing import Any

from app.agents.tools import to_json
from app.models.turn import TurnState
from app.triage.policy import Decision, doctor_urgency, rule


def supervisor_view(state: TurnState) -> str:
    context = state["context"]
    names = {m.id: m.name for m in context.members}
    writer = next((m for m in context.members if m.id == state.get("profile_id")), None)
    return to_json({  # stable first, this message and this run's progress last
        "household": [{"id": m.id, "name": m.name, "age_years": m.age_years, "conditions": m.conditions}
                      for m in context.members],
        "writer": writer and {"id": writer.id, "name": writer.name},
        "earlier_conversations": [{"summary": s.summary, "closed_at": s.closed_at} for s in context.prior_summaries],
        "open_episodes": [{"id": e.id, "member": names.get(e.member_id, e.member_id), "complaint": e.complaint,
                           "status": e.status, "questions_asked": e.questions_asked,
                           "check_back_after": e.check_back_after, "updated_at": e.updated_at}
                          for e in context.episodes],
        "recent_messages": [{"role": m.role, "content": m.content, "member": names.get(m.member_id),
                             "episode_id": m.episode_id, "decision": m.decision, "at": m.created_at}
                            for m in context.recent_messages],
        "screen": state["screen"].model_dump(),
        "text": state["text"], "now": datetime.now(UTC),
        "this_turn": progress(state),
    })


def progress(state: TurnState) -> dict[str, Any]:
    """What has been established so far in this run."""
    members = {m.id: m.name for m in state["context"].members}
    clarification = state.get("clarification")
    return {
        "member": state.get("member_id") and {"id": state["member_id"], "name": members.get(state["member_id"])},
        "clarification_needed": clarification and {"kind": clarification.kind,
                                                    "needed_info": clarification.needed_info},
        "episode": episode_summary(state),
        "triage": triage_summary(state),
        "decision": state.get("decision"), "urgency": state.get("urgency"),
        "draft_reply": state.get("draft_reply"), "doctor_note_saved": bool(state.get("handoff_path")),
    }


def episode_summary(state: TurnState) -> dict[str, Any] | None:
    selected = state.get("selected_episode")
    if not state.get("episode_id"):
        return selected and {"id": selected.id, "complaint": selected.complaint, "status": selected.status}
    return {"id": state["episode_id"], "new": selected is None, "complaint": state["extraction"].complaint,
            "status": selected.status if selected else "new",
            "questions_asked_before": selected.questions_asked if selected else 0}


def triage_summary(state: TurnState) -> dict[str, Any] | None:
    found = state.get("triage")
    if found is None:
        return None
    next_question = found.pending[0] if found.pending else None
    return {
        "protocol": found.protocol.title, "sufficient": found.sufficient, "disposition": found.disposition,
        "positive_findings": [q.finding for q in found.protocol.questions if found.states.get(q.id) is True],
        "ruled_out": [q.finding for q in found.protocol.questions if found.states.get(q.id) is False],
        "next_to_check": next_question and next_question.finding,
        "unanswered_last_time": state.get("unanswered_last_time", False),
    }


def policy_suggestion(state: TurnState) -> dict[str, Any]:
    """What Q's written policy decides for this case: guidance the supervisor weighs, not an order."""
    decision, reason = rule(state["extraction"], state["context"], state.get("clarification"), state.get("triage"))
    urgency = doctor_urgency(reason, state.get("triage")) if decision == Decision.DOCTOR else None
    return {"decision": decision, "reason": reason, "urgency": urgency}


def context_report(state: TurnState) -> str:
    clarification = state.get("clarification")
    return to_json({
        "member": progress(state)["member"],
        "episode": episode_summary(state),
        "clarification": clarification and {"kind": clarification.kind, "needed_info": clarification.needed_info,
                                             "how_to_ask": clarification.responder_instructions},
        "policy_suggestion": policy_suggestion(state) if clarification else None,
    })


def intake_report(state: TurnState) -> str:
    extraction = state["extraction"]
    return to_json({
        "intent": extraction.intent, "complaint": extraction.complaint, "resolved": extraction.resolved,
        "reported_values": extraction.slots.model_dump(exclude_none=True),
        "red_flags": [f.sign for f in extraction.red_flags],
        "doctor_flags": [f.flag for f in extraction.doctor_flags],
        "episode": episode_summary(state),
        "triage": triage_summary(state),
        "associated_symptoms": state.get("associated") or {},
        "policy_suggestion": policy_suggestion(state),
    })
