from uuid import uuid4

from app.policy import Decision
from app.store import EpisodeWrite, FactWrite, TurnWrite


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def make_turn(
    household_id, session_id, *, member_id=None, episode=None, facts=(), turn_id=None, pending=None, text=None
):
    return TurnWrite(
        household_id=household_id,
        session_id=session_id,
        turn_id=turn_id or new_id("t"),
        user_text=text or "he's been hot since last night",
        reply="How high is his temperature?",
        decision=Decision.ASK,
        member_id=member_id,
        episode=episode,
        facts=tuple(facts),
        pending_episode_id=pending,
    )


def fact(member_id, text, *, subject="fever", kind="observation", **fields):
    return FactWrite(member_id=member_id, episode_id=None, kind=kind, subject=subject, text=text, **fields)


def fever_episode(episode_id, member_id, slots, status="triaging"):
    return EpisodeWrite(
        id=episode_id,
        member_id=member_id,
        complaint="fever",
        status=status,
        slots=slots,
        questions_asked_delta=1,
        check_back_after=None,
    )


def post(client, household_id, session, text, **fields):
    return client.post(f"/households/{household_id}/sessions/{session}/messages", json={"text": text, **fields})
