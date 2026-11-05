"""Eval-only entailment judge: does each kept note statement say no more than the user's quoted words?

The code check (app/agents/handoff/faithfulness.py) proves a quote is real and is the user's; it can't
tell whether the sentence goes beyond it ("lips are a normal colour" citing "breathing is fine"). This
judge reads each kept statement next to its quotes and the user messages they come from, and nothing Q
said, so a denial can't be borrowed from Q's question. It runs in evals only and never changes a note.
"""

import json
import logging
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from app.prompts.judge import ENTAILMENT_PROMPT

logger = logging.getLogger(__name__)


NOT_JUDGED = "not_judged"  # a claim the judge returned no verdict for


class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str


class EntailmentResult(BaseModel):
    verdicts: list[ClaimVerdict]


Judge = Callable[[list[dict[str, Any]]], EntailmentResult]


def entailment_judge(model: BaseChatModel) -> Judge:
    """One strict structured call per note, judging all its claims together."""
    structured = model.with_structured_output(EntailmentResult, method="json_schema", strict=True,
                                              prompt_cache_key="q:EntailmentJudge")
    return lambda claims: structured.invoke([("system", ENTAILMENT_PROMPT),
                                             ("human", json.dumps({"claims": claims}))])


def claims_from(note: dict[str, Any]) -> list[dict[str, Any]]:
    """Every statement the code check kept, with its quotes and the user message each comes from."""
    user_words = {m["message_id"]: m["content"] for m in note.get("transcript") or [] if m["role"] == "user"}

    def evidence(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"quote": s["quote"], "message": user_words.get(s["message_id"])} for s in sources]

    history = [{"claim_id": f"h{i}", "kind": "history", "text": s["text"], "evidence": evidence(s["sources"])}
               for i, s in enumerate(note.get("history_sources") or [], 1)]
    timeline = [{"claim_id": f"t{i}", "kind": "timeline", "text": t["finding"], "evidence": evidence([t])}
                for i, t in enumerate(note.get("symptom_timeline") or [], 1)]
    return [*history, *timeline]


def judge_note(judge: Judge, note: dict[str, Any]) -> dict[str, Any]:
    """How many claims the judge found supported, and each one it didn't, with its reason."""
    claims = claims_from(note)
    if not claims:
        return {"claims": 0, "supported": 0, "flagged": []}
    try:
        verdicts = {v.claim_id: v for v in judge(claims).verdicts}
    except Exception as exc:  # an eval must report a failed judge, not abort the run
        logger.exception("Entailment judge failed on %d claims", len(claims))
        return {"claims": len(claims), "supported": 0, "flagged": [], "error": f"{type(exc).__name__}: {exc}"}
    judged = [{**c, "verdict": v.verdict if (v := verdicts.get(c["claim_id"])) else NOT_JUDGED,
               "reason": v.reason if v else "no verdict returned"} for c in claims]
    flagged = [c for c in judged if c["verdict"] != "supported"]
    return {"claims": len(claims), "supported": len(claims) - len(flagged), "flagged": flagged}
