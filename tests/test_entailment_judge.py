"""The eval-only entailment judge reads each kept note statement against the user's own words."""

import json

from langchain_core.runnables import RunnableLambda

from app.evals.entailment import (
    NOT_JUDGED,
    ClaimVerdict,
    EntailmentResult,
    claims_from,
    entailment_judge,
    judge_note,
)

NOTE = {
    "history_sources": [
        {"text": "Dry cough for two weeks.", "sources": [{"message_id": "1", "quote": "dry cough for two weeks"}]},
        {"text": "Lips are a normal colour.", "sources": [{"message_id": "current", "quote": "breathing is fine"}]},
    ],
    "symptom_timeline": [{"message_id": "1", "onset": "two weeks ago", "finding": "dry cough",
                          "quote": "dry cough for two weeks"}],
    "transcript": [
        {"message_id": "1", "role": "user", "content": "I have a dry cough for two weeks"},
        {"message_id": "2", "role": "assistant", "content": "Are the lips or face turning blue or grey?"},
        {"message_id": "current", "role": "user", "content": "No, breathing is fine"},
    ],
}


class FakeJudgeModel:
    def __init__(self, result):
        self.result, self.payloads = result, []

    def with_structured_output(self, schema, **kwargs):
        assert schema is EntailmentResult and kwargs["strict"] is True
        return RunnableLambda(self._answer)

    def _answer(self, messages):
        self.payloads.append(json.loads(messages[-1][1]))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def verdicts(*items):
    return EntailmentResult(verdicts=[ClaimVerdict(claim_id=c, verdict=v, reason=r) for c, v, r in items])


def test_each_kept_statement_and_timeline_entry_becomes_a_claim_with_its_users_message():
    claims = claims_from(NOTE)
    assert [(c["claim_id"], c["kind"], c["text"]) for c in claims] == [
        ("h1", "history", "Dry cough for two weeks."), ("h2", "history", "Lips are a normal colour."),
        ("t1", "timeline", "dry cough")]
    assert claims[1]["evidence"] == [{"quote": "breathing is fine", "message": "No, breathing is fine"}]


def test_the_judge_never_sees_qs_words():
    model = FakeJudgeModel(verdicts(("h1", "supported", ""), ("h2", "supported", ""), ("t1", "supported", "")))
    judge_note(entailment_judge(model), NOTE)
    assert "blue or grey" not in json.dumps(model.payloads)


def test_statements_that_go_beyond_their_quotes_are_flagged_with_the_reason():
    model = FakeJudgeModel(verdicts(("h1", "supported", ""), ("t1", "supported", ""),
                                    ("h2", "unsupported", "the quote says nothing about the lips")))
    result = judge_note(entailment_judge(model), NOTE)
    assert (result["claims"], result["supported"]) == (3, 2)
    assert [(f["claim_id"], f["verdict"], f["reason"]) for f in result["flagged"]] == [
        ("h2", "unsupported", "the quote says nothing about the lips")]


def test_a_claim_the_judge_skipped_is_flagged_as_not_judged_and_unknown_ids_are_ignored():
    model = FakeJudgeModel(verdicts(("h1", "supported", ""), ("t1", "supported", ""), ("x9", "unsupported", "")))
    result = judge_note(entailment_judge(model), NOTE)
    assert [(f["claim_id"], f["verdict"]) for f in result["flagged"]] == [("h2", NOT_JUDGED)]


def test_a_note_with_nothing_to_judge_makes_no_call():
    model = FakeJudgeModel(verdicts())
    result = judge_note(entailment_judge(model), {**NOTE, "history_sources": [], "symptom_timeline": []})
    assert result == {"claims": 0, "supported": 0, "flagged": []} and model.payloads == []


def test_a_failed_judge_call_is_reported_not_raised():
    result = judge_note(entailment_judge(FakeJudgeModel(TimeoutError("model timed out"))), NOTE)
    assert result["error"] == "TimeoutError: model timed out" and result["claims"] == 3
