"""The doctor note states only what was discussed, in the user's words, and says where each item was reported."""

import pytest

from app.agents.handoff.draft import IntakeNoteDraft
from app.agents.handoff.faithfulness import NO_NARRATIVE, narrative, pertinent_negatives, refers_back, reported_in
from app.triage.engine import profile, triage
from app.triage.protocol import load_protocols

EARLIER, NOW = "2026-10-28T17:38:00+00:00", "2026-10-29T18:07:00+00:00"
EVIDENCE = [
    {"message_id": "1", "role": "user", "content": "I have a dry cough for two weeks", "at": EARLIER, "session_id": "s1"},
    {"message_id": "2", "role": "assistant", "content": "Are the lips or face turning blue or grey?", "at": EARLIER,
     "session_id": "s1"},
    {"message_id": "3", "role": "user", "content": "No nothing of that sort", "at": EARLIER, "session_id": "s1"},
    {"message_id": "4", "role": "assistant", "content": "Any trouble breathing?", "at": NOW, "session_id": "s2"},
    {"message_id": "current", "role": "user", "content": "No, breathing is fine", "at": NOW, "session_id": "s2"},
]


def statement(text, *sources):
    return {"text": text, "sources": [{"message_id": m, "quote": q} for m, q in sources]}


def note(*history, timeline=(), reporter=None):
    draft = IntakeNoteDraft.model_validate({"chief_complaint": "Dry cough", "history": list(history),
                                            "symptom_timeline": list(timeline), "relevant_history": None})
    return narrative(draft, EVIDENCE, reporter=reporter, session_id="s2")


@pytest.mark.parametrize("quote, expected", [
    ("No", True), ("No nothing of that sort", True), ("not really", True), ("yes", True),
    ("breathing is fine", False), ("dry cough", False), ("No. It's completely dry", False),
])
def test_a_reply_that_only_refers_back_to_the_question_is_not_evidence(quote, expected):
    assert refers_back(quote) is expected


def test_a_denial_inferred_from_a_bare_reply_is_dropped_from_the_history():
    result = note(statement("Dry cough for two weeks.", ("1", "dry cough for two weeks")),
                  statement("Denies blue or grey lips.", ("3", "No nothing of that sort")),
                  statement("Denies breathing difficulty.", ("current", "breathing is fine")))
    assert result["history_of_present_illness"] == "Dry cough for two weeks. Denies breathing difficulty."
    assert [d["text"] for d in result["faithfulness"]["dropped"]] == ["Denies blue or grey lips."]


@pytest.mark.parametrize("sources", [
    [("2", "blue or grey")],  # Q's own words are not the user's report
    [("9", "dry cough")],  # no such message
    [("1", "wet cough")],  # not in the message it cites
    [],  # no source at all
])
def test_unsupported_history_statements_are_dropped(sources):
    result = note(statement("Something about the cough.", *sources))
    assert result["history_of_present_illness"] == NO_NARRATIVE
    assert len(result["faithfulness"]["dropped"]) == 1


def test_each_kept_statement_says_which_conversation_reported_it():
    result = note(statement("Dry cough for two weeks.", ("1", "dry cough for two weeks")))
    source, = result["history_sources"][0]["sources"]
    assert source["reported_in"]["conversation"] == "an earlier conversation"
    assert source["reported_in"]["label"] == "an earlier conversation, 17:38 UTC on 28 Oct 2026"


def test_the_timeline_follows_the_same_rules_in_the_order_things_were_reported():
    result = note(timeline=[
        {"message_id": "current", "onset": None, "finding": "Breathing fine", "quote": "breathing is fine"},
        {"message_id": "3", "onset": None, "finding": "No blue lips", "quote": "No nothing of that sort"},
        {"message_id": "1", "onset": "two weeks ago", "finding": "Dry cough", "quote": "dry cough"}])
    assert [(t["finding"], t["reported_in"]["conversation"]) for t in result["symptom_timeline"]] == [
        ("Dry cough", "an earlier conversation"), ("Breathing fine", "this conversation")]


def test_a_reporter_who_is_not_the_patient_is_named_by_code():
    result = note(statement("Dry cough.", ("1", "dry cough")), reporter="Ravi")
    assert result["history_of_present_illness"] == "Reported by Ravi. Dry cough."


def test_pertinent_negatives_list_only_what_was_asked_with_the_question_and_where():
    cough = load_protocols()["cough"]
    findings = {
        "blue_lips": {"value": "absent", "quote": "No nothing of that sort", "at": EARLIER, "session_id": "s1",
                      "asked": "Are the lips or face turning blue or grey?"},
        "struggling_to_breathe": {"value": "absent", "quote": "breathing is fine", "at": NOW, "session_id": "s2",
                                  "asked": None},
        "fever_with_cough": {"value": "resolved", "quote": "fever subsided", "at": NOW, "session_id": "s2"},
    }
    found = triage(cough, findings, {"duration_hours": 336}, profile(28, []))
    negatives = pertinent_negatives(found, findings, session_id="s2")
    assert negatives == [
        {"finding": "Lips or face turning blue or grey", "asked": "Are the lips or face turning blue or grey?",
         "answer": "No nothing of that sort", "reported_in": reported_in("s1", EARLIER, "s2")},
        {"finding": "Struggling to breathe", "asked": None, "answer": "breathing is fine",
         "reported_in": reported_in("s2", NOW, "s2")},
    ]  # unasked questions are left out, and a resolved fever is not a denial
