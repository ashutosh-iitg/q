import json

from ui.transcript import transcript_json, transcript_markdown

MESSAGES = [
    {"role": "user", "content": "Maya has a fever"},
    {"role": "assistant", "content": "How high is it?", "decision": "ask", "turn_id": "t1", "note": None},
    {"role": "assistant", "content": "Please see a doctor.", "decision": "doctor", "turn_id": "t2",
     "note": {"chief_complaint": "fever", "escalation": {"urgency": "doctor_24h"}}},
]


def test_markdown_lists_each_message_with_speaker_and_decision():
    text = transcript_markdown(MESSAGES, account="Sharma family", profile="Ravi", session_id="s1")

    assert text.startswith("# Q conversation · Sharma family")
    assert "**Ravi:** Maya has a fever" in text
    assert "**Q** _(ask)_: How high is it?" in text
    assert "_Doctor intake note saved: fever_" in text


def test_json_keeps_messages_and_session_details():
    data = json.loads(transcript_json(MESSAGES, account="Sharma family", profile="Ravi", session_id="s1"))

    assert data["session_id"] == "s1"
    assert data["profile"] == "Ravi"
    assert data["messages"][2]["note"]["chief_complaint"] == "fever"
    assert data["messages"][0] == {"role": "user", "content": "Maya has a fever"}
