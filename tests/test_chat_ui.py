from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app.demo import demo_accounts
from ui.client import ChatAPIError

APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"


def test_demo_batch_has_varied_unique_profiles():
    accounts = demo_accounts()
    assert len(accounts) == len({a["household_id"] for a in accounts}) == 10
    assert all(2 <= len(a["members"]) <= 5 for a in accounts)
    members = [m for a in accounts for m in a["members"]]
    assert len(members) == len({m["id"] for m in members}) == 35
    assert any(m["age_years"] < 1 for m in members)
    assert any(12 <= m["age_years"] <= 18 for m in members)
    assert any(m["age_years"] >= 65 for m in members)
    assert any(m["conditions"] for m in members)
    assert any(all(not m["conditions"] for m in a["members"]) for a in accounts)


def test_chat_switching_retry_and_new_session():
    calls = []
    fail_next = True

    def request(url, method, path, payload=None):
        nonlocal fail_next
        if path == "/demo/accounts":
            return demo_accounts()
        calls.append((path, payload))
        if path.endswith("/close"):
            return {"status": "consolidating"}
        if fail_next:
            fail_next = False
            raise ChatAPIError("Please retry.")
        return {"reply": "How long have you felt this way?", "decision": "ask"}

    with patch("ui.client.request", side_effect=request):
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception
        assert len(at.selectbox[0].options) == 10
        at.chat_input[0].set_value("I feel tired").run()
        assert at.error[0].value == "Please retry."
        failed_path, failed_payload = calls[-1]
        at.button(key="retry_message").click().run()
        assert not at.exception
        assert calls[-1] == (failed_path, failed_payload)
        assert len(at.chat_message) == 2
        assert failed_payload["profile_id"] == "demo-ravi"
        at.selectbox[1].select("demo-asha").run()
        assert len(at.chat_message) == 0
        at.chat_input[0].set_value("I feel dizzy").run()
        assert calls[-1][0] != failed_path
        assert calls[-1][1]["profile_id"] == "demo-asha"
        at.selectbox[0].select("demo-kapoor").run()
        assert len(at.chat_message) == 0
        assert len(at.selectbox[1].options) == 2
        at.selectbox[0].select("demo").run()
        at.selectbox[1].select("demo-ravi").run()
        assert len(at.chat_message) == 2
        at.button(key="new_chat").click().run()
        assert calls[-1][0] == failed_path.removesuffix("/messages") + "/close"
        assert len(at.chat_message) == 0
        at.chat_input[0].set_value("A new question").run()
        assert calls[-1][0] != failed_path
        assert not at.exception


def test_backend_unavailable_shows_reconnect(monkeypatch):
    monkeypatch.setenv("Q_API_URL", "http://unavailable.test")
    with patch("ui.client.request", side_effect=ChatAPIError("Cannot reach Q.")):
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception
        assert at.error[0].value == "Cannot reach Q."
        assert at.button[0].label == "Reconnect"


def test_http_timeout_becomes_retryable_error():
    from ui.client import request
    with patch("ui.client.httpx.request", side_effect=httpx.ReadTimeout("timed out")):
        with pytest.raises(ChatAPIError, match="Cannot reach Q"):
            request("http://localhost:8000", "POST", "/messages", {"text": "hello"})


NOTE = {"chief_complaint": "Intermittent chest pain", "history_of_present_illness": "Reported by her son.",
        "symptom_timeline": [{"finding": "Chest pain", "onset": "yesterday", "quote": "chest pain on and off"}],
        "relevant_history": "Hypertension", "escalation": {"urgency": "doctor_now", "reason": "protocol"},
        "triage": {"protocol": {"title": "Chest pain", "source": "synthetic-placeholder"}}}


def test_a_doctor_reply_shows_its_saved_intake_note():
    calls = []

    def request(url, method, path, payload=None):
        calls.append((method, path))
        if path == "/demo/accounts":
            return demo_accounts()
        if "/handoffs/" in path:
            return NOTE
        return {"reply": "This needs a doctor's review.", "decision": "doctor"}

    with patch("ui.client.request", side_effect=request):
        at = AppTest.from_file(APP_PATH).run()
        at.chat_input[0].set_value("Mum has chest pain").run()
        assert not at.exception
        turn_id = at.session_state["conversations"][("demo", "demo-ravi")]["messages"][-1]["turn_id"]
        assert ("GET", f"/households/demo/handoffs/{turn_id}") in calls
        assert at.expander[0].label == "Doctor intake note"
        assert any("Doctor review · within a few hours" in m.value for m in at.markdown)
        assert any("No doctor or clinic is contacted" in c.value for c in at.caption)


def test_an_example_button_sends_its_text():
    sent = []

    def request(url, method, path, payload=None):
        if path == "/demo/accounts":
            return demo_accounts()
        sent.append(payload["text"])
        return {"reply": "How high is the fever?", "decision": "ask"}

    with patch("ui.client.request", side_effect=request):
        at = AppTest.from_file(APP_PATH).run()
        at.button(key="example_0").click().run()
        assert not at.exception
        assert sent == ["Asha has had a fever since yesterday"]  # another member, not the writer
        assert len(at.chat_message) == 2
        assert at.get("download_button")  # transcript export appears once the chat has messages


def test_a_503_shows_the_apis_safety_message():
    from ui.client import request
    detail = "Q could not hand this case to a doctor. If symptoms are severe, call your local emergency number now."
    response = httpx.Response(503, json={"detail": detail}, request=httpx.Request("POST", "http://q/messages"))
    with patch("ui.client.httpx.request", return_value=response):
        with pytest.raises(ChatAPIError, match="call your local emergency number"):
            request("http://q", "POST", "/messages", {"text": "hi"})
