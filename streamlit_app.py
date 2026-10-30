"""Run with: uv run --env-file .env streamlit run streamlit_app.py"""

import json
import os
from uuid import uuid4

import streamlit as st

from ui.client import ChatAPIError, request
from ui.transcript import transcript_json, transcript_markdown

# What each decision means for the person reading, and its colour.
DECISIONS = {
    "ask": ("Follow-up questions", "#2563eb"),
    "ask_member": ("Checking who this is about", "#7c3aed"),
    "ask_episode": ("Checking which illness", "#7c3aed"),
    "answer": ("Health information", "#0f766e"),
    "home_monitoring": ("Home care", "#15803d"),
    "doctor": ("Doctor review", "#c2410c"),
    "emergency": ("Emergency", "#b91c1c"),
}
URGENCY = {"doctor_now": "within a few hours", "doctor_24h": "within 24 hours",
           "doctor_routine": "routine appointment"}
REASONS = {
    "protocol": "the triage questions", "doctor_requested": "you asked for a doctor",
    "doctor_flag": "a symptom that needs a doctor", "red_flag": "an emergency sign",
    "recurrence": "the illness came back", "not_improving": "not improving at check-back",
    "already_escalated": "already with a doctor", "intake_turn_limit": "the questions didn't settle it",
}
STYLE = """<style>
.q-chip {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem;
         font-weight:600; border:1px solid; margin-bottom:6px;}
.q-member {padding:8px 10px; border-radius:10px; border:1px solid rgba(128,128,128,0.25); margin-bottom:6px;}
.q-member small {opacity:0.7;}
</style>"""


def new_conversation() -> dict:
    return {"session_id": uuid4().hex, "messages": [], "pending": None}


@st.cache_data(ttl=60, show_spinner=False)
def accounts_for(api_url: str) -> list[dict]:
    return request(api_url, "GET", "/demo/accounts")


def age_label(age: float) -> str:
    return f"{round(age * 12)} months" if age < 1 else f"{age:g} years"


def chip(decision: str, extra: str = "") -> str:
    label, colour = DECISIONS.get(decision, (decision, "#64748b"))
    text = f"{label} · {extra}" if extra else label
    return f'<span class="q-chip" style="color:{colour}; border-color:{colour}; background:{colour}14">{text}</span>'


def show_note(note: dict, key: str) -> None:
    """The doctor intake note the turn saved. The demo stops here: no clinician receives it."""
    escalation = note.get("escalation") or {}
    with st.expander("Doctor intake note", expanded=False):
        st.markdown(f"**Chief complaint:** {note.get('chief_complaint') or '—'}")
        urgency = URGENCY.get(escalation.get("urgency"), escalation.get("urgency"))
        reason = REASONS.get(escalation.get("reason"), escalation.get("reason"))
        st.markdown(f"**Urgency:** {urgency} · **Why:** {reason}")
        st.markdown(note.get("history_of_present_illness") or "")
        timeline = note.get("symptom_timeline") or []
        if timeline:
            st.markdown("**Symptoms, in the order they were reported**")
            for entry in timeline:
                onset = f" ({entry['onset']})" if entry.get("onset") else ""
                st.markdown(f"- {entry['finding']}{onset}: “{entry['quote']}”{reported_label(entry)}")
        negatives = note.get("pertinent_negatives") or []
        if negatives:
            st.markdown("**Denied when asked**")
            for item in negatives:
                asked = f" (asked: “{item['asked']}”)" if item.get("asked") else ""
                st.markdown(f"- {item['finding']}: “{item['answer']}”{asked}{reported_label(item)}")
        dropped = (note.get("faithfulness") or {}).get("dropped") or []
        if dropped:
            st.caption(f"{len(dropped)} statement(s) left out because the patient's own words didn't support them.")
        if note.get("relevant_history"):
            st.markdown(f"**Relevant history:** {note['relevant_history']}")
        protocol = ((note.get("triage") or {}).get("protocol")) or {}
        if protocol:
            source = (protocol.get("source") or "").replace("-", " ")
            st.caption(f"Triage protocol: {protocol.get('title')} ({source})")
        st.download_button("Download note (JSON)", json.dumps(note, indent=2), file_name=f"intake-{key}.json",
                           mime="application/json", key=f"note_{key}")
    st.caption("Demo: the note is saved on the Q server. No doctor or clinic is contacted.")


def reported_label(item: dict) -> str:
    """Where and when an item was reported; an earlier conversation is called out in bold."""
    where = item.get("reported_in") or {}
    if not where.get("label"):
        return ""
    label = where["label"]
    return f" · **{label}**" if where.get("conversation") == "an earlier conversation" else f" · _{label}_"


def show_reply(message: dict) -> None:
    note = message.get("note") or {}
    urgency = URGENCY.get((note.get("escalation") or {}).get("urgency"), "")
    st.markdown(chip(message["decision"], urgency), unsafe_allow_html=True)
    st.markdown(message["content"])
    if note:
        show_note(note, message["turn_id"])


def sidebar(accounts: list[dict]) -> tuple[str, dict, dict]:
    with st.sidebar:
        st.header("Family")
        by_id = accounts_by_id(accounts)
        account_id = st.selectbox("Account", list(by_id), format_func=lambda key: by_id[key]["name"])
        members = {m["id"]: m for m in by_id[account_id]["members"]}
        profile_id = st.selectbox("Writing as", list(members), key=f"profile_{account_id}",
                                  format_func=lambda key: members[key]["name"])
        for member in members.values():
            you = " · you" if member["id"] == profile_id else ""
            conditions = ", ".join(member["conditions"]) or "No known conditions"
            st.markdown(f'<div class="q-member"><b>{member["name"]}</b>{you}<br>'
                        f'<small>{age_label(member["age_years"])} · {conditions}</small></div>',
                        unsafe_allow_html=True)
        st.caption("Synthetic demo families.")
    return account_id, members, members[profile_id]


def accounts_by_id(accounts: list[dict]) -> dict:
    return {a["household_id"]: a for a in accounts}


def export_transcript(conversation: dict, account: str, profile: str) -> None:
    messages, session_id = conversation["messages"], conversation["session_id"]
    if not messages:
        return
    st.caption("Export this chat")
    left, right = st.columns(2)
    labels = {key: label for key, (label, _) in DECISIONS.items()}
    left.download_button("Markdown", transcript_markdown(messages, account, profile, session_id, labels),
                         file_name=f"q-transcript-{session_id}.md", mime="text/markdown",
                         key="export_md", use_container_width=True)
    right.download_button("JSON", transcript_json(messages, account, profile, session_id),
                          file_name=f"q-transcript-{session_id}.json", mime="application/json",
                          key="export_json", use_container_width=True)


def examples(profile: dict, members: dict) -> list[str]:
    other = next((m for m in members.values() if m["id"] != profile["id"]), profile)
    return [f"{other['name']} has had a fever since yesterday", "What helps with a sore throat?",
            "I've had a headache on and off for 3 days"]


def main() -> None:
    st.set_page_config(page_title="Q · Family health", page_icon="🌿", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)
    api_url = os.environ.get("Q_API_URL", "http://localhost:8000").rstrip("/")
    st.title("🌿 Q")
    st.caption("A family health assistant: symptom questions, home care advice, and doctor handoff.")
    try:
        accounts = accounts_for(api_url)
    except ChatAPIError as exc:
        st.error(str(exc))
        st.code("uv run --env-file .env uvicorn api.main:app --reload", language="bash")
        if st.button("Reconnect"):
            st.rerun()
        st.stop()

    account_id, members, profile = sidebar(accounts)
    conversations = st.session_state.setdefault("conversations", {})
    conversation = conversations.setdefault((account_id, profile["id"]), new_conversation())
    path = f"/households/{account_id}/sessions/{conversation['session_id']}"
    with st.sidebar:
        sending = conversation["pending"] is not None and conversation["pending"]["error"] is None
        if st.button("New chat", key="new_chat", disabled=sending, use_container_width=True):
            try:
                if conversation["messages"]:
                    request(api_url, "POST", f"{path}/close")
            except ChatAPIError as exc:
                st.error(str(exc))
            else:
                conversations[(account_id, profile["id"])] = new_conversation()
                st.rerun()
        st.caption("Each profile keeps its own chat while this page is open. New chat ends it, and Q "
                   "summarises it into family memory in the background.")
        export_transcript(conversation, accounts_by_id(accounts)[account_id]["name"], profile["name"])

    st.markdown(f"Writing as **{profile['name']}**. Mention anyone in the family by name or relation, "
                "and Q works out who each message is about.")
    st.caption("Not medical advice. In an emergency, call your local emergency number.")

    chosen = None
    if not conversation["messages"]:
        st.info("Describe a symptom, ask a health question, or follow up on how someone is doing.")
        st.caption("Try an example")
        for index, example in enumerate(examples(profile, members)):
            if st.button(example, key=f"example_{index}", use_container_width=True):
                chosen = example
    for message in conversation["messages"]:
        with st.chat_message(message["role"], avatar="🌿" if message["role"] == "assistant" else None):
            if message["role"] == "assistant":
                show_reply(message)
            else:
                st.markdown(message["content"])

    text = st.chat_input("Describe a symptom or ask a question…", max_chars=2000,
                         disabled=conversation["pending"] is not None) or chosen
    if text and text.strip():
        conversation["messages"].append({"role": "user", "content": text.strip()})
        conversation["pending"] = {"text": text.strip(), "turn_id": uuid4().hex,
                                   "profile_id": profile["id"], "error": None}
        st.rerun()

    pending = conversation["pending"]
    if pending:
        if pending["error"]:
            st.error(pending["error"])
            if st.button("Retry message", key="retry_message"):
                pending["error"] = None
                st.rerun()
        else:
            try:
                with st.spinner("Q is thinking…"):
                    result = request(api_url, "POST", f"{path}/messages",
                                     {key: pending[key] for key in ("text", "turn_id", "profile_id")})
                    note = None
                    if result["decision"] == "doctor":
                        try:
                            note = request(api_url, "GET", f"/households/{account_id}/handoffs/{pending['turn_id']}")
                        except ChatAPIError:
                            note = None  # the reply still stands; the API saved the note before replying
            except ChatAPIError as exc:
                pending["error"] = str(exc)
            else:
                conversation["messages"].append({"role": "assistant", "content": result["reply"],
                                                 "decision": result["decision"], "turn_id": pending["turn_id"],
                                                 "note": note})
                conversation["pending"] = None
            st.rerun()


if __name__ == "__main__":
    main()
