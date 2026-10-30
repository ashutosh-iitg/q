"""Transcript export for the Streamlit chat: Markdown to read, JSON to keep the saved notes."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime


def transcript_markdown(messages: list[dict], account: str, profile: str, session_id: str,
                        labels: Mapping[str, str] | None = None) -> str:
    labels = labels or {}
    lines = [f"# Q conversation · {account}", "",
             f"Writing as {profile} · session `{session_id}` · exported {_now()}", ""]
    for message in messages:
        if message["role"] == "user":
            lines.append(f"**{profile}:** {message['content']}")
        else:
            decision = message.get("decision", "")
            lines.append(f"**Q** _({labels.get(decision, decision)})_: {message['content']}")
            note = message.get("note")
            if note:
                lines.append(f"\n_Doctor intake note saved: {note.get('chief_complaint') or 'see JSON export'}_")
        lines.append("")
    lines.append("_Synthetic demo. Not medical advice._")
    return "\n".join(lines)


def transcript_json(messages: list[dict], account: str, profile: str, session_id: str) -> str:
    return json.dumps({"account": account, "profile": profile, "session_id": session_id,
                       "exported_at": _now(), "messages": messages}, indent=2)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
