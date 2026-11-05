"""Run the four assignment scenarios against the real model and save their transcripts.

Run: uv run --env-file .env python -m scripts.run_scenarios [extra] [--judge]
--judge adds the eval-only entailment judge (app/evals/entailment.py) to every doctor note: one extra
model call per note, reported in the transcript, never written into the note. Each scenario gets a fresh copy of a synthetic demo family, so scenarios never share memory. Each
inner list is one session; a session is closed (and consolidated into memory) before the next.
"""

import os
import sys
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from api.main import create_app
from app.config import judge_model
from app.demo import demo_accounts
from app.evals.entailment import Judge, entailment_judge, judge_note

OUT = Path(os.environ.get("Q_TRANSCRIPTS_DIR") or Path(__file__).resolve().parents[1] / "transcripts")

SCENARIOS = [
    {"file": "1-child-fever-home-care.md", "family": "demo-reddy", "writer": "demo-reddy-lakshmi",
     "title": "A child's fever: implicit member, progressive intake, home monitoring",
     "sessions": [["My son has had a fever since last night, it's 38.4 now",
                   "No stiff neck, he's making sense, wakes up easily and there's no rash",
                   "He's drinking fine and peeing normally. He's tired but still playing, so mild"]]},
    {"file": "2-chest-pain-emergency.md", "family": "demo-patel", "writer": "demo-patel-priya",
     "title": "A parent's chest pain: triage that turns into an emergency",
     "sessions": [["Mum has had some chest pain on and off since this morning",
                   "Yes, it's happening right now, she's sweating a lot and it's going into her left arm"]]},
    {"file": "3-blood-pressure-across-sessions.md", "family": "demo-fernandes", "writer": "demo-fernandes-maria",
     "title": "High blood pressure across two sessions: memory, episode follow-up, doctor handoff",
     "sessions": [["My blood pressure was 150/95 this morning and I have a mild headache",
                   "No chest pain, no confusion, no weakness or numbness, my vision is fine and the headache "
                   "isn't severe. No dizziness."],
                  ["It's still high today, 165/100, and the headache is worse than yesterday"]]},
    {"file": "4-two-children-one-conversation.md", "family": "demo-khan", "writer": "demo-khan-sameer",
     "title": "A general question, then two children's symptoms in one conversation",
     "sessions": [["Is it safe to give children ibuprofen for a fever?",
                   "Ayaan has had a cough for two days",
                   "He's breathing normally, no blue lips, no blood, and the wheeze isn't getting worse",
                   "Also Zoya threw up twice this morning",
                   "She's still feeling sick now",
                   "Okay, let's go back to Ayaan"]]},
]

# Further behaviours, kept out of the four submitted transcripts: python -m scripts.run_scenarios extra
EXTRA_SCENARIOS = [
    {"file": "5-ambiguous-member.md", "family": "demo-reddy", "writer": "demo-reddy-lakshmi",
     "title": "An ambiguous \"she\": Q asks who, then continues the intake",
     "sessions": [["She's had a sore throat since yesterday", "My daughter, Anika",
                   "No, she's breathing fine and swallowing her drinks okay"]]},
    {"file": "6-home-care-then-worse.md", "family": "demo-das", "writer": "demo-das-ananya",
     "title": "Home care, then worse the next day: check-back escalation",
     "sessions": [["My mother has had a fever since yesterday, it's 38.2",
                   "No stiff neck, she's not confused, she's awake and talking, and no rash",
                   "She's drinking and peeing normally, and she feels mildly unwell"],
                  ["Mum's fever is worse today, 39.5, and she's barely drinking anything"]]},
    {"file": "7-baby-danger-sign-first.md", "family": "demo-iyer", "writer": "demo-iyer-meera",
     "title": "A baby's fever with a danger sign in the first message",
     "sessions": [["The baby has a fever of 38.6 and she's very hard to wake up"]]},
]


def run(client: TestClient, scenario: dict, judge: Judge | None = None) -> str:
    store = client.app.state.store
    family = next(a for a in demo_accounts() if a["household_id"] == scenario["family"])
    household = f"scenario-{uuid4().hex[:8]}"
    ids = {m["id"]: f"{household}-{m['id'].rsplit('-', 1)[-1]}" for m in family["members"]}
    store.upsert_members(household, [{**m, "id": ids[m["id"]]} for m in family["members"]])
    names = {ids[m["id"]]: m["name"] for m in family["members"]}
    writer = ids[scenario["writer"]]
    family_line = ", ".join(f"{m['name']} ({m['age_years']}{', ' + ', '.join(m['conditions']) if m['conditions'] else ''})"
                            for m in family["members"])
    lines = [f"# {scenario['title']}", "", f"Family: {family_line}. Writing as **{names[writer]}**.", ""]
    if len(scenario["sessions"]) > 1:
        lines += ["_Sessions are a day apart: after each one, the runner moves its messages, episodes and facts "
                  "back a day in the database. Notes are saved files written during the run, so they keep "
                  "the run's date._", ""]
    for number, messages in enumerate(scenario["sessions"], 1):
        session = f"{household}-s{number}"
        later = len(scenario["sessions"]) - number  # sessions still to come, one day apart
        if len(scenario["sessions"]) > 1:
            lines += [f"## Session {number} ({'today' if not later else f'{later} day(s) earlier'})", ""]
        for index, text in enumerate(messages):
            turn = uuid4().hex
            response = client.post(f"/households/{household}/sessions/{session}/messages",
                                   json={"text": text, "turn_id": turn, "profile_id": writer})
            response.raise_for_status()
            result = response.json()
            context = store.load_context(household, session)
            if number > 1 and index == 0:
                lines += ["> **Memory carried in:** " + " ".join(s.summary for s in context.prior_summaries), ""]
            lines += [f"**{names[writer]}:** {text}", "", f"**Q** (`{result['decision']}`):", "",
                      *(f"> {line}" for line in result["reply"].splitlines()), ""]
            episodes = "; ".join(f"{names.get(e.member_id, e.member_id)}: {e.complaint}, {e.status}"
                                 for e in context.episodes) or "none"
            lines += [f"_Episodes: {episodes}_", ""]
            if result["decision"] == "doctor":
                lines += note_lines(client.get(f"/households/{household}/handoffs/{turn}").json(), judge)
        if later:
            move_back_a_day(store, household)  # before consolidation, so the summary is dated that day
        client.post(f"/households/{household}/sessions/{session}/close")  # consolidates into memory
        if later:
            move_back_a_day(store, household, closed_only=True)
    return "\n".join(lines)


def move_back_a_day(store, household: str, closed_only: bool = False) -> None:
    """The run takes minutes; this makes what was said so far a day older, as the scenario describes."""
    day = "interval '1 day'"
    sessions = "SELECT id FROM sessions WHERE household_id = %(h)s"
    members = "SELECT id FROM members WHERE household_id = %(h)s"
    statements = [f"UPDATE sessions SET closed_at = closed_at - {day} WHERE household_id = %(h)s"] if closed_only else [
        f"UPDATE sessions SET started_at = started_at - {day}, last_active_at = last_active_at - {day} "
        "WHERE household_id = %(h)s",
        f"UPDATE messages SET created_at = created_at - {day} WHERE session_id IN ({sessions})",
        f"UPDATE episodes SET opened_at = opened_at - {day}, updated_at = updated_at - {day}, "
        f"check_back_after = check_back_after - {day} WHERE member_id IN ({members})",
        f"UPDATE member_facts SET created_at = created_at - {day}, effective_at = effective_at - {day} "
        f"WHERE member_id IN ({members})",
    ]
    with store._pool.connection() as conn:  # demo-only time travel; the app never rewrites history
        for statement in statements:
            conn.execute(statement, {"h": household})


def note_lines(note: dict, judge: Judge | None = None) -> list[str]:
    escalation = note["escalation"]
    timeline = [f"  - {t['finding']} ({t['onset'] or 'onset not stated'}): \"{t['quote']}\" [{t['reported_in']['label']}]"
                for t in note["symptom_timeline"]]
    negatives = [f"  - {n['finding']}: \"{n['answer']}\"" + (f" (asked: \"{n['asked']}\")" if n["asked"] else "")
                 + f" [{n['reported_in']['label']}]" for n in note.get("pertinent_negatives") or []]
    return ["<details><summary>Doctor intake note saved (no clinician is contacted in this demo)</summary>", "",
            f"- Chief complaint: {note['chief_complaint']}",
            f"- Urgency: {escalation['urgency']} (reason: {escalation['reason']})",
            f"- History: {note['history_of_present_illness']}", "- Timeline:", *timeline,
            "- Denied when asked:", *(negatives or ["  - none"]),
            f"- Relevant history: {note['relevant_history'] or 'none recorded'}",
            f"- Left out as unsupported: {len(note['faithfulness']['dropped'])}",
            *(judge_lines(judge_note(judge, note)) if judge else []), "", "</details>", ""]


def judge_lines(result: dict) -> list[str]:
    if "error" in result:
        return [f"- Entailment judge: failed ({result['error']})"]
    flagged = [f"  - {f['verdict']}: \"{f['text']}\" citing "
               + ", ".join(f"\"{e['quote']}\"" for e in f["evidence"]) + f" ({f['reason']})"
               for f in result["flagged"]]
    return [f"- Entailment judge: {result['supported']}/{result['claims']} supported", *flagged]


def main() -> None:
    args = sys.argv[1:]
    judge = entailment_judge(judge_model()) if "--judge" in args else None
    OUT.mkdir(exist_ok=True)
    with TestClient(create_app()) as client:
        for scenario in EXTRA_SCENARIOS if "extra" in args else SCENARIOS:
            (OUT / scenario["file"]).write_text(run(client, scenario, judge))
            print("saved", OUT / scenario["file"])


if __name__ == "__main__":
    main()
