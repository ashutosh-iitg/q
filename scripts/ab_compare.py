"""Run every scenario on both engines with the real model, and compare them.

Run: uv run --env-file .env python -m scripts.ab_compare [graph|deep ...]
Each engine gets fresh copies of the demo families. Writes per-engine transcripts to transcripts/ab/,
raw results to output/ab/results.json and the comparison to output/ab/deep-vs-graph.md.
"""

import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.callbacks import BaseCallbackHandler

from api.main import create_app
from app.demo import demo_accounts
from scripts.run_scenarios import EXTRA_SCENARIOS, SCENARIOS, move_back_a_day, note_lines

ROOT = Path(__file__).resolve().parents[1]
OUT, TRANSCRIPTS = ROOT / "output" / "ab", ROOT / "transcripts" / "ab"
ENGINES = ("graph", "deep")
# The outcome a clinician would expect at the turns that settle each scenario: (session, turn) -> decision.
EXPECTED = {
    "1-child-fever-home-care.md": {(1, 3): "home_monitoring"},
    "2-chest-pain-emergency.md": {(1, 2): "emergency"},
    "3-blood-pressure-across-sessions.md": {(2, 1): "doctor"},
    "4-two-children-one-conversation.md": {(1, 1): "answer"},
    "5-ambiguous-member.md": {(1, 1): "ask_member"},
    "6-home-care-then-worse.md": {(1, 3): "home_monitoring", (2, 1): "doctor"},
    "7-baby-danger-sign-first.md": {(1, 1): "emergency"},
}
SEVERITY = {"emergency": 3, "doctor": 2}  # under-triage: a lower outcome than these


class Meter(BaseCallbackHandler):
    """Model calls and tokens, per turn."""

    def __init__(self) -> None:
        self.lock, self.calls, self.tokens = threading.Lock(), 0, 0

    def reset(self) -> None:
        with self.lock:
            self.calls = self.tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        usage = sum((getattr(g.message, "usage_metadata", None) or {}).get("total_tokens", 0)
                    for gens in response.generations for g in gens if hasattr(g, "message"))
        with self.lock:
            self.calls += 1
            self.tokens += usage


def run_engine(engine: str) -> list[dict]:
    os.environ["Q_ENGINE"] = engine
    meter, records = Meter(), []
    with TestClient(create_app()) as client:
        client.app.state.turn_callbacks = [meter]
        for scenario in [*SCENARIOS, *EXTRA_SCENARIOS]:
            lines, turns = run_scenario(client, scenario, meter)
            records += turns
            (TRANSCRIPTS / engine).mkdir(parents=True, exist_ok=True)
            (TRANSCRIPTS / engine / scenario["file"]).write_text("\n".join(lines))
            print(engine, "done", scenario["file"], flush=True)
    return records


def run_scenario(client: TestClient, scenario: dict, meter: Meter) -> tuple[list[str], list[dict]]:
    store = client.app.state.store
    family = next(a for a in demo_accounts() if a["household_id"] == scenario["family"])
    household = f"ab-{uuid4().hex[:8]}"
    ids = {m["id"]: f"{household}-{m['id'].rsplit('-', 1)[-1]}" for m in family["members"]}
    store.upsert_members(household, [{**m, "id": ids[m["id"]]} for m in family["members"]])
    writer = ids[scenario["writer"]]
    lines, turns = [f"# {scenario['title']}", ""], []
    for number, messages in enumerate(scenario["sessions"], 1):
        session, later = f"{household}-s{number}", len(scenario["sessions"]) - number
        lines += [f"## Session {number}", ""]
        for index, text in enumerate(messages, 1):
            turn = uuid4().hex
            meter.reset()
            started = time.perf_counter()
            response = client.post(f"/households/{household}/sessions/{session}/messages",
                                   json={"text": text, "turn_id": turn, "profile_id": writer})
            seconds = time.perf_counter() - started
            result = response.json() if response.status_code == 200 else {"decision": f"error_{response.status_code}",
                                                                           "reply": response.text}
            saved = client.get(f"/households/{household}/handoffs/{turn}") if result["decision"] == "doctor" else None
            note = saved.json() if saved is not None and saved.status_code == 200 else None  # none: a reminder
            turns.append({"scenario": scenario["file"], "session": number, "turn": index, "text": text,
                          "decision": result["decision"], "reply": result["reply"], "seconds": round(seconds, 1),
                          "calls": meter.calls, "tokens": meter.tokens,
                          "dropped": len(note["faithfulness"]["dropped"]) if note else None,
                          "urgency": (note or {}).get("escalation", {}).get("urgency")})
            lines += [f"**User:** {text}", "", f"**Q** (`{result['decision']}`, {seconds:.1f}s, {meter.calls} calls):",
                      "", *(f"> {line}" for line in result["reply"].splitlines()), ""]
            if note:
                lines += note_lines(note)
        if later:
            move_back_a_day(store, household)
        client.post(f"/households/{household}/sessions/{session}/close")
        if later:
            move_back_a_day(store, household, closed_only=True)
    return lines, turns


def summary(records: list[dict]) -> dict:
    expected = [(r, EXPECTED.get(r["scenario"], {}).get((r["session"], r["turn"]))) for r in records]
    judged = [(r, e) for r, e in expected if e]
    seconds = sorted(r["seconds"] for r in records)
    return {
        "turns": len(records),
        "expected_matched": f"{sum(r['decision'] == e for r, e in judged)}/{len(judged)}",
        "under_triage": sum(SEVERITY.get(r["decision"], 0) < SEVERITY[e] for r, e in judged if e in SEVERITY),
        "errors": sum(r["decision"].startswith("error") for r in records),
        "seconds_mean": round(statistics.mean(seconds), 1),
        "seconds_p95": seconds[min(len(seconds) - 1, round(0.95 * (len(seconds) - 1)))],
        "calls_mean": round(statistics.mean(r["calls"] for r in records), 1),
        "tokens_mean": round(statistics.mean(r["tokens"] for r in records)),
        "note_statements_dropped": sum(r["dropped"] or 0 for r in records),
    }


def report(results: dict[str, list[dict]]) -> str:
    stats = {engine: summary(records) for engine, records in results.items()}
    rows = list(next(iter(stats.values())))
    lines = ["# Graph vs deep engine", "", "| Measure | " + " | ".join(stats) + " |",
             "|---|" + "---|" * len(stats)]
    lines += [f"| {row} | " + " | ".join(str(stats[e][row]) for e in stats) + " |" for row in rows]
    if set(results) == set(ENGINES):
        pairs = list(zip(results["graph"], results["deep"], strict=False))
        agree = sum(g["decision"] == d["decision"] for g, d in pairs)
        lines += ["", f"Same decision on {agree}/{len(pairs)} turns.", "", "## Turn by turn", "",
                  "| Scenario | S.T | Message | Expected | Graph | Deep | Calls g/d | Seconds g/d |",
                  "|---|---|---|---|---|---|---|---|"]
        for g, d in pairs:
            want = EXPECTED.get(g["scenario"], {}).get((g["session"], g["turn"]), "")
            mark = "" if g["decision"] == d["decision"] else " ≠"
            lines.append(f"| {g['scenario'].split('-', 1)[0]} | {g['session']}.{g['turn']} | {g['text'][:48]} | {want} | "
                         f"{g['decision']} | {d['decision']}{mark} | {g['calls']}/{d['calls']} | "
                         f"{g['seconds']}/{d['seconds']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engines = sys.argv[1:] or list(ENGINES)
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for engine in engines:
        results[engine] = run_engine(engine)
        path.write_text(json.dumps(results, indent=2))
    (OUT / "deep-vs-graph.md").write_text(report({e: results[e] for e in ENGINES if e in results}))
    print((OUT / "deep-vs-graph.md").read_text())


if __name__ == "__main__":
    main()
