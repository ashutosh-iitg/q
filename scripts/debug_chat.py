"""Terminal chat that runs the graph in-process, so breakpoint()/pdb work inside nodes.

Run: python -m scripts.debug_chat [household_id] [profile_id]   (loads the repo's .env)
Debug: python -m pdb -m scripts.debug_chat
"""

import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from app.config import chat_model, handoff_dir, require_env
from app.demo import demo_accounts
from app.graph.build import build_graph
from app.memory.store import Store

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_HOUSEHOLD = "demo"
DEFAULT_PROFILE = "demo-ravi"
EXIT_WORDS = {"", "quit", "exit"}


def main() -> None:
    load_dotenv(ENV_FILE)  # variables already set in the shell take precedence
    household_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOUSEHOLD
    profile_id = sys.argv[2] if len(sys.argv) > 2 else (DEFAULT_PROFILE if household_id == DEFAULT_HOUSEHOLD else None)
    store = Store.connect(require_env("DATABASE_URL"))
    try:
        store.init_schema()
        for account in demo_accounts():
            store.upsert_members(account["household_id"], account["members"])
        model = chat_model()
        graph = build_graph(store, model, handoff_dir=handoff_dir())
        session_id = f"debug-{uuid4().hex[:8]}"
        print(f"household={household_id} profile={profile_id} session={session_id} (empty line or 'quit' exits)")
        while (text := input("you> ").strip()) not in EXIT_WORDS:
            turn_id = uuid4().hex
            state = {"household_id": household_id, "session_id": session_id, "turn_id": turn_id,
                     "text": text, "profile_id": profile_id}
            config = {"run_name": "q.turn", "tags": ["turn", "debug"],  # traced like an API turn
                      "metadata": {"household_id": household_id, "session_id": session_id, "turn_id": turn_id}}
            final: dict = {}
            for step in graph.stream(state, config=config, stream_mode="updates"):
                node, update = next(iter(step.items()))
                print(f"  [{node}] {sorted(update or {})}")
                final = {**final, **(update or {})}
            print(f"  decision: {final.get('decision')} ({final.get('decision_reason')})")
            if final.get("handoff_path"):
                print(f"  handoff note: {final['handoff_path']}")
            print(f"q> {final.get('reply')}\n")
    finally:
        store.close()


if __name__ == "__main__":
    main()
