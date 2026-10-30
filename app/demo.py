"""Synthetic accounts shared by API seeding and the demo catalog."""

import json
from pathlib import Path

DEMO_PATH = Path(__file__).parents[1] / "fixtures" / "demo_families.json"


def demo_accounts() -> list[dict]:
    return json.loads(DEMO_PATH.read_text())["accounts"]
