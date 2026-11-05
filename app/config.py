import os
from pathlib import Path

from langchain_openai import ChatOpenAI

DEFAULT_HANDOFF_DIR = Path(__file__).resolve().parents[1] / "handoffs"


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


def chat_model() -> ChatOpenAI:
    """The model every role uses. Q_REASONING_EFFORT (e.g. low, medium, high) is optional."""
    return ChatOpenAI(model=require_env("Q_MODEL_MINI"), use_responses_api=True, timeout=60, max_retries=1,
                      reasoning_effort=os.environ.get("Q_REASONING_EFFORT", "").strip() or None)


def judge_model() -> ChatOpenAI:
    """The eval-only entailment judge: Q_JUDGE_MODEL, else the model Q itself uses."""
    name = os.environ.get("Q_JUDGE_MODEL", "").strip() or require_env("Q_MODEL_MINI")
    return ChatOpenAI(model=name, use_responses_api=True, timeout=60, max_retries=1)


def handoff_dir() -> Path:
    """Where doctor-handoff intake notes are saved, one JSON file per escalation."""
    return Path(os.environ.get("Q_HANDOFF_DIR", "").strip() or DEFAULT_HANDOFF_DIR)
