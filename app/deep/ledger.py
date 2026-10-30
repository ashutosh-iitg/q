"""The turn's state while the deep agents run.

Subagents are isolated: their messages never reach the supervisor's state, only their final report.
So each one's final tool records what it established here, and code saves the whole turn once, at the
end, exactly as the graph's save_turn does. Every update replaces the state with a new mapping.
"""

import threading
from types import MappingProxyType
from typing import Any

from app.models.turn import TurnState


class TurnLedger:
    def __init__(self, state: TurnState) -> None:
        self._state: MappingProxyType = MappingProxyType(dict(state))
        self._lock = threading.Lock()  # tools may run in parallel

    @property
    def state(self) -> TurnState:
        return dict(self._state)  # type: ignore[return-value]

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def apply(self, update: TurnState | dict[str, Any]) -> None:
        with self._lock:
            self._state = MappingProxyType({**self._state, **update})
