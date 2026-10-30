"""Extension points for the capabilities outside this harness's scope. Plug one in by passing an
Extensions to build_graph; each is None by default, and the harness runs without it.

- knowledge: a custom knowledge base. For a general health question, its passages reach the reply
  writer as reference material.
- lab_results: automated lab-report analysis. Analysed results for a member become the lab_results
  lookup for the intake and handoff agents; without an analyser the tool isn't offered at all.
- deliver_handoff: a live telemedicine integration. It receives each intake note after the note is
  saved; if it raises, the turn fails, so Q never tells the user a case was handed over when it wasn't.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Knowledge = Callable[[str], list[str]]  # the user's question -> relevant passages
LabResults = Callable[[str, str], list[dict[str, Any]]]  # (household_id, member_id) -> {test, value, range, date}
DeliverHandoff = Callable[[dict[str, Any]], None]  # the saved intake note -> sent to a clinician service


@dataclass(frozen=True)
class Extensions:
    knowledge: Knowledge | None = None
    lab_results: LabResults | None = None
    deliver_handoff: DeliverHandoff | None = None
