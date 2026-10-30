"""Sufficiency: when has Q learned enough to declare a disposition?

A disposition is sufficient once a question at its level is positive and every question at every
more urgent level is ruled out. So one positive emergency answer is enough on its own, while home
care needs every question answered negative. Until then, the engine names the next question, one
per turn like a clinician taking a history: the first open one at the most urgent open level.

A question is positive when the user says yes, when they don't know (an emergency can't be ruled
out on an unknown), or when its rule holds for the member's profile and reported values.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.triage.protocol import LEVELS, Disposition, Protocol, Question

NEW_SIGN = "new_sign"  # pending placeholder: the intake agent chooses the next sign at this level
ABSENT = {"absent", "no"}  # "no" is how findings were recorded before present/absent
RESOLVED = "resolved"  # there earlier in this illness, gone now
# A sign that came and went still needs urgent review; below these levels it is history, not a current sign.
# Placeholder pending clinical review, like the protocols.
URGENT_LEVELS = {"emergency", "doctor_now"}

@dataclass(frozen=True)
class Profile:
    age_years: float | None
    conditions: tuple[str, ...]  # seeded conditions plus remembered condition facts, normalised


@dataclass(frozen=True)
class Triage:
    protocol: Protocol
    disposition: Disposition | None  # set only when sufficient
    deciding: tuple[str, ...]  # positive question IDs at the disposition's level
    pending: tuple[Question, ...]  # the one question to ask next while not sufficient
    floor: str | None  # most urgent level already positive while a more urgent one is still open
    states: Mapping[str, bool | None]  # every question: positive, ruled out, or unanswered

    @property
    def sufficient(self) -> bool:
        return self.disposition is not None


def profile(age_years: float | None, conditions: Iterable[str]) -> Profile:
    return Profile(age_years, tuple(sorted({_normalise(c) for c in conditions if c.strip()})))


def triage(protocol: Protocol, findings: Mapping[str, Mapping[str, Any]], slots: Mapping[str, Any],
           member: Profile) -> Triage:
    states = {q.id: _state(q, findings, slots, member) for q in protocol.questions}
    positive_levels = [level for level in LEVELS if any(states[q.id] for q in protocol.at(level))]
    floor = positive_levels[0] if positive_levels else None
    for level in LEVELS:
        questions = protocol.at(level)
        deciding = tuple(q.id for q in questions if states[q.id] is True)
        if deciding:
            return Triage(protocol, level, deciding, (), None, states)
        open_ = tuple(q for q in questions if states[q.id] is None)
        if open_:
            return Triage(protocol, None, (), open_[:1], floor, states)
        if sum(q.authored for q in questions) < protocol.agent_signs.get(level, 0):
            return Triage(protocol, None, (), (Question(NEW_SIGN, level, None, None),), floor, states)
    # Negative answers to a few generated signs do not establish comprehensive coverage.
    return Triage(protocol, "home_care" if protocol.allow_home_care else "doctor_24h", (), (), None, states)


def _state(question: Question, findings: Mapping[str, Mapping[str, Any]], slots: Mapping[str, Any],
           member: Profile) -> bool | None:
    if question.rule is not None:
        value = evaluate(question.rule, slots, member)
        if value is not None:
            return value
    answer = findings.get(question.id, {}).get("value")
    if answer == RESOLVED:
        return question.level in URGENT_LEVELS
    return None if answer is None else answer not in ABSENT


def evaluate(rule: Mapping[str, Any], slots: Mapping[str, Any], member: Profile) -> bool | None:
    """True, False, or None when a reported value it needs is missing.

    Missing profile data counts as positive: an unknown age can't rule out a risk.
    """
    if "any" in rule:
        values = [evaluate(part, slots, member) for part in rule["any"]]
        return True if True in values else None if None in values else False
    if "all" in rule:
        values = [evaluate(part, slots, member) for part in rule["all"]]
        return False if False in values else None if None in values else True
    if "age_at_least" in rule:
        return member.age_years is None or member.age_years >= rule["age_at_least"]
    if "age_below" in rule:
        return member.age_years is None or member.age_years < rule["age_below"]
    if "condition_in" in rule:
        wanted = [_normalise(c) for c in rule["condition_in"]]
        return any(term in condition for condition in member.conditions for term in wanted)
    if "has_conditions" in rule:
        return bool(member.conditions) == rule["has_conditions"]
    value = slots.get(rule["slot"])
    if value is None:
        return None
    return value >= rule["at_least"] if "at_least" in rule else value == rule["equals"]


def _normalise(label: str) -> str:
    return "_".join(label.lower().replace("-", " ").split())
