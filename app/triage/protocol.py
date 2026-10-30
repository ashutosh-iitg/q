"""Triage protocols: data, loaded from YAML files in `protocols/`.

The format follows licensed telephone-triage guidelines: questions are grouped by disposition
level, most urgent first; the first positive answer sets the disposition, and all negative means
home care. Licensed content is loaded by dropping its files, in this format, into the directory.
Files marked `source: synthetic-placeholder` are not clinically validated.

Authoring rule: one sign per question. A question listing several signs lets a partial denial
("no sweating") read as ruling them all out, so bundled guideline questions are split on import.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

PROTOCOL_DIR = Path(__file__).with_name("protocols")
GENERIC = "generic"
LEVELS = ("emergency", "doctor_now", "doctor_24h", "doctor_routine")  # most urgent first
Disposition = Literal["emergency", "doctor_now", "doctor_24h", "doctor_routine", "home_care"]
RULE_KEYS = {"any", "all", "age_at_least", "age_below", "condition_in", "has_conditions", "slot"}
SLOT_TESTS = {"at_least", "equals"}


@dataclass(frozen=True)
class Question:
    id: str
    level: str
    ask: str | None  # guidance on what to ask; None: answered from the member's profile alone
    rule: Mapping[str, Any] | None  # evaluated by code; None: a question for the user
    sign: str | None = None  # the finding, stated as present; what an answer is recorded against
    authored: bool = False  # chosen by the intake agent for this episode, not part of the protocol file

    @property
    def finding(self) -> str | None:
        return self.sign or self.ask


@dataclass(frozen=True)
class Protocol:
    id: str
    title: str
    source: str
    version: str
    complaints: tuple[str, ...]  # complaint labels this protocol covers
    questions: tuple[Question, ...]
    home_care: tuple[str, ...]  # basic precautions, appended to a home-monitoring reply by code
    watch_for: tuple[str, ...]  # signs that mean the user should get help before check-back
    check_back_hours: int
    # A guideline rather than a script: how many signs the intake agent must choose at a level
    # (e.g. 3 complaint-specific emergency signs) before that level counts as ruled out.
    agent_signs: Mapping[str, int] = field(default_factory=dict)
    fallback_signs: tuple[Mapping[str, str], ...] = ()  # {sign, ask}, used when the agent can't choose
    allow_home_care: bool = True  # incomplete complaint coverage must end in human review

    def at(self, level: str) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.level == level)

    def question(self, question_id: str) -> Question | None:
        return next((q for q in self.questions if q.id == question_id), None)


def load_protocols(directory: Path = PROTOCOL_DIR) -> dict[str, Protocol]:
    protocols = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            protocol = _protocol(yaml.safe_load(path.read_text()))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid protocol {path.name}: {exc}") from exc
        if protocol.id in protocols:
            raise ValueError(f"Duplicate protocol id {protocol.id!r} in {path.name}")
        protocols[protocol.id] = protocol
    if GENERIC not in protocols:
        raise ValueError(f"No {GENERIC!r} protocol in {directory}")
    return protocols


def protocol_for(complaint: str | None, protocols: Mapping[str, Protocol]) -> Protocol:
    """The protocol covering a complaint label, or the generic one."""
    return next((p for p in protocols.values() if complaint in p.complaints), protocols[GENERIC])


def with_signs(protocol: Protocol, findings: Mapping[str, Mapping[str, Any]]) -> Protocol:
    """The episode's protocol: the file's questions plus the signs the intake agent chose, which are
    kept in the episode's findings. A chosen sign sits after the file's questions at its level."""
    added = [Question(id=k, level=v["level"], ask=v.get("ask"), rule=None, sign=v["sign"], authored=True)
             for k, v in findings.items() if "sign" in v and v.get("level") in LEVELS]
    ordered = tuple(q for level in LEVELS for q in (*protocol.at(level), *(q for q in added if q.level == level)))
    return replace(protocol, questions=ordered)


def _protocol(raw: Mapping[str, Any]) -> Protocol:
    questions = tuple(
        Question(id=q["id"], level=level, ask=q.get("ask"), rule=q.get("rule"), sign=q.get("sign"))
        for level in LEVELS for q in raw["questions"].get(level, [])
    )
    unknown_levels = set(raw["questions"]) - set(LEVELS)
    if unknown_levels:
        raise ValueError(f"unknown levels {sorted(unknown_levels)}")
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    for q in questions:
        if q.ask is None and q.rule is None:
            raise ValueError(f"question {q.id!r} needs ask, rule or both")
        if q.ask is not None and not q.sign:
            raise ValueError(f"question {q.id!r} needs a sign: the finding its answer is recorded against")
        if q.rule is not None:
            _check_rule(q.rule)
            if q.ask is None and _uses_slot(q.rule):
                raise ValueError(f"question {q.id!r} depends on a reported value, so it needs ask")
    agent_signs = {level: int(n) for level, n in raw.get("agent_signs", {}).items() if level in LEVELS}
    if len(raw.get("fallback_signs", ())) < max(agent_signs.values(), default=0):
        raise ValueError("agent_signs needs at least as many fallback_signs as its largest minimum")
    allow_home_care = raw.get("allow_home_care", raw["id"] != GENERIC)
    if type(allow_home_care) is not bool or (raw["id"] == GENERIC and allow_home_care):
        raise ValueError("allow_home_care must be boolean and false for the generic protocol")
    return Protocol(
        id=raw["id"], title=raw["title"], source=raw["source"], version=str(raw["version"]),
        complaints=tuple(raw.get("complaints", ())), questions=questions,
        home_care=tuple(raw["home_care"]), watch_for=tuple(raw["watch_for"]),
        check_back_hours=int(raw["check_back_hours"]),
        agent_signs=agent_signs,
        fallback_signs=tuple({"sign": s["sign"], "ask": s["ask"]} for s in raw.get("fallback_signs", ())),
        allow_home_care=allow_home_care,
    )


def _check_rule(rule: Mapping[str, Any]) -> None:
    key = "slot" if "slot" in rule else next(iter(rule), None)
    if key not in RULE_KEYS or (key != "slot" and len(rule) != 1):
        raise ValueError(f"a rule has exactly one known test: {dict(rule)}")
    if key in ("any", "all"):
        for part in rule[key]:
            _check_rule(part)
    if key == "slot" and len(set(rule) & SLOT_TESTS) != 1:
        raise ValueError(f"slot rule needs one of {sorted(SLOT_TESTS)}: {dict(rule)}")


def _uses_slot(rule: Mapping[str, Any]) -> bool:
    return "slot" in rule or any(_uses_slot(part) for part in rule.get("any", rule.get("all", ())))
