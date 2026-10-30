"""Checking that a model's quote really comes from the user's words."""

import re

ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
SENTENCE_END = re.compile(r"[.!?\n]")
MAX_GAP_CHARS = 40


def is_quoted(quote: str, text: str) -> bool:
    """True when the quote is in the text, allowing "..." for a few words left out.

    Fragments must appear in order, each gap short and inside one sentence, so "No ...
    breathlessness" matches "No sweating or breathlessness" but not "No, it's fine. She has
    breathlessness", where stitching the fragments would turn a symptom into a denial.
    """
    fragments = [f for f in ELLIPSIS.split(quote.strip()) if f]
    if not fragments:
        return False
    start = text.find(fragments[0])
    while start >= 0:
        if _rest_follows(fragments[1:], text, start + len(fragments[0])):
            return True
        start = text.find(fragments[0], start + 1)
    return False


def _rest_follows(fragments: list[str], text: str, position: int) -> bool:
    for fragment in fragments:
        found = text.find(fragment, position)
        gap = text[position:found]
        if found < 0 or len(gap) > MAX_GAP_CHARS or SENTENCE_END.search(gap):
            return False
        position = found + len(fragment)
    return True
