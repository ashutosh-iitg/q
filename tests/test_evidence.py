import pytest

from app.agents.evidence import is_quoted

TEXT = "No, it's not happening right now. No sweating or breathlessness, and it doesn't spread."


@pytest.mark.parametrize(("quote", "expected"), [
    ("No sweating or breathlessness", True),
    ("No ... breathlessness", True),  # a few words skipped within the sentence
    ("No … breathlessness", True),
    ("No ... sweating", True),  # the first "No" fails, the second matches
    ("breathlessness ... No", False),  # out of order
    ("No ... spread", False),  # gap too long
    ("happening ... sweating", False),  # gap crosses a sentence
    ("crushing pain", False),
    ("...", False),
    ("", False),
])
def test_quotes_must_come_from_one_place_in_the_users_words(quote, expected):
    assert is_quoted(quote, TEXT) is expected


def test_skipping_words_cannot_turn_a_symptom_into_a_denial():
    assert not is_quoted("No ... breathlessness", "No, it's fine. She has breathlessness")
