"""Checks the ambiguous/disambiguated split against the real dataset.

These hit the network (cached after the first run). They're the only place the condition
filtering is verified end to end -- the distinction matters because the whole validation story
rests on disambiguated items having a real, non-"unknown" correct answer.
"""

from src.data.bbq_loader import load_bbq_ambiguous, load_bbq_disambiguated


def test_disambiguated_items_never_have_unknown_as_the_correct_answer():
    # This is what makes disambiguated data usable to validate the scorer: if "unknown" were
    # ever correct here, a broken always-answer-unknown scorer could still look good.
    items = load_bbq_disambiguated("Religion")
    assert items
    assert all(item.label != item.unknown_index for item in items)


def test_ambiguous_items_always_have_unknown_as_the_correct_answer():
    items = load_bbq_ambiguous("Religion")
    assert items
    assert all(item.label == item.unknown_index for item in items)


def test_conditions_return_disjoint_question_ids():
    ambiguous = {item.question_id for item in load_bbq_ambiguous("Religion")}
    disambiguated = {item.question_id for item in load_bbq_disambiguated("Religion")}
    assert ambiguous and disambiguated
    assert not (ambiguous & disambiguated)
