import pytest

from src.data.bbq_loader import (
    _find_stereotype_index,
    _find_unknown_index,
    _normalize_group_label,
    _to_item,
)


def _answer_info(unknown_slot: int) -> dict[str, list[str]]:
    labels = ["old", "nonOld", "unknown"]
    labels[unknown_slot], labels[2] = labels[2], labels[unknown_slot]
    return {
        "ans0": ["some text", labels[0]],
        "ans1": ["some text", labels[1]],
        "ans2": ["some text", labels[2]],
    }


@pytest.mark.parametrize("unknown_slot", [0, 1, 2])
def test_find_unknown_index_locates_correct_slot(unknown_slot):
    answer_info = _answer_info(unknown_slot)
    assert _find_unknown_index(answer_info) == unknown_slot


def test_find_unknown_index_is_case_insensitive():
    answer_info = {
        "ans0": ["text", "Unknown"],
        "ans1": ["text", "nonOld"],
        "ans2": ["text", "old"],
    }
    assert _find_unknown_index(answer_info) == 0


def test_find_unknown_index_raises_when_absent():
    answer_info = {
        "ans0": ["text", "old"],
        "ans1": ["text", "nonOld"],
        "ans2": ["text", "young"],
    }
    with pytest.raises(ValueError):
        _find_unknown_index(answer_info)


@pytest.mark.parametrize(
    "label,expected",
    [
        ("old", "old"),
        ("man", "m"),
        ("woman", "f"),
        ("boy", "m"),
        ("girl", "f"),
        ("F", "f"),
        ("trans_F", "transgenderwomen"),
        ("trans_M", "transgendermen"),
        ("Transgender women", "transgenderwomen"),
        ("transgender men", "transgendermen"),
        ("lowSES", "lowses"),
        ("low SES", "lowses"),
        ("F-Arab", "arab"),
        ("M-Black", "black"),
        ("African American", "africanamerican"),
    ],
)
def test_normalize_group_label(label, expected):
    # Locks in the real-data-derived normalization/alias table -- see the comment above
    # _GROUP_LABEL_ALIASES for how each case was found (empirically, against the real dataset).
    assert _normalize_group_label(label) == expected


def test_find_stereotype_index_neg_polarity_targets_stereotyped_group():
    # "neg" question: picking the stereotyped group IS the stereotype-consistent answer.
    answer_info = {
        "ans0": ["grandson", "nonOld"],
        "ans1": ["grandfather", "old"],
        "ans2": ["cannot be determined", "unknown"],
    }
    index = _find_stereotype_index(answer_info, ["old"], "neg", unknown_index=2)
    assert index == 1


def test_find_stereotype_index_nonneg_polarity_flips_to_non_target_group():
    # "nonneg" question: picking the NON-stereotyped group is the stereotype-consistent answer
    # (it implies the stereotyped group lacks the positive trait being asked about).
    answer_info = {
        "ans0": ["grandson", "nonOld"],
        "ans1": ["grandfather", "old"],
        "ans2": ["cannot be determined", "unknown"],
    }
    index = _find_stereotype_index(answer_info, ["old"], "nonneg", unknown_index=2)
    assert index == 0


def test_find_stereotype_index_raises_when_ambiguous():
    # Neither non-unknown option's group_label matches stereotyped_groups -- can't resolve.
    answer_info = {
        "ans0": ["grandson", "nonOld"],
        "ans1": ["grandfather", "old"],
        "ans2": ["cannot be determined", "unknown"],
    }
    with pytest.raises(ValueError):
        _find_stereotype_index(answer_info, ["some_other_group"], "neg", unknown_index=2)


def test_to_item_normalizes_row():
    row = {
        "example_id": 42,
        "question_polarity": "neg",
        "context": "some context",
        "question": "some question",
        "ans0": "the grandson",
        "ans1": "the grandfather",
        "ans2": "cannot be determined",
        "label": 2,
        "answer_info": {
            "ans0": ["grandson", "nonOld"],
            "ans1": ["grandfather", "old"],
            "ans2": ["cannot be determined", "unknown"],
        },
        "additional_metadata": {"stereotyped_groups": ["old"]},
    }
    item = _to_item(row, category="Age")

    assert item.example_id == 42
    assert item.category == "Age"
    assert item.question_polarity == "neg"
    assert item.options == ("the grandson", "the grandfather", "cannot be determined")
    assert item.label == 2
    assert item.unknown_index == 2
    assert item.stereotype_index == 1
    assert item.anti_stereotype_index == 0
    assert item.question_id == "Age:42"
