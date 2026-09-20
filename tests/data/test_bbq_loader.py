import pytest

from src.data.bbq_loader import _find_unknown_index, _to_item


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
    }
    item = _to_item(row, category="Age")

    assert item.example_id == 42
    assert item.category == "Age"
    assert item.question_polarity == "neg"
    assert item.options == ("the grandson", "the grandfather", "cannot be determined")
    assert item.label == 2
    assert item.unknown_index == 2
    assert item.question_id == "Age:42"
