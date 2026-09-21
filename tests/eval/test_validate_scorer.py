from src.data.bbq_loader import BBQItem
from src.eval.validate_scorer import sample_items


def _item(example_id: int) -> BBQItem:
    return BBQItem(
        example_id=example_id,
        category="Age",
        question_polarity="neg",
        context="context",
        question="question",
        options=("a", "b", "unknown"),
        label=0,
        unknown_index=2,
        stereotype_index=1,
    )


def test_sample_items_returns_all_when_fewer_than_requested():
    items = [_item(i) for i in range(3)]
    assert sample_items(items, per_category=10, seed=0) == items


def test_sample_items_respects_requested_size():
    items = [_item(i) for i in range(100)]
    assert len(sample_items(items, per_category=10, seed=0)) == 10


def test_sample_items_is_deterministic_for_a_given_seed():
    items = [_item(i) for i in range(100)]
    first = sample_items(items, per_category=10, seed=0)
    second = sample_items(items, per_category=10, seed=0)
    assert first == second


def test_sample_items_differs_across_seeds():
    # Guards against the sample silently ignoring the seed -- which would make "rerun with a
    # different seed to check robustness" a no-op.
    items = [_item(i) for i in range(100)]
    assert sample_items(items, per_category=10, seed=0) != sample_items(items, per_category=10, seed=1)


def test_sample_items_does_not_mutate_input():
    items = [_item(i) for i in range(100)]
    before = list(items)
    sample_items(items, per_category=10, seed=0)
    assert items == before
