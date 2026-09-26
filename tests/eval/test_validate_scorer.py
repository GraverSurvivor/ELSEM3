from types import SimpleNamespace

from src.data.bbq_loader import BBQItem
from src.eval.validate_scorer import FP16_REFERENCE_ACCURACY, _report_quantized, sample_items


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


# --- --model hqq_3bit: biased or broken ----------------------------------------------------

_DEFAULT_ARGS = SimpleNamespace(per_category=100, seed=0)
_CATS = {"Age": 0.9, "SES": 0.8}


def test_quantized_well_above_chance_is_reported_as_biased_not_broken(capsys):
    _report_quantized(0.80, _CATS, _DEFAULT_ARGS)
    out = capsys.readouterr().out
    assert "BIASED, NOT BROKEN" in out
    assert f"{FP16_REFERENCE_ACCURACY:.3f}" in out and "88% retained" in out


def test_quantized_near_chance_is_reported_as_broken(capsys):
    _report_quantized(0.36, _CATS, _DEFAULT_ARGS)
    assert "BROKEN:" in capsys.readouterr().out


def test_quantized_names_the_weakest_category(capsys):
    _report_quantized(0.80, _CATS, _DEFAULT_ARGS)
    assert "Weakest category: SES (0.800)" in capsys.readouterr().out


def test_quantized_on_a_different_sample_does_not_claim_a_comparison_with_fp16(capsys):
    _report_quantized(0.80, _CATS, SimpleNamespace(per_category=25, seed=0))
    out = capsys.readouterr().out
    assert "not directly comparable" in out
    assert "retained" not in out
