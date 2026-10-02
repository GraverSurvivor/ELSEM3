"""Quantization error per module, for the largest-error baseline. Real HQQ on CPU, tiny tensors."""

import torch

from src.quant.quant_error import quantization_error, rank_by_error


def test_error_is_positive_and_shrinks_with_more_bits():
    torch.manual_seed(0)
    weight = torch.randn(128, 256)
    err3 = quantization_error(weight, nbits=3, group_size=64, device="cpu")
    err8 = quantization_error(weight, nbits=8, group_size=64, device="cpu")
    assert 0 < err8["rel_error"] < err3["rel_error"] < 1
    assert err3["abs_error"] > 0


def test_ranking_is_largest_first_with_a_stable_tie_break():
    entries = [
        {"path": "b", "rel_error": 0.1},
        {"path": "a", "rel_error": 0.3},
        {"path": "c", "rel_error": 0.3},
    ]
    assert [e["path"] for e in rank_by_error(entries)] == ["a", "c", "b"]
