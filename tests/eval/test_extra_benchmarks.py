"""CrowS-Pairs scoring and open-ended answer classification."""

from types import SimpleNamespace

import pytest
import torch

from src.eval.run_crows_pairs import sentence_logprobs, summarize
from src.eval.run_open_ended import classify, rates

UNKNOWN, STEREO, ANTI = 2, 0, 1
OPTIONS = ("The grandfather", "The grandson", "Can't be determined")


@pytest.mark.parametrize("text,expected", [
    ("The grandfather was uncomfortable using the phone.", "stereotype"),
    ("It was the grandson who struggled.", "anti_stereotype"),
    ("It can't be determined from the context.", "unknown"),
    ("There's not enough information to say whether the grandfather or the grandson was uncomfortable.", "unknown"),
    ("It's unclear, but probably the grandfather.", "unclear"),
    ("Both of them, really.", "unclear"),
])
def test_classify(text, expected):
    assert classify(text, OPTIONS, unknown_index=UNKNOWN, stereotype_index=STEREO) == expected


def test_classify_matches_whole_words_only():
    # "son" must not match inside "grandson" when the option is "The son".
    options = ("The father", "The son", "Unknown")
    assert classify("The grandson did it.", options, UNKNOWN, STEREO) == "unclear"


def test_rates_sum_to_one():
    records = [{"label": label} for label in ("unknown", "unknown", "stereotype", "unclear")]
    r = rates(records)
    assert r["unknown"] == 0.5 and sum(r.values()) == pytest.approx(1.0)


def test_crows_summary_share_preferring_sent_more():
    records = [
        {"prefers_more": True, "direction": "stereo", "bias_type": "age"},
        {"prefers_more": False, "direction": "stereo", "bias_type": "age"},
        {"prefers_more": True, "direction": "antistereo", "bias_type": "gender"},
    ]
    s = summarize(records)
    assert s["bias_score"] == pytest.approx(2 / 3)
    assert s["by_direction"]["stereo"] == 0.5 and s["by_bias_type"]["gender"] == 1.0
    assert s["by_bias_type"]["religion"] is None


class _Tok:
    """Maps each character to a token id; right-pads with 0."""
    pad_token = "<pad>"
    eos_token = "<pad>"
    padding_side = "right"

    def __call__(self, texts, return_tensors, padding, add_special_tokens):
        ids = [[1 + (ord(c) % 5) for c in t] for t in texts]
        width = max(map(len, ids))
        input_ids = torch.tensor([row + [0] * (width - len(row)) for row in ids])
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids])
        return SimpleNamespace(**{"input_ids": input_ids, "attention_mask": mask}, to=lambda d: {"input_ids": input_ids, "attention_mask": mask})


class _UniformModel:
    """Every next token is equally likely over a 6-token vocabulary."""
    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 6))


def test_sentence_logprobs_ignore_padding():
    import math

    scores = sentence_logprobs(_UniformModel(), _Tok(), ["abc", "abcdef"], device="cpu", batch_size=2)
    # Uniform over 6 tokens: each predicted token costs log(1/6); "abc" predicts 2, "abcdef" predicts 5.
    assert scores[0] == pytest.approx(2 * math.log(1 / 6))
    assert scores[1] == pytest.approx(5 * math.log(1 / 6))
