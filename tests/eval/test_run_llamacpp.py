"""The llama.cpp scorer, against a fake llama.cpp model: same argmax and confidence as option_scorer,
and its runs can only be compared with other llama.cpp runs."""

import math

import pytest

from src.eval.run_llamacpp import condition_name, score_with_llama
from src.stats.metrics import ConditionMismatchError, assert_comparable, compute_dtype


class FakeLlama:
    """Records what it was fed; its last-position logits are set per prompt."""

    def __init__(self, logits_per_prompt, vocab=10):
        self.logits_per_prompt = list(logits_per_prompt)
        self.vocab = vocab
        self.fed = []
        self.n_tokens = 0
        self.scores = None

    def reset(self):
        self.n_tokens = 0

    def eval(self, ids):
        self.fed.append(list(ids))
        self.n_tokens = len(ids)
        row = [0.0] * self.vocab
        for token, value in self.logits_per_prompt.pop(0).items():
            row[token] = value
        self.scores = [[0.0] * self.vocab for _ in range(len(ids) - 1)] + [row]


def test_scores_the_three_label_tokens_at_the_last_position():
    llm = FakeLlama([{3: 2.0, 4: 0.0, 5: 0.0, 9: 50.0}, {3: 0.0, 4: 0.0, 5: 1.0}])
    predictions, confidences = score_with_llama(llm, [[1, 2], [1, 2, 7]], label_ids=[3, 4, 5])
    assert predictions == [0, 2]  # token 9 is not a label, so its huge logit is ignored
    assert confidences[0] == pytest.approx(math.exp(2) / (math.exp(2) + 2))
    assert llm.fed == [[1, 2], [1, 2, 7]]  # token ids go in unchanged


def test_condition_names_and_compute_type():
    assert condition_name("q3km_fairgate") == "llamacpp_q3km_fairgate"
    with pytest.raises(ValueError):
        condition_name("q4")
    assert compute_dtype("llamacpp_q3km") == "llamacpp"
    assert compute_dtype("llamacpp_f16_disambig") == "llamacpp"


def test_llamacpp_runs_never_compare_with_transformers_runs():
    class Row:
        def __init__(self, condition):
            self.condition = condition

    with pytest.raises(ConditionMismatchError):
        assert_comparable([Row("fp16")], [Row("llamacpp_q3km")])
    assert_comparable([Row("llamacpp_f16")], [Row("llamacpp_q3km_fairgate")])
