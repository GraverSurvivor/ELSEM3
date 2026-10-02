"""FairGate: the gate, the exports, and the report, on synthetic runs."""

from src.attribution.patch import Patch
from src.experiments.schema import QuestionResult
from src.product.fairgate import build_report, check, hqq_config, llamacpp_overrides

UNKNOWN, STEREO, ANTI = 2, 0, 1
N = 200


def _run(answers, condition):
    return [
        QuestionResult(run_id=condition, condition=condition, question_id=f"Age:{i}", category="Age",
                       predicted_index=a, label=UNKNOWN, unknown_index=UNKNOWN,
                       is_unknown_predicted=a == UNKNOWN, is_correct=a == UNKNOWN,
                       is_stereotyped_predicted=a == STEREO)
        for i, a in enumerate(answers)
    ]


BASE = _run([UNKNOWN] * 180 + [STEREO] * 20, "fp16")
BROKEN = _run([UNKNOWN] * 80 + [STEREO] * 120, "hqq_3bit")
REPAIRED = _run([UNKNOWN] * 179 + [STEREO] * 21, "hqq_repaired")
PATCH = Patch(name="search_top5",
              modules=("model.layers.1.self_attn.qkv_proj", "model.layers.10.self_attn.o_proj",
                       "model.layers.2.mlp.down_proj", "model.layers.0.mlp.gate_up_proj"),
              nbits=8, source={"kind": "search"})


def test_gate_fails_a_real_regression():
    result = check(BASE, BROKEN)
    assert result["verdict"] == "FAIL" and len(result["failures"]) == 2
    assert result["safe_answers_flipped_to_stereotype"] == 100
    assert "points" in result["failures"][0]


def test_gate_passes_a_change_within_noise():
    result = check(BASE, REPAIRED)
    assert result["verdict"] == "PASS" and result["failures"] == []
    assert result["answers_changed"] == 1  # label-free: just counts disagreements


def test_llamacpp_names_and_anchoring():
    lines = llamacpp_overrides(PATCH)
    assert lines == [
        '--tensor-type "blk\\.1\\.attn_qkv=q8_0"',
        '--tensor-type "blk\\.10\\.attn_output=q8_0"',
        '--tensor-type "blk\\.2\\.ffn_down=q8_0"',
        '--tensor-type "blk\\.0\\.ffn_up=q8_0"',
    ]


def test_hqq_config_lists_every_override():
    config = hqq_config(PATCH)
    assert config["base"]["nbits"] == 3
    assert set(config["overrides"]) == set(PATCH.modules)
    assert all(o == {"nbits": 8, "group_size": 64} for o in config["overrides"].values())


def test_report_contains_the_verdicts_and_the_patch():
    html = build_report(BASE, BROKEN, REPAIRED, PATCH, controls=None, groups=None)
    assert html.startswith("<!doctype html>")
    assert "FAIL" in html and "PASS" in html
    assert "blk\\.10\\.attn_output=q8_0" in html
    assert "Not yet run" in html  # controls missing is reported, not hidden
    assert html.count("<table") == html.count("</table></div>")
