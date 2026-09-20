from src.experiments import logger
from src.experiments.schema import CategorySummary, QuestionResult


def _result(question_id: str) -> QuestionResult:
    return QuestionResult(
        run_id="run1",
        condition="bf16",
        question_id=question_id,
        category="Age",
        predicted_index=2,
        label=2,
        unknown_index=2,
        is_unknown_predicted=True,
        is_correct=True,
    )


def _summary(n_stereotyped=None, srs=None) -> CategorySummary:
    return CategorySummary(
        run_id="run1",
        condition="bf16",
        category="Age",
        n_items=10,
        n_unknown=8,
        usr=0.8,
        n_stereotyped=n_stereotyped,
        srs=srs,
    )


def test_new_run_id_includes_condition_and_is_unique():
    id_a = logger.new_run_id("bf16")
    id_b = logger.new_run_id("hqq_3bit")
    assert id_a.startswith("bf16_")
    assert id_b.startswith("hqq_3bit_")
    assert id_a != id_b


def test_question_results_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
    results = [_result("Age:1"), _result("Age:2")]

    logger.write_question_results(results, run_id="run1")
    loaded = logger.read_question_results(run_id="run1")

    assert loaded == results


def test_category_summary_round_trip_with_stereotype_data(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
    summaries = [_summary(n_stereotyped=3, srs=0.3)]

    logger.write_category_summary(summaries, run_id="run1")
    loaded = logger.read_category_summary(run_id="run1")

    assert loaded == summaries


def test_category_summary_round_trip_without_stereotype_data(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
    summaries = [_summary(n_stereotyped=None, srs=None)]

    logger.write_category_summary(summaries, run_id="run1")
    loaded = logger.read_category_summary(run_id="run1")

    assert loaded == summaries
    assert loaded[0].n_stereotyped is None
    assert loaded[0].srs is None
