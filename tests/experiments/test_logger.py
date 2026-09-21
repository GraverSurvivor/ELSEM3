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


class TestCheckpointing:
    """Behaviour a long, interruptible run depends on.

    A baseline run checkpoints after every category, so these files get rewritten repeatedly
    while a run is in progress. That turns two normally-theoretical failures into real ones: an
    interrupt landing mid-write, and a read of a run that doesn't exist yet.
    """

    def test_no_temp_file_is_left_behind(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
        logger.write_question_results([_result("Age:1")], run_id="run1")
        logger.write_category_summary([_summary()], run_id="run1")

        assert list(tmp_path.rglob("*.tmp")) == []

    def test_repeated_writes_replace_rather_than_append(self, tmp_path, monkeypatch):
        # Each checkpoint rewrites the full accumulated list, so writing twice must not double
        # the rows -- that would silently inflate every count derived from the file.
        monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)

        logger.write_question_results([_result("Age:1")], run_id="run1")
        logger.write_question_results([_result("Age:1"), _result("Age:2")], run_id="run1")

        assert len(logger.read_question_results(run_id="run1")) == 2

    def test_failed_write_leaves_the_previous_checkpoint_intact(self, tmp_path, monkeypatch):
        # The point of writing atomically: a crash partway through must not destroy the last
        # good checkpoint, or an interrupted run becomes unresumable instead of just stale.
        monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
        logger.write_question_results([_result("Age:1")], run_id="run1")

        class _Boom(Exception):
            pass

        def exploding_dumps(*args, **kwargs):
            raise _Boom("interrupted mid-write")

        monkeypatch.setattr(logger.json, "dumps", exploding_dumps)
        try:
            logger.write_question_results([_result("Age:1"), _result("Age:2")], run_id="run1")
        except _Boom:
            pass

        monkeypatch.undo()
        monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
        recovered = logger.read_question_results(run_id="run1")
        assert [r.question_id for r in recovered] == ["Age:1"]

    def test_missing_ok_returns_empty_for_an_unstarted_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
        assert logger.read_question_results(run_id="never-ran", missing_ok=True) == []

    def test_missing_ok_does_not_create_the_run_directory(self, tmp_path, monkeypatch):
        # Probing for a resumable run shouldn't litter results/ with empty directories that
        # later look like real runs.
        monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
        logger.read_question_results(run_id="never-ran", missing_ok=True)

        assert not (tmp_path / "never-ran").exists()

    def test_missing_run_still_raises_without_missing_ok(self, tmp_path, monkeypatch):
        import pytest

        monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            logger.read_question_results(run_id="never-ran")

    def test_partial_run_reads_back_only_finished_categories(self, tmp_path, monkeypatch):
        # What resume actually relies on: categories are written whole, so the set of categories
        # present in the file is exactly the set that finished.
        monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)

        age = [_result("Age:1"), _result("Age:2")]
        logger.write_question_results(age, run_id="run1")

        loaded = logger.read_question_results(run_id="run1", missing_ok=True)
        assert {r.category for r in loaded} == {"Age"}
