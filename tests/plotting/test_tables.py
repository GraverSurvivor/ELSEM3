import csv

import pytest

from src.plotting.tables import comparison_rows, to_markdown, write_table


def _block(n, base, deg, rep):
    rates = lambda u: {"usr": u, "srs": 1 - u}  # noqa: E731
    ci = {"point": 0.9, "lower": 0.85, "upper": 0.95}
    delta = {"delta": -0.036, "lower": -0.043, "upper": -0.029}
    return {
        "n_items": n, "baseline": rates(base), "hqq_3bit": rates(deg), "repaired": rates(rep),
        "recovery": {"usr": ci, "srs": ci},
        "gap_left": {"usr": delta, "srs": delta},
        "repair_effect": {"chisq_p": 1e-9},
    }


def test_comparison_rows_canonical_order_with_overall_last():
    comparison = {"overall": _block(20, 0.8, 0.4, 0.7), "SES": _block(10, 0.8, 0.4, 0.7), "Age": _block(10, 0.8, 0.4, 0.7)}
    rows = comparison_rows(comparison)
    assert [r["Category"] for r in rows] == ["Age", "SES", "Overall"]
    row = rows[0]
    assert row["USR baseline"] == "80.0%"
    assert row["USR recovered"] == "90% [85%, 95%]"
    assert row["USR gap left"] == "-3.6 pp [-4.3, -2.9]"
    assert row["Repair p (chi2)"] == "<0.001"


def test_markdown_and_csv_agree(tmp_path):
    rows = [{"A": 1, "B": "x"}, {"A": 2, "B": "y"}]
    csv_path, md_path = write_table(rows, tmp_path / "sub" / "t")
    with csv_path.open(encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == [{"A": "1", "B": "x"}, {"A": "2", "B": "y"}]
    assert md_path.read_text(encoding="utf-8") == to_markdown(rows)
    assert to_markdown(rows).splitlines()[0] == "| A | B |"


def test_empty_table_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_table([], tmp_path / "t")
