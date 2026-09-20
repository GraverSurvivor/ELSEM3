import matplotlib

matplotlib.use("Agg")  # no display in CI/headless test runs

import matplotlib.pyplot as plt

from src.plotting.style import CATEGORY_PALETTE, save_figure, set_house_style


def test_set_house_style_runs_without_error():
    set_house_style()


def test_category_palette_covers_all_five_categories():
    expected = {"Age", "Gender_identity", "Race_ethnicity", "Religion", "SES"}
    assert set(CATEGORY_PALETTE.keys()) == expected


def test_save_figure_writes_png(tmp_path, monkeypatch):
    import src.plotting.style as style_module

    monkeypatch.setattr(style_module, "FIGURES_DIR", tmp_path)

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    path = save_figure(fig, "smoke_test")

    assert path.exists()
    assert path.name == "smoke_test.png"
    plt.close(fig)


def test_save_figure_scopes_to_run_id(tmp_path, monkeypatch):
    import src.plotting.style as style_module

    monkeypatch.setattr(style_module, "FIGURES_DIR", tmp_path)

    fig, ax = plt.subplots()
    path = save_figure(fig, "smoke_test", run_id="run1")

    assert path == tmp_path / "run1" / "smoke_test.png"
    assert path.exists()
    plt.close(fig)
