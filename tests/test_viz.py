"""Tests for step 3: exploratory visualization.

The properties that matter most: the auto-generated batch stays linear in feature count
(nothing pairwise happens without being asked for), nothing is silently omitted without
being named, and every figure lands on disk and in the log.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import dsa
from dsa.profile import profile_frame
from dsa.viz.figures import categorical_bar_charts, correlation_heatmap, numeric_box_plots, pair_plot


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    """Test-only hygiene -- production code never closes a figure (see dsa.viz.figures'
    module docstring), but leaving hundreds open across a full test run just trips
    matplotlib's too-many-open-figures warning for no reason."""
    yield
    plt.close("all")


@pytest.fixture
def session(tmp_path):
    return dsa.new_session(project_root=tmp_path)


@pytest.fixture
def analyzed_session(session):
    """A session past repairs/transforms approval, ready for dsa.analyze()."""
    rng = np.random.default_rng(0)
    n = 50
    frame = pd.DataFrame({
        "city": rng.choice(["london", "leeds"], n),                       # low-cardinality categorical
        "many_groups": rng.choice([f"grp-{i}" for i in range(15)], n),    # >= threshold: skipped
        "active": rng.choice([True, False], n),                           # boolean
        "age": rng.normal(40, 10, n),                                     # numeric
        "income": rng.normal(50_000, 12_000, n),                          # numeric
        "joined": pd.date_range("2024-01-01", periods=n),                 # datetime
        "y": rng.integers(0, 2, n),                                       # target
    })
    session.raw = frame
    session.rebuild()
    session.target = "y"

    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)
    dsa.approve_transforms(session)
    return session


# --- pure figure builders --------------------------------------------------------------

def test_pair_plot_numeric_numeric_produces_a_scatter():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    fig = pair_plot(frame, profile_frame(frame), "a", "b")
    assert len(fig.axes[0].collections) == 1  # the scatter PathCollection


def test_pair_plot_categorical_numeric_normalizes_the_categorical_to_the_x_axis():
    frame = pd.DataFrame({"cat": ["x", "y", "x", "y"], "num": [1.0, 2.0, 3.0, 4.0]})
    fig = pair_plot(frame, profile_frame(frame), "num", "cat")  # numeric passed first
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["x", "y"]


def test_pair_plot_categorical_categorical_produces_a_heatmap():
    frame = pd.DataFrame({"a": ["x", "y", "x", "y"], "b": ["p", "p", "q", "q"]})
    fig = pair_plot(frame, profile_frame(frame), "a", "b")
    assert len(fig.axes[0].collections) == 1  # the heatmap's QuadMesh


def test_pair_plot_rejects_datetime():
    frame = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=3), "n": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="does not support"):
        pair_plot(frame, profile_frame(frame), "d", "n")


def test_categorical_bar_charts_skips_high_cardinality_and_reports_it():
    frame = pd.DataFrame({
        "low": ["a", "b"] * 10,
        "high": [f"v{i % 12}" for i in range(20)],  # 12 distinct values: over the threshold
    })
    figures, skipped = categorical_bar_charts(frame, profile_frame(frame), max_categories=10)
    names = {name for name, _ in figures}
    assert "low" in names
    assert "high" not in names
    assert skipped == ["high"]


def test_categorical_bar_charts_label_each_bar_with_count_and_proportion():
    frame = pd.DataFrame({"cat": ["a", "a", "a", "b"]})  # a: 3 (75%), b: 1 (25%)
    figures, _ = categorical_bar_charts(frame, profile_frame(frame))
    _, fig = figures[0]
    labels = {t.get_text() for t in fig.axes[0].texts}
    assert "3\n(75.0%)" in labels
    assert "1\n(25.0%)" in labels


def test_numeric_box_plots_covers_every_numeric_column():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "c": ["x", "y", "z"]})
    figures = numeric_box_plots(frame, profile_frame(frame))
    assert {name for name, _ in figures} == {"a", "b"}


def test_numeric_box_plots_annotate_the_five_number_summary_and_mean():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    figures = numeric_box_plots(frame, profile_frame(frame))
    _, fig = figures[0]
    text = fig.axes[0].texts[0].get_text()
    assert "min:" in text and "1.00" in text
    assert "Q1:" in text
    assert "median:" in text and "3.00" in text
    assert "Q3:" in text
    assert "max:" in text and "5.00" in text
    assert "mean:" in text and "3.00" in text


def test_correlation_heatmap_needs_at_least_two_numeric_columns():
    one_numeric = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    assert correlation_heatmap(one_numeric, profile_frame(one_numeric)) is None

    two_numeric = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    assert correlation_heatmap(two_numeric, profile_frame(two_numeric)) is not None


# --- dsa.analyze -------------------------------------------------------------------------

def test_analyze_requires_transforms_to_be_approved(session):
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    session.raw = frame
    session.rebuild()
    session.target = "y"
    with pytest.raises(dsa.GateRequired, match="propose_transforms"):
        dsa.analyze(session)


def test_analyze_makes_bar_charts_only_for_low_cardinality_categorical_and_boolean(analyzed_session):
    summary = dsa.analyze(analyzed_session)
    named = {p.stem for p in summary.bar_charts}
    assert named == {"bar_city", "bar_active"}
    assert summary.skipped_categorical == ("many_groups",)


def test_analyze_makes_box_plots_for_every_numeric_column_including_the_target(analyzed_session):
    summary = dsa.analyze(analyzed_session)
    assert {p.stem for p in summary.box_plots} == {"box_age", "box_income", "box_y"}


def test_analyze_makes_one_correlation_heatmap(analyzed_session):
    summary = dsa.analyze(analyzed_session)
    assert summary.correlation is not None
    assert summary.correlation.exists()


def test_analyze_skips_correlation_with_fewer_than_two_numeric_columns(session):
    frame = pd.DataFrame({"city": ["a", "b", "a", "b"], "y": [0, 1, 0, 1]})
    session.raw = frame
    session.rebuild()
    session.target = "y"
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)
    dsa.approve_transforms(session)

    summary = dsa.analyze(session)
    assert summary.correlation is None


def test_analyze_saves_every_figure_and_logs_it(analyzed_session):
    summary = dsa.analyze(analyzed_session)
    all_paths = [*summary.bar_charts, *summary.box_plots, *([summary.correlation] if summary.correlation else [])]
    assert all_paths
    for path in all_paths:
        assert path.exists()
        assert path.parent == analyzed_session.figures_dir

    entry = next(e for e in analyzed_session.log.entries if e.op == "viz.analyze")
    assert set(entry.artifacts) == {str(p) for p in all_paths}


def test_analyze_opens_the_figures_gate_and_proceed_closes_it(analyzed_session):
    dsa.analyze(analyzed_session)
    assert not analyzed_session.gates["figures"].answered

    dsa.proceed(analyzed_session, "figures")
    assert analyzed_session.gates["figures"].answered


# --- dsa.plot_pair -----------------------------------------------------------------------

def test_plot_pair_requires_analyze_to_have_run_first(analyzed_session):
    with pytest.raises(KeyError, match="no gate named"):
        dsa.plot_pair(analyzed_session, "age", "income")


def test_plot_pair_after_the_figures_gate_is_closed_is_rejected(analyzed_session):
    dsa.analyze(analyzed_session)
    dsa.proceed(analyzed_session, "figures")
    with pytest.raises(ValueError, match="already closed"):
        dsa.plot_pair(analyzed_session, "age", "income")


def test_plot_pair_rejects_an_unknown_column(analyzed_session):
    dsa.analyze(analyzed_session)
    with pytest.raises(ValueError, match="no such column"):
        dsa.plot_pair(analyzed_session, "nope", "age")


def test_plot_pair_saves_and_logs_the_figure(analyzed_session):
    dsa.analyze(analyzed_session)
    path = dsa.plot_pair(analyzed_session, "age", "income")

    assert path.exists()
    assert path.parent == analyzed_session.figures_dir
    entry = next(e for e in analyzed_session.log.entries if e.op == "viz.plot_pair")
    assert entry.artifacts == [str(path)]


def test_plot_pair_does_not_close_the_figures_gate(analyzed_session):
    """Asking for more plots is not the same as finishing the review."""
    dsa.analyze(analyzed_session)
    dsa.plot_pair(analyzed_session, "age", "income")
    assert not analyzed_session.gates["figures"].answered
