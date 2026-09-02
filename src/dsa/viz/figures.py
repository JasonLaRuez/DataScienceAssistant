"""Step 3a: building the actual plots.

Every function here takes a frame and its already-computed ``DataProfile`` and returns
one or more ``matplotlib.figure.Figure`` objects. No session, no file I/O, no gates --
same shape as :mod:`dsa.clean.detect`: pure observation, nothing here decides what a
human should do about what it shows.

Figures are never closed by this module. Jupyter's inline backend auto-displays every
figure still open at the end of a cell, so leaving them open is what makes a notebook
cell that just calls ``dsa.analyze(s)`` show every plot with no extra display code.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dsa.profile import BOOLEAN, CATEGORICAL, NUMERIC, DataProfile

# A column with at least this many distinct values is not "low-cardinality" -- comparable
# in spirit to dsa.clean.detect.HIGH_CARDINALITY, but independently configurable since
# this governs a display decision, not a modelling one.
DEFAULT_MAX_CATEGORIES = 10

_DISCRETE_KINDS = (CATEGORICAL, BOOLEAN)

# One flat color for every univariate plot -- position and label already distinguish the
# categories in a bar chart, so a different hue per bar would encode nothing real.
_COLOR = sns.color_palette("colorblind")[0]

# A single box has nothing to fill matplotlib's default 6.4-inch figure width -- ~30%
# narrower than default, same default height.
_BOX_PLOT_FIGSIZE = (4.5, 4.8)


def categorical_bar_charts(
    frame: pd.DataFrame, profile: DataProfile, max_categories: int = DEFAULT_MAX_CATEGORIES
) -> tuple[list[tuple[str, Figure]], list[str]]:
    """One value-count bar chart per categorical or boolean column under ``max_categories``.

    Returns ``(figures, skipped)`` -- ``figures`` as ``(column_name, Figure)`` pairs, and
    ``skipped`` naming every categorical/boolean column that had too many distinct values
    to chart, so a caller can report what was left out rather than it vanishing silently.
    """
    figures: list[tuple[str, Figure]] = []
    skipped: list[str] = []

    for column in profile.columns:
        if column.kind not in _DISCRETE_KINDS:
            continue
        if column.n_unique >= max_categories:
            skipped.append(column.name)
            continue

        counts = frame[column.name].value_counts()
        total = int(counts.sum())
        fig, ax = plt.subplots()
        bars = ax.bar(counts.index.astype(str), counts.to_numpy(), color=_COLOR)
        ax.bar_label(
            bars,
            labels=[f"{c:,}\n({c / total:.1%})" for c in counts.to_numpy()],
            padding=3,
        )
        # bar_label draws above the bar; without headroom the top label clips against
        # the axes frame.
        ax.margins(y=0.15)
        ax.set_title(f"{column.name}: value counts")
        ax.set_xlabel(column.name)
        ax.set_ylabel("count")
        fig.tight_layout()
        figures.append((column.name, fig))

    return figures, skipped


def numeric_box_plots(frame: pd.DataFrame, profile: DataProfile) -> list[tuple[str, Figure]]:
    """One box-and-whisker plot per numeric column, to surface outliers.

    Annotated with the five-number summary (min, Q1, median, Q3, max) plus the mean,
    since a box plot's whiskers and quartile lines show these visually but don't give
    the reader the exact values.
    """
    figures: list[tuple[str, Figure]] = []

    for column in profile.columns:
        if column.kind != NUMERIC:
            continue
        values = frame[column.name].dropna()
        fig, ax = plt.subplots(figsize=_BOX_PLOT_FIGSIZE)
        sns.boxplot(y=values, color=_COLOR, ax=ax)
        ax.set_title(f"{column.name}: distribution")
        ax.set_ylabel(column.name)
        _annotate_five_number_summary(ax, values)
        fig.tight_layout()
        figures.append((column.name, fig))

    return figures


def _annotate_five_number_summary(ax: Axes, values: pd.Series) -> None:
    """Draw a min/Q1/median/Q3/max/mean text box in a plot's top-right corner."""
    summary = (
        f"min:    {values.min():.2f}\n"
        f"Q1:     {values.quantile(0.25):.2f}\n"
        f"median: {values.median():.2f}\n"
        f"Q3:     {values.quantile(0.75):.2f}\n"
        f"max:    {values.max():.2f}\n"
        f"mean:   {values.mean():.2f}"
    )
    ax.text(
        0.97,
        0.97,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray", "alpha": 0.85},
    )


def correlation_heatmap(frame: pd.DataFrame, profile: DataProfile) -> Figure | None:
    """One feature-feature Pearson correlation heatmap over every numeric column.

    Returns ``None`` if fewer than two numeric columns exist -- a correlation of one
    thing with itself is not a plot. Diverging colormap centered at 0: correlation is a
    polarity value (negative/positive), so a sequential or rainbow map would misread it.
    """
    numeric_columns = [c.name for c in profile.columns if c.kind == NUMERIC]
    if len(numeric_columns) < 2:
        return None

    correlations = frame[numeric_columns].corr()
    side = 0.6 * len(numeric_columns) + 2
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(correlations, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("feature-feature correlation")
    fig.tight_layout()
    return fig


def categorical_association_heatmap(frame: pd.DataFrame, profile: DataProfile) -> Figure | None:
    """One Cramer's V association heatmap over every categorical/boolean column -- the
    categorical analog of :func:`correlation_heatmap`.

    Returns ``None`` if fewer than two categorical/boolean columns exist. Cramer's V
    ranges over ``[0, 1]`` with no negative direction (unlike Pearson correlation), so
    this uses a sequential colormap rather than ``correlation_heatmap``'s diverging one.
    Not bias-corrected: plain Cramer's V is slightly biased upward on small samples or
    sparse contingency tables, which is an acceptable simplification for exploratory use
    but worth knowing before reading too much into a borderline value.
    """
    columns = [c.name for c in profile.columns if c.kind in _DISCRETE_KINDS]
    if len(columns) < 2:
        return None

    n = len(columns)
    associations = pd.DataFrame(np.eye(n), index=columns, columns=columns)
    for i, a in enumerate(columns):
        for j in range(i + 1, n):
            b = columns[j]
            v = _cramers_v(frame[a], frame[b])
            associations.iloc[i, j] = v
            associations.iloc[j, i] = v

    side = 0.6 * n + 2
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(associations, cmap="rocket", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("categorical feature association (Cramer's V)")
    fig.tight_layout()
    return fig


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V association between two categorical/boolean series, in [0, 1]."""
    table = pd.crosstab(x, y).to_numpy(dtype=float)
    r, c = table.shape
    min_dim = min(r - 1, c - 1)
    n = table.sum()
    if min_dim <= 0 or n == 0:
        return float("nan")

    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))

    return float(np.sqrt(chi2 / (n * min_dim)))


def scatter_matrix(
    frame: pd.DataFrame, profile: DataProfile, columns: tuple[str, ...] | None = None
) -> Figure:
    """A grid of pairwise scatter plots over numeric columns, for use when the feature
    count is small enough that the whole grid stays legible.

    ``columns`` defaults to every ``NUMERIC``-kind column in ``profile``; pass an
    explicit subset to plot fewer than that. Feature names appear only on the left
    column and bottom row (``sharex``/``sharey`` plus ``label_outer`` suppress the
    redundant interior tick labels a naive grid would otherwise repeat on every cell).
    The diagonal is left empty -- a feature scattered against itself adds nothing.
    """
    if columns is None:
        columns = tuple(c.name for c in profile.columns if c.kind == NUMERIC)
    if len(columns) < 2:
        raise ValueError(f"scatter_matrix needs at least 2 numeric columns; got {columns}")

    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n), sharex="col", sharey="row")

    for row, y in enumerate(columns):
        for col, x in enumerate(columns):
            ax = axes[row, col]
            if row != col:
                ax.scatter(frame[x], frame[y], alpha=0.6, color=_COLOR)
            # label_outer() runs even on the (data-free) diagonal cells: the top-left
            # and bottom-right diagonal cells are also the grid's edge cells, so they
            # still need their outer tick labels shown for the left-column/bottom-row
            # labeling below to land anywhere.
            ax.label_outer()

    for col, x in enumerate(columns):
        axes[-1, col].set_xlabel(x)
    for row, y in enumerate(columns):
        axes[row, 0].set_ylabel(y)

    fig.suptitle("pairwise feature scatter matrix")
    fig.tight_layout()
    return fig


def pair_plot(frame: pd.DataFrame, profile: DataProfile, x: str, y: str) -> Figure:
    """Plot ``x`` against ``y``, dispatched on each column's kind.

    Axis order in the call never determines the plot's layout: a categorical/numeric
    pair always puts the numeric value on the value axis regardless of which one was
    named ``x``. Datetime columns are not supported yet -- there is no meaningful
    default for them here, so this raises rather than guessing at one.
    """
    x_kind = profile[x].kind
    y_kind = profile[y].kind

    fig, ax = plt.subplots()

    if x_kind == NUMERIC and y_kind == NUMERIC:
        ax.scatter(frame[x], frame[y], alpha=0.6, color=_COLOR)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")
    elif x_kind in _DISCRETE_KINDS and y_kind in _DISCRETE_KINDS:
        table = pd.crosstab(frame[x], frame[y])
        sns.heatmap(table, cmap="rocket", annot=True, fmt="d", ax=ax)
        ax.set_title(f"{x} vs {y}")
    elif x_kind in _DISCRETE_KINDS and y_kind == NUMERIC:
        sns.boxplot(x=frame[x], y=frame[y], color=_COLOR, ax=ax)
        ax.set_title(f"{y} by {x}")
    elif x_kind == NUMERIC and y_kind in _DISCRETE_KINDS:
        sns.boxplot(x=frame[y], y=frame[x], color=_COLOR, ax=ax)
        ax.set_title(f"{x} by {y}")
    else:
        plt.close(fig)
        raise ValueError(
            f"plot_pair does not support {x!r} ({x_kind}) vs {y!r} ({y_kind}) yet; "
            "supported combinations are numeric/numeric, categorical-or-boolean/numeric, "
            "and categorical-or-boolean/categorical-or-boolean"
        )

    fig.tight_layout()
    return fig
