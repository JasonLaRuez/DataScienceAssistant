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
import pandas as pd
import seaborn as sns
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
        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.to_numpy(), color=_COLOR)
        ax.set_title(f"{column.name}: value counts")
        ax.set_xlabel(column.name)
        ax.set_ylabel("count")
        fig.tight_layout()
        figures.append((column.name, fig))

    return figures, skipped


def numeric_box_plots(frame: pd.DataFrame, profile: DataProfile) -> list[tuple[str, Figure]]:
    """One box-and-whisker plot per numeric column, to surface outliers."""
    figures: list[tuple[str, Figure]] = []

    for column in profile.columns:
        if column.kind != NUMERIC:
            continue
        fig, ax = plt.subplots()
        sns.boxplot(y=frame[column.name].dropna(), color=_COLOR, ax=ax)
        ax.set_title(f"{column.name}: distribution")
        ax.set_ylabel(column.name)
        fig.tight_layout()
        figures.append((column.name, fig))

    return figures


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
