"""Step 3b: running the figure-builders in dsa.viz.figures against a Session.

Saves what dsa.viz.figures produces to session.figures_dir, logs each figure, and opens
the "figures" review gate -- the same relationship dsa.clean.proposals has to
dsa.clean.detect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

from dsa.gates import REVIEW, GateRequired, open_gate, revise
from dsa.profile import profile_frame
from dsa.session import Session
from dsa.viz.figures import (
    DEFAULT_MAX_CATEGORIES,
    categorical_bar_charts,
    correlation_heatmap,
    numeric_box_plots,
    pair_plot,
)

STEP = 3


@dataclass(frozen=True)
class AnalysisSummary:
    """What :func:`analyze` produced -- the step-7 write-up reads these paths rather
    than needing to recompute them."""

    bar_charts: tuple[Path, ...]
    box_plots: tuple[Path, ...]
    correlation: Path | None
    skipped_categorical: tuple[str, ...]

    def describe(self) -> str:
        header = (
            f"{len(self.bar_charts)} bar chart(s), {len(self.box_plots)} box plot(s), "
            + (
                "1 correlation heatmap"
                if self.correlation
                else "no correlation heatmap (fewer than 2 numeric columns)"
            )
        )
        lines = [header]
        if self.skipped_categorical:
            lines.append(
                "  skipped (too many categories): " + ", ".join(self.skipped_categorical)
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()


def analyze(session: Session, max_categories: int = DEFAULT_MAX_CATEGORIES) -> AnalysisSummary:
    """Generate the initial exploratory figure set (step 3).

    Bar charts for low-cardinality categorical/boolean columns, box plots for every
    numeric column, and one feature-feature correlation heatmap -- each linear in
    feature count, unlike the pairwise plots :func:`plot_pair` makes on request.

    Requires repairs and transforms to already be approved: the plots are drawn from
    ``session.df`` in its natural units, and the approval check is only a "step 2 is
    finished" marker, not a data dependency of the plots themselves.
    """
    transform_gate = session.gates.get("transform_plan")
    if transform_gate is None or not transform_gate.answered:
        raise GateRequired(
            "repairs and transforms must be approved before analyzing.\n"
            "  call dsa.propose_transforms(s) and dsa.approve_transforms(s) first"
        )
    if session.df is None:
        raise ValueError("no data loaded; call a loader first")

    profile = profile_frame(session.df)

    with session.log.record(
        STEP, "viz.analyze", {"max_categories": max_categories}, session.df.shape
    ) as rec:
        bar_figures, skipped = categorical_bar_charts(session.df, profile, max_categories)
        box_figures = numeric_box_plots(session.df, profile)
        corr_figure = correlation_heatmap(session.df, profile)

        bar_paths = tuple(_save(session, f"bar_{name}", fig) for name, fig in bar_figures)
        box_paths = tuple(_save(session, f"box_{name}", fig) for name, fig in box_figures)
        corr_path = _save(session, "correlation_matrix", corr_figure) if corr_figure is not None else None

        rec.artifacts = [str(p) for p in (*bar_paths, *box_paths, *([corr_path] if corr_path else []))]
        rec.notes = (
            f"{len(bar_paths)} bar charts, {len(box_paths)} box plots, "
            f"{1 if corr_path else 0} correlation heatmap"
        )

    summary = AnalysisSummary(
        bar_charts=bar_paths,
        box_plots=box_paths,
        correlation=corr_path,
        skipped_categorical=tuple(skipped),
    )

    open_gate(
        session,
        key="figures",
        kind=REVIEW,
        question="Look over these figures. Ask for more with dsa.plot_pair(s, x, y), or say what should change.",
        step=STEP,
        context=summary.describe(),
    )
    return summary


def plot_pair(session: Session, x: str, y: str, reason: str = "") -> Path:
    """Plot one feature against another, on request (step 3 continued).

    The auto-generated batch in :func:`analyze` is deliberately linear in feature count;
    exhaustively plotting every pair would be exponential. This is the escape hatch for
    exactly the pair you want, after seeing the initial figures.

    Requires :func:`analyze` to have been called and the "figures" gate not yet closed --
    enforced by :func:`dsa.gates.revise`, reused here for its existing guard and logging.
    """
    revise(session, "figures", reason or f"plot {x} vs {y}")

    if session.df is None:
        raise ValueError("no data loaded; call a loader first")
    unknown = [name for name in (x, y) if name not in session.df.columns]
    if unknown:
        raise ValueError(f"no such column(s) in the working frame: {unknown}")

    profile = profile_frame(session.df)

    with session.log.record(STEP, "viz.plot_pair", {"x": x, "y": y}, session.df.shape) as rec:
        figure = pair_plot(session.df, profile, x, y)
        path = _save(session, f"pair_{x}_vs_{y}", figure)
        rec.artifacts = [str(path)]

    return path


def _save(session: Session, name: str, figure: Figure) -> Path:
    path = session.figures_dir / f"{name}.png"
    figure.savefig(path)
    return path
