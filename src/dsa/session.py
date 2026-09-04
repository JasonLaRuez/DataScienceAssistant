"""The Session object.

Every operation in the toolkit takes a :class:`Session` as its first argument (CLAUDE.md
rule 4: notebooks import and display, they hold no state of their own). Making the state
explicit rather than a module-level global means two datasets can be worked on side by
side, a partially-restarted kernel is never ambiguous, and every function is testable
without setup.

The two-tier data model lives here:

* ``raw``     -- the frame exactly as loaded. Never modified.
* ``repairs`` -- approved tier-1 repairs (dtype coercions, duplicate drops). Not learned
                 from the data, so applying them cannot leak information.
* ``df``      -- the working frame, produced by replaying ``repairs`` against ``raw``.

Because ``df`` is rebuilt from ``raw`` whenever the repair list changes, revising a
decision never requires re-running the notebook from the top, and a repair can never be
applied twice by a re-run cell.

Tier-2 learned transforms (imputation, scaling, encoding, selection) are deliberately
absent from this object: they are unfitted Pipeline steps fitted inside each CV fold, so
that leakage is structurally impossible rather than a thing to remember.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

from dsa.gates import Gate
from dsa.runlog import RunLog, environment_snapshot

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from dsa.clean.plan import Proposal, RepairPlan, TransformPlan

# A tier-1 repair is a named, deterministic frame -> frame function. Keeping the shape
# this simple is what makes replaying the whole list cheap and order-explicit.
Repair = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class Session:
    """Everything one analysis run knows about itself."""

    run_id: str
    run_dir: Path
    project_root: Path
    log: RunLog
    seed: int = 0

    # --- data -------------------------------------------------------------------
    raw: pd.DataFrame | None = None
    df: pd.DataFrame | None = None
    source: str | None = None  # provenance string, e.g. a Kaggle dataset slug

    # --- problem definition (set only through gates, never inferred) -------------
    target: str | None = None
    task: str | None = None  # "classification" | "regression"
    groups: str | None = None  # column defining groups for group-aware splitting

    # --- accumulated decisions ---------------------------------------------------
    repairs: list[tuple[str, Repair]] = field(default_factory=list)
    gates: dict[str, Gate] = field(default_factory=dict)

    # The most recent proposal set for each phase, and the tier-2 transforms approved
    # from the second. Kept separate because a repair plan and a transform plan are
    # proposed and reviewed at different times (repairs first, transforms only after).
    repair_plan: RepairPlan | None = None
    transform_plan: TransformPlan | None = None
    transforms: tuple[Proposal, ...] = ()

    # --- step 4: feature selection and engineering --------------------------------
    # feature_df is one layer above df the same way df is one layer above raw: replaying
    # engineered against df, not against raw, because engineering builds on the cleaned
    # frame. selected_features/pca_components are recorded decisions, not applied data --
    # nothing here mutates feature_df.
    feature_df: pd.DataFrame | None = None
    engineered: list[tuple[str, Repair]] = field(default_factory=list)
    selected_features: tuple[str, ...] | None = None
    pca_components: int | None = None

    @property
    def figures_dir(self) -> Path:
        """Where exploratory figures are written. Gitignored; only figures the write-up
        references are later copied into ``reports/figures/``."""
        path = self.run_dir / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def rebuild(self) -> pd.DataFrame:
        """Replay the approved repairs against ``raw`` to produce the working frame."""
        if self.raw is None:
            raise ValueError("no data loaded; call a loader first")
        frame = self.raw
        for _name, repair in self.repairs:
            frame = repair(frame)
        self.df = frame
        return frame

    def rebuild_features(self) -> pd.DataFrame:
        """Replay engineered features against ``df`` to produce the feature frame.

        One layer above :meth:`rebuild`, same reasoning: revising an earlier engineered
        feature and re-running never compounds, because this always starts fresh from
        ``df`` rather than building on the previous ``feature_df``.
        """
        if self.df is None:
            raise ValueError("no data loaded; call a loader first")
        frame = self.df
        for _name, engineer in self.engineered:
            frame = engineer(frame)
        self.feature_df = frame
        return frame

    def summary(self) -> str:
        """Short human-readable status, for display at the top of a notebook."""
        if self.raw is None:
            data = "no data loaded"
        else:
            rows, cols = self.raw.shape
            data = f"{rows:,} rows x {cols} cols"
            if self.df is not None and self.df.shape != self.raw.shape:
                data += f" -> working {self.df.shape[0]:,} x {self.df.shape[1]}"
        open_gates = [g.key for g in self.gates.values() if not g.answered]
        return "\n".join(
            [
                f"run {self.run_id}  (seed {self.seed})",
                f"  source : {self.source or '-'}",
                f"  data   : {data}",
                f"  target : {self.target or '- (not set)'}",
                f"  task   : {self.task or '- (not set)'}",
                f"  repairs: {len(self.repairs)} approved",
                f"  gates  : {len(open_gates)} open" + (f" -> {', '.join(open_gates)}" if open_gates else ""),
                f"  log    : {self.log.path}",
            ]
        )

    def __repr__(self) -> str:  # notebooks display the repr of a bare `s` cell
        return self.summary()


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` looking for pyproject.toml.

    Notebooks run with their own directory as the working directory, so resolving the
    root explicitly is what keeps ``runs/`` in one place instead of scattering it.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def new_session(project_root: Path | str | None = None, seed: int = 0) -> Session:
    """Start a run: create ``runs/<run_id>/``, snapshot the environment, open the log."""
    root = Path(project_root) if project_root else find_project_root()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Written before anything else so that a run which crashes immediately is still
    # reproducible from what was recorded.
    snapshot = environment_snapshot(seed)
    (run_dir / "env.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    session = Session(
        run_id=run_id,
        run_dir=run_dir,
        project_root=root,
        log=RunLog(run_dir / "run.jsonl"),
        seed=seed,
    )
    with session.log.record(0, "session.start", {"seed": seed, "python": snapshot["python"]}):
        pass
    return session
