"""Step 4b: running dsa.features.recommend against a Session, and recording your
feature-engineering/selection/dimensionality-reduction decisions.

Saves nothing to disk (no figures here) -- logs each operation and manages the
"features" review gate, the same relationship dsa.clean.proposals has to
dsa.clean.detect.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from dsa.features.recommend import FeatureRecommendations
from dsa.features.recommend import recommend_features as _recommend_features
from dsa.gates import REVIEW, GateRequired, open_gate, revise
from dsa.profile import profile_frame
from dsa.session import Session

STEP = 4


def propose_features(session: Session) -> pd.DataFrame:
    """Start step 4 (opens the "features" review gate).

    Requires repairs and transforms already approved (same guard :func:`dsa.analyze`
    uses) -- feature work reasons about the cleaned frame, not a mid-repair one.
    Initializes ``session.feature_df`` from ``session.df`` and returns it for a first
    look.
    """
    transform_gate = session.gates.get("transform_plan")
    if transform_gate is None or not transform_gate.answered:
        raise GateRequired(
            "repairs and transforms must be approved before proposing features.\n"
            "  call dsa.propose_transforms(s) and dsa.approve_transforms(s) first"
        )
    if session.df is None:
        raise ValueError("no data loaded; call a loader first")

    with session.log.record(STEP, "features.propose", {}, session.df.shape) as rec:
        frame = session.rebuild_features()
        rec.output_shape = frame.shape

    open_gate(
        session,
        key="features",
        kind=REVIEW,
        question="Engineer, select, and optionally reduce features. Proceed when finished.",
        step=STEP,
        context=f"{frame.shape[1]} columns available: {', '.join(str(c) for c in frame.columns)}",
    )
    return frame


def engineer_feature(
    session: Session, name: str, fn: Callable[[pd.DataFrame], pd.Series], reason: str = ""
) -> pd.DataFrame:
    """Add a deterministic derived column (step 4 continued).

    ``fn`` receives the current feature frame and returns the new column's values,
    e.g. ``dsa.engineer_feature(s, "family_size", lambda df: df["SibSp"] + df["Parch"] + 1)``.

    Deliberately an arbitrary callable, not a fixed kind vocabulary like repairs/
    transforms: feature engineering is inherently open-ended (title-from-name, binning,
    log-transforms, whatever the domain calls for) in a way a small enum can't cover.
    Only fixed-formula functions belong here -- anything needing fitted statistics (e.g.
    target encoding) would leak if computed this way and isn't supported.

    Reuses :func:`dsa.gates.revise`'s guard (requires :func:`propose_features` called,
    gate still open) and round logging, same pattern as
    :func:`dsa.propose_manual_repair`. Raises if ``name`` collides with an existing
    column. The new step is validated against the full replay before anything on
    ``session`` is changed, so a broken ``fn`` never leaves ``session.engineered`` in a
    half-applied state.
    """
    revise(session, "features", reason or f"engineer {name}")

    if name in session.feature_df.columns:
        raise ValueError(f"{name!r} already exists as a column; choose a different name")

    def step(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(**{name: fn(frame)})

    candidate_engineered = [*session.engineered, (name, step)]
    frame = session.df
    for _name, engineer in candidate_engineered:
        frame = engineer(frame)

    with session.log.record(STEP, "features.engineer", {"name": name}, session.feature_df.shape) as rec:
        session.engineered = candidate_engineered
        session.feature_df = frame
        rec.output_shape = frame.shape

    return frame


def recommend_features(session: Session) -> FeatureRecommendations:
    """Text-only signals about the current feature frame -- near-zero variance,
    collinear pairs, weak target association (see
    :func:`dsa.features.recommend.recommend_features` for what each means).

    No gate required, same ungated category as :func:`dsa.plot_missingness`/
    :func:`dsa.describe_dataset` -- callable any time, including before
    :func:`propose_features` (falls back to ``session.df``), and again after each
    :func:`engineer_feature` call to see how the picture changed.
    """
    frame = session.feature_df if session.feature_df is not None else session.df
    if frame is None:
        raise ValueError("no data loaded; call a loader first")

    profile = profile_frame(frame)
    with session.log.record(STEP, "features.recommend", {}, frame.shape):
        recommendations = _recommend_features(frame, profile, session.target)
    return recommendations


def select_features(
    session: Session, exclude: tuple[str, ...] = (), include: tuple[str, ...] | None = None
) -> Session:
    """Record the final feature set (step 4 continued).

    ``include``, if given, is the exact set (validated to exist in the feature frame);
    otherwise every feature-frame column except the target and anything named in
    ``exclude``. Replaces the previous selection rather than accumulating -- changing
    your mind and re-calling is safe, the same re-run guarantee
    :func:`dsa.approve_repairs`/:func:`dsa.approve_transforms` already give.

    This only *records* the choice in ``session.selected_features`` -- it does not
    remove columns from ``session.feature_df``.
    """
    revise(session, "features", "select features")

    frame = session.feature_df
    if frame is None:
        raise ValueError("no feature frame; call dsa.propose_features(s) first")

    if include is not None:
        unknown = [name for name in include if name not in frame.columns]
        if unknown:
            raise ValueError(f"no such column(s) in the feature frame: {unknown}")
        selected = tuple(include)
    else:
        unknown = [name for name in exclude if name not in frame.columns]
        if unknown:
            raise ValueError(f"no such column(s) in the feature frame: {unknown}")
        selected = tuple(c for c in frame.columns if c != session.target and c not in exclude)

    with session.log.record(STEP, "features.select", {"selected": list(selected)}, frame.shape):
        session.selected_features = selected

    return session


def reduce_dimensions(session: Session, n_components: int, reason: str = "") -> Session:
    """Record a PCA choice (step 4 continued).

    Only records ``session.pca_components`` -- does not fit a ``PCA`` or build a
    ``Pipeline``. PCA needs already-numeric, already-imputed/scaled/encoded input, which
    only exists once a full training pipeline is assembled; that composition belongs to
    a later step, not this one.
    """
    revise(session, "features", reason or f"reduce to {n_components} components")

    if n_components < 1:
        raise ValueError(f"n_components must be at least 1; got {n_components}")

    with session.log.record(STEP, "features.reduce_dimensions", {"n_components": n_components}):
        session.pca_components = n_components

    return session
