"""Turning approved tier-2 proposals into an unfitted scikit-learn pipeline.

Nothing here touches the working frame. The result is a `ColumnTransformer` that has not
been fitted to anything; step 5 fits it inside each cross-validation fold, on that fold's
training rows only. That is what makes leakage structurally impossible rather than a
discipline someone has to remember (CLAUDE.md).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dsa.clean.plan import Proposal

# Treatments reported by :func:`column_treatments`, and the reason each column gets one.
PASSTHROUGH = "passthrough"
DROPPED = "dropped"


def _by_kind(transforms: tuple[Proposal, ...]) -> dict[str, Proposal]:
    """Index approved transforms by kind. At most one proposal per kind is produced."""
    return {proposal.kind: proposal for proposal in transforms}


def build_preprocessor(
    transforms: tuple[Proposal, ...], feature_columns: list[str]
) -> ColumnTransformer:
    """Compose approved transforms into an unfitted `ColumnTransformer`.

    ``feature_columns`` is every column that is not the target. Columns not covered by an
    approved transform are passed through untouched rather than silently dropped -- a
    column disappearing without anyone deciding it should would violate rule 9.
    """
    approved = _by_kind(transforms)

    numeric_columns = _union(approved, ("impute_numeric", "scale_numeric"), feature_columns)
    categorical_columns = _union(
        approved, ("impute_categorical", "onehot_categorical"), feature_columns
    )
    # Every kind whose whole purpose is to remove columns from the feature matrix.
    dropped: set[str] = set()
    for kind in ("drop_high_cardinality", "drop_datetime"):
        if kind in approved:
            dropped.update(approved[kind].columns)

    # A column named by both branches would be transformed twice; categorical wins,
    # because only the categorical branch can encode it into something a model accepts.
    numeric_columns = [c for c in numeric_columns if c not in categorical_columns]

    branches: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        steps: list[tuple[str, object]] = []
        if "impute_numeric" in approved:
            steps.append(
                ("impute", SimpleImputer(strategy=approved["impute_numeric"].params["strategy"]))
            )
        if "scale_numeric" in approved:
            steps.append(("scale", StandardScaler()))
        if steps:
            branches.append(("numeric", Pipeline(steps), numeric_columns))

    if categorical_columns:
        steps = []
        if "impute_categorical" in approved:
            steps.append(
                (
                    "impute",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=approved["impute_categorical"].params["fill_value"],
                    ),
                )
            )
        if "onehot_categorical" in approved:
            # handle_unknown="ignore" so a level unseen during training does not blow up
            # at predict time; sparse_output=False keeps the output inspectable.
            steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        if steps:
            branches.append(("categorical", Pipeline(steps), categorical_columns))

    transformers: list[tuple[str, object, list[str]]] = [
        (name, pipe, columns) for name, pipe, columns in branches
    ]
    if dropped:
        transformers.append(("dropped", "drop", sorted(dropped)))

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def column_treatments(
    transforms: tuple[Proposal, ...], feature_columns: list[str]
) -> pd.DataFrame:
    """Report what the pipeline will do to each feature column.

    Makes the preprocessing legible before anything is fitted, so an unintended
    passthrough or drop is visible at review time rather than discovered in step 6.
    """
    approved = _by_kind(transforms)
    rows = []
    for column in feature_columns:
        applied = [
            proposal.kind for proposal in transforms if column in proposal.columns
        ]
        if "drop_high_cardinality" in applied or "drop_datetime" in applied:
            treatment = DROPPED
        elif applied:
            treatment = " -> ".join(applied)
        else:
            treatment = PASSTHROUGH
        rows.append({"column": column, "treatment": treatment})
    return pd.DataFrame(rows)


def _union(
    approved: dict[str, Proposal], kinds: tuple[str, ...], feature_columns: list[str]
) -> list[str]:
    """Columns named by any of ``kinds``, in the order they appear in the frame."""
    named: set[str] = set()
    for kind in kinds:
        if kind in approved:
            named.update(approved[kind].columns)
    return [column for column in feature_columns if column in named]
