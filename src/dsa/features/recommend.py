"""Step 4a: detecting signals worth mentioning about the current feature set.

Every function here takes a frame and its already-computed DataProfile and returns
text -- not a Proposal, not a Plan. Nothing here is meant to be accepted or rejected by
id; dsa.select_features/dsa.engineer_feature are the actual decision points, and you're
free to ignore any of this. No session, no gates -- same shape as dsa.clean.detect and
dsa.viz.figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dsa.profile import BOOLEAN, CATEGORICAL, NUMERIC, DataProfile
from dsa.viz.figures import cramers_v

_DISCRETE_KINDS = (CATEGORICAL, BOOLEAN)

# A numeric column this flat relative to its own scale carries little signal.
NEAR_ZERO_VARIANCE_CV = 0.01  # coefficient of variation (std / |mean|) below this

# A categorical/boolean column this dominated by a single value carries little signal.
NEAR_ZERO_VARIANCE_DOMINANCE = 0.99  # single most-frequent value's share of rows

# Two numeric columns (Pearson), or two categorical/boolean columns (Cramer's V), this
# strongly related are largely redundant with each other.
COLLINEARITY_THRESHOLD = 0.9

# A feature this weakly related to the target is unlikely to help predict it.
WEAK_TARGET_ASSOCIATION = 0.05


@dataclass(frozen=True)
class FeatureRecommendations:
    """Plain-text signals about the current feature set -- not a plan. Nothing here is
    applied or needs to be accepted/rejected by id; read it, then decide with
    dsa.select_features()/dsa.engineer_feature() directly."""

    near_zero_variance: tuple[str, ...]
    collinear_pairs: tuple[tuple[str, str, float], ...]
    weak_target_association: tuple[tuple[str, float], ...]

    def describe(self) -> str:
        lines = []
        if self.near_zero_variance:
            lines.append("near-zero variance: " + ", ".join(self.near_zero_variance))
        if self.collinear_pairs:
            lines.append(
                "collinear pairs: "
                + "; ".join(f"{a} <-> {b} ({v:.2f})" for a, b, v in self.collinear_pairs)
            )
        if self.weak_target_association:
            lines.append(
                "weak target association: "
                + ", ".join(f"{c} ({v:.2f})" for c, v in self.weak_target_association)
            )
        return "\n".join(lines) if lines else "No signals worth flagging."

    def _repr_markdown_(self) -> str:
        if not (self.near_zero_variance or self.collinear_pairs or self.weak_target_association):
            return "No signals worth flagging."
        parts = ["### Feature recommendations (informational -- nothing here is applied)"]
        if self.near_zero_variance:
            parts.append(
                "**Near-zero variance** (little signal by itself):\n"
                + "\n".join(f"- {c}" for c in self.near_zero_variance)
            )
        if self.collinear_pairs:
            parts.append(
                "**Highly collinear pairs** (consider keeping only one):\n"
                + "\n".join(f"- {a} <-> {b} ({v:.2f})" for a, b, v in self.collinear_pairs)
            )
        if self.weak_target_association:
            parts.append(
                "**Weak target association**:\n"
                + "\n".join(f"- {c} ({v:.2f})" for c, v in self.weak_target_association)
            )
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return self.describe()


def recommend_features(
    frame: pd.DataFrame, profile: DataProfile, target: str | None
) -> FeatureRecommendations:
    """Text-only signals about ``frame``'s columns: near-zero variance, collinear
    pairs, weak target association. Mixed numeric/categorical pairs are not checked --
    that needs a correlation-ratio/ANOVA-style measure this does not attempt.
    """
    features = [c for c in profile.columns if c.name != target]

    near_zero_set = {c.name for c in features if _is_near_zero_variance(frame[c.name], c.kind)}
    near_zero = tuple(sorted(near_zero_set))

    # A column already flagged near-zero-variance is excluded from collinearity/target-
    # association checks below: correlating a near-constant column is both meaningless
    # (it carries no signal to be "collinear" or "associated" with anything) and, for a
    # genuinely constant one, mathematically undefined (0/0).
    numeric = [c.name for c in features if c.kind == NUMERIC and c.name not in near_zero_set]
    discrete = [
        c.name for c in features if c.kind in _DISCRETE_KINDS and c.name not in near_zero_set
    ]

    collinear: list[tuple[str, str, float]] = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1 :]:
            r = frame[a].corr(frame[b])
            if pd.notna(r) and abs(r) > COLLINEARITY_THRESHOLD:
                collinear.append((a, b, abs(float(r))))
    for i, a in enumerate(discrete):
        for b in discrete[i + 1 :]:
            v = cramers_v(frame[a], frame[b])
            if pd.notna(v) and v > COLLINEARITY_THRESHOLD:
                collinear.append((a, b, float(v)))
    collinear.sort(key=lambda item: -item[2])

    weak: list[tuple[str, float]] = []
    if target is not None and target in frame.columns:
        target_kind = profile[target].kind
        if target_kind == NUMERIC:
            for c in numeric:
                r = frame[c].corr(frame[target])
                if pd.notna(r) and abs(r) < WEAK_TARGET_ASSOCIATION:
                    weak.append((c, abs(float(r))))
        elif target_kind in _DISCRETE_KINDS:
            for c in discrete:
                v = cramers_v(frame[c], frame[target])
                if pd.notna(v) and v < WEAK_TARGET_ASSOCIATION:
                    weak.append((c, float(v)))
    weak.sort(key=lambda item: item[1])

    return FeatureRecommendations(
        near_zero_variance=near_zero,
        collinear_pairs=tuple(collinear),
        weak_target_association=tuple(weak),
    )


def _is_near_zero_variance(series: pd.Series, kind: str) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    if kind == NUMERIC:
        mean = values.mean()
        if mean == 0:
            return False
        cv = values.std() / abs(mean)
        return bool(pd.notna(cv) and cv < NEAR_ZERO_VARIANCE_CV)
    if kind in _DISCRETE_KINDS:
        dominance = values.value_counts(normalize=True).iloc[0]
        return bool(dominance > NEAR_ZERO_VARIANCE_DOMINANCE)
    return False
