"""Step 2a: describing the data.

Profiling answers the questions a data scientist asks in the first five minutes -- how
big is it, what is missing, what is constant, what is secretly a string -- and produces
the evidence that the cleaning proposals in :mod:`dsa.clean` are argued from.

Nothing here modifies data. Profiling is pure observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as pdt

from dsa.session import Session

STEP = 2

# Column kinds drive both the proposals and the eventual preprocessing pipeline.
NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"
BOOLEAN = "boolean"
OTHER = "other"


def column_kind(series: pd.Series) -> str:
    """Classify a column for downstream handling.

    Under pandas 3 a text column has dtype ``str``, not ``object``, so this dispatches on
    the pandas type helpers rather than comparing against ``object`` (see CLAUDE.md).
    Order matters: booleans and datetimes are numeric-ish to some checks, so they are
    tested first.
    """
    if pdt.is_bool_dtype(series):
        return BOOLEAN
    if pdt.is_datetime64_any_dtype(series):
        return DATETIME
    if pdt.is_numeric_dtype(series):
        return NUMERIC
    if isinstance(series.dtype, pd.CategoricalDtype) or pdt.is_string_dtype(series):
        return CATEGORICAL
    return OTHER


@dataclass(frozen=True)
class ColumnProfile:
    """What is known about one column."""

    name: str
    dtype: str
    kind: str
    n_missing: int
    pct_missing: float
    n_unique: int
    examples: tuple[Any, ...]  # up to three non-null values, for making evidence concrete
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def is_constant(self) -> bool:
        """A column carrying no information. Missing values do not count as variation."""
        return self.n_unique <= 1


@dataclass(frozen=True)
class DataProfile:
    """What is known about the frame as a whole."""

    n_rows: int
    n_cols: int
    memory_mb: float
    n_duplicate_rows: int
    columns: tuple[ColumnProfile, ...]

    def __getitem__(self, name: str) -> ColumnProfile:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def of_kind(self, kind: str) -> tuple[ColumnProfile, ...]:
        return tuple(c for c in self.columns if c.kind == kind)

    def to_frame(self) -> pd.DataFrame:
        """Tabular view, for display in a notebook."""
        return pd.DataFrame(
            [
                {
                    "column": c.name,
                    "dtype": c.dtype,
                    "kind": c.kind,
                    "missing": c.n_missing,
                    "missing_%": round(c.pct_missing, 2),
                    "unique": c.n_unique,
                    "examples": ", ".join(str(v) for v in c.examples),
                }
                for c in self.columns
            ]
        )

    def describe(self) -> str:
        """Compact text summary, used as gate context and in the write-up."""
        header = (
            f"{self.n_rows:,} rows x {self.n_cols} columns, {self.memory_mb:.1f} MB"
            f" | {self.n_duplicate_rows:,} duplicate rows"
        )
        missing = [c for c in self.columns if c.n_missing]
        constant = [c for c in self.columns if c.is_constant]
        lines = [header]
        if missing:
            worst = sorted(missing, key=lambda c: -c.pct_missing)[:5]
            lines.append(
                "  missing: "
                + ", ".join(f"{c.name} ({c.pct_missing:.1f}%)" for c in worst)
                + (f" and {len(missing) - len(worst)} more" if len(missing) > len(worst) else "")
            )
        if constant:
            lines.append("  constant: " + ", ".join(c.name for c in constant))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()


def profile_frame(frame: pd.DataFrame) -> DataProfile:
    """Profile a frame. Pure: no session, no logging, no mutation."""
    n_rows = len(frame)
    columns = []

    for name in frame.columns:
        series = frame[name]
        n_missing = int(series.isna().sum())
        non_null = series.dropna()
        kind = column_kind(series)

        stats: dict[str, Any] = {}
        if kind == NUMERIC and not non_null.empty:
            stats = {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
            }
        elif kind == CATEGORICAL and not non_null.empty:
            counts = non_null.value_counts()
            stats = {"top": str(counts.index[0]), "top_freq": int(counts.iloc[0])}

        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                kind=kind,
                n_missing=n_missing,
                pct_missing=100.0 * n_missing / n_rows if n_rows else 0.0,
                n_unique=int(series.nunique(dropna=True)),
                examples=tuple(non_null.head(3).tolist()),
                stats=stats,
            )
        )

    return DataProfile(
        n_rows=n_rows,
        n_cols=frame.shape[1],
        memory_mb=float(frame.memory_usage(deep=True).sum()) / 1_048_576,
        n_duplicate_rows=int(frame.duplicated().sum()),
        columns=tuple(columns),
    )


def profile(session: Session) -> DataProfile:
    """Profile the session's working frame and record it (step 2)."""
    if session.df is None:
        raise ValueError("no data loaded; call a loader first")

    with session.log.record(STEP, "profile.compute", input_shape=session.df.shape) as rec:
        result = profile_frame(session.df)
        rec.output_shape = session.df.shape
        rec.notes = result.describe().splitlines()[0]
    return result
