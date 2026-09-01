"""Detecting what needs cleaning.

Every detector returns proposals carrying counts and concrete example values, because a
proposal without evidence cannot be approved on any basis but trust.

Detectors observe; they never modify the frame.
"""

from __future__ import annotations

import warnings

import pandas as pd
from pandas.api import types as pdt

from dsa.clean.plan import REPAIR, TRANSFORM, Plan, Proposal
from dsa.clean.repairs import build_repair
from dsa.profile import CATEGORICAL, DATETIME, NUMERIC, DataProfile, profile_frame

# A string column is worth coercing only if nearly all of it parses. Below this, the
# failures are the story and coercing would quietly manufacture nulls.
COERCE_THRESHOLD = 0.90

# Above this many distinct values, one-hot encoding stops being reasonable.
HIGH_CARDINALITY = 20

# A column this close to unique-per-row is identifying rows rather than describing them.
# A ratio rather than exact equality, because duplicate rows pending removal would
# otherwise hide an identifier behind a handful of repeated values.
IDENTIFIER_RATIO = 0.99

# Below this many rows, near-uniqueness means nothing: in a 10-row frame almost any
# column is unique per row.
IDENTIFIER_MIN_ROWS = 20

# Parsing is tried on a sample; parsing a whole 100k-row column purely to decide whether
# to *propose* something is not worth the wait.
SAMPLE_SIZE = 500


def _examples(values: pd.Series, limit: int = 4) -> str:
    """Render a few distinct values with their frequency, for use as evidence."""
    counts = values.value_counts().head(limit)
    return ", ".join(f"{value!r} (x{count})" for value, count in counts.items())


def _parse_rate(series: pd.Series, parser) -> tuple[float, pd.Series]:
    """Fraction of a sample that ``parser`` converts, plus the values that failed."""
    non_null = series.dropna()
    if non_null.empty:
        return 0.0, non_null
    sample = non_null.sample(min(SAMPLE_SIZE, len(non_null)), random_state=0)
    with warnings.catch_warnings():
        # Parsing deliberately-unparseable columns is the normal case here.
        warnings.simplefilter("ignore")
        parsed = parser(sample)
    failed = sample[parsed.isna()]
    return 1.0 - (len(failed) / len(sample)), failed


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _looks_like_a_date(series: pd.Series) -> bool:
    """Whether ``series`` would itself be proposed for datetime coercion.

    Every calendar date is near-unique by construction -- one value per day -- so without
    this check the identifier heuristic below would condemn almost any date-like string
    column before the coercion/extraction detectors ever saw it.
    """
    rate, _failures = _parse_rate(series, _to_datetime)
    return rate >= COERCE_THRESHOLD


def detect(frame: pd.DataFrame, profile: DataProfile, target: str | None = None) -> Plan:
    """Build the full set of proposals for a frame.

    Detection runs in two phases, because the two tiers disagree about what the data is.
    A string column proposed for coercion to numeric is *not* a high-cardinality
    categorical column, even though that is exactly what it looks like right now. So the
    proposed repairs are applied to a throwaway copy, the result is re-profiled, and
    transforms are detected against that hypothetical frame.

    The consequence, and it is a real one: rejecting a repair invalidates the transforms
    that were derived assuming it. Re-run ``propose`` after dropping a repair.
    """
    repairs = _detect_repairs(frame, profile, target)

    hypothetical = frame
    for proposal in repairs:
        _name, repair = build_repair(proposal)
        hypothetical = repair(hypothetical)

    transforms = _detect_transforms(profile_frame(hypothetical), target)
    return Plan(repairs=tuple(repairs), transforms=tuple(transforms))


def _detect_repairs(
    frame: pd.DataFrame, profile: DataProfile, target: str | None
) -> list[Proposal]:
    """Tier 1: deterministic fixes, argued from the frame as it stands."""
    repairs: list[Proposal] = []
    counter = {"R": 0}

    def next_id(prefix: str) -> str:
        counter[prefix] += 1
        return f"{prefix}{counter[prefix]}"

    # Columns carrying no information are identified first so that later detectors do not
    # raise further proposals about columns already condemned.
    dead = _dead_columns(profile)

    if profile.n_duplicate_rows:
        pct = 100.0 * profile.n_duplicate_rows / profile.n_rows
        repairs.append(
            Proposal(
                id=next_id("R"),
                tier=REPAIR,
                kind="drop_duplicate_rows",
                columns=(),
                summary=f"drop {profile.n_duplicate_rows:,} exact duplicate rows",
                evidence=(
                    f"{profile.n_duplicate_rows:,} of {profile.n_rows:,} rows "
                    f"({pct:.1f}%) are exact duplicates"
                ),
                consequence=f"{profile.n_rows - profile.n_duplicate_rows:,} rows remain",
                alternatives=("keep them, if repeated measurements are meaningful here",),
            )
        )

    for column in profile.columns:
        if profile.n_rows and column.n_missing == profile.n_rows:
            repairs.append(
                Proposal(
                    id=next_id("R"),
                    tier=REPAIR,
                    kind="drop_columns",
                    columns=(column.name,),
                    summary=f"drop {column.name!r}: entirely empty",
                    evidence=f"all {profile.n_rows:,} values are missing",
                    consequence="column removed; no information is lost",
                    params={"columns": [column.name]},
                )
            )
        elif column.is_constant:
            value = column.examples[0] if column.examples else "<all missing>"
            repairs.append(
                Proposal(
                    id=next_id("R"),
                    tier=REPAIR,
                    kind="drop_columns",
                    columns=(column.name,),
                    summary=f"drop {column.name!r}: constant",
                    evidence=(
                        f"a single distinct value ({value!r}) across {profile.n_rows:,} rows"
                    ),
                    consequence="column removed; it cannot contribute to any model",
                    params={"columns": [column.name]},
                )
            )

    # Uniqueness is judged against the row count *after* duplicates are removed. A frame
    # with 6 duplicate rows makes a perfect identifier look 97% unique, which would hide
    # it behind the threshold below.
    effective_rows = profile.n_rows - profile.n_duplicate_rows

    for column in profile.columns:
        if column.name in dead or column.name == target:
            continue
        # A column unique in (almost) every row identifies rows rather than describing
        # them -- but only for types where that is surprising. A continuous float is
        # near-unique by nature, so measurements like income or latitude must not be
        # mistaken for identifiers; neither is a calendar date, which belongs to the
        # coercion/extraction detectors instead (see _looks_like_a_date).
        if (
            effective_rows >= IDENTIFIER_MIN_ROWS
            and column.n_unique >= IDENTIFIER_RATIO * effective_rows
            and column.kind in (NUMERIC, CATEGORICAL)
            and not pdt.is_float_dtype(frame[column.name])
            and not (column.kind == CATEGORICAL and _looks_like_a_date(frame[column.name]))
        ):
            repairs.append(
                Proposal(
                    id=next_id("R"),
                    tier=REPAIR,
                    kind="drop_columns",
                    columns=(column.name,),
                    summary=f"drop {column.name!r}: looks like an identifier",
                    evidence=(
                        f"{column.n_unique:,} distinct values across {effective_rows:,} "
                        "rows - essentially one per row"
                    ),
                    consequence="column removed; row identifiers do not generalise",
                    alternatives=(
                        "keep it, if the value is genuinely informative rather than a row id",
                    ),
                    params={"columns": [column.name]},
                )
            )

    # Columns already proposed for removal must not also attract a coercion proposal:
    # two contradictory proposals about one column, where whichever runs first silently
    # neutralises the other, is worse than either alone.
    condemned = dead | {
        name
        for proposal in repairs
        if proposal.kind == "drop_columns"
        for name in proposal.params["columns"]
    }

    for column in profile.of_kind(CATEGORICAL):
        if column.name in condemned or column.name == target:
            continue
        series = frame[column.name]

        numeric_rate, numeric_failures = _parse_rate(series, _to_numeric)
        if numeric_rate >= COERCE_THRESHOLD:
            repairs.append(
                Proposal(
                    id=next_id("R"),
                    tier=REPAIR,
                    kind="coerce_numeric",
                    columns=(column.name,),
                    summary=f"coerce {column.name!r}: {column.dtype} -> numeric",
                    evidence=_coercion_evidence(numeric_rate, numeric_failures, "numbers"),
                    consequence=(
                        "unparseable values become missing"
                        if len(numeric_failures)
                        else "no values are lost"
                    ),
                    alternatives=(
                        "treat as categorical",
                        "extract the numeric part with a custom rule",
                    ),
                    params={"column": column.name},
                )
            )
            continue  # a column being read as numeric is not also a date

        date_rate, date_failures = _parse_rate(series, _to_datetime)
        if date_rate >= COERCE_THRESHOLD:
            repairs.append(
                Proposal(
                    id=next_id("R"),
                    tier=REPAIR,
                    kind="coerce_datetime",
                    columns=(column.name,),
                    summary=f"coerce {column.name!r}: {column.dtype} -> datetime",
                    evidence=_coercion_evidence(date_rate, date_failures, "dates"),
                    consequence="unparseable values become missing",
                    alternatives=("treat as categorical",),
                    params={"column": column.name},
                )
            )

    if target and target in frame.columns:
        missing_target = int(frame[target].isna().sum())
        if missing_target:
            repairs.append(
                Proposal(
                    id=next_id("R"),
                    tier=REPAIR,
                    kind="drop_rows_missing_target",
                    columns=(target,),
                    summary=f"drop {missing_target:,} rows with no {target!r} value",
                    evidence=f"{missing_target:,} rows are missing the target",
                    consequence=(
                        f"{profile.n_rows - missing_target:,} rows remain; "
                        "unlabelled rows cannot be trained on"
                    ),
                    alternatives=("keep them for semi-supervised use (out of scope in v1)",),
                    params={"column": target},
                )
            )

    return repairs


def _detect_transforms(profile: DataProfile, target: str | None) -> list[Proposal]:
    """Tier 2: learned preprocessing, argued from the frame *as it will be* after repairs.

    Specified, never applied. Each of these becomes an unfitted pipeline step.
    """
    transforms: list[Proposal] = []
    counter = {"T": 0}

    def next_id(prefix: str) -> str:
        counter[prefix] += 1
        return f"{prefix}{counter[prefix]}"

    features = [c for c in profile.columns if c.name != target]

    numeric = [c for c in features if c.kind == NUMERIC]
    categorical = [c for c in features if c.kind == CATEGORICAL]

    numeric_missing = [c for c in numeric if c.n_missing]
    if numeric_missing:
        transforms.append(
            Proposal(
                id=next_id("T"),
                tier=TRANSFORM,
                kind="impute_numeric",
                columns=tuple(c.name for c in numeric_missing),
                summary=f"impute {len(numeric_missing)} numeric column(s) with the median",
                evidence=_missingness_evidence(numeric_missing),
                consequence=(
                    "fitted per CV fold on training rows only, so no test information leaks"
                ),
                alternatives=("mean", "a constant sentinel", "iterative imputation", "drop the rows"),
                params={"strategy": "median"},
            )
        )

    categorical_missing = [c for c in categorical if c.n_missing]
    if categorical_missing:
        transforms.append(
            Proposal(
                id=next_id("T"),
                tier=TRANSFORM,
                kind="impute_categorical",
                columns=tuple(c.name for c in categorical_missing),
                summary=f"fill {len(categorical_missing)} categorical column(s) with 'missing'",
                evidence=_missingness_evidence(categorical_missing),
                consequence="absence becomes its own category rather than being guessed at",
                alternatives=("most frequent value", "drop the rows"),
                params={"fill_value": "missing"},
            )
        )

    if numeric:
        transforms.append(
            Proposal(
                id=next_id("T"),
                tier=TRANSFORM,
                kind="scale_numeric",
                columns=tuple(c.name for c in numeric),
                summary=f"standardise {len(numeric)} numeric column(s)",
                evidence="; ".join(_range_of(c) for c in numeric[:5]),
                consequence="zero mean, unit variance; needed by linear models, ignored by trees",
                alternatives=("min-max scaling", "robust scaling if outliers dominate", "no scaling"),
            )
        )

    low_card = [c for c in categorical if c.n_unique <= HIGH_CARDINALITY]
    if low_card:
        widened = sum(c.n_unique for c in low_card)
        transforms.append(
            Proposal(
                id=next_id("T"),
                tier=TRANSFORM,
                kind="onehot_categorical",
                columns=tuple(c.name for c in low_card),
                summary=f"one-hot encode {len(low_card)} categorical column(s)",
                evidence="; ".join(f"{c.name} ({c.n_unique} levels)" for c in low_card[:5]),
                consequence=(
                    f"{len(low_card)} columns become roughly {widened}; "
                    "levels unseen at predict time are ignored"
                ),
                alternatives=("ordinal encoding", "target encoding (needs care to avoid leakage)"),
            )
        )

    high_card = [c for c in categorical if c.n_unique > HIGH_CARDINALITY]
    if high_card:
        transforms.append(
            Proposal(
                id=next_id("T"),
                tier=TRANSFORM,
                kind="drop_high_cardinality",
                columns=tuple(c.name for c in high_card),
                summary=f"drop {len(high_card)} high-cardinality column(s)",
                evidence="; ".join(f"{c.name} ({c.n_unique:,} levels)" for c in high_card[:5]),
                consequence="one-hot encoding these would add a very large number of columns",
                alternatives=(
                    "target encoding",
                    "group rare levels into 'other'",
                    "keep them for a tree model",
                ),
            )
        )

    datetimes = [c for c in features if c.kind == DATETIME]
    if datetimes:
        transforms.append(
            Proposal(
                id=next_id("T"),
                tier=TRANSFORM,
                kind="extract_datetime_parts",
                columns=tuple(c.name for c in datetimes),
                summary=f"extract year/month/day/day-of-week from {len(datetimes)} datetime column(s)",
                evidence="; ".join(f"{c.name} ({c.n_unique:,} distinct timestamps)" for c in datetimes[:5]),
                consequence=(
                    f"{len(datetimes)} datetime column(s) replaced by "
                    f"{4 * len(datetimes)} integer part column(s); a missing timestamp "
                    "produces -1 for all of its parts"
                ),
                alternatives=(
                    "drop the column(s) instead",
                    "also encode month / day-of-week cyclically (sin/cos) for linear models",
                    "derive an elapsed-time feature relative to a reference date in step 4",
                ),
            )
        )

    return transforms


def _coercion_evidence(rate: float, failures: pd.Series, noun: str) -> str:
    text = f"{rate:.1%} of sampled values parse as {noun}"
    if len(failures):
        text += f"; failures: {_examples(failures)}"
    return text


def _missingness_evidence(columns) -> str:
    shown = "; ".join(f"{c.name} missing {c.pct_missing:.1f}%" for c in columns[:5])
    if len(columns) > 5:
        shown += f"; and {len(columns) - 5} more"
    return shown


def _range_of(column) -> str:
    low = column.stats.get("min")
    high = column.stats.get("max")
    if low is None or high is None:
        return f"{column.name} (no range: all missing)"
    return f"{column.name} range [{low:.3g}, {high:.3g}]"


def _dead_columns(profile: DataProfile) -> set[str]:
    """Columns carrying no information, which should not attract further proposals."""
    return {
        c.name
        for c in profile.columns
        if c.is_constant or (profile.n_rows and c.n_missing == profile.n_rows)
    }
