"""Turning approved tier-1 proposals into frame transformations.

Every repair here must be **idempotent**: the working frame is rebuilt by replaying the
whole approved list against ``Session.raw`` each time the list changes, so a repair may
legitimately run many times on data it has already been applied to.

Repairs are deterministic and learn nothing from the data, which is why applying them
eagerly cannot leak information between training and test rows. Anything that *is*
learned belongs in :mod:`dsa.clean.pipeline` instead.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from dsa.clean.plan import Proposal

Repair = Callable[[pd.DataFrame], pd.DataFrame]


def _drop_duplicate_rows(_: Proposal) -> Repair:
    return lambda frame: frame.drop_duplicates()


def _drop_columns(proposal: Proposal) -> Repair:
    columns = list(proposal.params["columns"])
    # errors="ignore" keeps the repair idempotent: replaying it after the columns are
    # already gone is a no-op rather than a KeyError.
    return lambda frame: frame.drop(columns=columns, errors="ignore")


def _coerce_numeric(proposal: Proposal) -> Repair:
    column = proposal.params["column"]

    def repair(frame: pd.DataFrame) -> pd.DataFrame:
        if column not in frame.columns:
            return frame
        return frame.assign(**{column: pd.to_numeric(frame[column], errors="coerce")})

    return repair


def _coerce_datetime(proposal: Proposal) -> Repair:
    column = proposal.params["column"]

    def repair(frame: pd.DataFrame) -> pd.DataFrame:
        if column not in frame.columns:
            return frame
        return frame.assign(
            **{column: pd.to_datetime(frame[column], errors="coerce", format="mixed")}
        )

    return repair


def _drop_rows_missing_target(proposal: Proposal) -> Repair:
    column = proposal.params["column"]

    def repair(frame: pd.DataFrame) -> pd.DataFrame:
        if column not in frame.columns:
            return frame
        return frame.dropna(subset=[column])

    return repair


_BUILDERS: dict[str, Callable[[Proposal], Repair]] = {
    "drop_duplicate_rows": _drop_duplicate_rows,
    "drop_columns": _drop_columns,
    "coerce_numeric": _coerce_numeric,
    "coerce_datetime": _coerce_datetime,
    "drop_rows_missing_target": _drop_rows_missing_target,
}

# The complete set of repair kinds this module knows how to execute. Exposed so that
# dsa.clean.proposals.propose_manual_repair can validate a human-specified kind against
# the same vocabulary the detectors use, rather than a second hardcoded copy.
REPAIR_KINDS = tuple(_BUILDERS)


def build_repair(proposal: Proposal) -> tuple[str, Repair]:
    """Return a ``(name, callable)`` pair for an approved repair proposal.

    The name is what appears in ``Session.repairs`` and in the write-up, so it identifies
    both the proposal and what it does.
    """
    builder = _BUILDERS.get(proposal.kind)
    if builder is None:
        raise ValueError(f"no repair implementation for kind {proposal.kind!r}")

    label = f"{proposal.id}:{proposal.kind}"
    if proposal.columns:
        label += f"({', '.join(proposal.columns)})"
    return label, builder(proposal)
