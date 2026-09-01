"""Tests for step 2: proposal detection, approval, and pipeline construction.

The properties that matter most here are that nothing is applied without approval, that
proposals do not contradict each other, and that the resulting pipeline is unfitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

import dsa
from dsa.clean.detect import detect
from dsa.profile import profile_frame


@pytest.fixture
def session(tmp_path):
    return dsa.new_session(project_root=tmp_path)


def loaded(session, frame, target=None):
    """Put a frame into a session the way a loader would."""
    session.raw = frame
    session.rebuild()
    session.target = target
    return session


@pytest.fixture
def messy():
    """A frame containing one instance of each problem the detectors look for."""
    rng = np.random.default_rng(0)
    n = 100
    frame = pd.DataFrame({
        "row_id": range(n),                                    # identifier
        "age": [str(v) for v in rng.integers(18, 80, n)],      # numeric hiding in a string
        "joined": pd.date_range("2024-01-01", periods=n).astype(str),  # date in a string
        "city": rng.choice(["london", "leeds"], n),            # low-cardinality categorical
        "constant": ["x"] * n,
        "empty": [None] * n,
        "income": rng.normal(50_000, 12_000, n),
        "churn": rng.integers(0, 2, n),
    })
    frame.loc[:4, "income"] = np.nan
    frame.loc[:2, "city"] = None
    frame.loc[3, "age"] = "unknown"
    return pd.concat([frame, frame.iloc[:5]], ignore_index=True)  # duplicate rows


def kinds(proposals) -> list[str]:
    return [p.kind for p in proposals]


# --- detection -----------------------------------------------------------------------

def test_each_problem_is_detected(messy):
    plan = detect(messy, profile_frame(messy), target="churn")

    assert "drop_duplicate_rows" in kinds(plan.repairs)
    assert "coerce_numeric" in kinds(plan.repairs)
    dropped = {c for p in plan.repairs if p.kind == "drop_columns" for c in p.columns}
    assert {"constant", "empty", "row_id"} <= dropped


def test_proposals_carry_concrete_evidence(messy):
    plan = detect(messy, profile_frame(messy), target="churn")
    coercion = next(p for p in plan.repairs if p.kind == "coerce_numeric")

    # The value that fails to parse is named, not merely counted.
    assert "unknown" in coercion.evidence
    assert "%" in coercion.evidence
    assert coercion.consequence
    assert coercion.alternatives


def test_no_column_gets_contradictory_proposals(messy):
    """A column proposed for dropping must not also be proposed for coercion: whichever
    ran first would silently neutralise the other."""
    plan = detect(messy, profile_frame(messy), target="churn")

    dropped = {c for p in plan.repairs if p.kind == "drop_columns" for c in p.columns}
    coerced = {c for p in plan.repairs if p.kind.startswith("coerce") for c in p.columns}
    assert dropped.isdisjoint(coerced)


def test_transforms_see_columns_as_repairs_will_leave_them(messy):
    """'age' is a string now but numeric after coercion, so it belongs to the numeric
    branch -- not to the high-cardinality categorical drop it would otherwise attract."""
    plan = detect(messy, profile_frame(messy), target="churn")

    scaling = next(p for p in plan.transforms if p.kind == "scale_numeric")
    assert "age" in scaling.columns

    high_card = [p for p in plan.transforms if p.kind == "drop_high_cardinality"]
    assert not any("age" in p.columns for p in high_card)


def test_the_target_is_never_proposed_for_transformation(messy):
    plan = detect(messy, profile_frame(messy), target="churn")
    for proposal in plan.transforms:
        assert "churn" not in proposal.columns


def test_rows_missing_the_target_are_proposed_for_removal():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, None, 0.0]})
    plan = detect(frame, profile_frame(frame), target="y")

    proposal = next(p for p in plan.repairs if p.kind == "drop_rows_missing_target")
    assert "1 rows are missing the target" in proposal.evidence


def test_a_clean_frame_produces_no_repairs():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    plan = detect(frame, profile_frame(frame), target="y")
    assert plan.repairs == ()


# --- the propose / approve interaction -----------------------------------------------

def test_propose_changes_nothing(session, messy):
    """Rule 9: proposing is measurement, not action."""
    loaded(session, messy, target="churn")
    before = session.df.copy()

    dsa.propose(session)

    pd.testing.assert_frame_equal(session.df, before)
    assert session.repairs == []


def test_propose_opens_a_review_gate_and_approve_closes_it(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose(session)
    assert not session.gates["cleaning_plan"].answered

    dsa.approve(session)
    assert session.gates["cleaning_plan"].answered


def test_approve_applies_repairs_and_records_transforms(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose(session)

    dsa.approve(session)

    assert "row_id" not in session.df.columns
    assert "constant" not in session.df.columns
    assert pd.api.types.is_numeric_dtype(session.df["age"])
    assert len(session.df) < len(session.raw)  # duplicates gone
    assert session.transforms  # tier-2 specs kept, not applied


def test_rejected_proposals_are_not_applied(session, messy):
    loaded(session, messy, target="churn")
    plan = dsa.propose(session)
    identifier = next(
        p for p in plan.repairs if p.kind == "drop_columns" and "row_id" in p.columns
    )

    dsa.approve(session, drop=(identifier.id,))

    assert "row_id" in session.df.columns
    entry = next(e for e in session.log.entries if e.op == "clean.approve")
    assert entry.params["rejected"] == [identifier.id]


def test_rejecting_an_unknown_proposal_is_an_error(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose(session)
    with pytest.raises(KeyError, match="no such proposal"):
        dsa.approve(session, drop=("R99",))


def test_approving_twice_is_safe_and_does_not_compound(session, messy):
    """Changing your mind re-approves against raw rather than the already-repaired frame."""
    loaded(session, messy, target="churn")
    dsa.propose(session)

    dsa.approve(session)
    first = session.df.copy()
    dsa.approve(session)

    pd.testing.assert_frame_equal(session.df, first)


def test_approve_without_a_plan_is_a_clear_error(session, messy):
    loaded(session, messy, target="churn")
    with pytest.raises(ValueError, match="call dsa.propose"):
        dsa.approve(session)


def test_raw_survives_approval_untouched(session, messy):
    loaded(session, messy, target="churn")
    before = session.raw.copy()
    dsa.propose(session)
    dsa.approve(session)
    pd.testing.assert_frame_equal(session.raw, before)


# --- the preprocessing pipeline ------------------------------------------------------

def test_preprocessor_is_unfitted(session, messy):
    """Fitting happens inside CV folds in step 5, never here."""
    loaded(session, messy, target="churn")
    dsa.propose(session)
    dsa.approve(session)

    pre = dsa.preprocessor(session)
    assert isinstance(pre, ColumnTransformer)
    with pytest.raises(NotFittedError):
        pre.transform(session.df.drop(columns=["churn"]))


def test_preprocessor_produces_a_usable_numeric_matrix(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose(session)
    dsa.approve(session)

    features = session.df.drop(columns=["churn"])
    matrix = dsa.preprocessor(session).fit_transform(features, session.df["churn"])

    assert matrix.shape[0] == len(session.df)
    assert not np.isnan(matrix).any()  # imputation covered every gap


def test_treatments_report_every_feature_column(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose(session)
    dsa.approve(session)

    report = dsa.treatments(session)
    assert set(report["column"]) == set(session.df.columns) - {"churn"}
    assert "impute_numeric -> scale_numeric" in set(report["treatment"])


def test_unhandled_columns_pass_through_rather_than_vanish(session):
    """A column disappearing without anyone deciding it should would violate rule 9."""
    frame = pd.DataFrame({"kept": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    loaded(session, frame, target="y")
    dsa.propose(session)
    dsa.approve(session, drop=tuple(session.plan.ids))  # approve nothing at all

    matrix = dsa.preprocessor(session).fit_transform(frame[["kept"]], frame["y"])
    assert matrix.shape == (4, 1)
    np.testing.assert_array_equal(matrix.ravel().astype(float), frame["kept"].to_numpy())


def test_a_continuous_float_column_is_not_mistaken_for_an_identifier():
    """Precise measurements are unique per row by nature; dropping them would be wrong."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"income": rng.normal(50_000, 1, 100), "y": rng.integers(0, 2, 100)})

    plan = detect(frame, profile_frame(frame), target="y")

    assert frame["income"].nunique() == 100  # genuinely unique per row
    assert plan.repairs == ()


def test_identifier_detection_needs_enough_rows():
    """In a tiny frame almost every column is unique per row, so the signal is worthless."""
    frame = pd.DataFrame({"code": ["a", "b", "c", "d"], "y": [0, 1, 0, 1]})
    plan = detect(frame, profile_frame(frame), target="y")
    assert not any(p.kind == "drop_columns" for p in plan.repairs)
