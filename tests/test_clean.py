"""Tests for step 2: proposal detection, approval, and pipeline construction.

The properties that matter most here are that nothing is applied without approval, that
proposals do not contradict each other, that transforms are only ever proposed against
the frame as it actually is (never a hypothetical one), and that the resulting pipeline
is unfitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

import dsa
from dsa.clean.detect import detect_repairs, detect_transforms
from dsa.clean.pipeline import DatetimeFeaturizer
from dsa.clean.repairs import build_repair
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


def apply_repairs(frame: pd.DataFrame, repairs) -> pd.DataFrame:
    """Apply detected repairs the way approve_repairs would, so a test can see the frame
    exactly as detect_transforms will see it."""
    out = frame
    for proposal in repairs:
        _name, repair = build_repair(proposal)
        out = repair(out)
    return out


# --- detection -----------------------------------------------------------------------

def test_each_problem_is_detected(messy):
    repairs = detect_repairs(messy, profile_frame(messy), target="churn")

    assert "drop_duplicate_rows" in kinds(repairs)
    assert "coerce_numeric" in kinds(repairs)
    dropped = {c for p in repairs if p.kind == "drop_columns" for c in p.columns}
    assert {"constant", "empty", "row_id"} <= dropped
    assert "coerce_datetime" in kinds(repairs)

    # 'joined' is a date hiding in a string: coerce_datetime (tier 1) first, then seen as
    # a real datetime column once repairs are applied and proposed for extraction (tier 2).
    repaired = apply_repairs(messy, repairs)
    transforms = detect_transforms(profile_frame(repaired), target="churn")
    extraction = next(p for p in transforms if p.kind == "extract_datetime_parts")
    assert "joined" in extraction.columns


def test_proposals_carry_concrete_evidence(messy):
    repairs = detect_repairs(messy, profile_frame(messy), target="churn")
    coercion = next(p for p in repairs if p.kind == "coerce_numeric")

    # The value that fails to parse is named, not merely counted.
    assert "unknown" in coercion.evidence
    assert "%" in coercion.evidence
    assert coercion.consequence
    assert coercion.alternatives


def test_no_column_gets_contradictory_proposals(messy):
    """A column proposed for dropping must not also be proposed for coercion: whichever
    ran first would silently neutralise the other."""
    repairs = detect_repairs(messy, profile_frame(messy), target="churn")

    dropped = {c for p in repairs if p.kind == "drop_columns" for c in p.columns}
    coerced = {c for p in repairs if p.kind.startswith("coerce") for c in p.columns}
    assert dropped.isdisjoint(coerced)


def test_transforms_see_columns_as_repairs_will_leave_them(messy):
    """'age' is a string now but numeric after coercion, so it belongs to the numeric
    branch -- not to the high-cardinality categorical drop it would otherwise attract."""
    repairs = detect_repairs(messy, profile_frame(messy), target="churn")
    repaired = apply_repairs(messy, repairs)
    transforms = detect_transforms(profile_frame(repaired), target="churn")

    scaling = next(p for p in transforms if p.kind == "scale_numeric")
    assert "age" in scaling.columns

    high_card = [p for p in transforms if p.kind == "drop_high_cardinality"]
    assert not any("age" in p.columns for p in high_card)


def test_the_target_is_never_proposed_for_transformation(messy):
    repairs = detect_repairs(messy, profile_frame(messy), target="churn")
    repaired = apply_repairs(messy, repairs)
    transforms = detect_transforms(profile_frame(repaired), target="churn")
    for proposal in transforms:
        assert "churn" not in proposal.columns


def test_rows_missing_the_target_are_proposed_for_removal():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, None, 0.0]})
    repairs = detect_repairs(frame, profile_frame(frame), target="y")

    proposal = next(p for p in repairs if p.kind == "drop_rows_missing_target")
    assert "1 rows are missing the target" in proposal.evidence


def test_a_clean_frame_produces_no_repairs():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    repairs = detect_repairs(frame, profile_frame(frame), target="y")
    assert repairs == []


def test_a_continuous_float_column_is_not_mistaken_for_an_identifier():
    """Precise measurements are unique per row by nature; dropping them would be wrong."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"income": rng.normal(50_000, 1, 100), "y": rng.integers(0, 2, 100)})

    repairs = detect_repairs(frame, profile_frame(frame), target="y")

    assert frame["income"].nunique() == 100  # genuinely unique per row
    assert repairs == []


def test_identifier_detection_needs_enough_rows():
    """In a tiny frame almost every column is unique per row, so the signal is worthless."""
    frame = pd.DataFrame({"code": ["a", "b", "c", "d"], "y": [0, 1, 0, 1]})
    repairs = detect_repairs(frame, profile_frame(frame), target="y")
    assert not any(p.kind == "drop_columns" for p in repairs)


# --- phase 1: propose / approve repairs -----------------------------------------------

def test_propose_repairs_changes_nothing(session, messy):
    """Rule 9: proposing is measurement, not action."""
    loaded(session, messy, target="churn")
    before = session.df.copy()

    dsa.propose_repairs(session)

    pd.testing.assert_frame_equal(session.df, before)
    assert session.repairs == []


def test_propose_repairs_opens_a_review_gate_and_approve_closes_it(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    assert not session.gates["repair_plan"].answered

    dsa.approve_repairs(session)
    assert session.gates["repair_plan"].answered


def test_approve_repairs_applies_them_to_the_working_frame(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)

    dsa.approve_repairs(session)

    assert "row_id" not in session.df.columns
    assert "constant" not in session.df.columns
    assert pd.api.types.is_numeric_dtype(session.df["age"])
    assert len(session.df) < len(session.raw)  # duplicates gone


def test_rejected_repairs_are_not_applied(session, messy):
    loaded(session, messy, target="churn")
    plan = dsa.propose_repairs(session)
    identifier = next(
        p for p in plan.repairs if p.kind == "drop_columns" and "row_id" in p.columns
    )

    dsa.approve_repairs(session, drop=(identifier.id,))

    assert "row_id" in session.df.columns
    entry = next(e for e in session.log.entries if e.op == "clean.approve_repairs")
    assert entry.params["rejected"] == [identifier.id]


def test_rejecting_an_unknown_repair_is_an_error(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    with pytest.raises(KeyError, match="no such proposal"):
        dsa.approve_repairs(session, drop=("R99",))


def test_approving_repairs_twice_is_safe_and_does_not_compound(session, messy):
    """Changing your mind re-approves against raw rather than the already-repaired frame."""
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)

    dsa.approve_repairs(session)
    first = session.df.copy()
    dsa.approve_repairs(session)

    pd.testing.assert_frame_equal(session.df, first)


def test_approve_repairs_without_a_plan_is_a_clear_error(session, messy):
    loaded(session, messy, target="churn")
    with pytest.raises(ValueError, match="call dsa.propose_repairs"):
        dsa.approve_repairs(session)


def test_raw_survives_repair_approval_untouched(session, messy):
    loaded(session, messy, target="churn")
    before = session.raw.copy()
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    pd.testing.assert_frame_equal(session.raw, before)


# --- phase 2: propose / approve transforms, and the ordering guarantee ----------------

def test_propose_transforms_before_repairs_are_approved_is_a_clear_error(session, messy):
    loaded(session, messy, target="churn")
    with pytest.raises(dsa.GateRequired, match="propose_repairs"):
        dsa.propose_transforms(session)

    dsa.propose_repairs(session)  # proposed but not yet approved
    with pytest.raises(dsa.GateRequired, match="propose_repairs"):
        dsa.propose_transforms(session)


def test_propose_transforms_opens_a_review_gate_and_approve_closes_it(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)

    dsa.propose_transforms(session)
    assert not session.gates["transform_plan"].answered

    dsa.approve_transforms(session)
    assert session.gates["transform_plan"].answered


def test_approve_transforms_records_but_does_not_apply_them(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)

    columns_before = list(session.df.columns)
    dsa.approve_transforms(session)

    assert session.transforms  # tier-2 specs kept, not applied
    assert list(session.df.columns) == columns_before


def test_rejected_transforms_are_not_recorded(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    plan = dsa.propose_transforms(session)
    scaling = next(p for p in plan.transforms if p.kind == "scale_numeric")

    dsa.approve_transforms(session, drop=(scaling.id,))

    assert scaling.id not in {p.id for p in session.transforms}


def test_transforms_are_proposed_against_the_real_repaired_frame(session, messy):
    """The guarantee this two-phase split exists for: a column removed by an approved
    repair can never show up in a transform proposal, because transforms are only
    proposed after repairs are actually applied -- no hypothetical frame involved."""
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    assert "row_id" not in session.df.columns

    plan = dsa.propose_transforms(session)

    for proposal in plan.transforms:
        assert "row_id" not in proposal.columns


def test_reapproving_repairs_invalidates_approved_transforms(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)
    dsa.approve_transforms(session)
    assert session.transforms

    dsa.propose_repairs(session)
    dsa.approve_repairs(session)  # changed their mind about repairs

    assert session.transforms == ()
    assert session.transform_plan is None
    assert "transform_plan" not in session.gates
    with pytest.raises(ValueError, match="call dsa.propose_transforms"):
        dsa.approve_transforms(session)


# --- the preprocessing pipeline ------------------------------------------------------

def test_preprocessor_is_unfitted(session, messy):
    """Fitting happens inside CV folds in step 5, never here."""
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)
    dsa.approve_transforms(session)

    pre = dsa.preprocessor(session)
    assert isinstance(pre, ColumnTransformer)
    with pytest.raises(NotFittedError):
        pre.transform(session.df.drop(columns=["churn"]))


def test_preprocessor_produces_a_usable_numeric_matrix(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)
    dsa.approve_transforms(session)

    features = session.df.drop(columns=["churn"])
    matrix = dsa.preprocessor(session).fit_transform(features, session.df["churn"])

    assert matrix.shape[0] == len(session.df)
    assert not np.isnan(matrix).any()  # imputation covered every gap


def test_treatments_report_every_feature_column(session, messy):
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)
    dsa.approve_transforms(session)

    report = dsa.treatments(session)
    assert set(report["column"]) == set(session.df.columns) - {"churn"}
    assert "impute_numeric -> scale_numeric" in set(report["treatment"])


def test_unhandled_columns_pass_through_rather_than_vanish(session):
    """A column disappearing without anyone deciding it should would violate rule 9."""
    frame = pd.DataFrame({"kept": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    loaded(session, frame, target="y")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    plan = dsa.propose_transforms(session)
    dsa.approve_transforms(session, drop=tuple(plan.ids))  # approve nothing at all

    matrix = dsa.preprocessor(session).fit_transform(frame[["kept"]], frame["y"])
    assert matrix.shape == (4, 1)
    np.testing.assert_array_equal(matrix.ravel().astype(float), frame["kept"].to_numpy())


# --- datetime extraction ---------------------------------------------------------------

def test_datetime_featurizer_extracts_calendar_parts():
    frame = pd.DataFrame({"joined": pd.to_datetime(["2024-03-15", "2023-12-31"])})

    out = DatetimeFeaturizer().fit_transform(frame)

    assert list(out.columns) == [
        "joined_year", "joined_month", "joined_day", "joined_dayofweek",
    ]
    assert out.loc[0].tolist() == [2024, 3, 15, 4]  # 2024-03-15 is a Friday
    assert out.loc[1].tolist() == [2023, 12, 31, 6]  # 2023-12-31 is a Sunday


def test_datetime_featurizer_gives_a_missing_timestamp_all_sentinels():
    """A NaT must not partially populate parts -- that would look like real data."""
    frame = pd.DataFrame({"joined": pd.to_datetime(["2024-03-15", None])})

    out = DatetimeFeaturizer().fit_transform(frame)

    assert out.loc[1].tolist() == [-1, -1, -1, -1]


def test_datetime_featurizer_reports_its_output_feature_names():
    frame = pd.DataFrame({"a": pd.to_datetime(["2024-01-01"]), "b": pd.to_datetime(["2024-01-01"])})
    featurizer = DatetimeFeaturizer().fit(frame)

    names = list(featurizer.get_feature_names_out())
    assert names == [
        "a_year", "a_month", "a_day", "a_dayofweek",
        "b_year", "b_month", "b_day", "b_dayofweek",
    ]


def test_treatments_reports_datetime_extraction(session, messy):
    """'joined' is coerced string -> datetime by a tier-1 repair (not reported here --
    column_treatments only covers tier-2 transforms) and then extracted by tier 2."""
    loaded(session, messy, target="churn")
    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)
    dsa.approve_transforms(session)

    report = dsa.treatments(session)
    joined = report.loc[report["column"] == "joined", "treatment"].iloc[0]
    assert joined == "extract_datetime_parts"
