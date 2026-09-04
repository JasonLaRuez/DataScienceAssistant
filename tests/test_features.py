"""Tests for step 4: feature selection and engineering.

The properties that matter most: engineering/selection/reduction never apply without
the "features" gate open, recommendations are pure observation (never mutate
feature_df), and select_features/reduce_dimensions only ever *record* a choice rather
than changing feature_df itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dsa
from dsa.features.recommend import COLLINEARITY_THRESHOLD, WEAK_TARGET_ASSOCIATION


@pytest.fixture
def session(tmp_path):
    return dsa.new_session(project_root=tmp_path)


@pytest.fixture
def feature_session(session):
    """A session past repairs/transforms approval, ready for dsa.propose_features().

    Deterministically constructed rather than left to RNG chance: 'a'/'b' are
    near-perfectly collinear and both strongly correlated with the target; 'noise' is
    constructed to have *exactly* zero correlation with the target (identical values
    replayed across both target groups), well below WEAK_TARGET_ASSOCIATION regardless
    of RNG version drift.
    """
    rng = np.random.default_rng(0)
    n = 60
    half = n // 2
    a = np.sort(rng.normal(0, 1, n))
    b = a * 2 + rng.normal(0, 0.01, n)
    y = np.array([0] * half + [1] * half)
    noise_pool = rng.normal(0, 1, half)
    noise = np.concatenate([noise_pool, noise_pool])

    frame = pd.DataFrame({"a": a, "b": b, "noise": noise, "y": y})
    session.raw = frame
    session.rebuild()
    session.target = "y"

    dsa.propose_repairs(session)
    dsa.approve_repairs(session)
    dsa.propose_transforms(session)
    dsa.approve_transforms(session)
    return session


# --- dsa.propose_features ---------------------------------------------------------------

def test_propose_features_requires_transforms_approved(session):
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    session.raw = frame
    session.rebuild()
    session.target = "y"
    with pytest.raises(dsa.GateRequired, match="propose_transforms"):
        dsa.propose_features(session)


def test_propose_features_initializes_feature_df_equal_to_df(feature_session):
    frame = dsa.propose_features(feature_session)
    pd.testing.assert_frame_equal(frame, feature_session.df)
    pd.testing.assert_frame_equal(feature_session.feature_df, feature_session.df)


def test_propose_features_opens_the_features_gate(feature_session):
    dsa.propose_features(feature_session)
    assert not feature_session.gates["features"].answered


# --- dsa.engineer_feature ----------------------------------------------------------------

def test_engineer_feature_adds_a_computed_column(feature_session):
    dsa.propose_features(feature_session)
    frame = dsa.engineer_feature(feature_session, "a_plus_b", lambda df: df["a"] + df["b"])

    assert "a_plus_b" in frame.columns
    np.testing.assert_allclose(frame["a_plus_b"], frame["a"] + frame["b"])


def test_engineer_feature_rejects_a_name_collision(feature_session):
    dsa.propose_features(feature_session)
    with pytest.raises(ValueError, match="already exists"):
        dsa.engineer_feature(feature_session, "a", lambda df: df["a"])


def test_engineer_feature_composes_across_multiple_calls(feature_session):
    dsa.propose_features(feature_session)
    dsa.engineer_feature(feature_session, "sum_ab", lambda df: df["a"] + df["b"])
    dsa.engineer_feature(feature_session, "sum_ab_noise", lambda df: df["sum_ab"] + df["noise"])

    assert {"sum_ab", "sum_ab_noise"}.issubset(feature_session.feature_df.columns)


def test_engineer_feature_does_not_corrupt_state_on_failure(feature_session):
    dsa.propose_features(feature_session)
    before_engineered = list(feature_session.engineered)
    before_columns = list(feature_session.feature_df.columns)

    def broken(df):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        dsa.engineer_feature(feature_session, "bad", broken)

    assert feature_session.engineered == before_engineered
    assert list(feature_session.feature_df.columns) == before_columns


def test_rebuild_features_is_idempotent(feature_session):
    dsa.propose_features(feature_session)
    dsa.engineer_feature(feature_session, "sum_ab", lambda df: df["a"] + df["b"])
    first = feature_session.feature_df.copy()

    feature_session.rebuild_features()

    pd.testing.assert_frame_equal(feature_session.feature_df, first)


def test_engineer_feature_requires_propose_features_first(feature_session):
    with pytest.raises(KeyError, match="no gate named"):
        dsa.engineer_feature(feature_session, "x", lambda df: df["a"])


def test_engineer_feature_after_features_gate_closed_is_rejected(feature_session):
    dsa.propose_features(feature_session)
    dsa.proceed(feature_session, "features")
    with pytest.raises(ValueError, match="already closed"):
        dsa.engineer_feature(feature_session, "x", lambda df: df["a"])


# --- dsa.recommend_features ---------------------------------------------------------------

def test_recommend_features_works_before_propose_features(session):
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    session.raw = frame
    session.rebuild()
    session.target = "y"

    recommendations = dsa.recommend_features(session)

    assert isinstance(recommendations, dsa.FeatureRecommendations)


def test_recommend_features_flags_near_zero_variance(feature_session):
    dsa.propose_features(feature_session)
    dsa.engineer_feature(feature_session, "flat", lambda df: pd.Series(1.0, index=df.index))

    recommendations = dsa.recommend_features(feature_session)

    assert "flat" in recommendations.near_zero_variance


def test_recommend_features_flags_collinear_pairs(feature_session):
    dsa.propose_features(feature_session)

    recommendations = dsa.recommend_features(feature_session)

    pairs = {frozenset((a, b)) for a, b, _ in recommendations.collinear_pairs}
    assert frozenset(("a", "b")) in pairs
    for _, _, strength in recommendations.collinear_pairs:
        assert strength > COLLINEARITY_THRESHOLD


def test_recommend_features_flags_weak_target_association(feature_session):
    dsa.propose_features(feature_session)

    recommendations = dsa.recommend_features(feature_session)

    weak_names = {c for c, _ in recommendations.weak_target_association}
    assert "noise" in weak_names
    for _, strength in recommendations.weak_target_association:
        assert strength < WEAK_TARGET_ASSOCIATION


def test_recommend_features_does_not_mutate_the_frame(feature_session):
    dsa.propose_features(feature_session)
    before = feature_session.feature_df.copy()

    dsa.recommend_features(feature_session)

    pd.testing.assert_frame_equal(feature_session.feature_df, before)


# --- dsa.select_features -------------------------------------------------------------------

def test_select_features_requires_propose_features_first(feature_session):
    with pytest.raises(KeyError, match="no gate named"):
        dsa.select_features(feature_session)


def test_select_features_exclude_removes_from_the_default_set(feature_session):
    dsa.propose_features(feature_session)
    dsa.select_features(feature_session, exclude=("noise",))

    assert "noise" not in feature_session.selected_features
    assert set(feature_session.selected_features) == {"a", "b"}


def test_select_features_include_is_used_exactly(feature_session):
    dsa.propose_features(feature_session)
    dsa.select_features(feature_session, include=("a",))

    assert feature_session.selected_features == ("a",)


def test_select_features_rejects_an_unknown_column(feature_session):
    dsa.propose_features(feature_session)
    with pytest.raises(ValueError, match="no such column"):
        dsa.select_features(feature_session, exclude=("nope",))


def test_select_features_replaces_rather_than_accumulates(feature_session):
    dsa.propose_features(feature_session)
    dsa.select_features(feature_session, exclude=("noise",))
    dsa.select_features(feature_session, exclude=("a",))

    assert set(feature_session.selected_features) == {"b", "noise"}


def test_select_features_only_records_a_choice_does_not_drop_columns(feature_session):
    dsa.propose_features(feature_session)
    dsa.select_features(feature_session, exclude=("noise",))

    assert "noise" in feature_session.feature_df.columns  # still present in the frame
    assert "noise" not in feature_session.selected_features  # just not selected


# --- dsa.reduce_dimensions ------------------------------------------------------------------

def test_reduce_dimensions_requires_propose_features_first(feature_session):
    with pytest.raises(KeyError, match="no gate named"):
        dsa.reduce_dimensions(feature_session, n_components=2)


def test_reduce_dimensions_records_but_builds_nothing(feature_session):
    dsa.propose_features(feature_session)
    dsa.reduce_dimensions(feature_session, n_components=2)

    assert feature_session.pca_components == 2


def test_reduce_dimensions_rejects_a_nonpositive_component_count(feature_session):
    dsa.propose_features(feature_session)
    with pytest.raises(ValueError, match="at least 1"):
        dsa.reduce_dimensions(feature_session, n_components=0)


# --- closing the gate ------------------------------------------------------------------------

def test_proceed_closes_the_features_gate(feature_session):
    dsa.propose_features(feature_session)
    dsa.select_features(feature_session)

    dsa.proceed(feature_session, "features")

    assert feature_session.gates["features"].answered
