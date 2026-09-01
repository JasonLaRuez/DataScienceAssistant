"""Tests for profiling: column classification and frame-level measurement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dsa
from dsa.profile import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, column_kind, profile_frame


@pytest.fixture
def session(tmp_path):
    return dsa.new_session(project_root=tmp_path)


def test_column_kind_uses_pandas_type_helpers():
    """Under pandas 3 a text column is dtype 'str', so an `== object` check would miss it."""
    assert column_kind(pd.Series(["a", "b"])) == CATEGORICAL
    assert column_kind(pd.Series([1, 2])) == NUMERIC
    assert column_kind(pd.Series([1.5, 2.5])) == NUMERIC
    assert column_kind(pd.Series([True, False])) == BOOLEAN
    assert column_kind(pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))) == DATETIME
    assert column_kind(pd.Series(["a", "b"], dtype="category")) == CATEGORICAL


def test_booleans_are_not_reported_as_numeric():
    """Booleans satisfy is_numeric_dtype, so classification order matters."""
    assert column_kind(pd.Series([True, False])) != NUMERIC


def test_profile_measures_missingness_uniqueness_and_duplicates():
    frame = pd.DataFrame({"a": [1, 1, None, 2], "b": ["x", "x", "x", "x"]})
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    profile = profile_frame(frame)
    assert profile.n_rows == 5
    assert profile.n_cols == 2
    assert profile.n_duplicate_rows == 2  # row 0 repeated twice more

    a = profile["a"]
    assert a.n_missing == 1
    assert a.n_unique == 2
    assert a.pct_missing == pytest.approx(20.0)
    assert not a.is_constant

    assert profile["b"].is_constant


def test_a_column_of_only_nulls_is_constant_and_has_no_examples():
    profile = profile_frame(pd.DataFrame({"empty": [None, None]}))
    column = profile["empty"]
    assert column.n_unique == 0
    assert column.is_constant
    assert column.examples == ()


def test_numeric_and_categorical_stats_are_populated():
    profile = profile_frame(pd.DataFrame({"n": [1.0, 3.0, 5.0], "c": ["a", "a", "b"]}))
    assert profile["n"].stats["min"] == 1.0
    assert profile["n"].stats["max"] == 5.0
    assert profile["n"].stats["mean"] == pytest.approx(3.0)
    assert profile["c"].stats == {"top": "a", "top_freq": 2}


def test_profile_of_an_empty_frame_does_not_divide_by_zero():
    profile = profile_frame(pd.DataFrame({"a": pd.Series([], dtype="float64")}))
    assert profile.n_rows == 0
    assert profile["a"].pct_missing == 0.0


def test_to_frame_renders_one_row_per_column():
    view = profile_frame(pd.DataFrame({"a": [1], "b": ["x"]})).to_frame()
    assert list(view["column"]) == ["a", "b"]
    assert set(view.columns) >= {"column", "dtype", "kind", "missing", "unique"}


def test_profile_records_the_operation(session):
    session.raw = pd.DataFrame({"a": [1, 2, 3]})
    session.rebuild()

    dsa.profile(session)

    entry = next(e for e in session.log.entries if e.op == "profile.compute")
    assert entry.input_shape == (3, 1)
    assert "3 rows" in entry.notes


def test_profile_without_data_is_a_clear_error(session):
    with pytest.raises(ValueError, match="no data loaded"):
        dsa.profile(session)


def test_unknown_column_lookup_raises():
    with pytest.raises(KeyError):
        profile_frame(pd.DataFrame({"a": [1]}))["nope"]
