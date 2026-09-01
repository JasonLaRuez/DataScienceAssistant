"""Tests for the Session: run directory setup, and the repair-replay data model."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import dsa


@pytest.fixture
def session(tmp_path):
    return dsa.new_session(project_root=tmp_path, seed=7)


def test_new_session_creates_a_reproducible_run_directory(session, tmp_path):
    assert session.run_dir.parent == tmp_path / "runs"

    # The environment snapshot is written before any work happens, so that a run which
    # crashes immediately is still reproducible.
    env = json.loads((session.run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["seed"] == 7
    assert "pandas" in env["packages"]

    assert [e.op for e in session.log.entries] == ["session.start"]


def test_repairs_replay_in_order_from_raw(session):
    session.raw = pd.DataFrame({"a": [1, 2, 2, 3]})
    session.repairs = [
        ("drop_duplicates", lambda df: df.drop_duplicates()),
        ("double_a", lambda df: df.assign(a=df["a"] * 2)),
    ]

    out = session.rebuild()
    assert out["a"].tolist() == [2, 4, 6]


def test_rebuild_is_idempotent(session):
    """Re-running a notebook cell must not apply a repair twice."""
    session.raw = pd.DataFrame({"a": [1, 2, 2, 3]})
    session.repairs = [("double_a", lambda df: df.assign(a=df["a"] * 2))]

    first = session.rebuild()
    second = session.rebuild()
    pd.testing.assert_frame_equal(first, second)


def test_revising_a_repair_does_not_require_reloading(session):
    """Dropping a repair rebuilds from raw rather than compounding on the working frame."""
    session.raw = pd.DataFrame({"a": [1, 2, 3]})
    session.repairs = [("double_a", lambda df: df.assign(a=df["a"] * 2))]
    session.rebuild()

    session.repairs = []
    pd.testing.assert_frame_equal(session.rebuild(), session.raw)


def test_raw_is_never_mutated_by_a_rebuild(session):
    session.raw = pd.DataFrame({"a": [1, 2, 3]})
    before = session.raw.copy()
    session.repairs = [("double_a", lambda df: df.assign(a=df["a"] * 2))]
    session.rebuild()
    pd.testing.assert_frame_equal(session.raw, before)


def test_rebuild_without_data_is_a_clear_error(session):
    with pytest.raises(ValueError, match="no data loaded"):
        session.rebuild()


def test_summary_reports_open_gates(session):
    dsa.open_gate(session, "target", "decision", "Which column is the target?", step=1)
    assert "1 open" in session.summary()
    assert "target" in session.summary()


def test_plan_and_transforms_default_to_unset(session):
    """A fresh session has no cleaning plan proposed and nothing approved yet."""
    assert session.repair_plan is None
    assert session.transform_plan is None
    assert session.transforms == ()


def test_find_project_root_locates_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = tmp_path / "notebooks" / "deep"
    nested.mkdir(parents=True)
    assert dsa.find_project_root(nested) == tmp_path.resolve()
