"""Tests for the run log: durability, timing, failure capture, and redaction."""

from __future__ import annotations

import pytest

from dsa.runlog import RunEntry, RunLog, environment_snapshot


def test_append_is_durable_and_readable(tmp_path):
    """An appended entry is on disk immediately, not buffered until close."""
    log = RunLog(tmp_path / "run.jsonl")
    log.append(RunEntry(ts="2026-01-01T00:00:00+00:00", step=1, op="load.csv"))

    # Read through a separate call, i.e. without relying on the writer's in-memory list.
    reloaded = RunLog.read(tmp_path / "run.jsonl")
    assert len(reloaded) == 1
    assert reloaded[0].op == "load.csv"


def test_record_captures_timing_and_output_shape(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    with log.record(2, "clean.apply", {"n_repairs": 3}, input_shape=(10, 4)) as rec:
        rec.output_shape = (9, 4)
        rec.notes = "dropped one duplicate row"

    (entry,) = log.entries
    assert entry.status == "ok"
    assert entry.input_shape == (10, 4)
    assert entry.output_shape == (9, 4)
    assert entry.notes == "dropped one duplicate row"
    assert entry.duration_s is not None and entry.duration_s >= 0


def test_record_logs_failures_and_reraises(tmp_path):
    """A write-up that omitted failed operations would not be an honest account."""
    log = RunLog(tmp_path / "run.jsonl")

    with pytest.raises(ValueError, match="boom"):
        with log.record(3, "viz.plot") as rec:
            rec.notes = "this note must not mask the error"
            raise ValueError("boom")

    (entry,) = log.entries
    assert entry.status == "error"
    assert "ValueError: boom" in entry.notes


def test_credential_shaped_params_are_redacted(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    with log.record(1, "load.kaggle", {
        "api_key": "sensitive",
        "kaggle_token": "sensitive",
        "password": "sensitive",
        "key_columns": ["id"],   # must survive: not a credential
        "dataset": "titanic",
    }):
        pass

    (entry,) = log.entries
    assert entry.params["api_key"] == "<redacted>"
    assert entry.params["kaggle_token"] == "<redacted>"
    assert entry.params["password"] == "<redacted>"
    assert entry.params["key_columns"] == ["id"]
    assert entry.params["dataset"] == "titanic"


def test_environment_snapshot_reports_versions():
    snapshot = environment_snapshot(seed=42)
    assert snapshot["seed"] == 42
    assert snapshot["python"].startswith("3.")
    assert "pandas" in snapshot["packages"]
