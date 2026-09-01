"""Tests for decision and review gates."""

from __future__ import annotations

import pytest

import dsa
from dsa.gates import DECISION, REVIEW


@pytest.fixture
def session(tmp_path):
    return dsa.new_session(project_root=tmp_path)


def test_require_raises_until_answered(session):
    dsa.open_gate(session, "target", DECISION, "Which column is the target?", step=1)

    with pytest.raises(dsa.GateRequired) as excinfo:
        dsa.require(session, "target")
    # The traceback is the whole user interface here, so it must carry the call to make.
    assert 'dsa.decide(s, "target", <value>)' in str(excinfo.value)

    dsa.decide(session, "target", "survived")
    assert dsa.require(session, "target") == "survived"


def test_reopening_a_gate_does_not_discard_an_answer(session):
    """Re-running a notebook cell must not silently reset a decision."""
    dsa.open_gate(session, "task", DECISION, "Classification or regression?", step=1)
    dsa.decide(session, "task", "classification")

    dsa.open_gate(session, "task", DECISION, "Classification or regression?", step=1)
    assert dsa.require(session, "task") == "classification"


def test_decide_rejects_a_value_outside_the_options(session):
    dsa.open_gate(session, "task", DECISION, "Which task?", step=1,
                  options=("classification", "regression"))
    with pytest.raises(ValueError, match="not one of the options"):
        dsa.decide(session, "task", "clustering")


def test_review_gate_loops_then_closes(session):
    dsa.open_gate(session, "figures", REVIEW, "Are these figures right?", step=3)

    dsa.revise(session, "figures", "add a correlation heatmap")
    dsa.revise(session, "figures", "log-scale the fare axis")
    gate = session.gates["figures"]
    assert gate.rounds == 2
    assert not gate.answered
    assert gate.history == ["add a correlation heatmap", "log-scale the fare axis"]

    dsa.proceed(session, "figures")
    assert dsa.require(session, "figures") == "proceed"
    assert dsa.pending(session) == []


def test_gate_kinds_are_not_interchangeable(session):
    dsa.open_gate(session, "d", DECISION, "?", step=1)
    dsa.open_gate(session, "r", REVIEW, "?", step=3)

    with pytest.raises(ValueError, match="decision gate"):
        dsa.revise(session, "d", "nope")
    with pytest.raises(ValueError, match="review gate"):
        dsa.decide(session, "r", "nope")


def test_gate_activity_is_logged(session):
    """The write-up reports what the user decided, so decisions must reach the log."""
    dsa.open_gate(session, "target", DECISION, "Which column?", step=1)
    dsa.decide(session, "target", "survived")

    ops = [e.op for e in session.log.entries]
    assert "gate.open" in ops and "gate.decide" in ops
    decided = next(e for e in session.log.entries if e.op == "gate.decide")
    assert decided.params == {"key": "target", "answer": "survived"}
