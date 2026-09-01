"""Human decision points.

The toolkit is deliberately not an autopilot (CLAUDE.md rule 8). Where a choice belongs
to the data scientist, an operation opens a *gate* and refuses to guess.

Gates are non-blocking. Rather than calling ``input()`` -- which stalls a kernel and
behaves badly when a cell is re-run -- an operation that needs an unmade decision raises
:class:`GateRequired` carrying the question, the options, and the exact call to make.
The user answers in the next cell and re-runs. That keeps notebooks re-runnable and keeps
the whole interaction visible in the notebook rather than hidden in a prompt.

Two kinds:

* **decision** -- one question, one answer, then closed (target column, task type,
  whether group-aware splitting applies, the evaluation metric, the final model).
* **review**   -- artifacts are produced, then the user loops on revise/proceed until
  satisfied (the step-2 cleaning plan, the step-3 figure set). Every round is logged, so
  the write-up can report honestly how much iteration a result took.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from dsa.session import Session

DECISION = "decision"
REVIEW = "review"


class GateRequired(Exception):
    """Raised when an operation needs a human decision that has not been made yet.

    The message is written to be read in a notebook traceback, so it states the question
    and the literal call that answers it.
    """


@dataclass
class Gate:
    """A single decision point and its current state."""

    key: str
    kind: str
    question: str
    step: int
    options: tuple[str, ...] | None = None
    context: str = ""  # rendered evidence the user needs in order to decide
    answer: Any = None
    answered: bool = False
    rounds: int = 0  # review gates: how many revise cycles have happened
    history: list[str] = field(default_factory=list)  # review gates: what was asked for

    def describe(self) -> str:
        """Notebook-readable summary, used in both tracebacks and gate listings."""
        lines = [f"[{self.kind} gate: {self.key}] {self.question}"]
        if self.options:
            lines.append("  options: " + " | ".join(self.options))
        if self.context:
            lines.append(self.context)
        if self.kind == DECISION:
            lines.append(f'  answer with: dsa.decide(s, "{self.key}", <value>)')
        else:
            lines.append(f'  continue with: dsa.revise(s, "{self.key}", "<what to change>")')
            lines.append(f'  or finish with: dsa.proceed(s, "{self.key}")')
        return "\n".join(lines)


def open_gate(
    session: "Session",
    key: str,
    kind: str,
    question: str,
    step: int,
    options: tuple[str, ...] | None = None,
    context: str = "",
) -> Gate:
    """Register a gate, or return the existing one if it is already open.

    Re-opening is idempotent so that re-running a notebook cell does not reset an answer
    the user has already given.
    """
    if kind not in (DECISION, REVIEW):
        raise ValueError(f"unknown gate kind {kind!r}; expected {DECISION!r} or {REVIEW!r}")

    existing = session.gates.get(key)
    if existing is not None:
        # Refresh the evidence -- the underlying data may have changed -- but never the
        # answer already recorded.
        existing.context = context or existing.context
        return existing

    gate = Gate(key=key, kind=kind, question=question, step=step, options=options, context=context)
    session.gates[key] = gate
    with session.log.record(step, "gate.open", {"key": key, "kind": kind, "question": question}):
        pass
    return gate


def require(session: "Session", key: str) -> Any:
    """Return a gate's answer, or raise :class:`GateRequired` if it is still open."""
    gate = session.gates.get(key)
    if gate is None:
        raise KeyError(f"no gate named {key!r} has been opened")
    if not gate.answered:
        raise GateRequired("\n" + gate.describe())
    return gate.answer


def decide(session: "Session", key: str, value: Any) -> Gate:
    """Answer a decision gate.

    The answer is logged, which is what lets the step-7 write-up state what *the user*
    decided rather than only what the code did.
    """
    gate = session.gates.get(key)
    if gate is None:
        raise KeyError(f"no gate named {key!r} has been opened")
    if gate.kind != DECISION:
        raise ValueError(f"gate {key!r} is a {gate.kind} gate; use revise()/proceed()")
    if gate.options and value not in gate.options:
        raise ValueError(f"{value!r} is not one of the options for {key!r}: {gate.options}")

    gate.answer = value
    gate.answered = True
    with session.log.record(gate.step, "gate.decide", {"key": key, "answer": value}):
        pass
    return gate


def revise(session: "Session", key: str, request: str) -> Gate:
    """Ask for changes at a review gate. The gate stays open and the round is recorded."""
    gate = _review_gate(session, key)
    gate.rounds += 1
    gate.history.append(request)
    with session.log.record(gate.step, "gate.revise", {"key": key, "round": gate.rounds, "request": request}):
        pass
    return gate


def proceed(session: "Session", key: str) -> Gate:
    """Close a review gate as satisfied."""
    gate = _review_gate(session, key)
    gate.answer = "proceed"
    gate.answered = True
    with session.log.record(gate.step, "gate.proceed", {"key": key, "rounds": gate.rounds}):
        pass
    return gate


def pending(session: "Session") -> list[Gate]:
    """Every gate still awaiting an answer, in the order they were opened."""
    return [g for g in session.gates.values() if not g.answered]


def _review_gate(session: "Session", key: str) -> Gate:
    gate = session.gates.get(key)
    if gate is None:
        raise KeyError(f"no gate named {key!r} has been opened")
    if gate.kind != REVIEW:
        raise ValueError(f"gate {key!r} is a {gate.kind} gate; use decide()")
    if gate.answered:
        raise ValueError(f"review gate {key!r} is already closed")
    return gate
