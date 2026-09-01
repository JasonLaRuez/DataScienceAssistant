"""The vocabulary of a cleaning plan.

A proposal is an argument, not an instruction: it carries the evidence that motivated it,
what applying it would do, and what else could reasonably be done instead. That is what
makes approving one a real decision rather than a rubber stamp (CLAUDE.md rule 9).

Pure data. Detection lives in :mod:`dsa.clean.detect`, and execution in
:mod:`dsa.clean.repairs` and :mod:`dsa.clean.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REPAIR = "repair"  # tier 1: deterministic, applied to the working frame on approval
TRANSFORM = "transform"  # tier 2: learned, fitted inside CV folds and never applied eagerly


@dataclass(frozen=True)
class Proposal:
    """One suggested change, with the evidence for it."""

    id: str  # "R1", "T3" - short enough to name when dropping or revising
    tier: str
    kind: str  # dispatch key for repairs.py / pipeline.py
    columns: tuple[str, ...]
    summary: str  # one line: what this would do
    evidence: str  # what was measured, with counts
    consequence: str  # what the data would look like afterwards
    alternatives: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"{self.id}  {self.summary}", f"    evidence : {self.evidence}"]
        if self.consequence:
            lines.append(f"    if applied: {self.consequence}")
        if self.alternatives:
            lines.append("    alternatives: " + " | ".join(self.alternatives))
        return "\n".join(lines)


@dataclass(frozen=True)
class Plan:
    """A full set of proposals, split by tier."""

    repairs: tuple[Proposal, ...] = ()
    transforms: tuple[Proposal, ...] = ()

    def __getitem__(self, proposal_id: str) -> Proposal:
        for proposal in (*self.repairs, *self.transforms):
            if proposal.id == proposal_id:
                return proposal
        raise KeyError(proposal_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in (*self.repairs, *self.transforms))

    def without(self, dropped: tuple[str, ...]) -> "Plan":
        """A copy with the named proposals removed, for approving a subset."""
        unknown = set(dropped) - set(self.ids)
        if unknown:
            raise KeyError(f"no such proposal(s): {sorted(unknown)}; plan has {self.ids}")
        return Plan(
            repairs=tuple(p for p in self.repairs if p.id not in dropped),
            transforms=tuple(p for p in self.transforms if p.id not in dropped),
        )

    def describe(self) -> str:
        if not self.repairs and not self.transforms:
            return "No changes proposed: nothing detected that needs repair or preprocessing."

        lines: list[str] = []
        if self.repairs:
            lines.append("REPAIRS (applied to the working frame on approval)")
            lines.extend(p.describe() for p in self.repairs)
        if self.transforms:
            if lines:
                lines.append("")
            lines.append("TRANSFORMS (unfitted pipeline steps; fitted inside each CV fold)")
            lines.extend(p.describe() for p in self.transforms)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()
