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
class RepairPlan:
    """A set of tier-1 repair proposals, argued from the working frame as it stands."""

    repairs: tuple[Proposal, ...] = ()

    def __getitem__(self, proposal_id: str) -> Proposal:
        for proposal in self.repairs:
            if proposal.id == proposal_id:
                return proposal
        raise KeyError(proposal_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.repairs)

    def without(self, dropped: tuple[str, ...]) -> "RepairPlan":
        """A copy with the named proposals removed, for approving a subset."""
        unknown = set(dropped) - set(self.ids)
        if unknown:
            raise KeyError(f"no such proposal(s): {sorted(unknown)}; plan has {self.ids}")
        return RepairPlan(repairs=tuple(p for p in self.repairs if p.id not in dropped))

    def describe(self) -> str:
        if not self.repairs:
            return "No repairs proposed: nothing detected that needs fixing."
        lines = ["REPAIRS (applied to the working frame on approval)"]
        lines.extend(p.describe() for p in self.repairs)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()


@dataclass(frozen=True)
class TransformPlan:
    """A set of tier-2 transform proposals, argued from the (already-repaired) working
    frame. Never applied eagerly -- see CLAUDE.md's two-tier data model."""

    transforms: tuple[Proposal, ...] = ()

    def __getitem__(self, proposal_id: str) -> Proposal:
        for proposal in self.transforms:
            if proposal.id == proposal_id:
                return proposal
        raise KeyError(proposal_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.transforms)

    def without(self, dropped: tuple[str, ...]) -> "TransformPlan":
        """A copy with the named proposals removed, for approving a subset."""
        unknown = set(dropped) - set(self.ids)
        if unknown:
            raise KeyError(f"no such proposal(s): {sorted(unknown)}; plan has {self.ids}")
        return TransformPlan(transforms=tuple(p for p in self.transforms if p.id not in dropped))

    def describe(self) -> str:
        if not self.transforms:
            return "No transforms proposed: nothing detected that needs preprocessing."
        lines = ["TRANSFORMS (unfitted pipeline steps; fitted inside each CV fold)"]
        lines.extend(p.describe() for p in self.transforms)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()
