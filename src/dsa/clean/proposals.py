"""Step 2b/2c: proposing repairs, then transforms, and applying what you approve.

The interaction is split into two phases so the tier-1/tier-2 distinction is unavoidable
rather than something to notice in a combined plan's text: repairs are proposed and
approved first, then transforms are proposed against the frame as it actually is *after*
those repairs -- never a hypothetical one. Nothing before a repair is approved modifies
the data, so a repair plan can be read, argued with, and partially rejected without
anything having already happened to the frame.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer

from dsa.clean.detect import detect_repairs, detect_transforms
from dsa.clean.pipeline import build_preprocessor, column_treatments
from dsa.clean.plan import RepairPlan, TransformPlan
from dsa.clean.repairs import build_repair
from dsa.gates import REVIEW, GateRequired, open_gate, proceed
from dsa.profile import profile_frame
from dsa.session import Session

STEP = 2


def propose_repairs(session: Session) -> RepairPlan:
    """Profile the working frame and propose tier-1 repairs (step 2a).

    Opens a review gate: the plan is a starting point to be argued with, not a verdict.
    Re-running after a revision is cheap and safe, because nothing has been applied.
    """
    if session.df is None:
        raise ValueError("no data loaded; call a loader first")

    with session.log.record(STEP, "clean.propose_repairs", {"target": session.target}, session.df.shape) as rec:
        profile = profile_frame(session.df)
        plan = RepairPlan(repairs=tuple(detect_repairs(session.df, profile, session.target)))
        rec.notes = f"{len(plan.repairs)} repairs proposed"

    session.repair_plan = plan
    open_gate(
        session,
        key="repair_plan",
        kind=REVIEW,
        question="Approve these repairs, or say what should change.",
        step=STEP,
        context=plan.describe(),
    )
    return plan


def approve_repairs(
    session: Session, drop: tuple[str, ...] = (), plan: RepairPlan | None = None
) -> Session:
    """Apply the approved repairs to the working frame.

    ``drop`` names proposals to reject by id (``dsa.approve_repairs(s, drop=("R3",))``).
    Approving replaces the whole repair list rather than appending to it, so changing
    your mind and re-approving is safe: the working frame is rebuilt from ``raw`` either
    way, and no repair is ever applied twice.

    Any already-approved transforms were derived from the working frame as it stood
    *before* this call, so they are invalidated: cleared, with the transform review gate
    removed so :func:`propose_transforms` must be run again against the now-current frame.
    """
    plan = plan or session.repair_plan
    if plan is None:
        raise ValueError("no repair plan to approve; call dsa.propose_repairs(s) first")

    approved = plan.without(tuple(drop))
    session.repairs = [build_repair(proposal) for proposal in approved.repairs]

    with session.log.record(
        STEP,
        "clean.approve_repairs",
        {"approved": list(approved.ids), "rejected": list(drop)},
        session.raw.shape if session.raw is not None else None,
    ) as rec:
        frame = session.rebuild()
        rec.output_shape = frame.shape

    # A review gate closes once; re-approving after a revision should not fail.
    gate = session.gates.get("repair_plan")
    if gate is not None and not gate.answered:
        proceed(session, "repair_plan")

    _invalidate_transforms(session)
    return session


def propose_transforms(session: Session) -> TransformPlan:
    """Profile the (now-repaired) working frame and propose tier-2 transforms (step 2b).

    Requires repairs to already be approved, so transforms are always argued from the
    frame as it actually is rather than a hypothetical one that repairs might produce.
    """
    repair_gate = session.gates.get("repair_plan")
    if repair_gate is None or not repair_gate.answered:
        raise GateRequired(
            "repairs must be proposed and approved before transforms can be proposed.\n"
            "  call dsa.propose_repairs(s) and dsa.approve_repairs(s) first"
        )

    if session.df is None:
        raise ValueError("no data loaded; call a loader first")

    with session.log.record(STEP, "clean.propose_transforms", {"target": session.target}, session.df.shape) as rec:
        profile = profile_frame(session.df)
        plan = TransformPlan(transforms=tuple(detect_transforms(profile, session.target)))
        rec.notes = f"{len(plan.transforms)} transforms proposed"

    session.transform_plan = plan
    open_gate(
        session,
        key="transform_plan",
        kind=REVIEW,
        question="Approve these transforms, or say what should change.",
        step=STEP,
        context=plan.describe(),
    )
    return plan


def approve_transforms(
    session: Session, drop: tuple[str, ...] = (), plan: TransformPlan | None = None
) -> Session:
    """Record the approved transforms onto the session.

    ``drop`` names proposals to reject by id. Never applied eagerly -- step 5 fits the
    resulting pipeline inside each CV fold, on that fold's training rows only.
    """
    plan = plan or session.transform_plan
    if plan is None:
        raise ValueError("no transform plan to approve; call dsa.propose_transforms(s) first")

    approved = plan.without(tuple(drop))

    with session.log.record(
        STEP, "clean.approve_transforms", {"approved": list(approved.ids), "rejected": list(drop)}
    ):
        session.transforms = approved.transforms

    gate = session.gates.get("transform_plan")
    if gate is not None and not gate.answered:
        proceed(session, "transform_plan")

    return session


def _invalidate_transforms(session: Session) -> None:
    """Clear any transforms approved against a since-changed working frame.

    Called whenever repairs are (re-)approved. Removing the gate rather than merely
    reopening it means the next :func:`propose_transforms` starts clean, with no memory
    of a review round that argued about a plan derived from a frame that no longer exists.
    """
    gate = session.gates.pop("transform_plan", None)
    if gate is None and not session.transforms:
        return  # nothing to invalidate

    session.transform_plan = None
    session.transforms = ()
    with session.log.record(STEP, "clean.invalidate_transforms", {}):
        pass


def feature_columns(session: Session) -> list[str]:
    """Every column of the working frame except the target."""
    if session.df is None:
        raise ValueError("no data loaded; call a loader first")
    return [str(c) for c in session.df.columns if c != session.target]


def preprocessor(session: Session) -> ColumnTransformer:
    """The unfitted preprocessing pipeline implied by the approved transforms.

    Unfitted by design: step 5 fits it inside each CV fold, on training rows only.
    """
    return build_preprocessor(session.transforms, feature_columns(session))


def treatments(session: Session):
    """Per-column report of what the pipeline will do, for review before fitting."""
    return column_treatments(session.transforms, feature_columns(session))
