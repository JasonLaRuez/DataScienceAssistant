"""Step 2b: proposing a cleaning plan, and applying the part of it you approve.

The interaction is deliberately two-step. ``propose`` measures and argues; ``approve``
acts. Nothing between loading and approval modifies the data, so a plan can be read,
argued with, and partially rejected without anything having already happened to the frame.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer

from dsa.clean.pipeline import build_preprocessor, column_treatments
from dsa.clean.plan import Plan
from dsa.clean.repairs import build_repair
from dsa.gates import REVIEW, open_gate, proceed
from dsa.profile import profile_frame
from dsa.session import Session

STEP = 2


def propose(session: Session) -> Plan:
    """Profile the working frame and propose repairs and transforms (step 2).

    Opens a review gate: the plan is a starting point to be argued with, not a verdict.
    Re-running after a revision is cheap and safe, because nothing has been applied.
    """
    if session.df is None:
        raise ValueError("no data loaded; call a loader first")

    with session.log.record(STEP, "clean.propose", {"target": session.target}, session.df.shape) as rec:
        profile = profile_frame(session.df)
        plan = detect_plan(session, profile)
        rec.notes = f"{len(plan.repairs)} repairs, {len(plan.transforms)} transforms proposed"

    session.plan = plan
    open_gate(
        session,
        key="cleaning_plan",
        kind=REVIEW,
        question="Approve this cleaning plan, or say what should change.",
        step=STEP,
        context=plan.describe(),
    )
    return plan


def detect_plan(session: Session, profile) -> Plan:
    """Run detection against the session's working frame. Split out so tests and the
    write-up can rebuild a plan without re-opening a gate."""
    from dsa.clean.detect import detect

    return detect(session.df, profile, session.target)


def approve(session: Session, drop: tuple[str, ...] = (), plan: Plan | None = None) -> Session:
    """Apply the approved repairs and record the approved transforms.

    ``drop`` names proposals to reject by id (``dsa.approve(s, drop=("R3", "T2"))``).
    Approving replaces the whole repair list rather than appending to it, so changing
    your mind and re-approving is safe: the working frame is rebuilt from ``raw`` either
    way, and no repair is ever applied twice.
    """
    plan = plan or session.plan
    if plan is None:
        raise ValueError("no plan to approve; call dsa.propose(s) first")

    approved = plan.without(tuple(drop))

    session.repairs = [build_repair(proposal) for proposal in approved.repairs]
    session.transforms = approved.transforms

    with session.log.record(
        STEP,
        "clean.approve",
        {
            "approved": list(approved.ids),
            "rejected": list(drop),
            "repairs": [name for name, _ in session.repairs],
        },
        session.raw.shape if session.raw is not None else None,
    ) as rec:
        frame = session.rebuild()
        rec.output_shape = frame.shape

    # A review gate closes once; re-approving after a revision should not fail.
    gate = session.gates.get("cleaning_plan")
    if gate is not None and not gate.answered:
        proceed(session, "cleaning_plan")

    return session


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
