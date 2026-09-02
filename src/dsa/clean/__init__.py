"""Step 2: cleaning and preprocessing.

Split by responsibility so each part can be reviewed on its own:

* :mod:`dsa.clean.plan`      - the vocabulary: a Proposal, a RepairPlan, a TransformPlan
* :mod:`dsa.clean.detect`    - measuring the frame and arguing for changes
* :mod:`dsa.clean.repairs`   - executing approved tier-1 repairs (idempotent)
* :mod:`dsa.clean.pipeline`  - composing approved tier-2 transforms, unfitted
* :mod:`dsa.clean.proposals` - the two-phase propose / approve interaction: repairs, then
  transforms
"""

from dsa.clean.detect import detect_repairs, detect_transforms
from dsa.clean.plan import Proposal, RepairPlan, TransformPlan
from dsa.clean.pipeline import TRANSFORM_KINDS, build_preprocessor, column_treatments
from dsa.clean.proposals import (
    approve_repairs,
    approve_transforms,
    feature_columns,
    preprocessor,
    propose_manual_repair,
    propose_manual_transform,
    propose_repairs,
    propose_transforms,
    treatments,
)
from dsa.clean.repairs import REPAIR_KINDS, build_repair

__all__ = [
    "RepairPlan",
    "TransformPlan",
    "Proposal",
    "detect_repairs",
    "detect_transforms",
    "propose_repairs",
    "approve_repairs",
    "propose_manual_repair",
    "propose_transforms",
    "approve_transforms",
    "propose_manual_transform",
    "preprocessor",
    "treatments",
    "feature_columns",
    "build_preprocessor",
    "build_repair",
    "column_treatments",
    "REPAIR_KINDS",
    "TRANSFORM_KINDS",
]
