"""Step 2: cleaning and preprocessing.

Split by responsibility so each part can be reviewed on its own:

* :mod:`dsa.clean.plan`      - the vocabulary: a Proposal and a Plan (pure data)
* :mod:`dsa.clean.detect`    - measuring the frame and arguing for changes
* :mod:`dsa.clean.repairs`   - executing approved tier-1 repairs (idempotent)
* :mod:`dsa.clean.pipeline`  - composing approved tier-2 transforms, unfitted
* :mod:`dsa.clean.proposals` - the propose / approve interaction
"""

from dsa.clean.detect import detect
from dsa.clean.plan import Plan, Proposal
from dsa.clean.pipeline import build_preprocessor, column_treatments
from dsa.clean.proposals import (
    approve,
    feature_columns,
    preprocessor,
    propose,
    treatments,
)
from dsa.clean.repairs import build_repair

__all__ = [
    "Plan",
    "Proposal",
    "detect",
    "propose",
    "approve",
    "preprocessor",
    "treatments",
    "feature_columns",
    "build_preprocessor",
    "build_repair",
    "column_treatments",
]
