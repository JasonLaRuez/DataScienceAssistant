"""Step 4: feature selection and engineering.

Split by responsibility, mirroring dsa.clean and dsa.viz:

* :mod:`dsa.features.recommend` - pure detection: a frame and its DataProfile in,
  plain-text signals out (not a Proposal/Plan -- nothing here is accepted or rejected
  by id). No session, no gates.
* :mod:`dsa.features.features` - session-facing orchestration: engineers/selects/
  records decisions against a Session, logs each operation, and manages the "features"
  review gate.
"""

from dsa.features.features import (
    engineer_feature,
    propose_features,
    recommend_features,
    reduce_dimensions,
    select_features,
)
from dsa.features.recommend import FeatureRecommendations

__all__ = [
    "propose_features",
    "engineer_feature",
    "recommend_features",
    "select_features",
    "reduce_dimensions",
    "FeatureRecommendations",
]
