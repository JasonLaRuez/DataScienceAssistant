"""DataScienceAssistant: a thin, auditable toolkit for tabular supervised learning.

Notebooks import from this module and nothing else. Every public operation takes a
``Session`` as its first argument, records what it did to that session's ``RunLog``,
and proposes rather than applies (see CLAUDE.md).

Typical opening of a notebook::

    import dsa
    s = dsa.new_session()
    s = dsa.load_kaggle(s, "titanic/titanic")
    dsa.pending(s)          # what still needs a human decision
"""

from dsa.clean import (
    Proposal,
    RepairPlan,
    TransformPlan,
    approve_repairs,
    approve_transforms,
    preprocessor,
    propose_manual_repair,
    propose_manual_transform,
    propose_repairs,
    propose_transforms,
    treatments,
)
from dsa.gates import (
    Gate,
    GateRequired,
    decide,
    open_gate,
    pending,
    proceed,
    require,
    revise,
)
from dsa.io import (
    CredentialsMissing,
    credential_source,
    find_tables,
    read_table,
    table_format,
)
from dsa.io.kaggle import load_file, load_kaggle
from dsa.profile import ColumnProfile, DataProfile, profile, profile_frame
from dsa.runlog import RunEntry, RunLog, environment_snapshot
from dsa.session import Session, find_project_root, new_session
from dsa.viz import AnalysisSummary, analyze, plot_pair, plot_scatter_matrix

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # session
    "Session",
    "new_session",
    "find_project_root",
    # run log
    "RunLog",
    "RunEntry",
    "environment_snapshot",
    # gates
    "Gate",
    "GateRequired",
    "open_gate",
    "require",
    "decide",
    "revise",
    "proceed",
    "pending",
    # loading (step 1)
    "load_kaggle",
    "load_file",
    "read_table",
    "find_tables",
    "table_format",
    "credential_source",
    "CredentialsMissing",
    # profiling and cleaning (step 2)
    "profile",
    "profile_frame",
    "DataProfile",
    "ColumnProfile",
    "propose_repairs",
    "approve_repairs",
    "propose_manual_repair",
    "propose_transforms",
    "approve_transforms",
    "propose_manual_transform",
    "preprocessor",
    "treatments",
    "RepairPlan",
    "TransformPlan",
    "Proposal",
    # exploratory visualization (step 3)
    "analyze",
    "plot_pair",
    "plot_scatter_matrix",
    "AnalysisSummary",
]
