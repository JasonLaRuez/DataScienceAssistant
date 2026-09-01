"""DataScienceAssistant: a thin, auditable toolkit for tabular supervised learning.

Notebooks import from this module and nothing else. Every public operation takes a
``Session`` as its first argument, records what it did to that session's ``RunLog``,
and proposes rather than applies (see CLAUDE.md).

Typical opening of a notebook::

    import dsa
    s = dsa.new_session()
    s
"""

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
from dsa.runlog import RunEntry, RunLog, environment_snapshot
from dsa.session import Session, find_project_root, new_session

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
]
