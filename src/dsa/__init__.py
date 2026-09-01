"""DataScienceAssistant: a thin, auditable toolkit for tabular supervised learning.

Notebooks import from this module and nothing else. Every public operation takes a
``Session`` as its first argument, records what it did to that session's ``RunLog``,
and proposes rather than applies (see CLAUDE.md).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
