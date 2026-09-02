"""Step 3: exploratory visualization.

Split by responsibility, mirroring dsa.clean:

* :mod:`dsa.viz.figures` - pure plot-building: a frame and its DataProfile in, one or
  more unclosed matplotlib Figures out. No session, no I/O, no gates.
* :mod:`dsa.viz.analyze` - session-facing orchestration: saves what figures.py builds,
  logs each figure, and manages the "figures" review gate.
"""

from dsa.viz.analyze import AnalysisSummary, analyze, plot_pair
from dsa.viz.figures import (
    DEFAULT_MAX_CATEGORIES,
    categorical_bar_charts,
    correlation_heatmap,
    numeric_box_plots,
    pair_plot,
)

__all__ = [
    "AnalysisSummary",
    "analyze",
    "plot_pair",
    "categorical_bar_charts",
    "numeric_box_plots",
    "correlation_heatmap",
    "pair_plot",
    "DEFAULT_MAX_CATEGORIES",
]
