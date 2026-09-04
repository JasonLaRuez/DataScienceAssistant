"""Step 3: exploratory visualization.

Split by responsibility, mirroring dsa.clean:

* :mod:`dsa.viz.figures` - pure plot-building: a frame and its DataProfile in, one or
  more unclosed matplotlib Figures out. No session, no I/O, no gates.
* :mod:`dsa.viz.analyze` - session-facing orchestration: saves what figures.py builds,
  logs each figure, and manages the "figures" review gate.
"""

from dsa.viz.analyze import AnalysisSummary, analyze, plot_missingness, plot_pair, plot_scatter_matrix
from dsa.viz.figures import (
    DEFAULT_MAX_CATEGORIES,
    categorical_association_heatmap,
    categorical_bar_charts,
    correlation_heatmap,
    cramers_v,
    missingness_bar_chart,
    numeric_box_plots,
    pair_plot,
    scatter_matrix,
)

__all__ = [
    "AnalysisSummary",
    "analyze",
    "plot_pair",
    "plot_scatter_matrix",
    "plot_missingness",
    "categorical_bar_charts",
    "numeric_box_plots",
    "correlation_heatmap",
    "categorical_association_heatmap",
    "cramers_v",
    "missingness_bar_chart",
    "pair_plot",
    "scatter_matrix",
    "DEFAULT_MAX_CATEGORIES",
]
