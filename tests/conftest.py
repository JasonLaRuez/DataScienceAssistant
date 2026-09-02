"""Shared test configuration.

matplotlib's default backend resolution can try to open a GUI (e.g. TkAgg) just from
constructing a Figure, before anything is ever shown -- fine interactively, but wrong,
and on this machine broken, in a headless test run. Force the non-interactive Agg
backend for tests only; production code (src/dsa/viz) never sets a backend, so a real
Jupyter kernel's own inline backend is untouched.
"""

import matplotlib

matplotlib.use("Agg")
