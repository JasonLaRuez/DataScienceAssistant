# DataScienceAssistant

A toolkit for moving quickly through a tabular machine-learning workflow in a Jupyter
notebook, pausing only at the decisions a human should make.

Not an autopilot. The point is to compress the mechanical work between meaningful choices:
you point it at a dataset, it loads and profiles the data and *proposes* what to do next
with evidence attached, and you approve, revise, or redirect.

## Scope

Single-table tabular supervised learning (classification and regression) on data that fits
in memory, roughly 10k-100k rows, sourced from the Kaggle API. Time series, NLP, and
computer vision are out of scope.

## How it works

- **All logic lives in `src/dsa/`.** Notebooks import and display; they hold no logic.
- **Nothing touches your data without approval.** Operations return proposals carrying
  counts, example values, and the consequence of applying them.
- **Leakage is structurally prevented.** Learned transforms are never applied eagerly; they
  are unfitted `Pipeline` steps fitted inside each CV fold on that fold's training data.
- **Every operation is logged** to `runs/<run_id>/run.jsonl` as it happens. The final
  write-up is rendered from that log rather than from recollection.

## Pipeline

1. Load - 2. Clean and preprocess - 3. Analyze and visualize - 4. Feature selection and
engineering - 5. Train - 6. Evaluate - 7. Write-up

## Setup

The environment is managed by [uv](https://docs.astral.sh/uv/) against a hermetic
Python 3.12. A fresh clone needs four steps.

**1. Create the environment.** This also installs the package itself in editable mode, so
`import dsa` works from a notebook with no path manipulation.

```bash
uv sync --no-python-downloads
```

`--no-python-downloads` works around a uv 0.12.8 bug on Windows: it downloads the
interpreter correctly but then fails creating the minor-version junction. Plain
`uv sync` is fine once an interpreter is already installed.

**2. Register the Jupyter kernel.** This environment is separate from any Anaconda
install on the machine, and selecting the wrong kernel is the most common early
confusion.

```bash
.venv/Scripts/python.exe -m ipykernel install --user --name dsa --display-name "Python 3.12 (dsa)"
```

**3. Install the notebook output filter.** *Required after every clone.* The
`*.ipynb filter=nbstripout` binding is committed in `.gitattributes`, but the filter it
refers to is defined in `.git/config`, which git does not clone. Without this step,
notebook outputs are committed.

```bash
.venv/Scripts/nbstripout.exe --install
```

**4. Provide Kaggle credentials.** For a modern `KGAT_` token, set it once as a user
environment variable. `Read-Host` keeps the value out of your shell history:

```bash
$t = Read-Host "Kaggle token"; [Environment]::SetEnvironmentVariable('KAGGLE_API_TOKEN', $t, 'User'); Remove-Variable t
```

Restart the editor afterwards so the kernel inherits it. The classic
`KAGGLE_USERNAME`/`KAGGLE_KEY` pair and `~/.kaggle/kaggle.json` also work; `dsa` reports
which mechanism it found and never reads the value.

Verify everything with `.venv/Scripts/python.exe -m pytest`.

## Working in the notebook

Open the project folder in VS Code with the Python and Jupyter extensions, and select the
**Python 3.12 (dsa)** kernel.

Begin every notebook with autoreload, so edits to `src/dsa/` take effect on the next cell
run without restarting the kernel:

```python
%load_ext autoreload
%autoreload 2

import dsa
s = dsa.new_session()
```

Notebooks are thin clients (rule 4): they import, call, and display. Analysis lives in
`src/dsa/`, which is what makes it testable and reviewable.

Because notebook outputs are stripped on commit, anything the write-up needs to show must
be a real file: figures are saved under `runs/<run_id>/figures/` while exploring, and the
ones the step-7 write-up references are copied into `reports/figures/` and committed.
The write-up itself is a markdown cell at the bottom of the finished notebook, rendered
from the run log.

## Status

Steps 1 (loading) and the Session / RunLog / gate spine are implemented. Step 2 was built
and then deliberately reverted; it remains available at tag `v0.1-step2-clean` and will be
rebuilt from a fresh plan.

Development rules and architecture notes are in [CLAUDE.md](CLAUDE.md).
