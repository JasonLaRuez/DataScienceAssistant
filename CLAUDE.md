# DataScienceAssistant

A toolkit that lets a data scientist move quickly through a tabular ML workflow inside a
Jupyter notebook, pausing only at the decisions a human should actually make.

The goal is **not** automation. It is to minimise the work between meaningful human choices.

## Scope (v1)

- Single-table tabular **supervised** learning: classification and regression.
- Data fits in memory. Expected range: 10,000-100,000 rows.
- Data source: the Kaggle API.
- Explicitly out of scope: time series, NLP, computer vision, out-of-core data.

## Rules

These are binding in every session in this repository.

### 1. Ask before installing
Never install a package, library, or tool without explicit approval from the user first.
This includes transitive tooling (linters, formatters, kernels) and applies even when an
install would be convenient or obvious. Ask once for a batch rather than repeatedly.

### 2. Commit major changes automatically
Commit without being asked when a step is finished, a subsystem lands, or a behavioural
change passes its tests. Messages are concise and informative: what changed and why.
Do not commit on every file save. Commit directly to `main`; tag each finished pipeline
step with an annotated tag (`v0.1-step1-load`, `v0.1-step2-clean`, ...).

### 3. Write code to be read
The user reviews all code manually. Prefer clear names and typed signatures over cleverness.
Comments explain *why* a choice was made, not *what* the line does. Match the surrounding
style. No dense one-liners.

### 4. All logic lives in `src/dsa/`
Notebooks are thin clients: they import, call, and display. A notebook never defines a
function, never holds business logic, and never manipulates a DataFrame directly.

### 5. Every learned transform is a scikit-learn Transformer
Learned transforms are composed into a `Pipeline` and fitted inside CV folds. Never mutate
a DataFrame in place to achieve what a Transformer should do.

### 6. Log every operation
Every operation appends an entry to the `RunLog` before returning, and the log is flushed
to disk immediately so a kernel crash loses nothing. The step-7 write-up is rendered from
that log, never from recollection.

### 7. Never commit data or credentials
`data/` and `runs/` are gitignored; the kagglehub cache is pointed at `data/` so a
project stays self-contained. Kaggle credentials live outside the repository -- a
`KAGGLE_API_TOKEN` environment variable for modern `KGAT_` tokens, or the classic
`KAGGLE_USERNAME`/`KAGGLE_KEY` pair or `~/.kaggle/kaggle.json`. Code asks only *which*
mechanism is configured and never reads the value; the RunLog redacts parameter names
matching `token|secret|password|credential|api_key`.

### 8. Stop at gates
Never infer the target column, the task type, or whether group-aware splitting is needed.
Two kinds of gate:
- **Decision gate** - one question, one answer, then proceed.
- **Review gate** - produce artifacts, then loop on `proceed` / `add` / `revise` until the
  user says proceed.

### 9. Propose, never apply
No change touches the user's data without approval. Proposals carry evidence - counts,
concrete example values, and the consequence of applying - not bare assertions.

### 10. Plan a step before building it
When asked to work on a pipeline step, do not write code first. Propose a detailed plan
and wait for approval. The plan covers:

- the modules and public functions to be added, with their signatures
- which gates the step opens, and of which kind
- what is deliberately **out of scope** for the step
- every decision with more than one reasonable answer, stated as tradeoffs rather than
  a pick (rule 3 applies: the user is choosing, not being informed)
- what the tests will prove, and what would count as the step failing

Revise the plan as directed and re-confirm before writing any code. This applies to a
step being redone as much as to a step being started for the first time.

## Architecture

**Two-tier data model.** This exists so that data leakage is structurally impossible rather
than something a human has to remember.

- *Tier 1 - repairs*: dataset-level and not learned (drop duplicate rows, coerce dtypes,
  parse dates, drop constant columns). Applied to the working frame only after approval.
  The approved repair list is replayed from `Session.raw` whenever it changes, so revising
  a decision never requires re-running the notebook and a repair can never double-apply.
- *Tier 2 - learned transforms*: imputation, scaling, encoding, feature selection. Never
  applied eagerly. They accumulate as unfitted named steps in a plan and are fitted inside
  each CV fold on that fold's training data only.

**Session** is an explicit object passed as the first argument to every operation. It holds
the run id and directory, the RunLog, the raw and working frames, the target/task/groups
(set only through gates), the unfitted preprocessing plan, CV splits, and results.

**RunLog** appends one JSON object per operation to `runs/<run_id>/run.jsonl`, flushed on
write. A sibling `env.json` records Python and package versions, platform, and the seed.

## Decisions already made

Settled deliberately, with the rejected option and the reason. Do not re-propose these
without new information; treat them as closed.

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Environment | `uv` + `pyproject.toml` + committed `uv.lock` | conda (slow solves, awkward pip deps); bare venv+pip (no lockfile) |
| Notebooks in git | commit `.ipynb`, outputs stripped by nbstripout | jupytext pairing (two more packages, loses plots); committing outputs (unreadable diffs); gitignoring notebooks (no version history) |
| Session state | an explicit `Session` passed to every operation | a module-level active session (hidden global state, ambiguous after a partial kernel restart) |
| Models | scikit-learn + XGBoost + LightGBM | PyTorch (an MLP rarely beats trees on tabular data; ~2.5 GB install) - the model registry stays open for it later |
| Branching | commit to `main`, annotated tag per finished step | branch-per-step and PR review (ceremony for a solo project; the user reviews files directly in the editor) |
| Plotting | seaborn alongside matplotlib | matplotlib alone (roughly twice the plotting code for worse defaults) |
| Editor | VS Code + Python/Jupyter extensions | JupyterLab (silently overwrites a notebook edited underneath it; VS Code detects and prompts) |

**Working agreement on notebooks:** never edit a `.ipynb` the user may have open with
unsaved changes. Prefer appending cells over rewriting, and confirm the file is saved
first. Otherwise stay in `src/`.

**Step 7 output format:** a markdown cell at the bottom of the finished notebook, rendered
from the run log. Figures it references are committed files under `reports/figures/`,
because nbstripout removes rendered output from the notebook itself. Exploratory figures
stay under `runs/<run_id>/figures/`, which is gitignored.

## Pipeline steps

| Step | Module | Gate |
|------|--------|------|
| 1 Load | `io/` | target column |
| 2 Clean | `profile.py`, `clean/` | plan approval (review gate) |
| 3 Analyze | `viz/` | figure review (review gate) |
| 4 Features | `features/` | - |
| 5 Train | `models/` | grouping, metric |
| 6 Evaluate | `models/evaluate.py` | final model |
| 7 Write-up | `report/render.py` | - |

Default validation: 5-fold cross-validation.

## Stack

pandas, NumPy, scikit-learn, matplotlib, seaborn, XGBoost, LightGBM. Environment managed by
`uv` against a hermetic Python 3.12. PyTorch is deliberately out of v1 scope; the model
registry is designed so it can be added later without a rewrite.

## Environment notes

- **pandas 3.x is in use.** String columns have dtype `str`, not `object`. Never detect
  text columns with `dtype == object` - use `pandas.api.types.is_string_dtype` or select
  on the `str` dtype. Copy-on-write is the default, and chained assignment raises.
- **`uv sync` must be run with `--no-python-downloads`** if it fails with "Missing expected
  target directory for Python minor version link". This is a uv 0.12.8 bug in creating the
  minor-version junction on Windows; the interpreter itself installs correctly.
- Verified working together: pandas 3.0.5, scikit-learn 1.9.0, seaborn 0.13.2,
  XGBoost 3.4.1, LightGBM 4.7.0 (ColumnTransformer -> Pipeline -> 5-fold CV, and both
  boosters on categorical dtypes).
