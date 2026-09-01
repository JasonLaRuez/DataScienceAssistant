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

## Status

Early. Repository scaffolding only; the toolchain and package land next.

Development rules and architecture notes are in [CLAUDE.md](CLAUDE.md).
