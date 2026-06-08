# QZero Research Workspace

This folder is the research layer for the project.

## What the training code writes here

- `research/experiments/<experiment_id>/manifest.json` for the run configuration and Git metadata
- `research/experiments/<experiment_id>/metrics.jsonl` for training and evaluation metrics
- `research/experiments/<experiment_id>/summary.json` for the final run summary

## What to keep here manually

- `research/research_log.md` for experiment notes written in plain Markdown
- `research/paper.md` for the draft paper outline and evolving draft text
- `research/figures/` for charts and plots
- `research/games/` for notable game replays or curated examples
- `research/models/` for named checkpoint snapshots you want to cite later

## Suggested workflow

1. Give every meaningful run an `--experiment_id`.
2. Keep the run configuration in `manifest.json`.
3. Read `metrics.jsonl` when you want plots or comparisons.
4. Add a short summary to `research_log.md` after each milestone.
5. Keep `paper.md` updated while you train, not after the fact.

