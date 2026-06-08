# Research Tracking Best Practices

This document outlines how to track research progress for the QZero project using the improved research folder structure.

---

## Folder Structure Overview

```
research/
├── README.md               # Main research workspace guide
├── paper.md               # Draft paper outline + methodology
├── EXPERIMENTS.md         # Experiment metadata (new)
├── research_log.md        # Detailed experiment notes (improved)
├── RESULTS.md             # Training metrics & progress (new)
├── tracking.py            # Experiment logging utilities
├── experiments/           # Auto-generated experiment data
│   ├── E001/
│   │   ├── manifest.json      # Git metadata, config snapshot
│   │   ├── metrics.jsonl      # Continuous training telemetry
│   │   └── summary.json       # Final run summary
│   ├── E002/
│   └── [etc.]
├── figures/               # Charts, plots, diagrams (future)
├── games/                 # Notable game replays (future)
└── models/                # Named checkpoint snapshots (future)
```

---

## Workflow for Each Experiment

### Step 1: Plan (Before Training)

**Update `research/EXPERIMENTS.md`:**
- Add new experiment section
- Document hypothesis and configuration
- Note expected outcomes

**Example:**
```markdown
### E006: Test New Learning Rate
**Status:** Planned
**Planned Date:** 2026-06-15
**Hypothesis:** Doubling LR to 6e-4 will accelerate convergence

**Configuration:**
- MCTS simulations: 800
- Learning rate: 6e-4 (vs. 3e-4 baseline)
- Other params: Same as E002
```

### Step 2: Execute Training

**Use experiment ID in command:**
```bash
python scripts/train.py \
  --experiment_id E006 \
  --checkpoint_dir checkpoints/E006 \
  --buffer_path data/E006.npz \
  --num_workers 4 \
  --num_sims 800 \
  --train_steps 50000 \
  --device cuda
```

**What gets auto-logged:**
- `research/experiments/E006/manifest.json` — Git metadata, full config
- `research/experiments/E006/metrics.jsonl` — Training telemetry (one line per step)
- Training continues; metrics stream in real-time

### Step 3: Monitor (During Training)

**Check TensorBoard:**
```bash
tensorboard --logdir runs/
```

**Real-time observations:**
- Loss convergence rate
- Training speed (games/sec)
- Value calibration
- Gradient magnitudes

### Step 4: Document (After Training or Major Milestone)

**Update `research/research_log.md`:**

Add an entry using the template:

```markdown
## E006: Test New Learning Rate

**Date:** 2026-06-15 to 2026-06-20

**Change:**
- Increased learning rate from 3e-4 to 6e-4
- All other hyperparameters from E002

**Reason:**
- E002 showed stable training; may be able to converge faster
- Hypothesis: Higher LR reduces time-to-competency

**Settings:**
- Checkpoint dir: `checkpoints/E006/`
- Buffer path: `data/E006.npz`
- Device: CUDA
- MCTS sims: 800
- Workers: 4
- Batch size: 1024
- Train steps: 50,000
- Eval every: 2,000 training steps

**Results:**
- Final policy loss: [value]
- Final value loss: [value]
- Convergence speed: [faster/slower/same] vs. E002
- Final Elo estimate: ~[rating]
- Best checkpoint: Step [X]

**Notes:**
- [Any issues, surprises, or interesting findings]

**Conclusion:**
- [What did this teach us about learning rate sensitivity?]
```

### Step 5: Analyze & Conclude

**Update `research/EXPERIMENTS.md`:**
- Mark as "Completed"
- Summarize key findings
- Note decision for next experiment

**Update `research/RESULTS.md`:**
- Add metrics snapshot
- Compare to baseline (E002)
- Update recommendations

---

## Metrics Logging Details

### Auto-generated: `research/experiments/EXXX/metrics.jsonl`

Each line is a JSON object:

```json
{
  "step": 1000,
  "timestamp": "2026-06-08T12:30:45+00:00",
  "loss_policy": 2.34,
  "loss_value": 0.41,
  "loss_total": 2.75,
  "grad_norm": 0.82,
  "learning_rate": 3e-4,
  "games_per_sec": 820,
  "buffer_size": 145000
}
```

**How to analyze:**
```python
import json
import pandas as pd

# Load metrics stream
with open("research/experiments/E002/metrics.jsonl") as f:
    metrics = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(metrics)

# Plot
df.set_index("step")[["loss_policy", "loss_value"]].plot()
```

### Manual: TensorBoard Event Files

Location: `runs/events.out.tfevents.*`

**View with:**
```bash
tensorboard --logdir runs/ --port 6006
```

Navigate to `localhost:6006` to see:
- Real-time loss curves
- Scalar metrics
- Hyperparameter sweep comparisons

---

## Checklist: Before Submitting Results

Use this checklist before marking an experiment as "Completed":

- [ ] Training finished (or reached milestone)
- [ ] Model checkpoint saved
- [ ] Manifest.json created with git metadata
- [ ] Metrics.jsonl complete
- [ ] Entry added to `research_log.md`
- [ ] EXPERIMENTS.md updated
- [ ] Key metrics noted in RESULTS.md
- [ ] Game replays backed up (if notable)
- [ ] Next steps documented

---

## Comparison Protocol: Running Multiple Experiments

When comparing experiments (e.g., E002 vs. E006):

**1. Matching conditions:**
- Same self-play config (workers, sims, games per iteration)
- Same hardware (same GPU type)
- Same random seed (for reproducibility)
- Same batch size

**2. Evaluation (Promotion Match):**
```bash
# Test if new model beats old model at specific step
python evaluation/arena.py \
  --model1 checkpoints/E002/model_step_0014000.pt \
  --model2 checkpoints/E006/model_step_0014000.pt \
  --num_games 100
```

**3. Record results:**
- Win rate for model2 (target: >55% for promotion)
- Average move quality
- Opening repertoire comparison

---

## Special Cases

### Resuming an Experiment

If training was interrupted and you want to continue E006 from step 25,000:

```bash
python scripts/train.py \
  --experiment_id E006 \
  --checkpoint_dir checkpoints/E006 \
  --buffer_path data/E006.npz \
  --resume \
  --train_steps 50000  # Total, not additional
```

**Note:** Manifest and metrics.jsonl are appended to (not overwritten)

### Running Ablations

For controlled comparison (E004 vs. E002):

**E002 (baseline):** 20 SE blocks, LR augmentation enabled
**E004 (ablation):** 20 blocks NO SE, LR augmentation disabled

```bash
# E004 configuration differs only in network architecture
# Document this clearly in EXPERIMENTS.md
```

### Archiving Old Checkpoints

To save disk space, archive older checkpoints:

```bash
# Keep only best_model.pt and latest 2 checkpoints
tar -czf checkpoints/E002_archive.tar.gz \
  checkpoints/E002/model_step_0000*.pt \
  checkpoints/E002/model_step_0001*.pt
```

---

## Integration with Existing Code

### Using `research/tracking.py` Utilities

The `tracking.py` module provides helpers:

```python
from research.tracking import (
    generate_experiment_id,
    collect_git_info,
    append_jsonl,
    write_json,
    append_markdown_experiment,
)

# Auto-generate ID
exp_id = generate_experiment_id(prefix="exp")  # exp-20260608-123045

# Capture git state
git_meta = collect_git_info(pathlib.Path(__file__).parent)
# Returns: {"commit": "a1b2c3", "branch": "main", "dirty": False, "status": None}

# Log metrics
append_jsonl(
    pathlib.Path("research/experiments") / exp_id / "metrics.jsonl",
    {
        "step": 1000,
        "loss_total": 2.75,
        "learning_rate": 3e-4,
    }
)

# Write final summary
write_json(
    pathlib.Path("research/experiments") / exp_id / "manifest.json",
    {
        "experiment_id": exp_id,
        "git": git_meta,
        "config": full_config_dict,
        "hyperparameters": hparams_dict,
    }
)
```

### TensorBoard Integration

The training script should already be logging to `runs/` via TensorBoard. Verify:

```bash
ls runs/ | wc -l  # Should show multiple event files

tensorboard --logdir runs/
# Navigate to http://localhost:6006
```

---

## Recommended Cadence

### Daily
- Monitor current experiment in TensorBoard
- Note any anomalies

### Every 5,000 steps
- Update RESULTS.md with latest metrics
- Review game replays for quality changes

### Every milestone (10k, 20k, 50k steps)
- Update EXPERIMENTS.md with progress
- Run evaluation match (if applicable)
- Analyze loss curves and convergence rate

### When experiment completes
- Finalize research_log.md entry
- Archive checkpoints
- Plan next experiment

---

## Tools & Commands Reference

```bash
# View TensorBoard
tensorboard --logdir /root/furai-test/qzero/runs/

# Check experiment metadata
cat research/experiments/E002/manifest.json | jq .

# Stream metrics
tail -f research/experiments/E002/metrics.jsonl | jq .

# Compare checkpoints
ls -lh checkpoints/E002/model_step_*.pt | tail -10

# Resume training
python scripts/train.py --resume

# Run quick test
python run_training.py --quick
```

---

## Questions?

Refer to:
- `research/README.md` — Basic overview
- `research/paper.md` — Technical details
- `research/EXPERIMENTS.md` — Experiment registry
- `research/research_log.md` — Detailed notes
- `configs/default.yaml` — All hyperparameters
