# QZero Research Workspace

This folder contains research documentation, experiment tracking, and analysis for the Quoridor AlphaZero project.

---

## 📋 Quick Navigation

### 🎯 Strategic Documents (NEW!)
- **[PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md)** ← **Start here if thinking about publishing**
  - What makes QZero publishable?
  - 13-week experimental plan (MVP to excellence)
  - Target venues and paper structure
  - Beyond-academia (GitHub, blog, video)

- **[RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)** ← **Use to design experiments**
  - 7 core research questions to answer
  - Specific experiments that answer each question
  - Success criteria and publication claims
  - Experimental execution checklist

### 📊 Current Progress
- **[RESULTS.md](RESULTS.md)** — Live training metrics & progress (updated regularly)
- **[EXPERIMENTS.md](EXPERIMENTS.md)** — Registry of all experiments
- **[research_log.md](research_log.md)** — Detailed notes from each experiment

### 📚 Research & Writing
- **[paper.md](paper.md)** — Evolving paper draft with full methodology
- **[TRACKING_GUIDE.md](TRACKING_GUIDE.md)** — How to log experiments and track progress

### Auto-generated
- `experiments/E001/`, `experiments/E002/`, etc. — Manifest, metrics, summary for each experiment

---

## 🚀 What Makes QZero Publishable?

### The Key Insight
A working AI system ≠ publishable research paper.

To be publishable, QZero needs to **answer specific research questions**:

1. ✓ Can AlphaZero solve Quoridor? (vs. heuristic, minimax baselines)
2. ✓ How many MCTS simulations are needed? (diminishing returns curve)
3. ✓ How much do SE blocks help? (3-5% claimed; needs proof)
4. ✓ Does augmentation actually work? (2× dataset claim)
5. ✓ What strategy emerges? (analysis of game replays over time)
6. ✓ How does QZero compare to classical AI? (head-to-head)
7. ✓ What's the training curve? (steps to competence)

**Each question requires a specific experiment.** See [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) for the full design.

---

## 🎯 Current Status

| Aspect | Status |
|--------|--------|
| **Implementation** | ✓ Complete (smoke test passed) |
| **Training Progress** | E002 @ 14,000/50,000 steps (28% done) |
| **Baseline Agents** | ✗ Not yet implemented |
| **Ablation Studies** | ✗ Not yet started |
| **Comparative Benchmarks** | ✗ Planned for Phase 2 |
| **Strategy Analysis** | ✗ Pending game replays |
| **Paper Draft** | ◐ Methodology complete; results pending |

📊 **View live training:** `tensorboard --logdir runs/`

---

## 📖 How to Use This Research Folder

### For Planning Research
1. **Read:** [PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md) (big picture)
2. **Design:** [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) (specific experiments)
3. **Execute:** Follow experiment plan (Phase 1-6 timelines)

### For Running Experiments
1. **Plan:** Update [EXPERIMENTS.md](EXPERIMENTS.md) with hypothesis
2. **Execute:** Follow [TRACKING_GUIDE.md](TRACKING_GUIDE.md) workflow
3. **Monitor:** Watch TensorBoard in real-time
4. **Document:** Add entry to [research_log.md](research_log.md)

### For Writing the Paper
1. **Start:** Read [paper.md](paper.md) sections 1-3 (intro + methodology)
2. **Add results:** Update sections 4-5 as experiments complete
3. **Reference:** Use [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) for publication claims
4. **Tables:** Pull from [EXPERIMENTS.md](EXPERIMENTS.md) architecture/results sections

### For Checking Progress
1. **Daily:** Review [RESULTS.md](RESULTS.md) metrics
2. **Weekly:** Update milestones section
3. **Per experiment:** Add research_log.md entries at checkpoints

---

## 📚 What Each File Contains

### [PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md)
**Purpose:** Strategic guide to making QZero publishable

**Key Sections:**
- 7 core research questions
- 6-phase experimental plan (13 weeks)
- Reproducibility checklist
- Target venues (ArXiv → journals → conferences)
- GitHub, blog, video strategy
- Success metrics (MVP → strong → conference-ready)

**Use when:** Planning the full research program

### [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)
**Purpose:** Map research questions to specific experiments

**Key Sections:**
- Question 1-7: What to ask and how to answer scientifically
- Experiment designs with metrics and success criteria
- Publication claims for each finding
- Priority matrix (must-have, should-have, nice-to-have)
- Example ablation walkthrough (E004 template)

**Use when:** Designing individual experiments

### [RESULTS.md](RESULTS.md)
**Purpose:** Live training progress and results summary

**Contents:**
- Quick status snapshot
- Training metrics (loss curves, speed)
- Checkpoint timeline
- Performance indicators
- Next milestones
- Known issues & resolutions

**Use when:** Checking progress or planning next milestones

### [EXPERIMENTS.md](EXPERIMENTS.md)
**Purpose:** Centralized registry of all experiments

**Contents:**
- E001: Smoke test
- E002: Main GPU training
- E003-E005: Planned experiments
- Architecture improvements table
- Reproducibility instructions

**Use when:** Looking up what's been tried or planning new runs

### [research_log.md](research_log.md)
**Purpose:** Detailed experiment notes

**Contents:**
- Full entry template
- E001 smoke test details
- E002 main training progress
- Space for future entries

**Use when:** Writing detailed notes after milestones

### [paper.md](paper.md)
**Purpose:** Evolving paper draft

**Contents:**
- Abstract & introduction
- Complete methodology (3.1-3.5)
- Experiments description
- Results framework (to be filled in)
- Discussion & future work

**Use when:** Writing or referencing technical content

### [TRACKING_GUIDE.md](TRACKING_GUIDE.md)
**Purpose:** Workflow for tracking research progress

**Contents:**
- Step-by-step experiment workflow
- Metrics logging details
- Comparison protocol
- Tools and commands reference

**Use when:** Starting a new experiment or need logging guidance

---

## 🔄 Recommended Reading Order

### If you have 30 minutes:
1. [PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md) (sections 1-3)
2. [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) (core questions only)

### If you have 1 hour:
1. [PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md) (full)
2. [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) (first 5 questions)

### If you're starting work:
1. [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) (your next experiment)
2. [TRACKING_GUIDE.md](TRACKING_GUIDE.md) (logging procedures)
3. [EXPERIMENTS.md](EXPERIMENTS.md) (see what others did)

### If you're writing the paper:
1. [paper.md](paper.md) (current draft)
2. [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) (publication claims)
3. [EXPERIMENTS.md](EXPERIMENTS.md) (tables and results)
4. [RESULTS.md](RESULTS.md) (metrics and figures)

---

## 💡 The Publication Strategy in 3 Points

### 1. **Scientific Questions** (not just engineering)
❌ "I trained a model for 50k steps"  
✅ "SE blocks provide 3.2% strength improvement with 0.8ms latency"

### 2. **Controlled Experiments** (not just observations)
❌ "My model seems better"  
✅ "Model A vs. B, 100 games, 58% win rate (p=0.02)"

### 3. **Clear Documentation** (for reproducibility)
❌ "Run training.py"  
✅ "Python 3.10 + PyTorch 2.7, seed=42, see configs/default.yaml"

---

## 🛠️ Quick Start for Next 2 Weeks

### Week 1: Establish Baselines
1. [ ] Implement E-Random (random move agent)
2. [ ] Implement E-Heuristic (greedy path agent)
3. [ ] Implement E-Minimax (alpha-beta search)
4. [ ] Run arena: E002 @ 14k vs. each baseline
5. [ ] Document results in EXPERIMENTS.md

### Week 2: Design Ablations
1. [ ] Finalize E004 (SE block ablation) config
2. [ ] Finalize E016 (augmentation ablation) config
3. [ ] Plan E010-E012 (MCTS sim study)
4. [ ] Update EXPERIMENTS.md with all configs
5. [ ] Start running E004

### Week 3+: Execute Phase 2
1. Follow [PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md) Phase 2 schedule
2. Log each experiment to research_log.md
3. Update RESULTS.md with findings
4. Begin paper.md Results section

---

## 🎓 Key Insight

> **Your implementation is engineering-complete. Your research just started.**

The transition from "working system" to "publishable paper" requires:
- Research questions → Specific experiments → Quantified findings → Scientific claims

This folder now contains the strategy and templates to make that happen.

---

## 📊 Success Metrics

### Minimum Viable Publication
- QZero >85% vs. heuristic
- 2+ ablation studies completed
- Paper submitted to ArXiv
- Code open-sourced
- All results reproducible

### Strong Publication
- QZero >95% vs. classical AI
- 5+ rigorous ablations
- Published in workshop/conference
- 100+ GitHub stars
- Blog series (3+ posts)

### Excellent Publication
- Superhuman performance
- 10+ controlled experiments
- Published in top-tier venue
- 500+ GitHub stars
- Media coverage

---

## 🔗 External Resources

- **AlphaZero paper:** Silver et al., 2018
- **Quoridor rules:** [Wikipedia](https://en.wikipedia.org/wiki/Quoridor)
- **MCTS survey:** Browne et al., 2012
- **Benchmarking games:** [Computer Olympiad](https://icga.org/)

---

## 📞 Quick Command Reference

```bash
# View live training
tensorboard --logdir runs/

# Check experiment manifest
cat research/experiments/E002/manifest.json | jq .

# Stream metrics
tail -f research/experiments/E002/metrics.jsonl | jq .

# Run quick test
python run_training.py --quick

# Resume training
python scripts/train.py --resume --experiment_id E002

# Start new experiment
python scripts/train.py --experiment_id E004 --num_workers 4 --train_steps 10000
```

---

## Last Updated

**2026-06-08** — Added strategic roadmaps for publication-quality research  
**Current State:** E002 training at 14,000 steps; research questions documented; experimental plan ready to execute

