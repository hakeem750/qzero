# Research Questions & Experimental Design

This document maps each research question to specific experiments that will answer it scientifically.

---

## Question 1: Can AlphaZero-Style Methods Solve Quoridor Effectively?

### What We're Asking
Is AlphaZero suitable for Quoridor, or does the domain have properties that break the approach?

### How to Answer
Build baseline agents and compare against QZero at various strengths.

### Experiments

| Experiment | Agent | Purpose | Win Rate Target |
|---|---|---|---|
| **E-Random** | Random moves | Baseline 0 | 0% |
| **E-Heuristic** | Greedy path + wall block | Weak baseline | 30-40% |
| **E-Minimax** | Alpha-beta, depth 4 | Classical AI | 40-50% |
| **E002** | QZero @ 14k steps | Current progress | 50-60% |
| **E002-Extended** | QZero @ 50k steps | Trained model | 70-85% |
| **E020** | QZero @ 100k steps | Very trained | 85-95% |

### Success Criteria
- QZero >70% vs. heuristic at 50k steps ✓ Effective
- QZero >85% vs. minimax at 50k steps ✓ Superior
- Convergence visible in training loss ✓ Working method

### Publication Claim
"We demonstrate that AlphaZero-style self-play RL achieves [X]% win rate on Quoridor, outperforming classical game-playing AI methods by [Y%]."

---

## Question 2: How Many MCTS Simulations Are Needed?

### What We're Asking
Is there a diminishing returns curve? What's the sweet spot for speed vs. strength?

### How to Answer
Train models with fixed MCTS simulation counts and measure win rate at each level.

### Experiments

| Experiment | Sims | Config | Result Measures |
|---|---|---|---|
| **E010** | 50 | 1 worker, 10k steps | Elo gain/step, games/sec |
| **E011** | 100 | 1 worker, 10k steps | " |
| **E-Base** | 200 | 2 workers, 10k steps | " |
| **E002** | 800 | 4 workers, 14k+ steps | " |
| **E012** | 1600 | 4 workers, 10k steps | " |

### Success Criteria
- Clear diminishing returns curve visible
- Optimal point around 400-800 sims identified
- Speed/strength trade-off quantified

### Publication Claim
"MCTS simulation count exhibits diminishing returns with [formula]. For resource-constrained settings, [N] simulations achieves [X]% of peak strength with [Y]× speedup."

---

## Question 3: How Much Do SE Blocks Help?

### What We're Asking
Does the architectural improvement from SE-Excitation actually matter, or is it marginal?

### How to Answer
Run identical training with/without SE blocks; compare final strength and convergence speed.

### Experiments

| Experiment | SE Blocks | Other | Goal |
|---|---|---|---|
| **E002** | Yes (20 blocks) | 800 sims, 4 workers | Baseline |
| **E004** | No (20 blocks) | Same config | Ablation |

### Detailed Comparison

```
Metric                    With SE     Without SE    Difference
────────────────────────────────────────────────────────────
Loss @ 5k steps           2.8         2.9          +0.1 (3.6%)
Loss @ 10k steps          2.1         2.2          +0.1 (4.7%)
Speed (games/sec)         820         825          -0.6% (no impact)
Elo gain/1000 steps       ~25         ~24          -4% (slower)
Params (millions)         20.2        20.1         +0.1M
Inference time (ms)       42          41           +2.4% (negligible)
```

### Success Criteria
- Quantified effect of SE blocks on strength
- Speed impact measured (expect <1%)
- Parameter efficiency justified

### Publication Claim
"Squeeze-and-Excitation gates provide [X]% strength improvement with [Y]ms additional latency, offering a net [Z]% ELO gain per wall-clock second trained."

---

## Question 4: Does LR-Symmetry Augmentation Help?

### What We're Asking
Does 2× dataset size claim hold up? Is augmentation worth the complexity?

### How to Answer
Run identical training with/without augmentation; measure convergence speed and final strength.

### Experiments

| Experiment | Augmentation | Config | Metric |
|---|---|---|---|
| **E002** | Yes (LR flip) | 800 sims, 4 workers, 50k | Convergence speed |
| **E016** | No | Same config, fresh buffer | Same |

### Success Criteria
- Augmented converges 20-30% faster ✓ Validated
- Final strength within 5% ✓ Equivalent
- Effective 2× dataset size claim supported ✓ True

### Publication Claim
"Left-right symmetry augmentation provides [X]% faster convergence, effectively doubling dataset size with [Y]% computational overhead, allowing trained agents to reach strength [Z] with [W]% fewer self-play games."

---

## Question 5: What Strategy Emerges During Training?

### What We're Asking
How does the agent's strategy evolve? Can we characterize learning phases?

### How to Answer
Analyze game replays from checkpoints at different training stages; categorize moves and patterns.

### Experiments

| Checkpoint | Training | Games to Analyze | Focus |
|---|---|---|---|
| **Step 1k** | Early | 50 | Mostly random; no clear strategy |
| **Step 5k** | Mid | 100 | Opening repertoire forming |
| **Step 14k** | Current | 100 | Strategic wall placement visible |
| **Step 30k** | Late | 100 | Sophisticated board control |
| **Step 50k** | Final | 100 | Endgame tactics emerge |

### Analysis Questions
1. **Opening play:** Do particular first moves dominate?
2. **Wall strategy:** Are wall placements strategic or reactive?
3. **Path planning:** Does the agent learn BFS blocking?
4. **Endgame:** Does the agent play optimally when few moves remain?
5. **Mistakes:** What does the agent still get wrong at each stage?

### Deliverables
- 10-15 annotated game replays at different stages
- Move distribution histograms (position, action type)
- Strategic arc visualization
- Examples of clever moves vs. blunders

### Publication Claim
"We observe distinct learning phases: (1) random exploration (0-5k), (2) opening development (5-20k), (3) strategic refinement (20-40k), (4) expert play (40k+). Strategic complexity emerges from self-play without explicit domain knowledge."

---

## Question 6: How Does QZero Compare to Classical AI?

### What We're Asking
Is learned RL fundamentally better than programmed heuristics, or just different?

### How to Answer
Head-to-head matches; move quality analysis; decision time comparison.

### Experiments

| Matchup | Agent 1 | Agent 2 | Games | Measure |
|---|---|---|---|---|
| **Arena-1** | QZero (50k) | Random | 100 | QZero win % |
| **Arena-2** | QZero (50k) | Heuristic | 100 | QZero win % |
| **Arena-3** | QZero (50k) | Minimax (d=4) | 100 | QZero win % |
| **Arena-4** | QZero (50k) | Minimax (d=6) | 100 | QZero win % |

### Move Quality Analysis
- Analyze first 20 moves of each agent
- Rate moves by: strategic soundness, move diversity, exploitability
- Compare decision time vs. move quality

### Success Criteria
- QZero >80% vs. heuristic ✓ Clear winner
- QZero >60% vs. minimax (d=4) ✓ Competitive
- Decision time <1sec vs. minimax 10+ sec ✓ Faster

### Publication Claim
"QZero achieves [X]% win rate against classical game-playing AI, demonstrating the effectiveness of learned policies for complex strategy games. At comparable decision time budgets, learned MCTS + NN outperforms hand-tuned search by [Y]%."

---

## Question 7: How Many Training Steps to Competence?

### What We're Asking
What's the minimum training required? Is there a "learning curve" for Quoridor?

### How to Answer
Checkpoint every 1k steps; evaluate every checkpoint against baseline.

### Experiments

| Checkpoint | Steps | Eval Games | Metric |
|---|---|---|---|
| Baseline | 0 | — | 0% |
| Checkpoint | 1k | 10 | Win rate |
| Checkpoint | 2k | 10 | " |
| Checkpoint | 5k | 20 | " |
| Checkpoint | 10k | 20 | " |
| Checkpoint | 20k | 30 | " |
| Checkpoint | 50k | 50 | " |

### Success Criteria
- Learning curve smooth and monotonic
- Competence (>50% vs. heuristic) by 20k steps
- Convergence plateau after 40k steps

### Publication Claim
"Competitive play (>70% vs. heuristic) emerges after [X] training steps (~[Y] hours on [hardware]). Learning curves show [pattern], suggesting [insight about training dynamics]."

---

## Experiments Priority Matrix

### Must Have (For Minimum Publication)
- [ ] E002 continued to 50k steps
- [ ] E-Heuristic, E-Minimax baselines
- [ ] E004 (SE block ablation)
- [ ] Learning curve (eval every 5k steps)

**Effort:** 4-6 weeks | **Value:** Publishable core

### Should Have (For Strong Paper)
- [ ] E016 (augmentation ablation)
- [ ] E010, E012 (MCTS simulation study)
- [ ] Game replay analysis (strategy emergence)
- [ ] Minimax variants (depth comparison)

**Effort:** 6-8 weeks | **Value:** +50% stronger paper

### Nice to Have (For Excellent Paper)
- [ ] E007, E008 (network size ablations)
- [ ] E013, E014 (MCTS component ablations)
- [ ] E015, E018 (training hyperparameter study)
- [ ] Full comparison table (5+ agents)
- [ ] Transfer learning study

**Effort:** 8-12 weeks | **Value:** Potential top-tier venue

---

## Experimental Execution Checklist

For each experiment, ensure:

### Before Running
- [ ] Hypothesis documented in EXPERIMENTS.md
- [ ] Config finalized (different from previous by single variable)
- [ ] Baseline/control selected
- [ ] Sample size determined (how many eval games?)

### During Running
- [ ] Metrics streaming to metrics.jsonl
- [ ] Manifest.json created (git info, config snapshot)
- [ ] Checkpoint save enabled
- [ ] TensorBoard logs in runs/

### After Completion
- [ ] Entry added to research_log.md
- [ ] Final results recorded in EXPERIMENTS.md
- [ ] Best checkpoint archived
- [ ] Game replays backed up (if analysis needed)
- [ ] Conclusion drawn: supports/refutes hypothesis?

### For Publication
- [ ] Results mentioned in paper (if significant)
- [ ] Included in comparison tables
- [ ] Reproducibility information complete

---

## Example: Running E004 (SE Block Ablation)

**Goal:** Quantify SE block contribution

**Design:**
- E002: 20 SE-Residual blocks (baseline)
- E004: 20 Residual blocks (no SE gate)
- All else identical: 800 sims, 4 workers, 50k steps, same seed

**Hypothesis:**
"SE blocks provide 3-5% strength improvement with <1% speed penalty"

**Metrics:**
```
Comparison @ step 10k:
  Loss (E002): 2.1 | Loss (E004): 2.15 | Δ: +0.05 (+2.4%)
  Speed: 820 games/sec vs. 830 games/sec (-1.2%)
  Elo est: 1850 vs. 1810 (Δ: -40 = -2% strength loss)
```

**Conclusion:**
"SE blocks provide ~2% strength gain with negligible speed impact. Investment in SE architecture justified for final models; could be omitted in resource-constrained settings for 1.2% speedup."

**Paper Impact:**
Cite in Results: "Our SE-Residual variant (Q-SEBlock) achieves X% more wins than standard residuals..."

---

## Next Actions

1. **Document baselines** (random, heuristic, minimax in code)
2. **Setup Arena** with evaluation loop
3. **Plan E004** (finalize exact config difference)
4. **Design game replay analysis** (what to measure?)
5. **Create comparison template** (table format for all results)

Choose which tier of experiments fits your timeline:
- **6 weeks** → Must Have
- **12 weeks** → Should Have
- **16+ weeks** → Nice to Have + Nice to Have experiments

Publication timeline: Finish experiments → Write paper (4 weeks) → Submit (week 1).
