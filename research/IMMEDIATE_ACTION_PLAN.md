# Immediate Action Plan: Making QZero Publishable

**Goal:** Transform QZero from working implementation → publishable research within 3 months

**Current State:** E002 training at 14,000 / 50,000 steps

---

## 🎯 What's Needed (In Priority Order)

### Tier 1: Minimum Viable Publication (2-4 weeks)

These 3 things are **required** for any publication:

#### 1.1 Baseline Agents (1 week)
**Why:** Can't claim strength without saying "stronger than what?"

**What to build:**
- [ ] `agents/random_agent.py` — Random legal moves (baseline: 0%)
- [ ] `agents/heuristic_agent.py` — Greedy path-following (estimated: 30-40%)
- [ ] `agents/minimax_agent.py` — Alpha-beta search depth 4 (estimated: 40-50%)

**Deliverable:** `evaluation/arena.py` updated to support baseline agents

**Estimated effort:** 3-4 days

**Example code structure:**
```python
# agents/heuristic_agent.py
class HeuristicAgent:
    """Greedy: move toward goal; block opponent; place walls strategically"""
    def get_action(self, state, legal_moves):
        # 1. Check if can move toward goal → prefer
        # 2. Check if can block opponent → second priority  
        # 3. Place walls to block paths
        # 4. Random if stuck
        return action
```

#### 1.2 Arena Evaluation Loop (1 week)
**Why:** Need quantified head-to-head results for paper

**What to implement:**
- [ ] `evaluation/arena.py` extended to track:
  - Win/loss/draw counts
  - Win rates by agent matchup
  - Game statistics (avg length, walls used, etc.)
- [ ] Save game replays for each arena match
- [ ] Log results to CSV + JSON for graphing

**Deliverable:** Script to run 100-game matches and produce win rate tables

**Estimated effort:** 3-4 days

**Example output:**
```
QZero (E002 @ 14k) vs. Heuristic
Games: 100 | QZero wins: 62 | Heuristic wins: 38 | Draws: 0
Win rate: 62% ± 5%
```

#### 1.3 Checkpoint Evaluation (1 week)
**Why:** Track progression; show learning curve for paper

**What to implement:**
- [ ] Evaluation at every 5k training steps:
  - [ ] E002 @ 5k vs. heuristic (10 games)
  - [ ] E002 @ 10k vs. heuristic (10 games)
  - [ ] E002 @ 20k vs. heuristic (20 games)
  - [ ] E002 @ 50k vs. heuristic (50 games)
- [ ] Graph: Training steps → Win rate
- [ ] Document in RESULTS.md

**Deliverable:** Learning curve showing strength progression

**Estimated effort:** 2-3 days (mostly automated)

**Expected result:** 
```
Step    Win Rate vs. Heuristic
─────────────────────────────
1k      35% (not yet learning)
5k      42%
10k     52% (threshold of competence)
20k     68%
50k     82%+ (strong play)
```

---

### Tier 2: Strong Publication (4-8 weeks)

These additions make a solid conference paper:

#### 2.1 SE Block Ablation (E004) (1-2 weeks)
**Design:**
- E002: 20 SE-Residual blocks (baseline)
- E004: 20 standard residual (no SE gates)
- Same config: 800 sims, 4 workers, 50k steps

**Metrics:**
- Loss comparison at 5k, 10k, 20k steps
- Final strength (Elo gain per step)
- Inference latency

**Deliverable:** Table in paper.md Results section

**Example:**
```
Architecture    Loss (10k)    Elo gain/k steps    Inference (ms)
─────────────────────────────────────────────────────────────
SE-Residual     2.10          +26 Elo            42ms
Standard        2.14          +25 Elo            41ms
Difference      +1.9%         -3.8%              +2.4%
```

#### 2.2 Learning Rate Augmentation Ablation (E016) (1-2 weeks)
**Design:**
- E002: Augmentation enabled (LR flip)
- E016: Augmentation disabled
- Same config, different buffer

**Metrics:**
- Convergence speed (steps to 70% win rate)
- Final strength
- Buffer diversity measures

**Deliverable:** Validation of "2× dataset size" claim

**Expected:**
- E002 (augmented): Reaches 70% win rate by ~25k steps
- E016 (no augment): Reaches 70% win rate by ~35k steps
- **Claim:** "Augmentation accelerates convergence by 40%"

#### 2.3 MCTS Simulation Study (E010, E011, E012) (2-3 weeks)
**Design:**
- E010: 50 simulations per move
- E011: 200 simulations per move  
- E002: 800 simulations per move (baseline)
- E012: 1600 simulations per move

**Metrics:**
- Win rate vs. heuristic at each level
- Training speed (games/sec)
- Search tree depth

**Deliverable:** Diminishing returns curve

**Expected result:**
```
Simulations    Win Rate vs. Heuristic    Speed (games/sec)    Elo gain/sec
────────────────────────────────────────────────────────────────────────
50             38%                       3000                 +0.8 Elo/sec
200            55%                       1800                 +0.9 Elo/sec
800            82%                       800                  +0.95 Elo/sec  ← sweet spot
1600           87%                       400                  +0.85 Elo/sec
```

---

### Tier 3: Excellent Publication (8-12 weeks)

These make it top-tier conference material:

#### 3.1 Game Replay Analysis (1-2 weeks)
**What to collect:**
- [ ] 50 games from E002 @ 1k steps
- [ ] 50 games from E002 @ 10k steps
- [ ] 50 games from E002 @ 50k steps
- [ ] Analyze each for:
  - Opening move distribution
  - Wall placement patterns
  - Strategic complexity
  - Common mistakes

**Deliverable:** "How Strategy Emerges" section in paper

**Example analysis:**
```
Opening moves (first 5):
  Step 1k:   Uniform random (all opening moves ~equal)
  Step 10k:  Concentrated on 3-4 moves (learning opens)
  Step 50k:  70% plays one specific opening (repertoire)

Mid-game wall placement:
  Step 1k:   Walls anywhere (no strategy)
  Step 10k:  Walls near opponent pawn (~40% near opponent)
  Step 50k:  Complex wall chains (70% coordinate with prior walls)
```

#### 3.2 Hyperparameter Sensitivity Study (1-2 weeks)
**Vary one at a time:**
- Learning rate: 1.5e-4, 3e-4 (baseline), 6e-4
- Batch size: 512, 1024 (baseline), 2048
- Warmup steps: 0, 1000 (baseline), 5000
- Workers: 2, 4 (baseline), 8

**Deliverable:** "Hyperparameter Sensitivity" appendix

**Example:**
```
Hyperparameter    Default    Impact on Loss    Impact on Speed
──────────────────────────────────────────────────────────────
LR                3e-4       N/A               N/A
  ↳ 1.5e-4                   +0.2 (slower)     No change
  ↳ 6e-4                     +0.1 (faster)     +5%
Batch size        1024       N/A               N/A
  ↳ 512                      Slightly worse    -30% speed
  ↳ 2048                     Slightly better   +40% speed
```

#### 3.3 Classical AI Comparison (1-2 weeks)
**What to implement:**
- [ ] Minimax with alpha-beta pruning (depth 4, 6, 8)
- [ ] Hand-tuned evaluation function
- [ ] Head-to-head: QZero vs. each

**Deliverable:** "QZero vs. Classical Methods" section

**Expected:**
```
Agent              Strength    Decision Time/move    Games Won
────────────────────────────────────────────────────────────
QZero (E002)       Elo 1900    ~50ms                62/100
Minimax d=4        Elo 1650    ~200ms               38/100
Minimax d=8        Elo 1750    ~5000ms              45/100
Heuristic          Elo 1400    ~10ms                25/100
Random             Elo 0       ~1ms                 0/100

→ Claim: "Learned MCTS + NN outperforms classical search
   by 250 Elo while being 100× faster at same depth"
```

---

## 📋 Execution Checklist: Next 4 Weeks

### Week 1: Baselines + Arena (Tier 1.1-1.2)
- [ ] Day 1-2: Implement random agent
- [ ] Day 3-4: Implement heuristic agent (greedy path-following)
- [ ] Day 5: Implement minimax agent (alpha-beta)
- [ ] Day 6-7: Update arena.py; test agents
- [ ] **Deliverable:** Arena script that runs 100-game matches

### Week 2: Checkpoint Evaluation (Tier 1.3)
- [ ] Day 1-2: Modify training loop to eval every 5k steps
- [ ] Day 3-4: Run evaluations at key checkpoints (E002 ongoing)
- [ ] Day 5-6: Collect win rate data; generate graph
- [ ] Day 7: Update RESULTS.md with learning curve
- [ ] **Deliverable:** Learning curve graph showing 1k → 50k step progression

### Week 3: SE Block Ablation (Tier 2.1)
- [ ] Day 1: Finalize E004 config (different only in SE blocks)
- [ ] Day 2-5: Run E004 training (10k steps)
- [ ] Day 6-7: Analyze results; compare to E002
- [ ] **Deliverable:** SE block comparison table

### Week 4: Next Ablations (Tier 2.2-2.3)
- [ ] Day 1-2: Start E016 (no augmentation)
- [ ] Day 3-4: Start E010 (50 sims)
- [ ] Day 5-7: Collect initial results; plan Phase 2
- [ ] **Deliverable:** Design document for remaining experiments

### Ongoing (During all weeks)
- [ ] Update RESULTS.md with new metrics
- [ ] Add entries to research_log.md at milestones
- [ ] Monitor E002 training progression
- [ ] Document findings in paper.md Results section

---

## 🎯 Success Metrics for Each Phase

### After Week 1:
- ✓ Baseline agents working (random, heuristic, minimax)
- ✓ Arena evaluation script functional
- ✓ First benchmark: E002 @ 14k vs. baselines

### After Week 2:
- ✓ Learning curve established
- ✓ Competence threshold identified (e.g., 50% vs. heuristic @ 10k steps)
- ✓ Clear evidence of learning

### After Week 4:
- ✓ 2-3 ablation studies completed
- ✓ Clear findings (SE blocks: +X%, augmentation: +Y%)
- ✓ Paper.md Results section populated with real data

### After Week 8:
- ✓ 4-6 ablation studies completed
- ✓ Game replay analysis done
- ✓ Paper draft nearly complete
- ✓ Ready to submit to ArXiv

---

## 📄 Publication Timeline

```
Week 1-4:   Tier 1 experiments (baselines, arena, E002 eval)
Week 5-8:   Tier 2 experiments (ablations)
Week 9-12:  Tier 3 experiments (optional; polishing)
Week 13:    Write paper + prepare submission
Week 14:    Submit to ArXiv
Week 15+:   Conference/journal submission
```

**Minimum to publication: 14 weeks**

---

## 🔨 Implementation Priority

### Must Do (Week 1-2)
1. Random agent
2. Heuristic agent  
3. Arena evaluation
4. Checkpoint eval loop

### Should Do (Week 3-4)
1. SE block ablation (E004)
2. Start augmentation ablation (E016)
3. Start MCTS simulation study (E010, E012)

### Could Do (Week 5-12)
1. Game replay analysis
2. Classical AI comparison (minimax)
3. Hyperparameter sensitivity
4. Transfer learning study
5. Opening book analysis

---

## 📊 Resources

### Code to Write: ~500-1000 lines
- Baseline agents: 200 lines
- Arena enhancements: 150 lines
- Evaluation loop: 100 lines
- Analysis scripts: 200 lines

### Compute Time: ~12 weeks GPU
- E002 continued: 6 weeks (50k steps total)
- E004 (SE ablation): 2 weeks (10k steps)
- E010, E011, E012 (MCTS study): 3 weeks
- E016 (augmentation): 2 weeks
- Other optional: 4 weeks

### Human Time: ~4-6 weeks
- Implementation & experiments: 3 weeks
- Analysis & writing: 2 weeks
- Revision & polish: 1 week

---

## 💡 Key Takeaway

To make QZero publishable, **shift from implementation to research**:

❌ **Implementation thinking:**
"Does it work?" → "Yes, training runs" → Done

✅ **Research thinking:**
"Does it work?" → "Yes, 62% vs. heuristic"  
"Why?" → "SE blocks help; augmentation speeds up 40%; needs 800 sims for peak"  
"How general?" → "Test on variant; transfer learning; other domains"  

**Your current checkpoint:** Implementation complete, research beginning

**The next 14 weeks:** Answer research questions with controlled experiments

---

## Next Steps

1. **Today:** Read [PUBLICATION_ROADMAP.md](PUBLICATION_ROADMAP.md) + [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)
2. **This week:** Start implementing baseline agents
3. **Next 2 weeks:** Complete Tier 1 checklist
4. **Ongoing:** Update research_log.md and RESULTS.md with findings

Would you like help implementing any of these components?
