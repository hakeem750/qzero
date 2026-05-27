# Clean Restart Guide: Fixing Curriculum Stagnation

## Problem Diagnosis

Your 80k buffer was trained exclusively with short games (~60-120 moves). At these lengths, **the mathematically optimal strategy is to never place a wall** because:

- A wall placement costs 1 move
- In a 60-move game, walls never pay off strategically  
- Your network learned the correct strategy for the wrong game: **sprint to the goal**

When evaluation runs with longer move limits (300+), the network has no gradient signal for wall placement and reverts to random wall placement. This is not a bug—it's exactly what the network was trained to do.

## Root Cause: Missing Curriculum Advancement

The curriculum never progressed because diagnostic data wasn't being properly collected. Now with the added logging, you can verify the issue.

## Clean Restart Steps

### 0. Run Diagnostics First (Optional)

Before restarting, check what the untrained network returns:

```bash
python diagnose_winners.py
```

This runs 5 games with minimal MCTS and prints `game_winner` values. You should see a mix of 1, 2, and 0 outcomes.

### 1. Delete the Poisoned Buffer

The 80k existing samples will bias training toward sprinting for thousands of steps even with new data. **You must start fresh:**

```bash
rm data/replay_buffer.npz
```

### 2. Launch Fresh Training with Corrected Settings

```bash
python scripts/train.py \
  --num_sims 150 \
  --num_workers 4 \
  --eval_every 3000 \
  --eval_games 40 \
  --eval_sims 100 \
  --ckpt_every 1000 \
  --force_max_moves 400 \
  --min_buffer_size 8000 \
  --train_steps 500000 \
  --fresh_buffer
```

### Key Parameter Explanations

| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| `--force_max_moves` | N/A (300 default) | **400** | Long enough for wall blocking to matter, short enough games finish quickly |
| `--num_sims` | 50 | **150** | Wall strategy needs exploration of 128 wall placements + 12 moves at root; 50 barely scratches the surface |
| `--num_workers` | 6 | **4** | At 150 sims, GPU becomes bottleneck; 6 workers will queue-starve each other |
| `--train_steps` | 100k | **500k** | Learn wall strategy requires more gradient steps than sprinting |
| `--min_buffer_size` | 2k | **8k** | Ensures stable training before curriculum advances |

## What to Expect: First 5000 Steps

### Good Signs:
✓ Buffer growing by 1000-2000 samples per 100 training steps  
✓ `z` (target_value_std) staying near 1.0 → decisive outcomes  
✓ `v_loss` rising to 0.1-0.3 initially → value head encountering uncertainty  
✓ `v_loss` slowly falling after 5000 steps → learning to predict value  
✓ Wall placements appearing in self-play with **strategic clustering** near opponent's path  

### Red Flags:
✗ Buffer growing slowly (<500/100 steps) → game length too short  
✗ `z` near 0.0 → almost all draws, curriculum can't advance  
✗ `v_loss` staying near 0.001 after 5000 steps → value computation bug  
✗ Walls randomly scattered everywhere → network still learning nothing  

## Buffer Growth Expected

- **Moves 1-5000:** ~20k-30k samples (1000+ per 100 steps, longer games)
- **Moves 5000-20000:** ~40k-60k new samples (plateau as buffer mixes racing + strategy)
- **Moves 50000+:** Clear wall placement patterns in generated games

## Why This Works

1. **Longer games (400 moves)** → Wall placement has time to pay off strategically
2. **Stronger MCTS (150 sims)** → Explores both move and wall space thoroughly  
3. **Fewer workers (4)** → Each worker gets GPU time; inference server not bottlenecked
4. **Fresh buffer** → No racing samples biasing every batch
5. **Larger min_buffer_size (8k)** → Curriculum starts from stable foundation

## Monitoring Progress

```bash
# Watch training in real-time
tail -f training_output.log | grep "step"

# Key metrics to monitor:
# [step NNNNNNN] buf=NNNNNN  loss=X.XXXX  p=X.XXXX  v=X.XXXX  z=+0.XXX+/-0.XXX
#
# buf = replay buffer size (should grow steadily)
# v = value loss (should start high ~0.15 then fall)
# z = target value stats (std should stay ~1.0 for decisive games)
```

## If It Fails: Debugging Checklist

1. **Buffer not growing:** Game length limit too low, or inference server bottleneck
   - Check stdout for `[worker] game_winner=X len=Y` - if `len` is tiny, max_moves limit is too small

2. **All draws / discarded games:** self-play is hitting the move cap before a real winner
   - Check `[worker]` output: should see roughly equal 1s and 2s, not all 0/None

3. **v_loss stuck at 0.001:** Value head regression bug
   - Look for NaN/Inf in loss; check buffer outcome distribution

4. **Walls still random at 50k steps:** Value computation or MCTS depth issue
   - Try increasing `--num_sims` to 200; reduce temperature in `game_generator.py`
   - If still random after 50k: Check value head MSE loss calculation; verify MCTS tree is exploring wall actions

## Keeping Your Model Checkpoint

If you had a previous model that learned something useful (even just sprinting):

```bash
python scripts/train.py \
  --resume \
  --force_max_moves 400 \
  --fresh_buffer \
  ...other args
```

The `--resume` flag loads the model weights but `--fresh_buffer` discards the old buffer. This lets the network keep its learned sprint patterns while retraining on new wall-focused samples.

## Expected Timeline

- **0-5k steps:** Wall strategy discovery (v_loss high)
- **5k-50k:** Consolidating wall placement patterns (v_loss falling)
- **50k+:** Stable play with strategic blocking (v_loss stabilized)

**Critical checkpoint at 50k steps:** If walls are still random placement (not strategically clustered) → value computation or MCTS depth issue, not a tuning problem. This indicates the core learning signal isn't reaching the network.
