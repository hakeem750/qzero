# Quoridor AlphaZero

AlphaZero-style reinforcement learning for the board game Quoridor.

---

## Notable Improvements Over Blueprint

| Area | Blueprint | This Implementation |
|---|---|---|
| **Residual block** | Plain conv-BN-ReLU | + **Squeeze-and-Excitation** channel gates |
| **Precision** | FP16 | **BF16** — same throughput, no overflow |
| **LR schedule** | Cosine only | **Warmup + cosine** — stable early training |
| **MCTS parallel** | No virtual loss | **Virtual loss** prevents thread collisions |
| **Dataset size** | 1× self-play | **2× via LR-symmetry augmentation** (free!) |
| **Draw detection** | Not specified | **500-move limit** prevents infinite games |
| **Transposition** | No cache | **Hash table** reuses identical states |
| **Policy head** | 2 filters | **4 filters** for 140-action space |
| **Init** | PyTorch default | **Kaiming normal** — avoids vanishing grad |
| **Buffer weights** | Uniform | Uniform + **PER hook** for future upgrade |

---

## Project Structure

```
quoridor_alpha_zero/
├── env/              Game environment (pure functional)
│   ├── state.py      Immutable QuoridorState
│   ├── rules.py      Legal moves, BFS path check, apply_action
│   ├── actions.py    140-action encoding
│   ├── encoding.py   (17,9,9) tensor + LR-mirror augmentation
│   └── quoridor_env.py  Gym-style wrapper
├── network/
│   ├── residual_block.py   ResidualBlock + SEBlock
│   ├── policy_value_net.py PolicyValueNet (stem + 20 SE-res blocks)
│   └── inference_server.py Batched BF16 GPU inference
├── mcts/
│   ├── node.py       Node with virtual loss
│   └── search.py     MCTS: PUCT, Dirichlet, transposition table
├── replay/
│   └── buffer.py     Ring buffer, uint8/f16 compression, PER hook
├── selfplay/
│   └── game_generator.py  Full game → trajectory + augmentation
├── trainer/
│   └── train_loop.py  AdamW + warmup-cosine + BF16 AMP
├── evaluation/
│   └── arena.py      Promotion match + ELO tracker
├── tests/            Pytest suites
├── scripts/train.py  Main orchestrator
├── configs/default.yaml
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train (CPU, quick test)

```bash
python scripts/train.py --device cpu --num_sims 50 --num_workers 1 --train_steps 1000
```

### Train (GPU, full run)

```bash
python scripts/train.py --device cuda --num_sims 800 --num_workers 4 --train_steps 100000
```

`torch.compile` is disabled by default during self-play training because the
inference server uses small, variable CUDA batches while filling the replay
buffer. Use `--compile` only for explicit comparison runs.

### Resume

```bash
python scripts/train.py --resume
```

### Standalone evaluation

Run candidate vs best without starting training, and watch the first game in the terminal:

```bash
python scripts/evaluate.py --display board --num-games 1
```

If your checkpoints live under a different folder, point the evaluator at that directory:

```bash
python scripts/evaluate.py --checkpoint-dir checkpoints/progress_gpu --display board --num-games 1
```

To evaluate a specific checkpoint:

```bash
python scripts/evaluate.py --candidate checkpoints/model_step_0000003.pt --display moves
```

### Research logging

Every training run can write a manifest and metric stream under `research/experiments/<experiment_id>/`.

Use an explicit experiment id when you want a stable name:

```bash
python scripts/train.py --experiment_id E023 --checkpoint_dir checkpoints/E023 --buffer_path data/E023.npz ...
```

The Markdown research log template lives in `research/research_log.md`, and the paper draft outline lives in `research/paper.md`.

### Run tests

```bash
pytest tests/ -v
```

### Docker

```bash
docker build -t quoridor-az .
docker run --gpus all quoridor-az
```

---

## Architecture

```
Input (17, 9, 9)
    ↓
Stem Conv (3×3, 256ch)
    ↓
20 × SE-ResidualBlock (256ch)
    ├── Conv 3×3 → BN → ReLU
    ├── Conv 3×3 → BN
    ├── SE gate (256→64→256, sigmoid)
    └── + residual → ReLU
    ↓
┌───────────────┐   ┌─────────────────────┐
│  Policy Head  │   │    Value Head        │
│  Conv 1×1 ×4  │   │  Conv 1×1 ×1        │
│  BN → ReLU    │   │  BN → ReLU → Flat   │
│  Flatten      │   │  Linear(81→256)     │
│  Linear→140   │   │  ReLU               │
│  (logits)     │   │  Linear(256→1)      │
└───────────────┘   │  Tanh  ∈ [-1,1]    │
                    └─────────────────────┘
```

---

## Action Space (140 actions)

| Range | Meaning |
|---|---|
| 0–3 | N / S / E / W pawn move |
| 4–7 | Jump N / S / E / W over opponent |
| 8–11 | Diagonal side-step (NW / NE / SW / SE) |
| 12–75 | Horizontal wall at anchor (r, c) |
| 76–139 | Vertical wall at anchor (r, c) |

---

## Hyperparameters (default)

| Param | Value |
|---|---|
| MCTS simulations | 800 |
| c_puct | 1.5 |
| Dirichlet α | 0.3 |
| Noise fraction | 0.25 |
| Virtual loss | 3 |
| Batch size | 1024 |
| Learning rate | 3e-4 |
| LR warmup steps | 1 000 |
| Cosine steps | 100 000 |
| Weight decay | 1e-4 |
| Grad clip | 1.0 |
| Replay capacity | 500 000 |
| Promotion threshold | 55% |

---

## Research Extensions

Planned future work:
- Gumbel AlphaZero (improved MCTS with guaranteed improvement)
- MuZero learned dynamics model
- Transformer backbone (board-as-token sequence)
- Triton MCTS kernels
- Distributed Ray-based actor pool
- Prioritised Experience Replay (PER hook already in buffer)
- FP8 training (H100+)
