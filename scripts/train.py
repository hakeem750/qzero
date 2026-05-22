"""
scripts/train.py — Main AlphaZero training orchestrator.

Usage:
  python scripts/train.py
  python scripts/train.py --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import os
import threading
import time
import copy
import pathlib

import numpy as np
import torch

# Ensure project root is on path when run from scripts/
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from network.policy_value_net import build_net, PolicyValueNet
from network.inference_server import InferenceServer
from replay.buffer import ReplayBuffer
from selfplay.game_generator import GameGenerator
from trainer.train_loop import TrainLoop, TrainConfig
from evaluation.arena import Arena


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def set_seeds(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
CKPT_DIR = pathlib.Path("checkpoints")

def save_checkpoint(model: PolicyValueNet, step: int) -> pathlib.Path:
    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / f"model_step_{step:07d}.pt"
    torch.save(model.state_dict(), path)
    print(f"  ✔ saved {path}")
    return path

def load_latest_checkpoint(model: PolicyValueNet) -> int:
    ckpts = sorted(CKPT_DIR.glob("model_step_*.pt"))
    if not ckpts:
        return 0
    model.load_state_dict(torch.load(ckpts[-1], map_location="cpu"))
    step = int(ckpts[-1].stem.split("_")[-1])
    print(f"Resumed from {ckpts[-1]}")
    return step


# ---------------------------------------------------------------------------
# Self-play worker thread
# ---------------------------------------------------------------------------
def selfplay_worker(
    inference_fn,
    buffer: ReplayBuffer,
    num_simulations: int,
    stop_event: threading.Event,
) -> None:
    gen = GameGenerator(
        inference_fn=inference_fn,
        num_simulations=num_simulations,
        augment=True,
    )
    while not stop_event.is_set():
        try:
            steps = gen.generate()
            for step in steps:
                buffer.push(step.obs, step.policy, step.outcome)
        except Exception as e:
            print(f"[selfplay worker error] {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_sims",    type=int, default=200)  # lower for quick start
    parser.add_argument("--train_steps", type=int, default=100_000)
    parser.add_argument("--eval_every",  type=int, default=2_000)
    parser.add_argument("--ckpt_every",  type=int, default=1_000)
    parser.add_argument("--resume",      action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Build model
    model      = build_net(compile_model=(device.type == "cuda"), device=device)
    best_model = copy.deepcopy(model)

    start_step = 0
    if args.resume:
        start_step = load_latest_checkpoint(model)
        best_model.load_state_dict(model.state_dict())

    # Inference server (batched GPU)
    server = InferenceServer(model, device=str(device), batch_size=64)
    server.start()

    def inference_fn(obs_np, mask_np):
        future = server.submit(obs_np[0], mask_np[0])
        policy, value = future.result()
        return policy[np.newaxis], np.array([[value]])

    # Replay buffer
    buffer = ReplayBuffer(capacity=500_000)

    # Training
    cfg = TrainConfig(device=str(device))
    loop = TrainLoop(model, buffer, cfg)
    loop.step = start_step

    # Evaluation arena
    arena = Arena(num_games=20, num_sims=100, win_threshold=0.55, device=str(device))

    # Start self-play workers
    stop_event = threading.Event()
    workers = []
    for _ in range(args.num_workers):
        t = threading.Thread(
            target=selfplay_worker,
            args=(inference_fn, buffer, args.num_sims, stop_event),
            daemon=True,
        )
        t.start()
        workers.append(t)

    print("Self-play workers started. Waiting for buffer to fill …")
    while not buffer.is_ready(2_000):
        time.sleep(2)
        print(f"  Buffer: {buffer.size} / 2000")

    print("Training started.")
    for i in range(args.train_steps):
        metrics = loop.train_step()

        if (loop.step % args.ckpt_every) == 0:
            save_checkpoint(model, loop.step)

        if (loop.step % args.eval_every) == 0:
            print(f"\n[step {loop.step}] Running evaluation …")
            result = arena.evaluate(model, best_model)
            print(f"  win_rate={result['win_rate']:.3f}  "
                  f"W/D/L={result['wins']}/{result['draws']}/{result['losses']}  "
                  f"promoted={result['promoted']}")
            if result["promoted"]:
                best_model.load_state_dict(model.state_dict())
                print("  ★ New best model promoted!")

    stop_event.set()
    server.stop()
    save_checkpoint(model, loop.step)
    print("Training complete.")


if __name__ == "__main__":
    main()
