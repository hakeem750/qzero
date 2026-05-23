"""
scripts/train.py - Main AlphaZero training orchestrator.

Usage:
  python scripts/train.py
  python scripts/train.py --resume
  python scripts/train.py --fresh_buffer
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import threading
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from evaluation.arena import Arena
from network.inference_server import InferenceServer
from network.policy_value_net import PolicyValueNet, build_net
from replay.buffer import ReplayBuffer
from selfplay.game_generator import GameGenerator
from trainer.train_loop import TrainConfig, TrainLoop


CKPT_DIR = pathlib.Path("checkpoints")
DEFAULT_BUFFER_PATH = pathlib.Path("data") / "replay_buffer.npz"


def set_seeds(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_checkpoint(model: PolicyValueNet, step: int) -> pathlib.Path:
    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / f"model_step_{step:07d}.pt"
    torch.save(model.state_dict(), path)
    print(f"  saved checkpoint {path}")
    return path


def load_latest_checkpoint(model: PolicyValueNet) -> int:
    ckpts = sorted(CKPT_DIR.glob("model_step_*.pt"))
    if not ckpts:
        return 0
    model.load_state_dict(torch.load(ckpts[-1], map_location="cpu"))
    step = int(ckpts[-1].stem.split("_")[-1])
    print(f"Resumed model from {ckpts[-1]}")
    return step


def save_replay_buffer(buffer: ReplayBuffer, path: pathlib.Path) -> None:
    t0 = time.time()
    saved_path = buffer.save(path)
    print(f"  saved replay buffer ({buffer.size} samples) to {saved_path} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Cold-start: fill buffer with random games — zero inference cost.
# Requires GameGenerator to support inference_fn=None / random_policy=True.
# If your GameGenerator doesn't support this, remove the call in main().
# ---------------------------------------------------------------------------
def fill_buffer_randomly(buffer: ReplayBuffer, target: int) -> None:
    print(f"Cold-start: filling buffer with random games to {target} samples...")
    try:
        # Cold-start with random play — still use higher Dirichlet for
        # better exploration diversity of action types (moves vs walls)
        gen = GameGenerator(
            inference_fn=None,
            num_simulations=0,
            augment=True,
            dirichlet_alpha=1.0,  # Encourages diverse action exploration
            noise_frac=0.5,       # Even though random, structure helps
        )
        while buffer.size < target:
            steps = gen.generate()
            for step in steps:
                buffer.push(step.obs, step.policy, step.outcome)
            print(f"  cold-start: {buffer.size} / {target}", end="\r")
        print(f"\n  Cold-start done: {buffer.size} samples")
    except Exception:
        # GameGenerator doesn't support random mode — skip silently.
        print("  Cold-start skipped (GameGenerator does not support random_policy).")


# ---------------------------------------------------------------------------
# Selfplay worker: uses cheap warmup sims until buffer is ready, then
# switches to full sims. Full traceback on errors so nothing stays silent.
# ---------------------------------------------------------------------------
def selfplay_worker(
    inference_fn,
    buffer: ReplayBuffer,
    num_simulations: int,
    stop_event: threading.Event,
    warmup_sims: int = 100,
    min_buffer_size: int = 2_000,
) -> None:
    # Warmup: stronger Dirichlet noise (higher alpha) to encourage
    # diverse exploration across moves AND walls, not just one action type
    gen_warmup = GameGenerator(
        inference_fn=inference_fn,
        num_simulations=warmup_sims,
        dirichlet_alpha=1.0,     # Higher: more uniform exploration
        noise_frac=0.5,          # More influential noise during warmup
        augment=True,
    )
    # Full play: still encourage diverse exploration but slightly less than warmup
    # This maintains action diversity throughout training, not just early on
    gen_full = GameGenerator(
        inference_fn=inference_fn,
        num_simulations=num_simulations,
        dirichlet_alpha=0.5,     # Better than standard (0.3), encourages diversity
        noise_frac=0.35,         # Stronger than standard (0.25) for better exploration
        augment=True,
    )

    while not stop_event.is_set():
        gen = gen_warmup if buffer.size < min_buffer_size else gen_full
        try:
            steps = gen.generate()
            for step in steps:
                buffer.push(step.obs, step.policy, step.outcome)
        except Exception:
            traceback.print_exc()   # full stack trace — never silent
            time.sleep(1)           # avoid a tight error loop


def print_training_status(metrics: dict, buffer: ReplayBuffer, elapsed: float) -> None:
    print(
        f"[step {int(metrics.get('step', 0)):>7}] "
        f"buf={buffer.size:>6}  "
        f"loss={metrics.get('loss', 0):.4f}  "
        f"p={metrics.get('policy_loss', 0):.4f}  "
        f"v={metrics.get('value_loss', 0):.4f}  "
        f"net_H={metrics.get('entropy', 0):.3f}  "
        f"target_H={metrics.get('target_entropy', 0):.3f}  "
        f"top_p={metrics.get('target_top_prob', 0):.3f}  "
        f"z={metrics.get('target_value_mean', 0):+.3f}"
        f"+/-{metrics.get('target_value_std', 0):.3f}  "
        f"lr={metrics.get('lr', 0):.2e}  "
        f"t={elapsed:.1f}s"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,            default=42)
    parser.add_argument("--device",          type=str,            default="cuda" if torch.cuda.is_available() else "cpu")
    # Raised from 2 → 8 so there are enough concurrent requests to saturate
    # the inference server and reduce per-call latency during buffer fill.
    parser.add_argument("--num_workers",     type=int,            default=8)
    parser.add_argument("--num_sims",        type=int,            default=200)
    # Cheap sims used only until min_buffer_size is reached (~10× faster fill).
    parser.add_argument("--warmup_sims",     type=int,            default=20)
    parser.add_argument("--train_steps",     type=int,            default=100_000)
    parser.add_argument("--eval_every",      type=int,            default=2_000)
    parser.add_argument("--ckpt_every",      type=int,            default=500)
    parser.add_argument("--log_every",       type=int,            default=100)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--buffer_path",     type=pathlib.Path,   default=DEFAULT_BUFFER_PATH)
    parser.add_argument("--resume_buffer",   action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fresh_buffer",    action="store_true", help="Start with an empty replay buffer instead of loading the saved one.")
    parser.add_argument("--save_buffer_every", type=int,          default=1_000)
    parser.add_argument("--min_buffer_size", type=int,            default=2_000)
    # Set to 0 to disable the random cold-start fill.
    parser.add_argument("--cold_start_size", type=int,            default=1_000,
                        help="Fill this many positions with random games before starting workers. Set 0 to skip.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    set_seeds(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    model = build_net(compile_model=(device.type == "cuda"), device=device)
    best_model = copy.deepcopy(model)

    start_step = 0
    if args.resume:
        start_step = load_latest_checkpoint(model)
        best_model.load_state_dict(model.state_dict())

    # Inference batch size matched to worker count so the server flushes
    # frequently instead of waiting to accumulate 64 requests from 2 workers.
    inference_batch_size = min(args.num_workers, 32)
    server = InferenceServer(model, device=str(device), batch_size=inference_batch_size)
    server.start()
    print(f"Inference server started (batch_size={inference_batch_size})")

    def inference_fn(obs_np, mask_np):
        future = server.submit(obs_np[0], mask_np[0])
        policy, value = future.result()
        return policy[np.newaxis], np.array([[value]])

    resume_buffer = not args.fresh_buffer
    if resume_buffer and args.buffer_path.exists():
        buffer = ReplayBuffer.load(args.buffer_path, capacity=500_000)
        print(f"Loaded replay buffer from {args.buffer_path} ({buffer.size} samples)")
    else:
        buffer = ReplayBuffer(capacity=500_000)
        if resume_buffer:
            print(f"No replay buffer found at {args.buffer_path}; starting empty")
        else:
            print("Starting with a fresh replay buffer")

    # Cold-start: fill half of min_buffer_size with free random games so
    # workers immediately start with warmup sims rather than from zero.
    cold_target = min(args.cold_start_size, args.min_buffer_size // 2)
    if cold_target > 0 and buffer.size < cold_target:
        fill_buffer_randomly(buffer, cold_target)

    cfg = TrainConfig(device=str(device), log_every=args.log_every)
    loop = TrainLoop(model, buffer, cfg)
    loop.step = start_step

    arena = Arena(num_games=20, num_sims=100, win_thresh=0.55, device=str(device))

    stop_event = threading.Event()
    workers = []
    for _ in range(args.num_workers):
        t = threading.Thread(
            target=selfplay_worker,
            args=(inference_fn, buffer, args.num_sims, stop_event,
                  args.warmup_sims, args.min_buffer_size),
            daemon=True,
        )
        t.start()
        workers.append(t)

    print(
        f"Started {args.num_workers} self-play workers "
        f"(warmup_sims={args.warmup_sims}, full_sims={args.num_sims}). "
        f"Waiting for buffer to fill to {args.min_buffer_size}..."
    )
    while not buffer.is_ready(args.min_buffer_size):
        time.sleep(2)
        alive = sum(t.is_alive() for t in workers)
        print(f"  Buffer: {buffer.size:>5} / {args.min_buffer_size}  workers_alive={alive}/{len(workers)}")
        if alive == 0:
            print("ERROR: all selfplay workers have crashed. Check tracebacks above.")
            server.stop()
            sys.exit(1)

    print("Training started.")
    train_t0 = time.time()
    try:
        for _ in range(args.train_steps):
            metrics = loop.train_step()
            if metrics.get("skipped"):
                print(f"[step {loop.step:>7}] skipped non-finite loss")
                continue

            if (loop.step % args.log_every) == 0:
                print_training_status(metrics, buffer, time.time() - train_t0)

            if (loop.step % args.ckpt_every) == 0:
                save_checkpoint(model, loop.step)

            if args.save_buffer_every > 0 and (loop.step % args.save_buffer_every) == 0:
                save_replay_buffer(buffer, args.buffer_path)

            if (loop.step % args.eval_every) == 0:
                print(f"\n[step {loop.step}] Running evaluation...")
                result = arena.evaluate(model, best_model)
                print(
                    f"  win_rate={result['win_rate']:.3f}  "
                    f"W/D/L={result['wins']}/{result['draws']}/{result['losses']}  "
                    f"promoted={result['promoted']}"
                )
                if result["promoted"]:
                    best_model.load_state_dict(model.state_dict())
                    print("  New best model promoted!")
    except KeyboardInterrupt:
        print("\nInterrupted; saving checkpoint and replay buffer...")
    finally:
        stop_event.set()
        for t in workers:
            t.join(timeout=5)
        server.stop()
        save_checkpoint(model, loop.step)
        save_replay_buffer(buffer, args.buffer_path)

    print("Training complete.")


if __name__ == "__main__":
    main()