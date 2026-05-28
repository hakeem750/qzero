#!/usr/bin/env python3
"""
Training launcher with automatic environment setup.

This script handles:
  1. Setting CUBLAS_WORKSPACE_CONFIG for deterministic GPU behavior
  2. Showing diagnostics before training starts
  3. Launching the training orchestrator

Usage:
  python run_training.py [--quick|--medium|--full] [--device cuda|cpu] [--diagnostics]

Presets:
  --quick    : 1 worker, 50 sims, 1000 steps (test)
  --medium   : 4 workers, 200 sims, 10000 steps
  --full     : 8 workers, 800 sims, 100000 steps (default)

torch.compile is disabled by default for stable multi-worker CUDA self-play.
Use --compile only when deliberately testing compiled inference.
"""
import os
import sys
import subprocess
import argparse
import pathlib
import torch


def setup_environment():
    """Set environment variables for deterministic training."""
    if torch.cuda.is_available() and 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        print("✓ CUBLAS_WORKSPACE_CONFIG=:16:8 (deterministic mode)")
    
    # Disable warnings about deterministic algorithms
    os.environ['PYTHONWARNINGS'] = 'ignore'


def run_diagnostics(device: str):
    """Run diagnostic checks before training."""
    print("\n" + "=" * 70)
    print("RUNNING DIAGNOSTICS...")
    print("=" * 70)
    
    script_path = pathlib.Path(__file__).parent / "scripts" / "diagnose_workers.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--device", device],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n✗ Diagnostics failed. Fix errors above before training.")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Train Quoridor AlphaZero model")
    
    # Preset configurations
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--quick", action="store_true",
                        help="Quick test: 1 worker, 50 sims, 1k steps")
    preset.add_argument("--medium", action="store_true",
                        help="Medium run: 4 workers, 200 sims, 10k steps")
    preset.add_argument("--full", action="store_true",
                        help="Full training: 8 workers, 800 sims, 100k steps (default)")
    
    # General options
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run diagnostics before training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--fresh-buffer", action="store_true",
                        help="Start with fresh replay buffer")
    parser.add_argument("--compile", action="store_true",
                        help="Opt into torch.compile on CUDA. Disabled by default for stable buffer fill.")
    parser.add_argument("--no-compile", action="store_true",
                        help=argparse.SUPPRESS)
    
    # Training parameters (can override presets)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of self-play workers (overrides preset)")
    parser.add_argument("--num-sims", type=int, default=None,
                        help="MCTS simulations per move (overrides preset)")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Total training steps (overrides preset)")
    parser.add_argument("--eval-every", type=int, default=2000,
                        help="Evaluation frequency (default: 2000 steps)")
    parser.add_argument("--eval-games", type=int, default=4,
                        help="Games per evaluation (default: 4)")
    parser.add_argument("--eval-show", choices=["off", "moves", "board"], default="off",
                        help="Print evaluation matches to the console")
    parser.add_argument("--eval-show-games", type=int, default=1,
                        help="Number of evaluation games to show when --eval-show is enabled")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("✗ CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    
    print(f"✓ Device: {args.device}")
    if args.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Apply preset configurations
    if args.quick:
        num_workers = 1
        num_sims = 50
        train_steps = 1000
        preset_name = "QUICK"
    elif args.medium:
        num_workers = 4
        num_sims = 200
        train_steps = 10000
        preset_name = "MEDIUM"
    else:  # Default to full
        num_workers = 8
        num_sims = 800
        train_steps = 100000
        preset_name = "FULL"
    
    # Override with explicit arguments
    if args.num_workers is not None:
        num_workers = args.num_workers
    if args.num_sims is not None:
        num_sims = args.num_sims
    if args.train_steps is not None:
        train_steps = args.train_steps
    
    print(f"\n✓ Configuration: {preset_name}")
    print(f"  Workers: {num_workers}")
    print(f"  MCTS sims/move: {num_sims}")
    print(f"  Training steps: {train_steps}")
    print(f"  Evaluation every: {args.eval_every} steps ({args.eval_games} games)")
    if args.eval_show != "off":
        print(f"  Evaluation console view: {args.eval_show} ({args.eval_show_games} game(s))")
    print(f"  torch.compile: {'enabled' if args.compile and not args.no_compile else 'disabled'}")
    
    # Run diagnostics if requested
    if args.diagnostics:
        if not run_diagnostics(args.device):
            return 1
    
    # Build training command
    train_script = pathlib.Path(__file__).parent / "scripts" / "train.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--device", args.device,
        "--num_workers", str(num_workers),
        "--num_sims", str(num_sims),
        "--train_steps", str(train_steps),
        "--eval_every", str(args.eval_every),
        "--eval_games", str(args.eval_games),
        "--eval_show", args.eval_show,
        "--eval_show_games", str(args.eval_show_games),
    ]
    
    if args.resume:
        cmd.append("--resume")
    if args.fresh_buffer:
        cmd.append("--fresh_buffer")
    if args.compile and not args.no_compile:
        cmd.append("--compile")
    if args.no_compile:
        cmd.append("--no_compile")
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING...")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    
    # Launch training
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
