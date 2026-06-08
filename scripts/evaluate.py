"""
Standalone evaluation runner for Quoridor AlphaZero checkpoints.

Examples:
  python scripts/evaluate.py
  python scripts/evaluate.py --candidate checkpoints/model_step_0010000.pt
  python scripts/evaluate.py --candidate checkpoints/model_step_0010000.pt --best checkpoints/best_model.pt --display board
  python scripts/evaluate.py --num-games 10 --sims 50 --display moves
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from evaluation.arena import Arena
from network.policy_value_net import build_net
from scripts.watch_match import _clean_state_dict_keys


CKPT_DIR = pathlib.Path("checkpoints")
BEST_CKPT_PATH = CKPT_DIR / "best_model.pt"


def latest_checkpoint(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    ckpts = sorted(checkpoint_dir.glob("model_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_step_*.pt checkpoints found in {checkpoint_dir}")
    return ckpts[-1]


def load_model(path: pathlib.Path, device: torch.device):
    model = build_net(compile_model=False, device=device)
    data = torch.load(path, map_location=device)
    state_dict = data["model"] if isinstance(data, dict) and "model" in data else data
    model.load_state_dict(_clean_state_dict_keys(state_dict))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=pathlib.Path, default=None,
                        help="Candidate checkpoint. Defaults to latest model_step_*.pt.")
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=CKPT_DIR,
                        help="Directory containing model_step_*.pt and best_model.pt.")
    parser.add_argument("--best", type=pathlib.Path, default=BEST_CKPT_PATH,
                        help="Best-model checkpoint. Defaults to <checkpoint-dir>/best_model.pt.")
    parser.add_argument("--num-games", type=int, default=1,
                        help="Number of evaluation games to play.")
    parser.add_argument("--sims", type=int, default=25,
                        help="MCTS simulations per move.")
    parser.add_argument("--max-moves", type=int, default=300,
                        help="Move cap before adjudication.")
    parser.add_argument("--win-thresh", type=float, default=0.55,
                        help="Promotion threshold used for reporting.")
    parser.add_argument("--display", choices=["off", "moves", "board"], default="board",
                        help="How to show the first evaluated games.")
    parser.add_argument("--display-games", type=int, default=1,
                        help="Number of games to display.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    best_path = args.best if args.best != BEST_CKPT_PATH else checkpoint_dir / "best_model.pt"
    candidate_path = args.candidate or latest_checkpoint(checkpoint_dir)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_path}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Candidate: {candidate_path}")
    print(f"Best:      {best_path}")
    print(f"Device:    {device}")
    print(f"Games:     {args.num_games}")
    print(f"Sims:      {args.sims} per move")

    candidate = load_model(candidate_path, device)
    best = load_model(best_path, device)

    arena = Arena(
        num_games=args.num_games,
        num_sims=args.sims,
        win_thresh=args.win_thresh,
        max_moves=args.max_moves,
        device=str(device),
    )

    result = arena.evaluate(
        candidate,
        best,
        progress=True,
        display=args.display,
        display_games=args.display_games,
    )

    print(
        f"\nwin_rate={result['win_rate']:.3f}  "
        f"W/D/L={result['wins']}/{result['draws']}/{result['losses']}  "
        f"len={result['avg_game_length']:.1f}  "
        f"cutoff={result.get('cutoff_rate', 0.0):.2f}  "
        f"H={result['policy_entropy']:.3f}  "
        f"v_cal={result['value_calibration_mse']:.3f}  "
        f"elo={result['elo']:.1f}  "
        f"promoted={result['promoted']}"
    )


if __name__ == "__main__":
    main()
