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
from scripts.watch_match import _clean_state_dict_keys, latest_checkpoint


CKPT_DIR = pathlib.Path("checkpoints")
BEST_CKPT_PATH = CKPT_DIR / "best_model.pt"


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
    parser.add_argument("--best", type=pathlib.Path, default=BEST_CKPT_PATH,
                        help="Best-model checkpoint.")
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

    candidate_path = args.candidate or latest_checkpoint()
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    if not args.best.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {args.best}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Candidate: {candidate_path}")
    print(f"Best:      {args.best}")
    print(f"Device:    {device}")
    print(f"Games:     {args.num_games}")
    print(f"Sims:      {args.sims} per move")

    candidate = load_model(candidate_path, device)
    best = load_model(args.best, device)

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
