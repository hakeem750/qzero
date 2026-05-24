"""
Watch a candidate checkpoint play against the saved best model.

Examples:
  python scripts/watch_match.py
  python scripts/watch_match.py --sims 10 --delay 0.5
  python scripts/watch_match.py --best-is-p1 --max-moves 80
  python scripts/watch_match.py --no-render
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from env.quoridor_env import QuoridorEnv
from env.state import MAX_MOVES
from evaluation.arena import _greedy_inference
from mcts.search import MCTS
from network.policy_value_net import build_net
from scripts.play import BoardPane, describe_action
from selfplay.game_generator import (
    _adjudicated_winner,
    _state_cycle_key,
    select_action_with_progress,
)


CKPT_DIR = pathlib.Path("checkpoints")
BEST_CKPT_PATH = CKPT_DIR / "best_model.pt"


def latest_checkpoint() -> pathlib.Path:
    ckpts = sorted(CKPT_DIR.glob("model_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_step_*.pt checkpoints found in {CKPT_DIR}")
    return ckpts[-1]


def _clean_state_dict_keys(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_model(path: pathlib.Path, device: torch.device):
    model = build_net(compile_model=False, device=device)
    data = torch.load(path, map_location=device)
    state_dict = data["model"] if isinstance(data, dict) and "model" in data else data
    model.load_state_dict(_clean_state_dict_keys(state_dict))
    model.eval()
    return model


def choose_action(
    mcts: MCTS,
    root,
    inference_fn,
    sims: int,
    state,
    seen_counts: dict[tuple, int],
) -> tuple[int, np.ndarray]:
    mcts.run_simulations_sync(root, inference_fn, sims, add_noise=False)
    probs = mcts.action_probs(root, temperature=1.0)
    action = select_action_with_progress(probs, state, temperature=0.0, seen_counts=seen_counts)
    return action, probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=pathlib.Path, default=None,
                        help="Candidate checkpoint. Defaults to latest model_step_*.pt.")
    parser.add_argument("--best", type=pathlib.Path, default=BEST_CKPT_PATH,
                        help="Best-model checkpoint.")
    parser.add_argument("--best-is-p1", action="store_true",
                        help="Let the best model play as Player 1. Default: candidate is Player 1.")
    parser.add_argument("--sims", type=int, default=25,
                        help="MCTS simulations per move for each side.")
    parser.add_argument("--max-moves", type=int, default=MAX_MOVES)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-render", action="store_true",
                        help="Print the board in the terminal; do not open the Tk board window.")
    parser.add_argument("--moves", action="store_true",
                        help="Print compact move lines instead of the terminal board.")
    parser.add_argument("--no-clear", action="store_true",
                        help="Do not clear the terminal between board frames.")
    parser.add_argument("--cell-size", type=int, default=58)
    args = parser.parse_args()

    candidate_path = args.candidate or latest_checkpoint()
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    if not args.best.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {args.best}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Candidate: {candidate_path}")
    print(f"Best:      {args.best}")
    print(f"Device:    {device}")
    print(f"Sims:      {args.sims} per move")

    candidate = load_model(candidate_path, device)
    best = load_model(args.best, device)
    candidate_fn = _greedy_inference(candidate, device, dtype)
    best_fn = _greedy_inference(best, device, dtype)

    candidate_is_p1 = not args.best_is_p1
    names = {
        1: "candidate" if candidate_is_p1 else "best",
        2: "best" if candidate_is_p1 else "candidate",
    }
    fns = {
        1: candidate_fn if candidate_is_p1 else best_fn,
        2: best_fn if candidate_is_p1 else candidate_fn,
    }
    mcts = {1: MCTS(c_puct=1.5, noise_frac=0.0), 2: MCTS(c_puct=1.5, noise_frac=0.0)}

    env = QuoridorEnv()
    env.reset(seed=args.seed)
    roots = {1: mcts[1].new_root(env.state), 2: mcts[2].new_root(env.state)}
    seen_counts = {_state_cycle_key(env.state): 1}
    pane = None if args.no_render else BoardPane(cell_size=args.cell_size)
    last_action = None

    def show_console_board() -> None:
        if not args.no_clear:
            os.system("cls" if os.name == "nt" else "clear")
        print(env.render())
        if last_action:
            print()
            print(last_action)
        print()
        print(f"candidate: {'P1' if candidate_is_p1 else 'P2'} | best: {'P2' if candidate_is_p1 else 'P1'}")

    try:
        while not env.is_terminal() and env.state.move_count < args.max_moves:
            if pane is not None:
                pane.update(env, last_action)
            elif args.no_render and not args.moves:
                show_console_board()

            player = env.state.current_player
            action, probs = choose_action(
                mcts[player],
                roots[player],
                fns[player],
                args.sims,
                env.state,
                seen_counts,
            )
            confidence = float(probs[action])
            last_action = (
                f"P{player} {names[player]} -> {describe_action(action)} "
                f"(p={confidence:.3f})"
            )
            if pane is not None or args.moves:
                print(f"move {env.state.move_count:>3}: {last_action}", flush=True)
            env.step(action)
            key = _state_cycle_key(env.state)
            seen_counts[key] = seen_counts.get(key, 0) + 1

            for side in (1, 2):
                if action in roots[side].children:
                    roots[side] = roots[side].children[action]
                else:
                    roots[side] = mcts[side].new_root(env.state)

            if pane is not None:
                pane.update(env, last_action)
            elif args.no_render and not args.moves:
                show_console_board()
            if args.delay > 0:
                time.sleep(args.delay)

        winner = _adjudicated_winner(env.state)
        if pane is not None:
            pane.update(env, last_action)
        elif args.no_render and not args.moves:
            show_console_board()
        if winner == 0:
            print("Game over: draw")
        else:
            print(f"Game over: Player {winner} wins ({names[winner]})")
        if pane is not None:
            print("Close the board window to exit.")
            while not pane.closed:
                pane.update(env, last_action)
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nWatch stopped.")


if __name__ == "__main__":
    main()
