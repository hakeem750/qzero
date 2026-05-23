"""
Interactive Quoridor gameplay.

Examples:
  python scripts/play.py
  python scripts/play.py --p2 random --seed 7
  python scripts/play.py --p1 random --p2 random --delay 0.2
  python scripts/play.py --p2 random --render-pane
  python scripts/play.py --no-save-buffer
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from env.encoding import encode_state, mirror_state_and_policy
from env.actions import (
    NUM_ACTIONS,
    action_name,
    action_to_h_wall,
    action_to_v_wall,
    h_wall_to_action,
    v_wall_to_action,
)
from env.quoridor_env import QuoridorEnv
from replay.buffer import ReplayBuffer


DEFAULT_BUFFER_PATH = pathlib.Path("data") / "replay_buffer.npz"


class BoardPane:
    def __init__(self, cell_size: int = 58) -> None:
        try:
            import tkinter as tk
        except ImportError as exc:
            raise RuntimeError("tkinter is required for --render-pane") from exc

        self.tk = tk
        self.cell = cell_size
        self.margin = 52
        self.wall = 8
        size = self.margin * 2 + self.cell * 9

        self.root = tk.Tk()
        self.root.title("Quoridor")
        self.root.resizable(False, False)
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas = tk.Canvas(self.root, width=size, height=size + 46, bg="#202124", highlightthickness=0)
        self.canvas.pack()

    def update(self, env: QuoridorEnv, last_action: str | None = None) -> None:
        if self.closed:
            return
        self.canvas.delete("all")
        s = env.state
        board_left = self.margin
        board_top = self.margin + 18

        status = (
            f"Move {s.move_count} | Player {s.current_player}'s turn | "
            f"P1 walls {s.p1_walls} | P2 walls {s.p2_walls}"
        )
        if last_action:
            status += f" | {last_action}"
        self.canvas.create_text(board_left, 18, text=status, fill="#f1f3f4", anchor="w", font=("Segoe UI", 11, "bold"))

        for c in range(9):
            x = board_left + c * self.cell + self.cell / 2
            self.canvas.create_text(x, board_top - 20, text=str(c), fill="#cfd8dc", font=("Segoe UI", 10))
        for r in range(9):
            y = board_top + (8 - r) * self.cell + self.cell / 2
            self.canvas.create_text(board_left - 22, y, text=str(r), fill="#cfd8dc", font=("Segoe UI", 10))

        for r in range(9):
            for c in range(9):
                x0 = board_left + c * self.cell
                y0 = board_top + (8 - r) * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                fill = "#d8b77a" if (r + c) % 2 == 0 else "#cfa968"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#8d6e3f", width=1)

        for r, c in s.h_walls:
            x0 = board_left + c * self.cell + 6
            y = board_top + (8 - r) * self.cell - self.wall / 2
            x1 = board_left + (c + 2) * self.cell - 6
            self.canvas.create_rectangle(x0, y, x1, y + self.wall, fill="#3949ab", outline="#283593")

        for r, c in s.v_walls:
            x = board_left + (c + 1) * self.cell - self.wall / 2
            y0 = board_top + (8 - (r + 1)) * self.cell + 6
            y1 = board_top + (8 - r) * self.cell + self.cell - 6
            self.canvas.create_rectangle(x, y0, x + self.wall, y1, fill="#3949ab", outline="#283593")

        self._draw_pawn(s.p1_pos, "1", "#e53935", board_left, board_top)
        self._draw_pawn(s.p2_pos, "2", "#1e88e5", board_left, board_top)
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True

    def _draw_pawn(self, pos: tuple[int, int], label: str, color: str, board_left: int, board_top: int) -> None:
        r, c = pos
        cx = board_left + c * self.cell + self.cell / 2
        cy = board_top + (8 - r) * self.cell + self.cell / 2
        radius = self.cell * 0.28
        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=color, outline="#ffffff", width=2)
        self.canvas.create_text(cx, cy, text=label, fill="#ffffff", font=("Segoe UI", 16, "bold"))

    def close(self) -> None:
        self.closed = True
        self.root.destroy()


MOVE_ALIASES = {
    "n": 0,
    "north": 0,
    "s": 1,
    "south": 1,
    "e": 2,
    "east": 2,
    "w": 3,
    "west": 3,
    "jn": 4,
    "jump_north": 4,
    "js": 5,
    "jump_south": 5,
    "je": 6,
    "jump_east": 6,
    "jw": 7,
    "jump_west": 7,
    "nw": 8,
    "diag_nw": 8,
    "ne": 9,
    "diag_ne": 9,
    "sw": 10,
    "diag_sw": 10,
    "se": 11,
    "diag_se": 11,
}


def describe_action(action: int) -> str:
    if action < 12:
        return f"{action:>3}: {action_name(action)}"
    if action < 76:
        r, c = action_to_h_wall(action)
        return f"{action:>3}: h {r} {c}"
    r, c = action_to_v_wall(action)
    return f"{action:>3}: v {r} {c}"


def print_legal_actions(actions: list[int]) -> None:
    moves = [a for a in actions if a < 12]
    h_walls = [a for a in actions if 12 <= a < 76]
    v_walls = [a for a in actions if a >= 76]

    print("Legal pawn moves:")
    print("  " + ", ".join(describe_action(a) for a in moves))
    print(f"Legal horizontal walls: {len(h_walls)}  examples: {', '.join(describe_action(a) for a in h_walls[:12])}")
    print(f"Legal vertical walls:   {len(v_walls)}  examples: {', '.join(describe_action(a) for a in v_walls[:12])}")


def parse_action(raw: str) -> int | None:
    parts = raw.strip().lower().replace(",", " ").split()
    if not parts:
        return None

    if len(parts) == 1:
        token = parts[0]
        if token.isdigit():
            action = int(token)
            return action if 0 <= action < NUM_ACTIONS else None
        return MOVE_ALIASES.get(token)

    if len(parts) == 3 and parts[0] in {"h", "v"}:
        try:
            r, c = int(parts[1]), int(parts[2])
        except ValueError:
            return None
        if not (0 <= r < 8 and 0 <= c < 8):
            return None
        return h_wall_to_action(r, c) if parts[0] == "h" else v_wall_to_action(r, c)

    return None


def human_action(env: QuoridorEnv) -> int:
    legal = set(env.legal_actions())
    while True:
        raw = input("Action (n/s/e/w, h r c, v r c, number, legal, quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if raw.lower() in {"l", "legal"}:
            print_legal_actions(sorted(legal))
            continue

        action = parse_action(raw)
        if action is None:
            print("Could not parse that action.")
            continue
        if action not in legal:
            print(f"Illegal action: {describe_action(action)}")
            continue
        return action


def choose_action(env: QuoridorEnv, player_kind: str, rng: random.Random) -> int:
    if player_kind == "random":
        return rng.choice(env.legal_actions())
    return human_action(env)


def action_policy(action: int) -> np.ndarray:
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[action] = 1.0
    return policy


def outcome_for_player(winner: int | None, player: int) -> float:
    if winner == 0 or winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def save_recorded_game(
    buffer: ReplayBuffer,
    path: pathlib.Path,
    raw_steps: list[tuple[np.ndarray, np.ndarray, int]],
    winner: int | None,
    augment: bool,
) -> None:
    saved = 0
    for obs, policy, player in raw_steps:
        outcome = outcome_for_player(winner, player)
        buffer.push(obs, policy, outcome)
        saved += 1
        if augment:
            m_obs, m_policy = mirror_state_and_policy(obs, policy)
            buffer.push(m_obs, m_policy, outcome)
            saved += 1

    buffer.save(path)
    print(f"Saved {saved} gameplay samples to {path} (buffer size={buffer.size})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", choices=["human", "random"], default="human")
    parser.add_argument("--p2", choices=["human", "random"], default="human")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--render-pane", action="store_true", help="Open an optional colored board window.")
    parser.add_argument("--save-buffer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-save-buffer", action="store_true", help="Do not save this played game into the replay buffer.")
    parser.add_argument("--buffer-path", type=pathlib.Path, default=DEFAULT_BUFFER_PATH)
    parser.add_argument("--no-augment", action="store_true", help="Disable mirror augmentation when saving gameplay.")
    args = parser.parse_args()
    save_buffer = not args.no_save_buffer

    rng = random.Random(args.seed)
    env = QuoridorEnv()
    env.reset(seed=args.seed)
    players = {1: args.p1, 2: args.p2}
    pane = BoardPane() if args.render_pane else None
    last_action = None
    raw_steps: list[tuple[np.ndarray, np.ndarray, int]] = []

    try:
        while not env.is_terminal():
            print()
            print(env.render())
            if pane is not None:
                pane.update(env, last_action)
            player = env.state.current_player
            player_kind = players[player]
            action = choose_action(env, player_kind, rng)
            if save_buffer:
                raw_steps.append((encode_state(env.state), action_policy(action), player))
            last_action = f"Player {player} -> {describe_action(action)}"
            print(last_action)
            _, reward, terminated, _, info = env.step(action)
            if pane is not None:
                pane.update(env, last_action)
            if args.delay > 0:
                time.sleep(args.delay)
            if terminated:
                print()
                print(env.render())
                if pane is not None:
                    pane.update(env, last_action)
                winner = info["winner"]
                if winner == 0:
                    print("Game over: draw")
                else:
                    print(f"Game over: Player {winner} wins (reward={reward:+.1f})")
                if save_buffer:
                    if args.buffer_path.exists():
                        buffer = ReplayBuffer.load(args.buffer_path)
                    else:
                        buffer = ReplayBuffer()
                    save_recorded_game(buffer, args.buffer_path, raw_steps, winner, augment=not args.no_augment)
                return
    except KeyboardInterrupt:
        print("\nGame stopped.")


if __name__ == "__main__":
    main()
