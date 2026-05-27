"""
QuoridorEnv - OpenAI Gym-style environment wrapping the pure rules engine.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional

from .state import QuoridorState, initial_state, BOARD_SIZE
from .rules import legal_actions, apply_action, is_terminal, winner
from .encoding import encode_state
from .actions import action_name


class QuoridorEnv:
    """
    Thin stateful wrapper around the pure functional engine.

    The underlying QuoridorState is immutable; each step creates a
    new state object rather than mutating in place.
    """

    def __init__(self) -> None:
        self.state: QuoridorState = initial_state()
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        self.state = initial_state()
        return encode_state(self.state.canonical())

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Returns (obs, reward, terminated, truncated, info).
        reward is from the perspective of the player who just moved.
        """
        assert not is_terminal(self.state), "Game is already over"
        assert action in self.legal_actions(), f"Illegal action {action_name(action)}"

        self.state = apply_action(self.state, action)
        obs = encode_state(self.state.canonical())

        w = winner(self.state)
        terminated = w is not None

        if not terminated:
            reward = 0.0
        elif w == 0:
            reward = 0.0
        else:
            prev_player = 3 - self.state.current_player
            reward = 1.0 if w == prev_player else -1.0

        info = {
            "winner": w,
            "move_count": self.state.move_count,
            "current_player": self.state.current_player,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    def legal_actions(self) -> List[int]:
        return legal_actions(self.state)

    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    def winner(self) -> Optional[int]:
        return winner(self.state)

    def encode(self) -> np.ndarray:
        return encode_state(self.state.canonical())

    def clone(self) -> "QuoridorEnv":
        env = QuoridorEnv.__new__(QuoridorEnv)
        env.state = self.state
        env._rng = np.random.default_rng()
        return env

    # ------------------------------------------------------------------
    def render(self) -> str:
        """Return an ASCII render of the current board."""
        s = self.state
        lines = [
            f"Move {s.move_count} | Player {s.current_player}'s turn | "
            f"P1 walls: {s.p1_walls} | P2 walls: {s.p2_walls}",
            "    " + "   ".join(str(c) for c in range(BOARD_SIZE)),
        ]

        for r in range(BOARD_SIZE - 1, -1, -1):
            cell_row = [f"{r}  "]
            for c in range(BOARD_SIZE):
                if (r, c) == s.p1_pos:
                    cell = "1"
                elif (r, c) == s.p2_pos:
                    cell = "2"
                else:
                    cell = "."

                cell_row.append(f" {cell} ")
                if c < BOARD_SIZE - 1:
                    wall = "|" if (r, c) in s.v_walls or (r - 1, c) in s.v_walls else " "
                    cell_row.append(wall)
            lines.append("".join(cell_row))

            if r > 0:
                wall_row = ["   "]
                for c in range(BOARD_SIZE):
                    wall = "---" if (r - 1, c) in s.h_walls or (r - 1, c - 1) in s.h_walls else "   "
                    wall_row.append(wall)
                    if c < BOARD_SIZE - 1:
                        wall_row.append(" ")
                lines.append("".join(wall_row))

        return "\n".join(lines)
