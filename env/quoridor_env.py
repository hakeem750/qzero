"""
QuoridorEnv — OpenAI Gym-style environment wrapping the pure rules engine.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional

from .state import QuoridorState, initial_state, BOARD_SIZE
from .rules import legal_actions, apply_action, is_terminal, winner
from .encoding import encode_state
from .actions import action_name, NUM_ACTIONS


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
        return encode_state(self.state)

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Returns (obs, reward, terminated, truncated, info).
        reward is from the perspective of the player who just moved.
        """
        assert not is_terminal(self.state), "Game is already over"
        assert action in self.legal_actions(), f"Illegal action {action_name(action)}"

        self.state = apply_action(self.state, action)
        obs = encode_state(self.state)

        w = winner(self.state)
        terminated = w is not None

        if not terminated:
            reward = 0.0
        elif w == 0:                      # draw
            reward = 0.0
        else:
            # The mover just *finished* its turn; current_player has already
            # flipped to the opponent.  The winner is 3 - current_player.
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
        return encode_state(self.state)

    def clone(self) -> "QuoridorEnv":
        env = QuoridorEnv.__new__(QuoridorEnv)
        env.state = self.state  # immutable — safe to share
        env._rng = np.random.default_rng()
        return env

    # ------------------------------------------------------------------
    def render(self) -> str:
        """ASCII render of the board."""
        s = self.state
        lines = []
        for r in range(BOARD_SIZE - 1, -1, -1):
            row_chars = []
            for c in range(BOARD_SIZE):
                if (r, c) == s.p1_pos:
                    row_chars.append("1")
                elif (r, c) == s.p2_pos:
                    row_chars.append("2")
                else:
                    row_chars.append(".")

                # east wall?
                if c < BOARD_SIZE - 1:
                    if (r, c) in s.v_walls or (r - 1, c) in s.v_walls:
                        row_chars.append("|")
                    else:
                        row_chars.append(" ")
            lines.append(" ".join(row_chars))

            # south wall row
            if r > 0:
                wall_row = []
                for c in range(BOARD_SIZE):
                    if (r - 1, c) in s.h_walls or (r - 1, c - 1) in s.h_walls:
                        wall_row.append("—")
                    else:
                        wall_row.append(" ")
                    if c < BOARD_SIZE - 1:
                        wall_row.append(" ")
                lines.append(" ".join(wall_row))

        header = (
            f"Move {s.move_count} | Player {s.current_player}'s turn | "
            f"P1 walls: {s.p1_walls} | P2 walls: {s.p2_walls}"
        )
        return header + "\n" + "\n".join(lines)
