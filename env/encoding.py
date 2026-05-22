"""
Tensor encoding: (17, 9, 9) float32.

Channel layout (current-player perspective):
  0   : current player pawn position
  1   : opponent pawn position
  2–3 : horizontal wall planes (anchor row, filled cell)
  4–5 : vertical wall planes   (anchor row, filled cell)
  6   : current player wall count  (scalar broadcast)
  7   : opponent wall count        (scalar broadcast)
  8   : current player id          (0.0 = P1, 1.0 = P2)
  9–16: reserved history planes (zeros until history is tracked)

IMPROVEMENT: Left-right mirror symmetry augmentation.
  Quoridor's goal rows are horizontal, so the board has perfect
  left-right (column) symmetry.  We exploit this to double the
  effective dataset size with zero extra self-play cost.
"""
from __future__ import annotations

import numpy as np

from .state import QuoridorState, BOARD_SIZE, WALL_GRID
from .actions import (
    NUM_ACTIONS, H_WALL_OFFSET, V_WALL_OFFSET, WALL_GRID as WG,
)

NUM_CHANNELS = 17
TENSOR_SHAPE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------

def encode_state(state: QuoridorState) -> np.ndarray:
    """
    Encode state into (17, 9, 9) float32 from current player's viewpoint.

    P1's perspective: row 0 is home, row 8 is goal (no flip needed).
    P2's perspective: we flip the board vertically so P2 always appears
      to be at the bottom heading toward the top row.
    """
    plane = np.zeros(TENSOR_SHAPE, dtype=np.float32)

    flip = (state.current_player == 2)

    def rflip(r: int) -> int:
        return (BOARD_SIZE - 1 - r) if flip else r

    def wflip(r: int) -> int:
        """Wall anchor row flip: anchor r covers rows r and r+1."""
        return (WALL_GRID - 1 - r) if flip else r

    # Channel 0: current player pawn
    cr, cc = state.active_pos
    plane[0, rflip(cr), cc] = 1.0

    # Channel 1: opponent pawn
    or_, oc = state.opponent_pos
    plane[1, rflip(or_), oc] = 1.0

    # Channels 2–3: horizontal walls (anchor fills both rows r and r+1)
    for r, c in state.h_walls:
        fr = wflip(r)
        plane[2, fr, c]     = 1.0
        plane[2, fr, c + 1] = 1.0
        plane[3, fr + 1 if not flip else fr - 1, c]     = 1.0
        plane[3, fr + 1 if not flip else fr - 1, c + 1] = 1.0

    # Channels 4–5: vertical walls (anchor fills both cols c and c+1)
    for r, c in state.v_walls:
        fr = wflip(r)
        plane[4, fr, c]     = 1.0
        plane[4, fr, c + 1] = 1.0
        plane[5, fr + 1 if not flip else fr - 1, c]     = 1.0
        plane[5, fr + 1 if not flip else fr - 1, c + 1] = 1.0

    # Channels 6–7: wall counts (scalar broadcast)
    plane[6, :, :] = state.active_walls   / 10.0
    plane[7, :, :] = state.opponent_walls / 10.0

    # Channel 8: player identity
    plane[8, :, :] = float(state.current_player - 1)

    # Channels 9–16: history (zeros for now; future work)

    return plane


# ---------------------------------------------------------------------------
# Left-right symmetry augmentation
# IMPROVEMENT: doubles training data cheaply; Quoridor is LR-symmetric.
# ---------------------------------------------------------------------------

def _mirror_lr_walls(wall_action_start: int, anchors: list[tuple[int,int]]) -> list[int]:
    """Reflect wall anchor columns for LR flip."""
    actions = []
    for r, c in anchors:
        mc = WALL_GRID - 1 - c
        # clamp to valid range (wall spans c and c+1, so mirror c → WG-1-c-1 = WG-2-c)
        mc = WALL_GRID - 2 - c
        if 0 <= mc < WALL_GRID:
            actions.append(wall_action_start + r * WALL_GRID + mc)
    return actions


def mirror_state_and_policy(
    obs: np.ndarray,
    policy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply left-right (column) flip to encoded observation and policy vector.

    Returns (mirrored_obs, mirrored_policy).
    """
    # Mirror obs: flip along last axis (columns)
    m_obs = np.flip(obs, axis=2).copy()

    m_policy = np.zeros_like(policy)

    # Pawn move actions: swap east↔west (2↔3), jump_east↔jump_west (6↔7),
    # diag_NW↔diag_NE (8↔9), diag_SW↔diag_SE (10↔11)
    pawn_mirror = {0:0, 1:1, 2:3, 3:2, 4:4, 5:5, 6:7, 7:6, 8:9, 9:8, 10:11, 11:10}
    for a, ma in pawn_mirror.items():
        m_policy[ma] = policy[a]

    # Horizontal walls: mirror column c → 6-c (wall spans c,c+1 → 6-c,7-c = 6-c,6-c+1)
    for idx in range(64):
        r, c = divmod(idx, WALL_GRID)
        mc = WALL_GRID - 2 - c
        if 0 <= mc < WALL_GRID:
            m_policy[H_WALL_OFFSET + r * WALL_GRID + mc] = policy[H_WALL_OFFSET + idx]

    # Vertical walls: mirror column c → 6-c
    for idx in range(64):
        r, c = divmod(idx, WALL_GRID)
        mc = WALL_GRID - 2 - c
        if 0 <= mc < WALL_GRID:
            m_policy[V_WALL_OFFSET + r * WALL_GRID + mc] = policy[V_WALL_OFFSET + idx]

    return m_obs, m_policy
