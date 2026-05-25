"""
Tensor encoding: (13, 9, 9) float32.

Channel layout (current-player perspective):
  0   : current player pawn position
  1   : opponent pawn position
  2–3 : horizontal wall planes (anchor fills rows r and r+1)
  4–5 : vertical wall planes   (anchor fills rows r and r+1)
  6   : current player wall count  (scalar broadcast)
  7   : opponent wall count        (scalar broadcast)
  8   : current player id          (0.0 = P1, 1.0 = P2)
  9   : north passage blocked plane (0-8, 0-8)
  10  : east passage blocked plane  (0-8, 0-8)
  11  : current player BFS distance map to goal (critical for wall strategy)
  12  : opponent BFS distance map to goal

IMPROVEMENTS:
  1. Passage planes (channels 9–10) make wall effects directly observable.
  2. Distance maps (channels 11–12) let the network see shortest path to goal
     without rediscovering BFS from game trajectories.
  3. Wall encoding fixed: uses rflip() consistently (not wflip) for 9×9 indexing.
  4. Left-right mirror symmetry augmentation doubles dataset size.
"""
from __future__ import annotations

import numpy as np

from .state import QuoridorState, BOARD_SIZE, WALL_GRID
from .actions import (
    NUM_ACTIONS, H_WALL_OFFSET, V_WALL_OFFSET, WALL_GRID as WG,
)
from .rules import bfs_distance_map, _blocked_ns, _blocked_ew

NUM_CHANNELS = 13
TENSOR_SHAPE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------

def encode_state(state: QuoridorState) -> np.ndarray:
    """
    Encode state into (13, 9, 9) float32 from current player's viewpoint.

    P1's perspective: row 0 is home, row 8 is goal (no flip needed).
    P2's perspective: we flip the board vertically so P2 always appears
      to be at the bottom heading toward the top row.
    """
    plane = np.zeros(TENSOR_SHAPE, dtype=np.float32)

    flip = (state.current_player == 2)

    def rflip(r: int) -> int:
        return (BOARD_SIZE - 1 - r) if flip else r

    # Channel 0: current player pawn
    cr, cc = state.active_pos
    plane[0, rflip(cr), cc] = 1.0

    # Channel 1: opponent pawn
    or_, oc = state.opponent_pos
    plane[1, rflip(or_), oc] = 1.0

    # Channels 2–3: horizontal walls (anchor (r,c) fills rows r and r+1)
    # BUGFIX: use rflip() consistently for 9×9 plane indexing, not wflip()
    for r, c in state.h_walls:
        plane[2, rflip(r), c]     = 1.0
        plane[2, rflip(r), c + 1] = 1.0
        plane[3, rflip(r + 1), c]     = 1.0
        plane[3, rflip(r + 1), c + 1] = 1.0

    # Channels 4–5: vertical walls (anchor (r,c) fills rows r and r+1)
    for r, c in state.v_walls:
        plane[4, rflip(r), c]     = 1.0
        plane[4, rflip(r), c + 1] = 1.0
        plane[5, rflip(r + 1), c]     = 1.0
        plane[5, rflip(r + 1), c + 1] = 1.0

    # Channels 6–7: wall counts (scalar broadcast)
    plane[6, :, :] = state.active_walls   / 10.0
    plane[7, :, :] = state.opponent_walls / 10.0

    # Channel 8: player identity
    plane[8, :, :] = float(state.current_player - 1)

    # Channel 9: north passage blocked (r,c)=1 means can't go north from (r,c)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if r < BOARD_SIZE - 1 and _blocked_ns(r, c, state.h_walls):
                plane[9, rflip(r), c] = 1.0

    # Channel 10: east passage blocked (r,c)=1 means can't go east from (r,c)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if c < BOARD_SIZE - 1 and _blocked_ew(r, c, state.v_walls):
                plane[10, rflip(r), c] = 1.0

    # Channel 11: current player BFS distance map to goal
    current_goal_row = 8 if state.current_player == 1 else 0
    current_dist_map = bfs_distance_map(
        state.active_pos, current_goal_row,
        state.h_walls, state.v_walls
    )
    if flip:
        current_dist_map = np.flip(current_dist_map, axis=0)
    plane[11, :, :] = current_dist_map.astype(np.float32) / 16.0

    # Channel 12: opponent BFS distance map to goal
    opponent_goal_row = 8 if state.current_player == 2 else 0
    opponent_dist_map = bfs_distance_map(
        state.opponent_pos, opponent_goal_row,
        state.h_walls, state.v_walls
    )
    if flip:
        opponent_dist_map = np.flip(opponent_dist_map, axis=0)
    plane[12, :, :] = opponent_dist_map.astype(np.float32) / 16.0

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
        actions.append(wall_action_start + r * WALL_GRID + mc)
    return actions


def mirror_state_and_policy(
    obs: np.ndarray,
    policy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply left-right (column) flip to encoded observation and policy vector.

    Channels 0–12 are all spatial and get flipped along the column (last) axis.
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

    # Horizontal walls: mirror column c -> 7-c.
    for idx in range(64):
        r, c = divmod(idx, WALL_GRID)
        mc = WALL_GRID - 1 - c
        m_policy[H_WALL_OFFSET + r * WALL_GRID + mc] = policy[H_WALL_OFFSET + idx]

    # Vertical walls: mirror column c -> 7-c.
    for idx in range(64):
        r, c = divmod(idx, WALL_GRID)
        mc = WALL_GRID - 1 - c
        m_policy[V_WALL_OFFSET + r * WALL_GRID + mc] = policy[V_WALL_OFFSET + idx]

    return m_obs, m_policy
