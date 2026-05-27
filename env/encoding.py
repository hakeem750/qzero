"""
AlphaZero-style spatial encoding for Quoridor.

The encoder returns a (20, 9, 9) float32 tensor from the current player's
perspective. It contains only board state and rule-legality information: no
shortest-path distances, no engine scores, and no handcrafted evaluation.

Channel layout:
  0   current player pawn
  1   opponent pawn
  2   horizontal wall upper cells
  3   horizontal wall lower cells
  4   vertical wall left cells
  5   vertical wall right cells
  6   current player walls remaining
  7   opponent walls remaining
  8   normalized move count
  9   current player id
  10  north/up passage blocked
  11  south/down passage blocked
  12  east/right passage blocked
  13  west/left passage blocked
  14  horizontal wall anchors
  15  vertical wall anchors
  16  current player's goal row
  17  opponent's goal row
  18  occupied cells
  19  constant bias plane
"""
from __future__ import annotations

import numpy as np

from .state import QuoridorState, BOARD_SIZE, WALL_GRID, MAX_MOVES
from .actions import H_WALL_OFFSET, V_WALL_OFFSET
from .rules import can_move

NUM_CHANNELS = 20
TENSOR_SHAPE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def encode_state(state: QuoridorState) -> np.ndarray:
    """Encode state into (20, 9, 9) float32 from current player's viewpoint."""
    plane = np.zeros(TENSOR_SHAPE, dtype=np.float32)
    flip = state.current_player == 2

    def rflip(r: int) -> int:
        return BOARD_SIZE - 1 - r if flip else r

    def unflip_pos(pr: int, pc: int) -> tuple[int, int]:
        return (BOARD_SIZE - 1 - pr, pc) if flip else (pr, pc)

    cr, cc = state.active_pos
    plane[0, rflip(cr), cc] = 1.0

    orow, oc = state.opponent_pos
    plane[1, rflip(orow), oc] = 1.0

    for r, c in state.h_walls:
        plane[2, rflip(r), c] = 1.0
        plane[2, rflip(r), c + 1] = 1.0
        plane[3, rflip(r + 1), c] = 1.0
        plane[3, rflip(r + 1), c + 1] = 1.0
        plane[14, rflip(r), c] = 1.0

    for r, c in state.v_walls:
        plane[4, rflip(r), c] = 1.0
        plane[4, rflip(r + 1), c] = 1.0
        plane[5, rflip(r), c + 1] = 1.0
        plane[5, rflip(r + 1), c + 1] = 1.0
        plane[15, rflip(r), c] = 1.0

    plane[6, :, :] = state.active_walls / 10.0
    plane[7, :, :] = state.opponent_walls / 10.0
    plane[8, :, :] = min(state.move_count, MAX_MOVES) / float(MAX_MOVES)
    plane[9, :, :] = float(state.current_player - 1)

    perspective_dirs = {
        10: (-1, 0),
        11: (1, 0),
        12: (0, 1),
        13: (0, -1),
    }
    for ch, (pdr, pdc) in perspective_dirs.items():
        for pr in range(BOARD_SIZE):
            for pc in range(BOARD_SIZE):
                nr, nc = pr + pdr, pc + pdc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    plane[ch, pr, pc] = 1.0
                    continue
                br, bc = unflip_pos(pr, pc)
                tr, tc = unflip_pos(nr, nc)
                if not can_move(br, bc, tr, tc, state.h_walls, state.v_walls):
                    plane[ch, pr, pc] = 1.0

    plane[16, BOARD_SIZE - 1, :] = 1.0
    plane[17, 0, :] = 1.0
    plane[18] = plane[0] + plane[1]
    plane[19, :, :] = 1.0
    return plane


def mirror_state_and_policy(
    obs: np.ndarray,
    policy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply left-right (column) flip to an encoded observation and policy vector.
    """
    m_obs = np.flip(obs, axis=2).copy()
    m_policy = np.zeros_like(policy)

    pawn_mirror = {
        0: 0, 1: 1, 2: 3, 3: 2,
        4: 4, 5: 5, 6: 7, 7: 6,
        8: 9, 9: 8, 10: 11, 11: 10,
    }
    for action, mirrored_action in pawn_mirror.items():
        m_policy[mirrored_action] = policy[action]

    for idx in range(64):
        r, c = divmod(idx, WALL_GRID)
        mc = WALL_GRID - 1 - c
        m_policy[H_WALL_OFFSET + r * WALL_GRID + mc] = policy[H_WALL_OFFSET + idx]
        m_policy[V_WALL_OFFSET + r * WALL_GRID + mc] = policy[V_WALL_OFFSET + idx]

    return m_obs, m_policy
