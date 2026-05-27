"""
Action space (140 total):

  0–11   : pawn moves
    0  north       (row -= 1)
    1  south       (row += 1)
    2  east        (col += 1)
    3  west        (col -= 1)
    4  jump north  (row -= 2, opponent adjacent north)
    5  jump south  (row += 2)
    6  jump east   (col += 2)
    7  jump west   (col -= 2)
    8  diag NW     jump blocked north  → go west
    9  diag NE     jump blocked north  → go east
   10  diag SW     jump blocked south  → go west
   11  diag SE     jump blocked south  → go east

  12–75  : horizontal wall anchors  (r*8 + c) + 12,  r,c ∈ [0,7]
  76–139 : vertical   wall anchors  (r*8 + c) + 76,  r,c ∈ [0,7]
"""
from __future__ import annotations

NUM_ACTIONS     = 140
NUM_MOVE_ACTIONS = 12
NUM_WALL_ACTIONS = 128   # 64 h + 64 v
H_WALL_OFFSET   = 12
V_WALL_OFFSET   = 76
WALL_GRID       = 8

# direction deltas for the 4 cardinal moves
MOVE_DELTAS = {
    0: (-1,  0),   # north
    1: ( 1,  0),   # south
    2: ( 0,  1),   # east
    3: ( 0, -1),   # west
}

# jump deltas (double step)
JUMP_DELTAS = {
    4: (-2,  0),
    5: ( 2,  0),
    6: ( 0,  2),
    7: ( 0, -2),
}

# diagonal side-steps: (primary_direction, side_direction_a, side_direction_b)
# action 8  = blocked north → side-step west (NW)
# action 9  = blocked north → side-step east (NE)
# action 10 = blocked south → side-step west (SW)
# action 11 = blocked south → side-step east (SE)
DIAG_SPEC = {
    8:  (0, 3),   # primary north(0), side west(3)
    9:  (0, 2),   # primary north(0), side east(2)
    10: (1, 3),   # primary south(1), side west(3)
    11: (1, 2),   # primary south(1), side east(2)
}


def action_is_wall(action: int) -> bool:
    return action >= H_WALL_OFFSET


def action_is_h_wall(action: int) -> bool:
    return H_WALL_OFFSET <= action < V_WALL_OFFSET


def action_is_v_wall(action: int) -> bool:
    return action >= V_WALL_OFFSET


def action_to_h_wall(action: int) -> tuple[int, int]:
    idx = action - H_WALL_OFFSET
    return divmod(idx, WALL_GRID)


def action_to_v_wall(action: int) -> tuple[int, int]:
    idx = action - V_WALL_OFFSET
    return divmod(idx, WALL_GRID)


def h_wall_to_action(r: int, c: int) -> int:
    return H_WALL_OFFSET + r * WALL_GRID + c


def v_wall_to_action(r: int, c: int) -> int:
    return V_WALL_OFFSET + r * WALL_GRID + c


def action_name(action: int) -> str:
    names = [
        "north", "south", "east", "west",
        "jump_north", "jump_south", "jump_east", "jump_west",
        "diag_NW", "diag_NE", "diag_SW", "diag_SE",
    ]
    if action < NUM_MOVE_ACTIONS:
        return names[action]
    if action_is_h_wall(action):
        r, c = action_to_h_wall(action)
        return f"h_wall({r},{c})"
    r, c = action_to_v_wall(action)
    return f"v_wall({r},{c})"


def legal_action_mask(state) -> "np.ndarray":
    """Return a boolean mask with True for legal actions."""
    import numpy as np
    from .rules import legal_actions

    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal_actions(state):
        mask[action] = True
    return mask


def mask_illegal_actions(policy: "np.ndarray", state, normalize: bool = True) -> "np.ndarray":
    """
    Zero illegal action probabilities and optionally renormalize.

    This is the numpy-side equivalent of PolicyValueNet.predict(), which masks
    logits before softmax for neural-network inference.
    """
    import numpy as np

    masked = np.asarray(policy, dtype=np.float64).copy()
    legal_mask = legal_action_mask(state)
    masked[~legal_mask] = 0.0
    if normalize:
        total = masked.sum(dtype=np.float64)
        if total > 0.0:
            masked /= total
        elif legal_mask.any():
            masked[legal_mask] = 1.0 / legal_mask.sum()
    return masked
