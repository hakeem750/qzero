"""
Core Quoridor rules engine.

All functions are pure — they take a QuoridorState and return results
without mutation, keeping the engine thread-safe.
"""
from __future__ import annotations

from collections import deque
from typing import List

from .state import QuoridorState, BOARD_SIZE
from .actions import (
    NUM_ACTIONS, MOVE_DELTAS, JUMP_DELTAS, DIAG_SPEC,
    H_WALL_OFFSET, V_WALL_OFFSET,
    action_to_h_wall, action_to_v_wall,
    h_wall_to_action, v_wall_to_action,
)

WALL_GRID = BOARD_SIZE - 1  # anchor grid is 8×8
# ---------------------------------------------------------------------------
# Wall-passage helpers
# ---------------------------------------------------------------------------

def _blocked_ns(r: int, c: int, h_walls: frozenset) -> bool:
    """True if passage between (r, c) and (r+1, c) is blocked (north↔south)."""
    # Horizontal wall (r, c) blocks cols c & c+1
    # Horizontal wall (r, c-1) blocks cols c-1 & c
    return (r, c) in h_walls or (r, c - 1) in h_walls


def _blocked_ew(r: int, c: int, v_walls: frozenset) -> bool:
    """True if passage between (r, c) and (r, c+1) is blocked (east↔west)."""
    # Vertical wall (r, c) blocks rows r & r+1
    # Vertical wall (r-1, c) blocks rows r-1 & r
    return (r, c) in v_walls or (r - 1, c) in v_walls


def can_move(fr: int, fc: int, tr: int, tc: int,
             h_walls: frozenset, v_walls: frozenset) -> bool:
    """Can a pawn step from (fr,fc) to the adjacent (tr,tc)?"""
    dr, dc = tr - fr, tc - fc
    if dr == -1 and dc == 0:   # north
        return not _blocked_ns(tr, fc, h_walls)
    if dr == 1 and dc == 0:    # south
        return not _blocked_ns(fr, fc, h_walls)
    if dr == 0 and dc == 1:    # east
        return not _blocked_ew(fr, fc, v_walls)
    if dr == 0 and dc == -1:   # west
        return not _blocked_ew(fr, fc - 1, v_walls)
    return False


# ---------------------------------------------------------------------------
# BFS path check
# ---------------------------------------------------------------------------

def shortest_path_length(
    start: tuple[int, int],
    goal_row: int,
    h_walls: frozenset,
    v_walls: frozenset,
) -> int | None:
    """Return shortest path length to goal_row, or None if no path exists."""
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), distance = queue.popleft()
        if r == goal_row:
            return distance
        for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if (nr, nc) not in visited and can_move(r, c, nr, nc, h_walls, v_walls):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), distance + 1))
    return None


def _bfs_path_exists(start: tuple[int, int], goal_row: int,
                     h_walls: frozenset, v_walls: frozenset) -> bool:
    """Return True if there exists any path from start to goal_row."""
    return shortest_path_length(start, goal_row, h_walls, v_walls) is not None


# ---------------------------------------------------------------------------
# Wall conflict check
# ---------------------------------------------------------------------------

def _walls_conflict(anchor, is_horiz, h_walls, v_walls):
    """
    Quoridor rules:
    - walls may touch
    - walls may not overlap
    - walls may not cross
    """
    r, c = anchor

    if is_horiz:
        return (
            (r, c) in h_walls
            or (r, c - 1) in h_walls
            or (r, c + 1) in h_walls
            or (r, c) in v_walls
        )

    return (
        (r, c) in v_walls
        or (r - 1, c) in v_walls
        or (r + 1, c) in v_walls
        or (r, c) in h_walls
    )


# ---------------------------------------------------------------------------
# Legal action enumeration
# ---------------------------------------------------------------------------

def legal_actions(state: QuoridorState) -> List[int]:
    """Return list of all legal action indices for the current player."""
    actions: List[int] = []
    cur = state.active_pos
    opp = state.opponent_pos
    hw, vw = state.h_walls, state.v_walls

    # ---- Pawn moves -------------------------------------------------------
    for action, (dr, dc) in MOVE_DELTAS.items():
        r, c = cur[0] + dr, cur[1] + dc
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            continue
        if not can_move(cur[0], cur[1], r, c, hw, vw):
            continue
        if (r, c) == opp:
            # Must jump or diagonal
            jr, jc = r + dr, c + dc
            if 0 <= jr < BOARD_SIZE and 0 <= jc < BOARD_SIZE and can_move(r, c, jr, jc, hw, vw):
                actions.append(action + 4)   # straight jump
            else:
                # diagonal side-steps in the jump direction
                for diag_action, (prim, side) in DIAG_SPEC.items():
                    if prim == action:
                        sdr, sdc = MOVE_DELTAS[side]
                        sr, sc = r + sdr, c + sdc
                        if 0 <= sr < BOARD_SIZE and 0 <= sc < BOARD_SIZE:
                            if can_move(r, c, sr, sc, hw, vw):
                                actions.append(diag_action)
        else:
            actions.append(action)

    # ---- Wall placements --------------------------------------------------
    cur_walls = state.active_walls
    if cur_walls > 0:
        for r in range(WALL_GRID):
            for c in range(WALL_GRID):
                # Horizontal
                if (r, c) not in hw and not _walls_conflict((r, c), True, hw, vw):
                    new_hw = hw | {(r, c)}
                    if (_bfs_path_exists(state.p1_pos, BOARD_SIZE - 1, new_hw, vw) and
                            _bfs_path_exists(state.p2_pos, 0, new_hw, vw)):
                        actions.append(h_wall_to_action(r, c))

                # Vertical
                if (r, c) not in vw and not _walls_conflict((r, c), False, hw, vw):
                    new_vw = vw | {(r, c)}
                    if (_bfs_path_exists(state.p1_pos, BOARD_SIZE - 1, hw, new_vw) and
                            _bfs_path_exists(state.p2_pos, 0, hw, new_vw)):
                        actions.append(v_wall_to_action(r, c))

    return actions


# ---------------------------------------------------------------------------
# Apply action → next state
# ---------------------------------------------------------------------------

def _resolve_pawn_destination(state: QuoridorState, action: int) -> tuple[int, int] | None:
    """
    Resolve a pawn move to its destination if it is legal in the current state.

    This centralizes pawn movement checks so both legal action generation and
    state application enforce the same wall-blocking rules.
    """
    cur = state.active_pos
    opp = state.opponent_pos
    hw, vw = state.h_walls, state.v_walls

    if action in MOVE_DELTAS:
        dr, dc = MOVE_DELTAS[action]
        nr, nc = cur[0] + dr, cur[1] + dc
        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
            return None
        if (nr, nc) == opp:
            return None
        if can_move(cur[0], cur[1], nr, nc, hw, vw):
            return nr, nc
        return None

    if action in JUMP_DELTAS:
        dr, dc = JUMP_DELTAS[action]
        mid = (cur[0] + dr // 2, cur[1] + dc // 2)
        landing = (cur[0] + dr, cur[1] + dc)
        if mid != opp:
            return None
        if not (0 <= landing[0] < BOARD_SIZE and 0 <= landing[1] < BOARD_SIZE):
            return None
        if can_move(opp[0], opp[1], landing[0], landing[1], hw, vw):
            return landing
        return None

    if action in DIAG_SPEC:
        prim, side = DIAG_SPEC[action]
        pdr, pdc = MOVE_DELTAS[prim]
        sdr, sdc = MOVE_DELTAS[side]
        mid = (cur[0] + pdr, cur[1] + pdc)
        jump_landing = (opp[0] + pdr, opp[1] + pdc)
        landing = (opp[0] + sdr, opp[1] + sdc)

        if mid != opp:
            return None
        if 0 <= jump_landing[0] < BOARD_SIZE and 0 <= jump_landing[1] < BOARD_SIZE:
            # If the straight jump square is open, the diagonal is not legal.
            if can_move(opp[0], opp[1], jump_landing[0], jump_landing[1], hw, vw):
                return None
        if not (0 <= landing[0] < BOARD_SIZE and 0 <= landing[1] < BOARD_SIZE):
            return None
        if can_move(opp[0], opp[1], landing[0], landing[1], hw, vw):
            return landing
        return None

    return None


def apply_action(state: QuoridorState, action: int, validate: bool = True) -> QuoridorState:
    """Return the state resulting from applying action. Pure function."""
    p1, p2 = state.p1_pos, state.p2_pos
    hw, vw = state.h_walls, state.v_walls
    p1w, p2w = state.p1_walls, state.p2_walls
    cur = state.current_player
    if validate and action not in legal_actions(state):
        raise ValueError(f"Illegal action: {action}")

    if action < 12:
        dest = _resolve_pawn_destination(state, action)
        if dest is None:
            raise ValueError(f"Illegal pawn action: {action}")
        if cur == 1:
            p1 = dest
        else:
            p2 = dest

    elif action < V_WALL_OFFSET:
        # Horizontal wall
        wr, wc = action_to_h_wall(action)
        hw = hw | {(wr, wc)}
        if cur == 1:
            p1w -= 1
        else:
            p2w -= 1
    else:
        # Vertical wall
        wr, wc = action_to_v_wall(action)
        vw = vw | {(wr, wc)}
        if cur == 1:
            p1w -= 1
        else:
            p2w -= 1

    return QuoridorState(p1, p2, hw, vw, p1w, p2w,
                         3 - cur, state.move_count + 1)


# ---------------------------------------------------------------------------
# Terminal check
# ---------------------------------------------------------------------------

def is_terminal(state: QuoridorState) -> bool:
    if state.p1_pos[0] == BOARD_SIZE - 1:
        return True
    if state.p2_pos[0] == 0:
        return True
    if state.is_draw:
        return True
    return False


def winner(state: QuoridorState) -> int | None:
    """Return 1, 2, or 0 (draw), or None if game is not over."""
    if state.p1_pos[0] == BOARD_SIZE - 1:
        return 1
    if state.p2_pos[0] == 0:
        return 2
    if state.is_draw:
        return 0
    return None


def adjudicate_winner(state: QuoridorState) -> int:
    """
    Resolve an artificial move-limit game by shortest path distance.

    Quoridor has no natural draw condition in this implementation; draws are
    introduced only by the training/evaluation move cap. When the cap is hit,
    the player closer to their goal is treated as the winner. Exact ties remain
    draws.
    """
    natural_winner = winner(state)
    if natural_winner not in (None, 0):
        return natural_winner

    p1_distance = shortest_path_length(state.p1_pos, BOARD_SIZE - 1, state.h_walls, state.v_walls)
    p2_distance = shortest_path_length(state.p2_pos, 0, state.h_walls, state.v_walls)

    if p1_distance is None and p2_distance is None:
        return 0
    if p1_distance is None:
        return 2
    if p2_distance is None:
        return 1
    if p1_distance < p2_distance:
        return 1
    if p2_distance < p1_distance:
        return 2

    p1_progress = state.p1_pos[0]
    p2_progress = BOARD_SIZE - 1 - state.p2_pos[0]
    if p1_progress > p2_progress:
        return 1
    if p2_progress > p1_progress:
        return 2
    return 0
