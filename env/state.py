"""
Immutable Quoridor game state.

Coordinates: (row, col), 0-indexed.
  P1 starts at row 0, goals at row 8.
  P2 starts at row 8, goals at row 0.

Wall anchors are (r, c) with r, c in [0, 7].
  A horizontal wall anchor (r, c) blocks movement
    between (r, c)↔(r+1, c) and (r, c+1)↔(r+1, c+1).
  A vertical wall anchor (r, c) blocks movement
    between (r, c)↔(r, c+1) and (r+1, c)↔(r+1, c+1).
"""
from __future__ import annotations
from dataclasses import dataclass

BOARD_SIZE = 9
MAX_WALLS   = 10
WALL_GRID   = 8      # anchor grid is 8×8
MAX_MOVES   = 300    # draw after this many half-moves (increased to allow decisive games)


@dataclass(frozen=True, slots=True)
class QuoridorState:
    p1_pos: tuple[int, int]
    p2_pos: tuple[int, int]

    # frozensets give O(1) membership and are hashable
    h_walls: frozenset[tuple[int, int]]
    v_walls: frozenset[tuple[int, int]]

    p1_walls: int
    p2_walls: int

    current_player: int   # 1 or 2
    move_count: int

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def active_pos(self) -> tuple[int, int]:
        return self.p1_pos if self.current_player == 1 else self.p2_pos

    @property
    def opponent_pos(self) -> tuple[int, int]:
        return self.p2_pos if self.current_player == 1 else self.p1_pos

    @property
    def active_walls(self) -> int:
        return self.p1_walls if self.current_player == 1 else self.p2_walls

    @property
    def opponent_walls(self) -> int:
        return self.p2_walls if self.current_player == 1 else self.p1_walls

    @property
    def is_draw(self) -> bool:
        return self.move_count >= MAX_MOVES

    def __hash__(self) -> int:  # slots=True drops the default; restore it
        return hash((
            self.p1_pos, self.p2_pos,
            self.h_walls, self.v_walls,
            self.p1_walls, self.p2_walls,
            self.current_player,
        ))


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def initial_state() -> QuoridorState:
    """Standard starting position."""
    return QuoridorState(
        p1_pos=(0, 4),
        p2_pos=(8, 4),
        h_walls=frozenset(),
        v_walls=frozenset(),
        p1_walls=MAX_WALLS,
        p2_walls=MAX_WALLS,
        current_player=1,
        move_count=0,
    )
