"""Anti-stall helpers for repeated positions and path-progress shaping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .actions import action_is_wall
from .rules import shortest_path_length
from .state import BOARD_SIZE, QuoridorState


def board_key(state: QuoridorState) -> Hashable:
    """Return a canonical board key that intentionally ignores move_count."""
    canonical = state.canonical()
    return (
        canonical.p1_pos,
        canonical.p2_pos,
        tuple(sorted(canonical.h_walls)),
        tuple(sorted(canonical.v_walls)),
        canonical.p1_walls,
        canonical.p2_walls,
        canonical.current_player,
    )


def player_distance(state: QuoridorState, player: int) -> int | None:
    if player == 1:
        return shortest_path_length(state.p1_pos, BOARD_SIZE - 1, state.h_walls, state.v_walls)
    return shortest_path_length(state.p2_pos, 0, state.h_walls, state.v_walls)


def progress_swing(before: QuoridorState, after: QuoridorState, player: int) -> float:
    """Positive when the mover improves their path or worsens the opponent path."""
    opponent = 3 - player
    own_before = player_distance(before, player)
    own_after = player_distance(after, player)
    opp_before = player_distance(before, opponent)
    opp_after = player_distance(after, opponent)

    own_delta = 0.0 if own_before is None or own_after is None else float(own_before - own_after)
    opp_delta = 0.0 if opp_before is None or opp_after is None else float(opp_after - opp_before)
    return own_delta + 0.5 * opp_delta


@dataclass(frozen=True, slots=True)
class AntiStallConfig:
    repetition_limit: int = 3
    stall_limit: int = 80
    progress_weight: float = 0.02
    repeat_penalty: float = 0.03
    non_progress_penalty: float = 0.002
    wall_no_progress_penalty: float = 0.01
    shaping_discount: float = 0.99


@dataclass(slots=True)
class AntiStallEvent:
    shaping: float
    progress_swing: float
    repeat_count: int
    repeated: bool
    stalled: bool


class AntiStallTracker:
    def __init__(self, config: AntiStallConfig | None = None) -> None:
        self.config = config or AntiStallConfig()
        self.seen: dict[Hashable, int] = {}
        self.non_progress_plies = 0

    def reset(self, state: QuoridorState) -> None:
        self.seen = {board_key(state): 1}
        self.non_progress_plies = 0

    def observe(
        self,
        before: QuoridorState,
        after: QuoridorState,
        player: int,
        action: int,
    ) -> AntiStallEvent:
        swing = progress_swing(before, after, player)
        if swing > 0.0:
            self.non_progress_plies = 0
        else:
            self.non_progress_plies += 1

        key = board_key(after)
        repeat_count = self.seen.get(key, 0) + 1
        self.seen[key] = repeat_count

        repeated = self.config.repetition_limit > 0 and repeat_count >= self.config.repetition_limit
        stalled = self.config.stall_limit > 0 and self.non_progress_plies >= self.config.stall_limit

        shaping = self.config.progress_weight * max(-2.0, min(2.0, swing))
        if repeat_count > 1:
            shaping -= self.config.repeat_penalty
        if swing <= 0.0:
            shaping -= self.config.non_progress_penalty
        if action_is_wall(action) and swing <= 0.0:
            shaping -= self.config.wall_no_progress_penalty

        return AntiStallEvent(
            shaping=shaping,
            progress_swing=swing,
            repeat_count=repeat_count,
            repeated=repeated,
            stalled=stalled,
        )
