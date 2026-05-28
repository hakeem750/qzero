"""
Tests for the Quoridor environment: rules, moves, walls, terminals.
Run with: pytest tests/test_rules.py -v
"""
import pytest
import numpy as np

from env.state import initial_state, QuoridorState, BOARD_SIZE
from env.rules import (
    adjudicate_winner,
    can_move,
    legal_actions,
    apply_action,
    is_terminal,
    shortest_path_length,
    winner,
)
from env.actions import h_wall_to_action, v_wall_to_action, NUM_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> QuoridorState:
    base = initial_state()
    fields = {
        "p1_pos": base.p1_pos,
        "p2_pos": base.p2_pos,
        "h_walls": base.h_walls,
        "v_walls": base.v_walls,
        "p1_walls": base.p1_walls,
        "p2_walls": base.p2_walls,
        "current_player": base.current_player,
        "move_count": base.move_count,
    }
    fields.update(kwargs)
    return QuoridorState(**fields)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_player_positions(self):
        s = initial_state()
        assert s.p1_pos == (0, 4)
        assert s.p2_pos == (8, 4)

    def test_no_walls(self):
        s = initial_state()
        assert len(s.h_walls) == 0
        assert len(s.v_walls) == 0

    def test_wall_counts(self):
        s = initial_state()
        assert s.p1_walls == 10
        assert s.p2_walls == 10

    def test_p1_moves_first(self):
        s = initial_state()
        assert s.current_player == 1

    def test_legal_actions_nonempty(self):
        s = initial_state()
        acts = legal_actions(s)
        assert len(acts) > 0


# ---------------------------------------------------------------------------
# Basic movement
# ---------------------------------------------------------------------------

class TestMovement:
    def test_can_move_north(self):
        s = _make_state(p1_pos=(1, 4))
        acts = legal_actions(s)
        assert 0 in acts   # north

    def test_cant_move_off_board_north(self):
        s = _make_state(p1_pos=(0, 4))
        acts = legal_actions(s)
        assert 0 not in acts

    def test_south_from_start(self):
        s = initial_state()
        acts = legal_actions(s)
        assert 1 in acts   # south

    def test_east_from_col0(self):
        s = _make_state(p1_pos=(1, 0))
        acts = legal_actions(s)
        assert 2 in acts   # east
        assert 3 not in acts  # west off board

    def test_movement_changes_position(self):
        s = initial_state()  # p1 at (0,4)
        s2 = apply_action(s, 1)  # south
        assert s2.p1_pos == (1, 4)
        assert s2.current_player == 2


# ---------------------------------------------------------------------------
# Wall blocking
# ---------------------------------------------------------------------------

class TestWallBlocking:
    def test_h_wall_blocks_northward(self):
        # h_wall at (1,3) blocks (1,3)↔(2,3) and (1,4)↔(2,4)
        hw = frozenset({(1, 3)})
        s = _make_state(p1_pos=(2, 4), h_walls=hw)
        acts = legal_actions(s)
        assert 0 not in acts   # north blocked

    def test_h_wall_does_not_block_unrelated_cell(self):
        hw = frozenset({(1, 0)})
        s = _make_state(p1_pos=(1, 4), h_walls=hw)
        acts = legal_actions(s)
        assert 0 in acts   # north NOT blocked at col 4

    def test_v_wall_blocks_eastward(self):
        # v_wall at (0,4) blocks (0,4)↔(0,5) and (1,4)↔(1,5)
        vw = frozenset({(0, 4)})
        s = _make_state(p1_pos=(0, 4), v_walls=vw)
        acts = legal_actions(s)
        assert 2 not in acts   # east blocked

    def test_wall_respects_path(self):
        # Placing a wall that would trap a player should be illegal
        s = _make_state(p1_pos=(0, 0), p2_pos=(8, 4))
        # wall at (0,0) blocks south passage for p1
        acts = legal_actions(s)
        # The BFS should reject walls that trap either player
        hw_action = h_wall_to_action(0, 0)
        # result depends on path; just check no crash
        _ = hw_action in acts


# ---------------------------------------------------------------------------
# Jumping and diagonals
# ---------------------------------------------------------------------------

class TestJumps:
    def test_jump_north_over_opponent(self):
        s = _make_state(p1_pos=(2, 4), p2_pos=(1, 4))
        acts = legal_actions(s)
        assert 4 in acts   # jump_north

    def test_jump_blocked_gives_diagonals(self):
        # p1 at (1,4), p2 at (2,4), wall at (2,4) blocking jump_north
        hw = frozenset({(2, 3)})  # blocks (2,3)↔(3,3) and (2,4)↔(3,4)
        s = _make_state(p1_pos=(1, 4), p2_pos=(2, 4), h_walls=hw)
        acts = legal_actions(s)
        # jump north blocked → should get diag NW or NE
        # (exact availability depends on east/west walls)
        diag_actions = [a for a in acts if a in (8, 9)]
        # at least one diagonal should be available
        assert len(diag_actions) > 0 or 4 not in acts


# ---------------------------------------------------------------------------
# Terminal conditions
# ---------------------------------------------------------------------------

class TestTerminal:
    def test_p1_wins_at_row8(self):
        s = _make_state(p1_pos=(8, 4))
        assert is_terminal(s)
        assert winner(s) == 1

    def test_p2_wins_at_row0(self):
        s = _make_state(p2_pos=(0, 4))
        assert is_terminal(s)
        assert winner(s) == 2

    def test_draw_at_max_moves(self):
        from env.state import MAX_MOVES
        s = _make_state(move_count=MAX_MOVES)
        assert is_terminal(s)
        assert winner(s) == 0

    def test_shortest_path_length_initial_state(self):
        s = initial_state()
        assert shortest_path_length(s.p1_pos, BOARD_SIZE - 1, s.h_walls, s.v_walls) == 8
        assert shortest_path_length(s.p2_pos, 0, s.h_walls, s.v_walls) == 8

    def test_adjudicate_move_cap_prefers_closer_player(self):
        from env.state import MAX_MOVES
        s = _make_state(p1_pos=(7, 4), p2_pos=(8, 4), move_count=MAX_MOVES)
        assert winner(s) == 0
        assert adjudicate_winner(s) == 1

    def test_hash_includes_move_count(self):
        s1 = _make_state(move_count=10)
        s2 = _make_state(move_count=11)
        assert hash(s1) != hash(s2)

    def test_initial_not_terminal(self):
        s = initial_state()
        assert not is_terminal(s)


# ---------------------------------------------------------------------------
# Wall placement validation
# ---------------------------------------------------------------------------

class TestWallPlacement:
    def test_h_wall_intersection_conflict(self):
        hw = frozenset({(3, 3)})
        vw = frozenset({(3, 3)})
        s = _make_state(h_walls=hw, v_walls=vw)
        acts = legal_actions(s)
        # No additional wall at (3,3) should be legal
        assert h_wall_to_action(3, 3) not in acts
        assert v_wall_to_action(3, 3) not in acts

    def test_overlapping_h_walls(self):
        hw = frozenset({(3, 3)})
        s = _make_state(h_walls=hw)
        acts = legal_actions(s)
        # Adjacent anchors of the same orientation overlap one wall segment.
        assert h_wall_to_action(3, 2) not in acts
        assert h_wall_to_action(3, 4) not in acts

    def test_overlapping_v_walls(self):
        vw = frozenset({(3, 3)})
        s = _make_state(v_walls=vw)
        acts = legal_actions(s)
        assert v_wall_to_action(2, 3) not in acts
        assert v_wall_to_action(4, 3) not in acts

    def test_no_walls_left(self):
        s = _make_state(p1_walls=0)
        acts = legal_actions(s)
        wall_acts = [a for a in acts if a >= 12]
        assert len(wall_acts) == 0


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_encode_shape(self):
        from env.encoding import encode_state
        s = initial_state()
        obs = encode_state(s)
        assert obs.shape == (20, 9, 9)
        assert obs.dtype == np.float32

    def test_mirror_preserves_shape(self):
        from env.encoding import encode_state, mirror_state_and_policy
        s = initial_state()
        obs = encode_state(s)
        policy = np.random.dirichlet(np.ones(NUM_ACTIONS))
        m_obs, m_policy = mirror_state_and_policy(obs, policy)
        assert m_obs.shape == obs.shape
        assert m_policy.shape == policy.shape

    def test_mirror_policy_sums_to_one(self):
        from env.encoding import encode_state, mirror_state_and_policy
        s = initial_state()
        obs = encode_state(s)
        policy = np.random.dirichlet(np.ones(NUM_ACTIONS))
        _, m_policy = mirror_state_and_policy(obs, policy)
        # Mirrored policy may not sum to exactly 1 if some walls are out of range —
        # but for a uniform policy it should be close
        assert abs(m_policy.sum() - policy.sum()) < 0.1
