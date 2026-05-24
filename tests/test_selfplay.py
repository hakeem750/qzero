from env.state import MAX_MOVES
from env.actions import NUM_ACTIONS
from tests.test_rules import _make_state
from selfplay.game_generator import _adjudicated_winner, select_action_with_progress

import numpy as np


class TestSelfPlayAdjudication:
    def test_max_move_draw_prefers_shorter_remaining_path(self):
        state = _make_state(
            p1_pos=(7, 4),
            p2_pos=(8, 4),
            move_count=MAX_MOVES,
        )

        assert _adjudicated_winner(state) == 1

    def test_exactly_equal_draw_stays_draw(self):
        state = _make_state(move_count=MAX_MOVES)

        assert _adjudicated_winner(state) == 0


class TestProgressSelection:
    def test_prefers_shortest_path_progress_on_tied_policy(self):
        state = _make_state(p1_pos=(3, 4), p2_pos=(8, 4), current_player=1)
        policy = np.zeros(NUM_ACTIONS, dtype=np.float64)
        policy[1] = 0.5  # south, closer to P1 goal row
        policy[2] = 0.5  # east, no progress

        action = select_action_with_progress(policy, state, temperature=0.0, seen_counts={})

        assert action == 1
