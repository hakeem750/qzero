from env.state import MAX_MOVES
from tests.test_rules import _make_state
from selfplay.game_generator import _adjudicated_winner


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
