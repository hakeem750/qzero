from env.anti_stall import AntiStallConfig, AntiStallTracker, board_key, progress_swing
from env.rules import apply_action
from env.state import initial_state
from tests.test_rules import _make_state


def test_board_key_ignores_move_count():
    s1 = _make_state(move_count=10)
    s2 = _make_state(move_count=20)

    assert board_key(s1) == board_key(s2)


def test_progress_swing_rewards_shorter_own_path():
    before = initial_state()
    after = apply_action(before, 1)

    assert progress_swing(before, after, player=1) > 0.0


def test_repetition_limit_detects_revisited_board():
    config = AntiStallConfig(repetition_limit=2, stall_limit=0)
    tracker = AntiStallTracker(config)
    before = initial_state()
    tracker.reset(before)
    after = _make_state(move_count=2)

    event = tracker.observe(before, after, player=1, action=2)

    assert event.repeated
    assert event.shaping < 0.0
