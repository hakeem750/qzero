import numpy as np

from env.actions import NUM_ACTIONS
from selfplay.game_generator import GameGenerator, select_action_from_policy
from tests.test_rules import _make_state


class TestPolicySelection:
    def test_deterministic_selection_uses_policy_argmax(self):
        state = _make_state(p1_pos=(3, 4), p2_pos=(8, 4), current_player=1)
        policy = np.zeros(NUM_ACTIONS, dtype=np.float64)
        policy[1] = 0.25
        policy[2] = 0.75

        action = select_action_from_policy(policy, state, deterministic=True)

        assert action == 2

    def test_illegal_policy_mass_is_masked(self):
        state = _make_state(p1_pos=(0, 4), p2_pos=(8, 4), current_player=1)
        policy = np.zeros(NUM_ACTIONS, dtype=np.float64)
        policy[0] = 1.0  # north is off-board and illegal

        action = select_action_from_policy(policy, state, deterministic=True)

        assert action in state.legal_actions()


def test_max_move_draw_generates_neutral_samples():
    steps = GameGenerator(
        inference_fn=None,
        num_simulations=0,
        augment=True,
        max_moves=2,
    ).generate()

    assert len(steps) == 4
    assert {step.outcome for step in steps} == {0.0}
