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


def test_resignation_keeps_assigned_winner():
    calls = 0

    def inference_fn(obs_batch, mask_batch):
        nonlocal calls
        calls += 1
        policy = mask_batch.astype(np.float64)
        totals = policy.sum(axis=1, keepdims=True)
        np.divide(policy, totals, out=policy, where=totals > 0)
        value = np.zeros((obs_batch.shape[0], 1), dtype=np.float32)
        if calls > 1:
            value[:] = -1.0
        return policy, value

    steps = GameGenerator(
        inference_fn=inference_fn,
        num_simulations=1,
        augment=False,
        max_moves=20,
        resign_threshold=-0.5,
    ).generate()

    assert steps
    assert steps[0].outcome == 1.0
