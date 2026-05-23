import numpy as np

from scripts.play import action_policy, outcome_for_player


def test_action_policy_is_one_hot():
    policy = action_policy(17)

    assert policy.shape == (140,)
    assert policy.dtype == np.float32
    assert policy.sum() == 1.0
    assert policy[17] == 1.0


def test_outcome_for_player():
    assert outcome_for_player(1, 1) == 1.0
    assert outcome_for_player(2, 1) == -1.0
    assert outcome_for_player(0, 1) == 0.0
    assert outcome_for_player(None, 1) == 0.0
