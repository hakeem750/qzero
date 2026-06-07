import numpy as np

from env.actions import (
    canonical_legal_action_mask,
    policy_from_canonical,
    policy_to_canonical,
)
from env.rules import apply_action
from env.state import initial_state


def test_p2_actions_are_flipped_into_canonical_network_space():
    state = apply_action(initial_state(), 1)
    assert state.current_player == 2

    mask = canonical_legal_action_mask(state)

    assert mask[1]
    assert not mask[0]

    real_policy = np.zeros(140, dtype=np.float32)
    real_policy[0] = 1.0

    canonical_policy = policy_to_canonical(real_policy, state)
    assert canonical_policy[1] == 1.0
    assert canonical_policy[0] == 0.0

    round_trip = policy_from_canonical(canonical_policy, state)
    assert np.array_equal(round_trip, real_policy)
