import numpy as np

from env.actions import (
    canonical_legal_action_mask,
    action_to_canonical,
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


def test_blocked_jump_is_blocked_in_canonical_space():
    # Real-board position for P2 to move with a wall blocking jump_north.
    state = apply_action(initial_state(), 1)
    state = state.__class__(
        p1_pos=(2, 4),
        p2_pos=(3, 4),
        h_walls=frozenset({(2, 3)}),
        v_walls=frozenset(),
        p1_walls=state.p1_walls,
        p2_walls=state.p2_walls,
        current_player=2,
        move_count=state.move_count,
    )

    real_legal = set(state.legal_actions())
    assert 4 not in real_legal  # P2 jump_north is blocked by the wall

    mask = canonical_legal_action_mask(state)
    assert not mask[action_to_canonical(4, state)]
