import pytest

from env.rules import apply_action, legal_actions
from tests.test_rules import _make_state


@pytest.mark.parametrize(
    "p1_pos,p2_pos,walls,blocked_jump,is_horizontal",
    [
        ((2, 4), (1, 4), frozenset({(0, 3)}), 4, True),
        ((1, 4), (2, 4), frozenset({(2, 3)}), 5, True),
        ((4, 3), (4, 4), frozenset({(4, 4)}), 6, False),
        ((4, 5), (4, 4), frozenset({(4, 3)}), 7, False),
    ],
)
def test_straight_jumps_do_not_cross_walls(
    p1_pos,
    p2_pos,
    walls,
    blocked_jump,
    is_horizontal,
):
    if is_horizontal:
        state = _make_state(p1_pos=p1_pos, p2_pos=p2_pos, h_walls=walls)
    else:
        state = _make_state(p1_pos=p1_pos, p2_pos=p2_pos, v_walls=walls)

    acts = legal_actions(state)
    assert blocked_jump not in acts

    with pytest.raises(ValueError):
        apply_action(state, blocked_jump, validate=False)
