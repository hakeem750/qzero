import numpy as np

from mcts.node import Node
from mcts.search import MCTS


def _root_with_visits(visits_by_action):
    root = Node(prior=1.0)
    for action, visits in visits_by_action.items():
        child = Node(prior=1.0)
        child.visit_count = visits
        root.children[action] = child
    return root


class TestMCTSActionProbs:
    def test_low_temperature_large_visits_stays_normalized(self):
        mcts = MCTS()
        root = _root_with_visits({3: 100_000, 7: 90_000, 11: 80_000})

        probs = mcts.action_probs(root, temperature=0.1)

        assert np.isfinite(probs).all()
        assert probs.min() >= 0.0
        assert probs.sum() == 1.0

    def test_zero_child_visits_falls_back_to_children(self):
        mcts = MCTS()
        root = _root_with_visits({4: 0, 8: 0})

        probs = mcts.action_probs(root, temperature=1.0)

        assert probs.sum() == 1.0
        assert probs[4] == 0.5
        assert probs[8] == 0.5
        assert np.count_nonzero(probs) == 2
