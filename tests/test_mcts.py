import numpy as np

from env.state import initial_state
from env.state import QuoridorState
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


class TestMCTSVirtualLoss:
    def test_sync_simulations_do_not_make_root_virtual_loss_negative(self):
        mcts = MCTS(virtual_loss=3)
        root = mcts.new_root(initial_state())

        def inference_fn(obs_batch, mask_batch):
            policy = mask_batch.astype(np.float64)
            policy /= policy.sum(axis=1, keepdims=True)
            value = np.zeros((obs_batch.shape[0], 1), dtype=np.float32)
            return policy, value

        mcts.run_simulations_sync(root, inference_fn, num_simulations=1)
        for _ in range(5):
            mcts.run_simulations_sync(root, inference_fn, num_simulations=10)
            assert root.virtual_loss == 0

        for child in root.children.values():
            assert child.virtual_loss >= 0

    def test_unexpanded_root_uses_requested_simulations(self):
        mcts = MCTS(virtual_loss=1)
        root = mcts.new_root(initial_state())
        calls = 0

        def inference_fn(obs_batch, mask_batch):
            nonlocal calls
            calls += 1
            policy = mask_batch.astype(np.float64)
            policy /= policy.sum(axis=1, keepdims=True)
            value = np.zeros((obs_batch.shape[0], 1), dtype=np.float32)
            return policy, value

        mcts.run_simulations_sync(root, inference_fn, num_simulations=5)

        assert calls == 5
        assert root.visit_count == 5


def test_mcts_prefers_immediate_win():
    state = QuoridorState(
        p1_pos=(7, 4),
        p2_pos=(8, 0),
        h_walls=frozenset(),
        v_walls=frozenset(),
        p1_walls=0,
        p2_walls=0,
        current_player=1,
        move_count=20,
    )

    def inference_fn(obs_batch, mask_batch):
        policy = mask_batch.astype(np.float64)
        policy /= policy.sum(axis=1, keepdims=True)
        value = np.zeros((obs_batch.shape[0], 1), dtype=np.float32)
        return policy, value

    mcts = MCTS(c_puct=1.5, noise_frac=0.0, virtual_loss=0)
    root = mcts.new_root(state)

    mcts.run_simulations_sync(root, inference_fn, num_simulations=100, add_noise=False)

    visits = {action: child.visit_count for action, child in root.children.items()}
    assert max(visits, key=visits.get) == 1
