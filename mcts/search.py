"""
MCTS search engine.

Improvements over blueprint:
  1. Transposition table (hash → Node) avoids re-expanding identical
     states reached via different paths — common in Quoridor with wall
     symmetries.
  2. Virtual loss for safe parallel batched leaf collection.
  3. Dirichlet noise applied only at root, with configurable alpha and
     frac — matches AlphaZero spec exactly.
  4. visit_softmax_temperature applied at policy extraction for smooth
     early-game exploration.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.state import QuoridorState
from env.rules import legal_actions, apply_action, is_terminal, winner
from env.encoding import encode_state
from env.actions import canonical_legal_action_mask, policy_from_canonical
from .node import Node


class MCTS:
    """
    Single-threaded MCTS with batching hooks.

    For batched usage: call select_leaves() to collect a batch,
    then call backup_batch() after inference.  This is the pattern
    used by selfplay/worker.py.
    """

    def __init__(
        self,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_frac: float = 0.25,
        virtual_loss: int = 3,          # IMPROVEMENT
        max_tree_size: int = 200_000,   # guard against memory blowup
        use_transpositions: bool = False,
    ) -> None:
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self.vl = virtual_loss
        self.max_tree_size = max_tree_size
        self.use_transpositions = use_transpositions

        # Transposition reuse is opt-in because priors/visits are edge stats.
        self._trans_table: Dict[int, Node] = {}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._trans_table.clear()

    # ------------------------------------------------------------------
    def new_root(self, state: QuoridorState) -> Node:
        h = hash(state)
        if self.use_transpositions and h in self._trans_table:
            node = self._trans_table[h]
        else:
            node = Node(prior=1.0, state=state, to_play=state.current_player)
            if self.use_transpositions:
                self._trans_table[h] = node
        node.action_from_parent = -1
        return node

    # ------------------------------------------------------------------
    def _select(self, root: Node) -> Tuple[List[Node], Node]:
        """
        Traverse from root to a leaf using PUCT, applying virtual loss.
        Returns (path, leaf).
        """
        path = [root]
        node = root
        node.add_virtual_loss(self.vl)
        while node.expanded and not is_terminal(node.state):
            best_action, best_child = self._best_child(node)
            # If best_child is None, node has no children despite being expanded
            # (terminal state or all actions blocked). Stop here.
            if best_child is None:
                break
            best_child.add_virtual_loss(self.vl)   # IMPROVEMENT
            path.append(best_child)
            node = best_child
        return path, node

    def _best_child(self, node: Node) -> Tuple[int, Node]:
        n_parent = node.visit_count + node.virtual_loss
        best_score = -math.inf
        best_action = -1
        best_child = None
        for action, child in node.children.items():
            score = child.puct_score(n_parent, self.c_puct, parent_to_play=node.to_play)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    # ------------------------------------------------------------------
    def expand(
        self,
        node: Node,
        policy: np.ndarray,   # (140,) masked & normalised
    ) -> None:
        """Expand a leaf node using the network policy prior."""
        if node.expanded:
            return
        actions = legal_actions(node.state)
        if not actions:
            node.expanded = True
            return
        for action in actions:
            child_state = apply_action(node.state, action, validate=False)
            h = hash(child_state)
            if self.use_transpositions and h in self._trans_table:
                child = self._trans_table[h]
            else:
                child = Node(
                    prior=float(policy[action]),
                    state=child_state,
                    to_play=child_state.current_player,
                    action_from_parent=action,
                )
                if self.use_transpositions:
                    self._trans_table[h] = child
            node.children[action] = child
        node.expanded = True

    # ------------------------------------------------------------------
    def _backup(self, path: List[Node], value: float) -> None:
        """
        Propagate value back up path.
        value is from the perspective of the player at the leaf.
        We negate at each ply because players alternate.
        """
        for node in reversed(path):
            node.revert_virtual_loss(self.vl)   # IMPROVEMENT
            node.backup(value)
            value = -value

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def add_dirichlet_noise(
        self,
        root: Node,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not root.children:
            return
        actions = list(root.children.keys())
        if rng is None:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        else:
            noise = rng.dirichlet([self.dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            child = root.children[a]
            # Ensure prior stays positive after noise mixing
            new_prior = (1 - self.noise_frac) * child.prior + self.noise_frac * n
            child.prior = max(1e-8, new_prior)  # avoid log(0) issues

    def run_simulations_sync(
        self,
        root: Node,
        inference_fn,      # callable(obs, mask) → (policy, value) arrays
        num_simulations: int,
        add_noise: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Synchronous (single-threaded) MCTS for evaluation and testing.
        inference_fn is called once per simulation.
        """
        simulations_remaining = num_simulations

        if not root.expanded:
            policy, value = _evaluate_state(root.state, inference_fn)
            self.expand(root, policy)
            if add_noise:
                self.add_dirichlet_noise(root, rng=rng)
            if num_simulations <= 0:
                return
            root.backup(value)
            simulations_remaining -= 1

        for _ in range(simulations_remaining):
            path, leaf = self._select(root)

            if is_terminal(leaf.state):
                w = winner(leaf.state)
                value = _terminal_value(w, leaf.to_play)
                self._backup(path, value)
                continue

            policy, value = _evaluate_state(leaf.state, inference_fn)
            self.expand(leaf, policy)
            self._backup(path, value)

    # ------------------------------------------------------------------
    def action_probs(
        self,
        root: Node,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Return policy target vector from visit counts.
        temperature=1.0 → proportional; temperature→0 → argmax.
        """
        from env.actions import NUM_ACTIONS
        visits = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if temperature == 0 or temperature < 1e-6:
            probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            if root.children:
                best = max(root.children, key=lambda action: visits[action])
            else:
                best = int(np.argmax(visits))
            probs[best] = 1.0
            return probs

        actions = list(root.children.keys())
        probs = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if actions:
            action_visits = visits[actions]
            positive = action_visits > 0
            if positive.any():
                scaled = np.full_like(action_visits, -np.inf, dtype=np.float64)
                scaled[positive] = np.log(action_visits[positive]) / temperature
                scaled -= np.max(scaled[positive])
                weights = np.exp(scaled)
                probs[actions] = weights
            else:
                probs[actions] = 1.0
        else:
            probs[:] = 1.0

        return _normalize_probs(probs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legal_mask(state: QuoridorState) -> np.ndarray:
    from env.actions import legal_action_mask
    return legal_action_mask(state)


def _canonical_legal_mask(state: QuoridorState) -> np.ndarray:
    return canonical_legal_action_mask(state)


def _evaluate_state(state: QuoridorState, inference_fn) -> tuple[np.ndarray, float]:
    """Evaluate a real state while presenting canonical actions to the network."""
    obs = encode_state(state.canonical())
    mask = _canonical_legal_mask(state)
    policy_canonical, value = inference_fn(obs[None], mask[None])
    policy = policy_from_canonical(policy_canonical[0], state)
    return policy, float(value[0, 0])


def _terminal_value(winner_id: Optional[int], to_play: int) -> float:
    if winner_id == 0 or winner_id is None:
        return 0.0
    return 1.0 if winner_id == to_play else -1.0


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    """
    Return a finite float64 probability vector whose sum is exactly 1.0.
    np.random.choice is strict about this, so absorb roundoff into the
    largest entry after normalization.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.where(np.isfinite(probs) & (probs > 0), probs, 0.0)
    total = probs.sum(dtype=np.float64)
    if total <= 0.0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= total

    residual = 1.0 - probs.sum(dtype=np.float64)
    probs[int(np.argmax(probs))] += residual
    return probs
