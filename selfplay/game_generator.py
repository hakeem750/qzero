"""
Self-play game generator.

Produces training trajectories via MCTS-guided self-play.

IMPROVEMENT: Symmetry augmentation baked in here.
  Every trajectory step is immediately mirrored (left-right flip)
  and both the original and mirrored samples are stored.
  This doubles the effective dataset size at zero extra game cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

import numpy as np

from env.quoridor_env import QuoridorEnv
from env.encoding import encode_state, mirror_state_and_policy
from env.rules import legal_actions, is_terminal, winner
from env.state import QuoridorState
from env.actions import NUM_ACTIONS
from mcts.search import MCTS, _legal_mask
from mcts.node import Node


@dataclass(slots=True)
class TrajectoryStep:
    obs: np.ndarray           # (17, 9, 9) float32
    policy: np.ndarray        # (140,)     float32
    outcome: float            # game result from current player's view


def _temperature(move_number: int) -> float:
    """
    Exploration schedule:
      - First 20 half-moves: temperature=1.0 (broad exploration)
      - After:               temperature=0.1 (near-greedy)
    """
    return 1.0 if move_number < 20 else 0.1


def _safe_sample_policy(policy: np.ndarray, state: QuoridorState) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64)
    policy = np.where(np.isfinite(policy) & (policy > 0), policy, 0.0)

    total = policy.sum(dtype=np.float64)
    if total <= 0.0:
        policy = np.zeros(NUM_ACTIONS, dtype=np.float64)
        actions = legal_actions(state)
        if actions:
            policy[actions] = 1.0 / len(actions)
        else:
            policy[:] = 1.0 / NUM_ACTIONS
    else:
        policy = policy / total

    policy[int(np.argmax(policy))] += 1.0 - policy.sum(dtype=np.float64)
    return policy


class GameGenerator:
    """
    Runs one complete self-play game using MCTS + neural network.

    Args:
        inference_fn: callable(obs_batch, mask_batch) → (policy, value)
          obs_batch:  (B, 17, 9, 9) float32
          mask_batch: (B, 140)      bool
          policy:     (B, 140)      float32
          value:      (B, 1)        float32
        num_simulations: MCTS simulations per move
        augment: whether to apply LR symmetry augmentation (IMPROVEMENT)
    """

    def __init__(
        self,
        inference_fn: Callable,
        num_simulations: int = 800,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_frac: float = 0.25,
        augment: bool = True,           # IMPROVEMENT
    ) -> None:
        self.inference_fn = inference_fn
        self.num_simulations = num_simulations
        self.augment = augment
        self.mcts = MCTS(
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            noise_frac=noise_frac,
        )

    # ------------------------------------------------------------------
    def generate(self) -> List[TrajectoryStep]:
        """
        Play one game to completion.
        Returns a list of TrajectorySteps (with augmentation if enabled).
        """
        env = QuoridorEnv()
        env.reset()
        self.mcts.reset()

        # Raw trajectory buffer: (obs, policy, player_at_step)
        raw: List[Tuple[np.ndarray, np.ndarray, int]] = []

        root = self.mcts.new_root(env.state)

        while not env.is_terminal():
            move_number = env.state.move_count
            cur_player  = env.state.current_player

            # MCTS search
            self.mcts.run_simulations_sync(
                root,
                self.inference_fn,
                num_simulations=self.num_simulations,
                add_noise=True,
            )

            # Policy target from visit counts
            temp   = _temperature(move_number)
            policy = self.mcts.action_probs(root, temperature=temp)
            policy = _safe_sample_policy(policy, env.state)

            # Record observation from current player's perspective
            obs = encode_state(env.state)
            raw.append((obs, policy, cur_player))

            # Pick action (sample if temp > 0, argmax if near zero)
            if temp > 0.05:
                action = int(np.random.choice(NUM_ACTIONS, p=policy))
            else:
                action = int(np.argmax(policy))

            # Tree reuse: advance root
            if action in root.children:
                root = root.children[action]
                root.action_from_parent = -1
            else:
                # Edge case: action not in tree (e.g. noise led to unexpected pick)
                env.step(action)
                root = self.mcts.new_root(env.state)
                continue

            env.step(action)

        game_winner = env.winner()

        # Assign outcomes (from each player's perspective at the time)
        steps: List[TrajectoryStep] = []
        for obs, policy, player in raw:
            if game_winner == 0 or game_winner is None:
                outcome = 0.0
            elif game_winner == player:
                outcome = 1.0
            else:
                outcome = -1.0

            steps.append(TrajectoryStep(obs=obs, policy=policy, outcome=outcome))

            # IMPROVEMENT: LR symmetry augmentation
            if self.augment:
                m_obs, m_policy = mirror_state_and_policy(obs, policy)
                steps.append(TrajectoryStep(obs=m_obs, policy=m_policy, outcome=outcome))

        return steps
