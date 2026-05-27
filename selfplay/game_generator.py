"""
Self-play game generator.

Produces training trajectories via MCTS-guided self-play.

IMPROVEMENT: Symmetry augmentation baked in here.
  Every trajectory step is immediately mirrored (left-right flip)
  and both the original and mirrored samples are stored.
  This doubles the effective dataset size at zero extra game cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from env.quoridor_env import QuoridorEnv
from env.encoding import encode_state, mirror_state_and_policy
from env.rules import legal_actions, winner
from env.state import QuoridorState
from env.actions import NUM_ACTIONS
from mcts.search import MCTS


@dataclass(slots=True)
class TrajectoryStep:
    obs: np.ndarray           # (20, 9, 9) float32
    policy: np.ndarray        # (140,)     float32
    outcome: float            # game result from current player's view


def _temperature(move_number: int) -> float:
    """
    Exploration schedule — encourage diverse action types (moves vs walls).
      - First 40 half-moves: temperature=1.0 (broad exploration)
      - Moves 40-80:         temperature=0.5 (moderate exploration)
      - After 80:            temperature=0.1 (near-greedy)
    """
    if move_number < 40:
        return 1.0
    elif move_number < 80:
        return 0.5
    else:
        return 0.1


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


def _uniform_inference(obs_batch: np.ndarray, mask_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    policy = mask_batch.astype(np.float64)
    totals = policy.sum(axis=1, keepdims=True)
    np.divide(policy, totals, out=policy, where=totals > 0)
    value = np.zeros((obs_batch.shape[0], 1), dtype=np.float32)
    return policy, value


def select_action_from_policy(
    policy: np.ndarray,
    state: QuoridorState,
    deterministic: bool = False,
) -> int:
    """Select a legal action directly from the MCTS visit distribution."""
    actions = legal_actions(state)
    if not actions:
        return int(np.argmax(policy))

    probs = np.zeros(NUM_ACTIONS, dtype=np.float64)
    probs[actions] = np.asarray(policy, dtype=np.float64)[actions]
    probs = _safe_sample_policy(probs, state)
    if deterministic:
        return int(max(actions, key=lambda action: probs[action]))
    return int(np.random.choice(NUM_ACTIONS, p=probs))


class GameGenerator:
    """
    Runs one complete self-play game using MCTS + neural network.

    Args:
        inference_fn: callable(obs_batch, mask_batch) → (policy, value)
          obs_batch:  (B, 20, 9, 9) float32
          mask_batch: (B, 140)      bool
        resign_threshold: float or None
          If set to a value between -1 and 1, terminate the game early if
          the value estimate drops below this threshold (indicating likely loss).
          Default None means resignation is disabled (standard AlphaZero)
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
        max_moves: int = 300,           # Optional move limit for curriculum
        resign_threshold: float | None = None,  # Optional: resign if value drops below threshold
    ) -> None:
        self.inference_fn = inference_fn or _uniform_inference
        self.num_simulations = num_simulations
        self.augment = augment
        self.max_moves = max_moves
        self.resign_threshold = resign_threshold
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
        move_count = 0

        while not env.is_terminal() and env.state.move_count < self.max_moves:
            move_number = env.state.move_count
            cur_player  = env.state.current_player

            # MCTS search
            try:
                self.mcts.run_simulations_sync(
                    root,
                    self.inference_fn,
                    num_simulations=self.num_simulations,
                    add_noise=True,
                )
            except Exception as e:
                import sys
                print(f"[MCTS error at move {move_count}] {e}", file=sys.stderr)
                raise

            # Optional: check for resignation (early termination if clearly losing)
            if self.resign_threshold is not None and root.q_value < self.resign_threshold:
                # Treat as a loss from current player's perspective
                game_winner = 3 - cur_player  # opponent wins
                break

            # Policy target from visit counts
            temp   = _temperature(move_number)
            policy = self.mcts.action_probs(root, temperature=temp)
            policy = _safe_sample_policy(policy, env.state)

            # Record observation from current player's perspective (MUST be canonical)
            obs = encode_state(env.state.canonical())
            raw.append((obs, policy, cur_player))

            action = select_action_from_policy(policy, env.state)

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

        game_winner = winner(env.state)

        # Assign outcomes (from each player's perspective at the time)
        steps: List[TrajectoryStep] = []
        for obs, policy, player in raw:
            if game_winner in (None, 0):
                outcome = 0.0
            else:
                outcome = 1.0 if game_winner == player else -1.0

            steps.append(TrajectoryStep(obs=obs, policy=policy, outcome=outcome))

            # IMPROVEMENT: LR symmetry augmentation
            if self.augment:
                m_obs, m_policy = mirror_state_and_policy(obs, policy)
                steps.append(TrajectoryStep(obs=m_obs, policy=m_policy, outcome=outcome))

        return steps
