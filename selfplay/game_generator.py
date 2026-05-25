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
from env.rules import legal_actions, can_move, apply_action
from env.state import QuoridorState, BOARD_SIZE
from env.actions import NUM_ACTIONS
from mcts.search import MCTS


@dataclass(slots=True)
class TrajectoryStep:
    obs: np.ndarray           # (13, 9, 9) float32
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


def _shortest_path_distance(
    start: tuple[int, int],
    goal_row: int,
    h_walls: frozenset[tuple[int, int]],
    v_walls: frozenset[tuple[int, int]],
) -> int:
    frontier = [(start, 0)]
    visited = {start}
    for (r, c), dist in frontier:
        if r == goal_row:
            return dist
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if (nr, nc) not in visited and can_move(r, c, nr, nc, h_walls, v_walls):
                    visited.add((nr, nc))
                    frontier.append(((nr, nc), dist + 1))
    return BOARD_SIZE * BOARD_SIZE


def _adjudicated_winner(state: QuoridorState) -> int:
    """
    Return the real winner, or adjudicate artificial max-move draws.

    Self-play games can otherwise teach the value head an all-zero target while
    weak agents learn to burn walls and shuffle. Shortest path is a stable
    Quoridor-specific tiebreak: closer to goal wins; equal distance falls back
    to raw pawn progress, then remains a draw if still tied.
    """
    if state.p1_pos[0] == BOARD_SIZE - 1:
        return 1
    if state.p2_pos[0] == 0:
        return 2

    p1_dist = _shortest_path_distance(state.p1_pos, BOARD_SIZE - 1, state.h_walls, state.v_walls)
    p2_dist = _shortest_path_distance(state.p2_pos, 0, state.h_walls, state.v_walls)
    if p1_dist < p2_dist:
        return 1
    if p2_dist < p1_dist:
        return 2

    p1_progress = state.p1_pos[0]
    p2_progress = BOARD_SIZE - 1 - state.p2_pos[0]
    if p1_progress > p2_progress:
        return 1
    if p2_progress > p1_progress:
        return 2
    return 0


def _state_cycle_key(state: QuoridorState) -> tuple:
    return (
        state.p1_pos,
        state.p2_pos,
        state.h_walls,
        state.v_walls,
        state.p1_walls,
        state.p2_walls,
        state.current_player,
    )


def _player_distance(state: QuoridorState, player: int) -> int:
    if player == 1:
        return _shortest_path_distance(state.p1_pos, BOARD_SIZE - 1, state.h_walls, state.v_walls)
    return _shortest_path_distance(state.p2_pos, 0, state.h_walls, state.v_walls)


def select_action_with_progress(
    policy: np.ndarray,
    state: QuoridorState,
    temperature: float,
    seen_counts: dict[tuple, int],
    progress_bias: float = 0.20,
    repeat_penalty: float = 0.35,
) -> int:
    """
    Select from MCTS policy while discouraging pathless shuffling.

    The policy target is left unchanged for training. This only affects the
    sampled game continuation, nudging near-ties toward shortest-path progress
    and away from repeated layouts.
    """
    actions = legal_actions(state)
    if not actions:
        return int(np.argmax(policy))

    base_dist = _player_distance(state, state.current_player)
    scores = np.full(NUM_ACTIONS, -np.inf, dtype=np.float64)
    for action in actions:
        child = apply_action(state, action)
        next_dist = _player_distance(child, state.current_player)
        progress = base_dist - next_dist
        repeat_count = seen_counts.get(_state_cycle_key(child), 0)
        scores[action] = float(policy[action]) + progress_bias * progress - repeat_penalty * repeat_count

    if temperature > 0.05:
        weights = np.zeros(NUM_ACTIONS, dtype=np.float64)
        action_scores = scores[actions]
        action_scores -= np.max(action_scores)
        weights[actions] = np.exp(action_scores / max(temperature, 1e-3))
        weights = _safe_sample_policy(weights, state)
        return int(np.random.choice(NUM_ACTIONS, p=weights))

    return int(np.argmax(scores))


class GameGenerator:
    """
    Runs one complete self-play game using MCTS + neural network.

    Args:
        inference_fn: callable(obs_batch, mask_batch) → (policy, value)
          obs_batch:  (B, 13, 9, 9) float32
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
        max_moves: int = 300,           # Optional move limit for curriculum
    ) -> None:
        self.inference_fn = inference_fn or _uniform_inference
        self.num_simulations = num_simulations
        self.augment = augment
        self.max_moves = max_moves
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
        seen_counts = {_state_cycle_key(env.state): 1}

        root = self.mcts.new_root(env.state)

        while not env.is_terminal() and env.state.move_count < self.max_moves:
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

            action = select_action_with_progress(policy, env.state, temp, seen_counts)

            # Tree reuse: advance root
            if action in root.children:
                root = root.children[action]
                root.action_from_parent = -1
            else:
                # Edge case: action not in tree (e.g. noise led to unexpected pick)
                env.step(action)
                key = _state_cycle_key(env.state)
                seen_counts[key] = seen_counts.get(key, 0) + 1
                root = self.mcts.new_root(env.state)
                continue

            env.step(action)
            key = _state_cycle_key(env.state)
            seen_counts[key] = seen_counts.get(key, 0) + 1

        game_winner = _adjudicated_winner(env.state)
        print(f"[worker] game_winner={game_winner} len={len(raw)}")  # DIAGNOSTIC

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
