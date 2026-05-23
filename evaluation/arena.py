# """
# Evaluation arena: candidate vs best_model head-to-head.

# Promotion threshold: 55% win rate.
# Match conditions:
#   - Equal first-player games
#   - Fixed seeds
#   - No Dirichlet noise
#   - Deterministic inference (temperature → 0)
# """
# from __future__ import annotations

# import copy
# from typing import Callable, Optional

# import numpy as np
# import torch

# from env.quoridor_env import QuoridorEnv
# from env.rules import is_terminal, winner
# from env.encoding import encode_state
# from env.actions import NUM_ACTIONS
# from mcts.search import MCTS, _legal_mask


# def _greedy_inference(model, device, dtype):
#     """Return a synchronous inference function (no noise)."""
#     model.eval()
#     @torch.no_grad()
#     def fn(obs_np, mask_np):
#         obs  = torch.from_numpy(obs_np ).to(device, dtype=torch.float32)
#         mask = torch.from_numpy(mask_np).to(device, dtype=torch.bool)
#         with torch.autocast(device_type=device.type, dtype=dtype):
#             policy, value = model.predict(obs, mask)
#         return policy.cpu().numpy(), value.cpu().numpy()
#     return fn


# def play_game(
#     model_a_fn,    # inference fn for model A
#     model_b_fn,    # inference fn for model B
#     a_is_p1: bool,
#     num_simulations: int = 400,
#     seed: int = 0,
# ) -> int:
#     """
#     Play one game. Returns winner: 1 or 2 (or 0 for draw).
#     No Dirichlet noise; greedy action selection.
#     """
#     np.random.seed(seed)
#     mcts_a = MCTS(c_puct=1.5, noise_frac=0.0)
#     mcts_b = MCTS(c_puct=1.5, noise_frac=0.0)

#     env = QuoridorEnv()
#     env.reset(seed=seed)

#     root_a = mcts_a.new_root(env.state)
#     root_b = mcts_b.new_root(env.state)

#     while not env.is_terminal():
#         cur = env.state.current_player
#         if (cur == 1) == a_is_p1:
#             mcts, fn, root = mcts_a, model_a_fn, root_a
#         else:
#             mcts, fn, root = mcts_b, model_b_fn, root_b

#         mcts.run_simulations_sync(root, fn, num_simulations, add_noise=False)
#         action = int(np.argmax(mcts.action_probs(root, temperature=0.0)))
#         env.step(action)

#         # Tree reuse
#         if action in root_a.children:
#             root_a = root_a.children[action]
#         else:
#             root_a = mcts_a.new_root(env.state)

#         if action in root_b.children:
#             root_b = root_b.children[action]
#         else:
#             root_b = mcts_b.new_root(env.state)

#     return env.winner() or 0


# class Arena:
#     """
#     Run a promotion match between candidate and best_model.
#     """

#     def __init__(
#         self,
#         num_games:   int   = 100,
#         num_sims:    int   = 400,
#         win_thresh:  float = 0.55,
#         device: str        = "cuda",
#     ) -> None:
#         self.num_games  = num_games
#         self.num_sims   = num_sims
#         self.win_thresh = win_thresh
#         self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
#         self.dtype      = torch.bfloat16

#     def evaluate(self, candidate, best_model) -> dict:
#         fn_cand = _greedy_inference(candidate,  self.device, self.dtype)
#         fn_best = _greedy_inference(best_model, self.device, self.dtype)

#         wins = draws = losses = 0
#         for i in range(self.num_games):
#             a_is_p1 = (i % 2 == 0)   # alternate starting sides
#             result = play_game(fn_cand, fn_best, a_is_p1=a_is_p1,
#                                num_simulations=self.num_sims, seed=i)
#             if result == 0:
#                 draws += 1
#             elif (result == 1) == a_is_p1:
#                 wins += 1
#             else:
#                 losses += 1

#         total = wins + draws + losses
#         win_rate = (wins + 0.5 * draws) / total
#         promoted = win_rate >= self.win_thresh

#         return {
#             "wins":     wins,
#             "draws":    draws,
#             "losses":   losses,
#             "win_rate": win_rate,
#             "promoted": promoted,
#         }


# # ---------------------------------------------------------------------------
# # ELO tracking
# # ---------------------------------------------------------------------------

# class EloTracker:
#     def __init__(self, initial_elo: float = 1000.0, k: float = 32.0) -> None:
#         self.elo = initial_elo
#         self.k   = k
#         self.history: list[float] = [initial_elo]

#     def update(self, win_rate: float) -> float:
#         """win_rate is candidate win rate against current best (0.5 = equal)."""
#         expected = 1 / (1 + 10 ** ((self.elo - self.elo) / 400))  # vs self = 0.5
#         delta = self.k * (win_rate - 0.5)
#         self.elo += delta
#         self.history.append(self.elo)
#         return self.elo
"""
Evaluation arena: candidate vs best_model head-to-head.

Promotion threshold: 55% win rate.
Match conditions:
  - Equal first-player games
  - Fixed seeds
  - No Dirichlet noise
  - Deterministic inference (temperature → 0)
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch

from env.quoridor_env import QuoridorEnv
from env.rules import is_terminal, winner
from env.encoding import encode_state
from env.actions import NUM_ACTIONS
from mcts.search import MCTS, _legal_mask


def _greedy_inference(model, device, dtype):
    """Return a synchronous inference function (no noise)."""
    model.eval()
    @torch.no_grad()
    def fn(obs_np, mask_np):
        obs  = torch.from_numpy(obs_np ).to(device, dtype=torch.float32)
        mask = torch.from_numpy(mask_np).to(device, dtype=torch.bool)
        with torch.autocast(device_type=device.type, dtype=dtype):
            policy, value = model.predict(obs, mask)
        # Cast to float32 before numpy() — bfloat16 is not supported by numpy
        return policy.float().cpu().numpy(), value.float().cpu().numpy()
    return fn


def play_game(
    model_a_fn,    # inference fn for model A
    model_b_fn,    # inference fn for model B
    a_is_p1: bool,
    num_simulations: int = 400,
    seed: int = 0,
) -> int:
    """
    Play one game. Returns winner: 1 or 2 (or 0 for draw).
    No Dirichlet noise; greedy action selection.
    """
    np.random.seed(seed)
    mcts_a = MCTS(c_puct=1.5, noise_frac=0.0)
    mcts_b = MCTS(c_puct=1.5, noise_frac=0.0)

    env = QuoridorEnv()
    env.reset(seed=seed)

    root_a = mcts_a.new_root(env.state)
    root_b = mcts_b.new_root(env.state)

    while not env.is_terminal():
        cur = env.state.current_player
        if (cur == 1) == a_is_p1:
            mcts, fn, root = mcts_a, model_a_fn, root_a
        else:
            mcts, fn, root = mcts_b, model_b_fn, root_b

        mcts.run_simulations_sync(root, fn, num_simulations, add_noise=False)
        action = int(np.argmax(mcts.action_probs(root, temperature=0.0)))
        env.step(action)

        # Tree reuse
        if action in root_a.children:
            root_a = root_a.children[action]
        else:
            root_a = mcts_a.new_root(env.state)

        if action in root_b.children:
            root_b = root_b.children[action]
        else:
            root_b = mcts_b.new_root(env.state)

    return env.winner() or 0


class Arena:
    """
    Run a promotion match between candidate and best_model.
    """

    def __init__(
        self,
        num_games:   int   = 100,
        num_sims:    int   = 400,
        win_thresh:  float = 0.55,
        device: str        = "cuda",
    ) -> None:
        self.num_games  = num_games
        self.num_sims   = num_sims
        self.win_thresh = win_thresh
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype      = torch.bfloat16

    def evaluate(self, candidate, best_model) -> dict:
        fn_cand = _greedy_inference(candidate,  self.device, self.dtype)
        fn_best = _greedy_inference(best_model, self.device, self.dtype)

        wins = draws = losses = 0
        for i in range(self.num_games):
            a_is_p1 = (i % 2 == 0)   # alternate starting sides
            result = play_game(fn_cand, fn_best, a_is_p1=a_is_p1,
                               num_simulations=self.num_sims, seed=i)
            if result == 0:
                draws += 1
            elif (result == 1) == a_is_p1:
                wins += 1
            else:
                losses += 1

        total = wins + draws + losses
        win_rate = (wins + 0.5 * draws) / total
        promoted = win_rate >= self.win_thresh

        return {
            "wins":     wins,
            "draws":    draws,
            "losses":   losses,
            "win_rate": win_rate,
            "promoted": promoted,
        }


# ---------------------------------------------------------------------------
# ELO tracking
# ---------------------------------------------------------------------------

class EloTracker:
    def __init__(self, initial_elo: float = 1000.0, k: float = 32.0) -> None:
        self.elo = initial_elo
        self.k   = k
        self.history: list[float] = [initial_elo]

    def update(self, win_rate: float) -> float:
        """win_rate is candidate win rate against current best (0.5 = equal)."""
        expected = 1 / (1 + 10 ** ((self.elo - self.elo) / 400))  # vs self = 0.5
        delta = self.k * (win_rate - 0.5)
        self.elo += delta
        self.history.append(self.elo)
        return self.elo