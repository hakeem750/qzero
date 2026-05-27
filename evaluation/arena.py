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
from env.state import MAX_MOVES
from env.rules import winner
from mcts.search import MCTS, _legal_mask
from selfplay.game_generator import select_action_from_policy


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
    max_moves: int = MAX_MOVES,
) -> dict:
    """
    Play one game. Returns winner plus evaluation telemetry.
    No Dirichlet noise; greedy action selection.
    """
    np.random.seed(seed)
    mcts_a = MCTS(c_puct=1.5, noise_frac=0.0)
    mcts_b = MCTS(c_puct=1.5, noise_frac=0.0)

    env = QuoridorEnv()
    env.reset(seed=seed)

    root_a = mcts_a.new_root(env.state)
    root_b = mcts_b.new_root(env.state)
    policy_entropies = []
    root_values = []

    while not env.is_terminal() and env.state.move_count < max_moves:
        cur = env.state.current_player
        if (cur == 1) == a_is_p1:
            mcts, fn, root = mcts_a, model_a_fn, root_a
        else:
            mcts, fn, root = mcts_b, model_b_fn, root_b

        mcts.run_simulations_sync(root, fn, num_simulations, add_noise=False)
        visit_policy = mcts.action_probs(root, temperature=1.0)
        deterministic_policy = mcts.action_probs(root, temperature=0.0)
        policy_entropies.append(float(-(visit_policy.clip(1e-8) * np.log(visit_policy.clip(1e-8))).sum()))
        root_values.append((cur, float(root.q_value)))
        action = select_action_from_policy(deterministic_policy, env.state, deterministic=True)
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

    winner_id = winner(env.state) or 0
    value_errors = []
    for player, value_pred in root_values:
        if winner_id == 0:
            value_target = 0.0
        else:
            value_target = 1.0 if winner_id == player else -1.0
        value_errors.append((value_pred - value_target) ** 2)

    return {
        "winner": winner_id,
        "game_length": env.state.move_count,
        "policy_entropy": float(np.mean(policy_entropies)) if policy_entropies else 0.0,
        "value_calibration_mse": float(np.mean(value_errors)) if value_errors else 0.0,
    }


class Arena:
    """
    Run a promotion match between candidate and best_model.
    Candidate is promoted if win_rate >= win_thresh.
    """

    def __init__(
        self,
        num_games:   int   = 100,
        num_sims:    int   = 400,
        win_thresh:  float = 0.55,
        max_moves:   int   = 300,
        device: str        = "cuda",
    ) -> None:
        self.num_games  = num_games
        self.num_sims   = num_sims
        self.win_thresh = win_thresh
        self.max_moves  = max_moves
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype      = torch.bfloat16
        self.elo        = 1000.0

    def evaluate(self, candidate, best_model, progress: bool = False) -> dict:
        """Evaluate candidate vs best_model. Returns stats and promotion decision."""
        fn_cand = _greedy_inference(candidate,  self.device, self.dtype)
        fn_best = _greedy_inference(best_model, self.device, self.dtype)

        wins = draws = losses = 0
        game_stats = []
        game_lengths = []
        policy_entropies = []
        value_calibration_mses = []

        for i in range(self.num_games):
            if progress:
                print(f"  game {i+1}/{self.num_games}...", end="\r")
            a_is_p1 = (i % 2 == 0)   # alternate starting sides
            result = play_game(fn_cand, fn_best, a_is_p1=a_is_p1,
                               num_simulations=self.num_sims, seed=i,
                               max_moves=self.max_moves)
            game_stats.append(result)
            
            winner_id = result["winner"]
            if winner_id == 0:
                draws += 1
            elif (winner_id == 1) == a_is_p1:
                wins += 1
            else:
                losses += 1
            
            game_lengths.append(result["game_length"])
            policy_entropies.append(result["policy_entropy"])
            value_calibration_mses.append(result["value_calibration_mse"])

        if progress:
            print("  evaluation complete.  ")

        total = wins + draws + losses
        win_rate = (wins + 0.5 * draws) / total
        promoted = win_rate >= self.win_thresh

        # Update ELO based on win rate
        delta_elo = 32.0 * (win_rate - 0.5)
        self.elo += delta_elo

        return {
            "wins":     wins,
            "draws":    draws,
            "losses":   losses,
            "win_rate": win_rate,
            "promoted": promoted,
            "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
            "policy_entropy": float(np.mean(policy_entropies)) if policy_entropies else 0.0,
            "value_calibration_mse": float(np.mean(value_calibration_mses)) if value_calibration_mses else 0.0,
            "elo": self.elo,
            "game_stats": game_stats,
        }




class Arena:
    """
    Run a promotion match between candidate and best_model.
    """

    def __init__(
        self,
        num_games:   int   = 100,
        num_sims:    int   = 400,
        win_thresh:  float = 0.55,
        max_moves:   int   = MAX_MOVES,
        device: str        = "cuda",
    ) -> None:
        self.num_games  = num_games
        self.num_sims   = num_sims
        self.win_thresh = win_thresh
        self.max_moves  = max_moves
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype      = torch.bfloat16
        self.elo_tracker = EloTracker()

    def evaluate(self, candidate, best_model, progress: bool = False) -> dict:
        import time

        fn_cand = _greedy_inference(candidate,  self.device, self.dtype)
        fn_best = _greedy_inference(best_model, self.device, self.dtype)

        wins = draws = losses = 0
        game_lengths = []
        policy_entropies = []
        value_calibration = []
        t0 = time.time()
        for i in range(self.num_games):
            a_is_p1 = (i % 2 == 0)   # alternate starting sides
            game = play_game(fn_cand, fn_best, a_is_p1=a_is_p1,
                             num_simulations=self.num_sims, seed=i,
                             max_moves=self.max_moves)
            result = game["winner"]
            game_lengths.append(game["game_length"])
            policy_entropies.append(game["policy_entropy"])
            value_calibration.append(game["value_calibration_mse"])
            if result == 0:
                draws += 1
            elif (result == 1) == a_is_p1:
                wins += 1
            else:
                losses += 1
            if progress:
                elapsed = time.time() - t0
                print(
                    f"  eval game {i + 1}/{self.num_games}: "
                    f"W/D/L={wins}/{draws}/{losses}  t={elapsed:.1f}s",
                    flush=True,
                )

        total = wins + draws + losses
        win_rate = (wins + 0.5 * draws) / total
        promoted = win_rate >= self.win_thresh
        elo = self.elo_tracker.update(win_rate)

        return {
            "wins":     wins,
            "draws":    draws,
            "losses":   losses,
            "win_rate": win_rate,
            "promoted": promoted,
            "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
            "policy_entropy": float(np.mean(policy_entropies)) if policy_entropies else 0.0,
            "value_calibration_mse": float(np.mean(value_calibration)) if value_calibration else 0.0,
            "elo": elo,
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
