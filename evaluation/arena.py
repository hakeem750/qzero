"""
Evaluation arena: candidate vs best_model head-to-head.

Promotion threshold: 55% win rate.
Match conditions:
  - Equal first-player games
  - Fixed seeds
  - No Dirichlet noise
  - Deterministic inference (temperature -> 0)
"""
from __future__ import annotations

import time

import numpy as np
import torch

from env.anti_stall import AntiStallConfig, AntiStallTracker
from env.actions import action_name
from env.quoridor_env import QuoridorEnv
from env.rules import winner
from env.state import MAX_MOVES
from mcts.search import MCTS
from selfplay.game_generator import select_action_from_policy


def _greedy_inference(model, device, dtype):
    """Return a synchronous inference function (no noise)."""
    model.eval()

    @torch.no_grad()
    def fn(obs_np, mask_np):
        obs = torch.from_numpy(obs_np).to(device, dtype=torch.float32)
        mask = torch.from_numpy(mask_np).to(device, dtype=torch.bool)
        with torch.autocast(device_type=device.type, dtype=dtype):
            policy, value = model.predict(obs, mask)
        # Cast to float32 before numpy() - bfloat16 is not supported by numpy
        return policy.float().cpu().numpy(), value.float().cpu().numpy()

    return fn


def play_game(
    model_a_fn,
    model_b_fn,
    a_is_p1: bool,
    num_simulations: int = 400,
    seed: int = 0,
    max_moves: int = MAX_MOVES,
    display: str = "off",
    game_index: int = 1,
    anti_stall_config: AntiStallConfig | None = None,
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
    anti_stall = AntiStallTracker(anti_stall_config or AntiStallConfig())
    anti_stall.reset(env.state)
    termination = "natural"
    repeated_positions = 0
    non_progress_moves = 0
    progress_swing_total = 0.0

    def player_label(player: int) -> str:
        model_name = "candidate" if (player == 1) == a_is_p1 else "best"
        return f"P{player} {model_name}"

    if display == "board":
        print(f"\n  eval game {game_index}: initial position")
        print(env.render())

    while not env.is_terminal() and env.state.move_count < max_moves:
        cur = env.state.current_player
        if (cur == 1) == a_is_p1:
            mcts, fn, root = mcts_a, model_a_fn, root_a
        else:
            mcts, fn, root = mcts_b, model_b_fn, root_b

        mcts.run_simulations_sync(root, fn, num_simulations, add_noise=False)
        visit_policy = mcts.action_probs(root, temperature=1.0)
        deterministic_policy = mcts.action_probs(root, temperature=0.0)
        entropy = -(visit_policy.clip(1e-8) * np.log(visit_policy.clip(1e-8))).sum()
        policy_entropies.append(float(entropy))
        root_values.append((cur, float(root.q_value)))
        action = select_action_from_policy(deterministic_policy, env.state, deterministic=True)
        move_number = env.state.move_count
        confidence = float(visit_policy[action])
        q_value = float(root.q_value)
        before_state = env.state
        env.step(action)
        event = anti_stall.observe(before_state, env.state, cur, action)
        repeated_positions += int(event.repeated)
        non_progress_moves += int(event.progress_swing <= 0.0)
        progress_swing_total += event.progress_swing

        if display == "moves":
            print(
                f"    eval game {game_index} move {move_number:>3}: "
                f"{player_label(cur)} -> {action_name(action)} "
                f"p={confidence:.3f} q={q_value:+.3f}",
                flush=True,
            )
        elif display == "board":
            print(
                f"\n  eval game {game_index} move {move_number}: "
                f"{player_label(cur)} -> {action_name(action)} "
                f"p={confidence:.3f} q={q_value:+.3f}"
            )
            print(env.render())

        if event.repeated:
            termination = "repetition"
            break
        if event.stalled:
            termination = "stall"
            break

        if action in root_a.children:
            root_a = root_a.children[action]
        else:
            root_a = mcts_a.new_root(env.state)

        if action in root_b.children:
            root_b = root_b.children[action]
        else:
            root_b = mcts_b.new_root(env.state)

    natural_winner = winner(env.state)
    cutoff = natural_winner in (None, 0) and env.state.move_count >= max_moves
    if cutoff:
        termination = "cutoff"
    winner_id = natural_winner or 0

    if display != "off":
        if winner_id == 0:
            result_text = "draw"
        else:
            result_text = f"{player_label(winner_id)} wins"
        cutoff_text = f" {termination}-draw" if winner_id == 0 and termination != "natural" else ""
        print(
            f"  eval game {game_index} result: {result_text} "
            f"moves={env.state.move_count}{cutoff_text}",
            flush=True,
        )

    value_errors = []
    for player, value_pred in root_values:
        if winner_id == 0:
            value_target = 0.0
        else:
            value_target = 1.0 if winner_id == player else -1.0
        value_errors.append((value_pred - value_target) ** 2)

    return {
        "winner": winner_id,
        "cutoff": cutoff,
        "termination": termination,
        "game_length": env.state.move_count,
        "repeated_positions": repeated_positions,
        "non_progress_moves": non_progress_moves,
        "avg_progress_swing": float(progress_swing_total / max(1, env.state.move_count)),
        "policy_entropy": float(np.mean(policy_entropies)) if policy_entropies else 0.0,
        "value_calibration_mse": float(np.mean(value_errors)) if value_errors else 0.0,
    }


class Arena:
    """Run a promotion match between candidate and best_model."""

    def __init__(
        self,
        num_games: int = 100,
        num_sims: int = 400,
        win_thresh: float = 0.55,
        max_moves: int = MAX_MOVES,
        device: str = "cuda",
        anti_stall_config: AntiStallConfig | None = None,
    ) -> None:
        self.num_games = num_games
        self.num_sims = num_sims
        self.win_thresh = win_thresh
        self.max_moves = max_moves
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.elo_tracker = EloTracker()
        self.anti_stall_config = anti_stall_config or AntiStallConfig()

    def evaluate(
        self,
        candidate,
        best_model,
        progress: bool = False,
        display: str = "off",
        display_games: int = 1,
    ) -> dict:
        """
        Run an evaluation match.

        If display is left off but display_games is positive, the first
        display_games are still shown in compact console mode so at least one
        game can be watched during evaluation.
        """
        fn_cand = _greedy_inference(candidate, self.device, self.dtype)
        fn_best = _greedy_inference(best_model, self.device, self.dtype)

        display_mode = display if display != "off" else ("moves" if display_games > 0 else "off")

        wins = draws = losses = 0
        game_lengths = []
        policy_entropies = []
        value_calibration = []
        cutoffs = []
        repetition_ends = []
        stall_ends = []
        non_progress_rates = []
        progress_swings = []
        t0 = time.time()

        for i in range(self.num_games):
            show_display = display_mode if i < display_games else "off"
            if progress and show_display != "off":
                print(f"\n  eval game {i + 1}/{self.num_games}:")

            a_is_p1 = (i % 2 == 0)
            game = play_game(
                fn_cand,
                fn_best,
                a_is_p1=a_is_p1,
                num_simulations=self.num_sims,
                seed=i,
                max_moves=self.max_moves,
                display=show_display,
                game_index=i + 1,
                anti_stall_config=self.anti_stall_config,
            )

            result = game["winner"]
            game_lengths.append(game["game_length"])
            policy_entropies.append(game["policy_entropy"])
            value_calibration.append(game["value_calibration_mse"])
            cutoffs.append(bool(game.get("cutoff", False)))
            repetition_ends.append(game.get("termination") == "repetition")
            stall_ends.append(game.get("termination") == "stall")
            non_progress_rates.append(float(game["non_progress_moves"]) / max(1.0, float(game["game_length"])))
            progress_swings.append(float(game["avg_progress_swing"]))

            if result == 0:
                draws += 1
            elif (result == 1) == a_is_p1:
                wins += 1
            else:
                losses += 1

            if progress and show_display == "off":
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
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "promoted": promoted,
            "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
            "policy_entropy": float(np.mean(policy_entropies)) if policy_entropies else 0.0,
            "value_calibration_mse": float(np.mean(value_calibration)) if value_calibration else 0.0,
            "cutoff_rate": float(np.mean(cutoffs)) if cutoffs else 0.0,
            "repetition_rate": float(np.mean(repetition_ends)) if repetition_ends else 0.0,
            "stall_rate": float(np.mean(stall_ends)) if stall_ends else 0.0,
            "non_progress_rate": float(np.mean(non_progress_rates)) if non_progress_rates else 0.0,
            "avg_progress_swing": float(np.mean(progress_swings)) if progress_swings else 0.0,
            "elo": elo,
        }


class EloTracker:
    def __init__(self, initial_elo: float = 1000.0, k: float = 32.0) -> None:
        self.elo = initial_elo
        self.k = k
        self.history: list[float] = [initial_elo]

    def update(self, win_rate: float) -> float:
        """win_rate is candidate win rate against current best (0.5 = equal)."""
        delta = self.k * (win_rate - 0.5)
        self.elo += delta
        self.history.append(self.elo)
        return self.elo
