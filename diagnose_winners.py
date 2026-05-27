#!/usr/bin/env python3
"""
Diagnostic script to check game_winner values from 5 self-play games.
This verifies whether the curriculum advancement threshold can be reached.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from network.policy_value_net import build_net
from network.inference_server import InferenceServer
from selfplay.game_generator import GameGenerator
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build a fresh untrained model
    model = build_net(compile_model=False, device=device)
    
    # Start inference server
    server = InferenceServer(model, device=str(device), batch_size=1)
    server.start()
    print("Inference server started")
    
    def inference_fn(obs_np, mask_np):
        future = server.submit(obs_np[0], mask_np[0])
        policy, value = future.result()
        return policy[np.newaxis], np.array([[value]])
    
    # Run 5 games with minimal MCTS (just to get outcomes fast)
    gen = GameGenerator(
        inference_fn=inference_fn,
        num_simulations=50,
        dirichlet_alpha=0.3,
        noise_frac=0.25,
        augment=False,  # Disable augmentation for cleaner count
    )
    
    print("\n=== Running 5 diagnostic games ===\n")
    winners = []
    for game_num in range(5):
        print(f"Game {game_num + 1}...", end="", flush=True)
        steps = gen.generate()
        print(f" ({len(steps)} half-moves)\n")
        # Extract winner from outcome values (all outcomes in a game should have same winner)
        if steps and steps[0].outcome != 0.0:
            winner = 1 if steps[0].outcome > 0 else 2
            winners.append(winner)
        else:
            winners.append(0)  # Draw
    
    print("\n=== Results ===")
    print(f"Winners: {winners}")
    print(f"Decisive games: {sum(1 for w in winners if w != 0)} / 5")
    print(f"Decisive rate: {100 * sum(1 for w in winners if w != 0) / 5:.1f}%")
    print(f"\nCurriculum threshold: 30% decisive required")
    print(f"Expected for curriculum advancement: {'YES' if sum(1 for w in winners if w != 0) >= 2 else 'NO'}")
    
    server.stop()

if __name__ == "__main__":
    main()
