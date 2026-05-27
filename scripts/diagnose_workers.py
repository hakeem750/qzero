"""
Diagnostic script to test if selfplay workers can communicate with the inference server.

Usage:
  python scripts/diagnose_workers.py --device cuda
"""
import os
import sys
import pathlib
import torch
import numpy as np

# Ensure deterministic behavior
if torch.cuda.is_available() and 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from network.policy_value_net import build_net
from network.inference_server import InferenceServer
from env.quoridor_env import QuoridorEnv
from env.encoding import encode_state
from mcts.search import _legal_mask
import time


def test_inference_server(device_str: str = "cuda"):
    """Test that inference server works."""
    print("=" * 60)
    print("TEST 1: Inference Server")
    print("=" * 60)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build and start inference server
    model = build_net(compile_model=(device.type == "cuda"), device=device)
    server = InferenceServer(model, device=str(device), batch_size=32)
    server.start()
    print("✓ Inference server started")
    
    # Test single inference
    env = QuoridorEnv()
    env.reset()
    obs = encode_state(env.state.canonical()).astype(np.float32)
    mask = _legal_mask(env.state).astype(bool)
    
    obs_batch = obs[np.newaxis]  # (1, 20, 9, 9)
    mask_batch = mask[np.newaxis]  # (1, 140)
    
    print(f"Sending inference request...")
    t0 = time.time()
    future = server.submit(obs_batch[0], mask_batch[0])
    policy, value = future.result(timeout=10)
    elapsed = time.time() - t0
    
    print(f"✓ Got response in {elapsed:.3f}s")
    print(f"  Policy shape: {policy.shape}, sum: {policy.sum():.4f}")
    # value is a scalar from inference_server (value_np[i, 0])
    print(f"  Value: {float(value):.4f}")
    
    # Test batched inference
    print(f"\nSending 8 batched requests...")
    futures = []
    t0 = time.time()
    for i in range(8):
        env_i = QuoridorEnv()
        env_i.reset()
        obs_i = encode_state(env_i.state.canonical()).astype(np.float32)
        mask_i = _legal_mask(env_i.state).astype(bool)
        future = server.submit(obs_i, mask_i)
        futures.append(future)
    
    results = []
    for i, future in enumerate(futures):
        policy, value = future.result(timeout=10)
        results.append((policy, value))
    
    elapsed = time.time() - t0
    print(f"✓ Got 8 responses in {elapsed:.3f}s (avg {elapsed/8:.3f}s each)")
    
    server.stop()
    print("✓ Inference server stopped\n")
    return True


def test_game_generation(device_str: str = "cuda"):
    """Test that game generation works."""
    print("=" * 60)
    print("TEST 2: Single Game Generation")
    print("=" * 60)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # Build model and inference function
    model = build_net(compile_model=(device.type == "cuda"), device=device)
    server = InferenceServer(model, device=str(device), batch_size=32)
    server.start()
    print("✓ Inference server started")
    
    def inference_fn(obs_np, mask_np):
        """Synchronous inference wrapper.
        
        MCTS passes batched inputs (batch_size=1):
          obs_np shape: (1, 20, 9, 9)
          mask_np shape: (1, 140)
        
        Server expects single unbatched observations, so unbatch, infer, and re-batch.
        """
        # Unbatch the single observation
        obs_single = obs_np[0]  # (20, 9, 9)
        mask_single = mask_np[0]  # (140,)
        
        # Submit to server
        future = server.submit(obs_single, mask_single)
        policy, value = future.result(timeout=10)
        
        # Re-batch for MCTS
        return policy[np.newaxis], np.array([[value]])
    
    # Create game generator
    from selfplay.game_generator import GameGenerator
    gen = GameGenerator(
        inference_fn=inference_fn,
        num_simulations=50,  # Quick test
        augment=False,
        max_moves=300,
        resign_threshold=None,
    )
    
    print("Generating one game...")
    t0 = time.time()
    try:
        steps = gen.generate()
        elapsed = time.time() - t0
        print(f"✓ Game completed in {elapsed:.2f}s")
        print(f"  Trajectory length: {len(steps)} steps")
        if steps:
            print(f"  Sample step obs shape: {steps[0].obs.shape}")
            print(f"  Sample step policy shape: {steps[0].policy.shape}")
            print(f"  Sample step outcome: {steps[0].outcome}")
    except Exception as e:
        print(f"✗ Game generation failed: {e}")
        import traceback
        traceback.print_exc()
        server.stop()
        return False
    
    server.stop()
    print("✓ Inference server stopped\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    try:
        success = test_inference_server(args.device)
        if success:
            success = test_game_generation(args.device)
        
        if success:
            print("=" * 60)
            print("✓ ALL TESTS PASSED - workers should work")
            print("=" * 60)
        else:
            print("\n✗ TESTS FAILED - check errors above")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
