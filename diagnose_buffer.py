#!/usr/bin/env python3
"""Diagnose why the buffer is not filling up."""
import os
import sys
import pathlib
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from replay.buffer import ReplayBuffer
from env.encoding import encode_state, NUM_CHANNELS, TENSOR_SHAPE
from env.quoridor_env import QuoridorEnv
from selfplay.game_generator import GameGenerator
from network.policy_value_net import build_net
from network.inference_server import InferenceServer


def test_buffer_push():
    """Test that buffer can accept observations."""
    print("=" * 60)
    print("TEST 1: Buffer Push")
    print("=" * 60)
    
    buffer = ReplayBuffer(capacity=1000)
    
    # Create a dummy observation
    obs = np.random.randn(*TENSOR_SHAPE).astype(np.float32)
    obs = np.clip(obs, 0, 1)
    policy = np.random.randn(140).astype(np.float32)
    value = 0.5
    
    print(f"Observation shape: {obs.shape} (expected {TENSOR_SHAPE})")
    print(f"Observation dtype: {obs.dtype}")
    print(f"NUM_CHANNELS: {NUM_CHANNELS}")
    
    try:
        buffer.push(obs, policy, value)
        print(f"✓ Buffer accepted observation")
        print(f"  Buffer size: {buffer.size}")
    except Exception as e:
        print(f"✗ Buffer rejected observation: {e}")
        return False
    
    return True


def test_encoding():
    """Test that encoding produces correct shape."""
    print("\n" + "=" * 60)
    print("TEST 2: State Encoding")
    print("=" * 60)
    
    env = QuoridorEnv()
    env.reset()
    
    try:
        obs = encode_state(env.state.canonical())
        print(f"Encoded observation shape: {obs.shape}")
        print(f"Expected shape: {TENSOR_SHAPE}")
        print(f"Dtype: {obs.dtype}")
        print(f"Min: {obs.min():.4f}, Max: {obs.max():.4f}")
        
        if obs.shape == TENSOR_SHAPE:
            print(f"✓ Encoding produces correct shape")
            return True
        else:
            print(f"✗ Shape mismatch! Got {obs.shape}, expected {TENSOR_SHAPE}")
            return False
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_buffer_and_encoding():
    """Test that encoded observations can be stored in buffer."""
    print("\n" + "=" * 60)
    print("TEST 3: Buffer + Encoding Integration")
    print("=" * 60)
    
    buffer = ReplayBuffer(capacity=1000)
    env = QuoridorEnv()
    env.reset()
    
    try:
        obs = encode_state(env.state.canonical())
        print(f"Encoded shape: {obs.shape}")
        
        # Try pushing
        policy = np.random.randn(140).astype(np.float32)
        value = 0.5
        
        buffer.push(obs, policy, value)
        print(f"✓ Successfully pushed encoded observation to buffer")
        print(f"  Buffer size: {buffer.size}")
        return True
    except Exception as e:
        print(f"✗ Failed to push: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_generation():
    """Test that game generation works."""
    print("\n" + "=" * 60)
    print("TEST 4: Game Generation (with real inference)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = build_net(device=device)
    server = InferenceServer(model, device=device, batch_size=32)
    server.start()
    print("✓ Inference server started")
    
    def inference_fn(obs_np, mask_np):
        future = server.submit(obs_np[0], mask_np[0])
        policy, value = future.result(timeout=10)
        return policy[np.newaxis], np.array([[value]])
    
    try:
        gen = GameGenerator(
            inference_fn=inference_fn,
            num_simulations=10,  # Quick test with few sims
            augment=False,  # Disable augmentation for now
        )
        
        print("Generating one game...")
        t0 = time.time()
        steps = gen.generate()
        elapsed = time.time() - t0
        
        print(f"✓ Generated {len(steps)} steps in {elapsed:.2f}s")
        
        if steps:
            step = steps[0]
            print(f"  Sample step obs shape: {step.obs.shape}")
            print(f"  Sample step policy shape: {step.policy.shape}")
            print(f"  Sample step outcome: {step.outcome}")
            
            if step.obs.shape == TENSOR_SHAPE:
                print(f"✓ Game generation produces correct shape")
                return True
            else:
                print(f"✗ Shape mismatch in game generation")
                return False
        else:
            print(f"✗ Game generation returned no steps")
            return False
            
    except Exception as e:
        print(f"✗ Game generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        server.stop()


def test_buffer_filling():
    """Test buffer filling with generated games."""
    print("\n" + "=" * 60)
    print("TEST 5: Buffer Filling with Games")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = build_net(device=device)
    server = InferenceServer(model, device=device, batch_size=32)
    server.start()
    
    def inference_fn(obs_np, mask_np):
        future = server.submit(obs_np[0], mask_np[0])
        policy, value = future.result(timeout=10)
        return policy[np.newaxis], np.array([[value]])
    
    buffer = ReplayBuffer(capacity=10000)
    
    try:
        gen = GameGenerator(
            inference_fn=inference_fn,
            num_simulations=10,
            augment=False,
        )
        
        print(f"Generating 5 games...")
        for game_num in range(5):
            steps = gen.generate()
            for step in steps:
                try:
                    buffer.push(step.obs, step.policy, step.outcome)
                except Exception as e:
                    print(f"✗ Failed to push step: {e}")
                    return False
            print(f"  Game {game_num+1}: {len(steps)} steps, buffer size: {buffer.size}")
        
        print(f"✓ Successfully filled buffer to {buffer.size} samples")
        return True
        
    except Exception as e:
        print(f"✗ Buffer filling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        server.stop()


if __name__ == "__main__":
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
    results = {}
    results["buffer_push"] = test_buffer_push()
    results["encoding"] = test_encoding()
    results["integration"] = test_buffer_and_encoding()
    
    # Only run expensive tests if basic tests pass
    if results["buffer_push"] and results["encoding"] and results["integration"]:
        results["game_gen"] = test_game_generation()
        if results["game_gen"]:
            results["buffer_fill"] = test_buffer_filling()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(results.values())
    print("\n" + ("✓ All tests passed!" if all_passed else "✗ Some tests failed"))
    sys.exit(0 if all_passed else 1)
