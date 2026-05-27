#!/usr/bin/env python3
"""Check the shape of the saved replay buffer."""
import sys
import pathlib
from pathlib import Path
import numpy as np

buffer_path = Path("data/replay_buffer.npz")

if not buffer_path.exists():
    print(f"Buffer file not found: {buffer_path}")
    sys.exit(1)

print(f"Loading buffer from {buffer_path}...")
try:
    with np.load(buffer_path, allow_pickle=False) as data:
        print(f"\nBuffer contents:")
        for key in data.keys():
            arr = data[key]
            print(f"  {key:12} : shape={arr.shape}, dtype={arr.dtype}")
        
        obs_shape = data["obs"].shape
        print(f"\n✓ Observation shape: {obs_shape}")
        print(f"  Channel dimension: {obs_shape[1]}")
        print(f"  Current code expects: (N, 20, 9, 9)")
        
        if obs_shape[1] != 20:
            print(f"\n✗ MISMATCH! Saved buffer has {obs_shape[1]} channels, but code expects 20")
            print(f"  This is likely why the buffer fails to load and training can't continue.")
            print(f"\nSOLUTION: Run training with --fresh_buffer flag to create a new buffer:")
            print(f"  python scripts/train.py --fresh_buffer")
        else:
            print(f"\n✓ Channel dimension matches (20)")
            
except Exception as e:
    print(f"✗ Failed to load buffer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
