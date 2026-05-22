"""
Replay buffer with compressed ring-buffer storage.

Storage layout (per slot):
  obs:     uint8  (17, 9, 9)  — scaled ×255, saves 4× vs float32
  policy:  float16 (140,)
  value:   float16 scalar

Capacity: 500,000 positions ≈ 500k × (17×81 + 140 + 1) bytes
  ≈ 500k × 1520 bytes ≈ 760 MB  (fits in RAM for most training rigs)

IMPROVEMENT over blueprint: the buffer also stores a sample_weight
array for easy extension to prioritized experience replay (PER).
Currently uniform (all 1.0) but can be updated externally without
changing the buffer interface.
"""
from __future__ import annotations

import threading
from typing import Tuple

import numpy as np


class ReplayBuffer:

    def __init__(self, capacity: int = 500_000) -> None:
        self.capacity = capacity
        self._lock = threading.Lock()
        self._size = 0
        self._ptr  = 0

        # Pre-allocate storage
        self._obs    = np.zeros((capacity, 17, 9, 9), dtype=np.uint8)
        self._policy = np.zeros((capacity, 140),     dtype=np.float16)
        self._value  = np.zeros((capacity,),          dtype=np.float16)
        # IMPROVEMENT: sample weights for PER extension
        self._weights = np.ones((capacity,),          dtype=np.float32)

    # ------------------------------------------------------------------
    def push(
        self,
        obs:    np.ndarray,    # (17, 9, 9) float32  in [0, 1]
        policy: np.ndarray,    # (140,)     float32
        value:  float,
        weight: float = 1.0,
    ) -> None:
        """Insert a single transition (thread-safe)."""
        with self._lock:
            # Quantise obs to uint8 to save memory
            self._obs   [self._ptr] = (obs * 255).clip(0, 255).astype(np.uint8)
            self._policy[self._ptr] = policy.astype(np.float16)
            self._value [self._ptr] = np.float16(value)
            self._weights[self._ptr] = weight

            self._ptr  = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def push_batch(
        self,
        obs:    np.ndarray,    # (N, 17, 9, 9)
        policy: np.ndarray,    # (N, 140)
        value:  np.ndarray,    # (N,)
    ) -> None:
        for i in range(len(obs)):
            self.push(obs[i], policy[i], float(value[i]))

    # ------------------------------------------------------------------
    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (obs, policy, value) float32 tensors of shape
        (B, 17, 9, 9), (B, 140), (B,).
        """
        with self._lock:
            assert self._size >= batch_size, "Buffer too small to sample"
            # Weighted sampling (uniform by default)
            w = self._weights[:self._size]
            probs = w / w.sum()
            idx = np.random.choice(self._size, size=batch_size, replace=False, p=probs)
            obs    = self._obs   [idx].astype(np.float32) / 255.0
            policy = self._policy[idx].astype(np.float32)
            value  = self._value [idx].astype(np.float32)
        return obs, policy, value

    # ------------------------------------------------------------------
    def update_weights(self, indices: np.ndarray, weights: np.ndarray) -> None:
        """IMPROVEMENT: PER weight update hook."""
        with self._lock:
            self._weights[indices] = weights

    @property
    def size(self) -> int:
        return self._size

    def is_ready(self, min_size: int) -> bool:
        return self._size >= min_size
