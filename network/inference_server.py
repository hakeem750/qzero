"""
InferenceServer — batched GPU inference for MCTS workers.

IMPROVEMENT: Uses BF16 (bfloat16) instead of FP16.
  BF16 has the same dynamic range as FP32 but reduced precision,
  virtually eliminating the overflow/NaN issues that plague FP16
  during training and inference on deep residual networks.
  Supported natively on A100/H100 and via emulation on RTX 30/40xx.

Design:
  Workers push (obs, legal_mask) requests to a queue.
  The server drains the queue into a batch, runs GPU inference,
  and delivers results back via per-request Futures.

  This single-GPU-process model prevents CUDA context contention
  when multiple CPU workers spawn CUDA operations concurrently.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from concurrent.futures import Future
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .policy_value_net import PolicyValueNet


@dataclass
class _InferenceRequest:
    obs: np.ndarray          # (17, 9, 9) float32
    legal_mask: np.ndarray   # (140,) bool
    future: Future = field(default_factory=Future)


class InferenceServer:
    """
    Thread-safe batched inference server.

    Usage:
        server = InferenceServer(model, device="cuda", batch_size=64)
        server.start()

        future = server.submit(obs, legal_mask)
        policy, value = future.result()   # blocks until inference done

        server.stop()
    """

    def __init__(
        self,
        model: PolicyValueNet,
        device: str | torch.device = "cuda",
        batch_size: int = 64,
        timeout: float = 0.005,   # seconds to wait before flushing partial batch
        dtype: torch.dtype = torch.bfloat16,   # IMPROVEMENT: BF16
    ) -> None:
        self.model = model.eval().to(device)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.timeout = timeout
        self.dtype = dtype

        self._queue: queue.Queue[_InferenceRequest | None] = queue.Queue()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._running = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)  # sentinel
        self._thread.join()

    # ------------------------------------------------------------------
    def submit(self, obs: np.ndarray, legal_mask: np.ndarray) -> Future:
        """Non-blocking: enqueue an inference request and return a Future."""
        req = _InferenceRequest(obs=obs, legal_mask=legal_mask)
        self._queue.put(req)
        return req.future

    # ------------------------------------------------------------------
    def _serve(self) -> None:
        """Main inference loop — drains queue into batches."""
        while self._running:
            batch: List[_InferenceRequest] = []

            # Block for first request
            try:
                first = self._queue.get(timeout=0.1)
                if first is None:
                    break
                batch.append(first)
            except queue.Empty:
                continue

            # Drain up to batch_size with a short timeout
            deadline = torch.cuda.Event(enable_timing=False) if self.device.type == "cuda" else None
            while len(batch) < self.batch_size:
                try:
                    req = self._queue.get_nowait()
                    if req is None:
                        self._running = False
                        break
                    batch.append(req)
                except queue.Empty:
                    break

            self._run_batch(batch)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_batch(self, batch: List[_InferenceRequest]) -> None:
        obs_np   = np.stack([r.obs        for r in batch], axis=0)
        mask_np  = np.stack([r.legal_mask for r in batch], axis=0)

        obs_t  = torch.from_numpy(obs_np ).to(self.device, dtype=self.dtype)
        mask_t = torch.from_numpy(mask_np).to(self.device, dtype=torch.bool)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            policy, value = self.model.predict(obs_t.float(), mask_t)

        policy_np = policy.cpu().float().numpy()
        value_np  = value.cpu().float().numpy()

        for i, req in enumerate(batch):
            req.future.set_result((policy_np[i], value_np[i, 0]))
