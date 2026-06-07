"""
Training loop.

Improvements over blueprint:
  1. LR warmup — ramps from 0 to peak over the first N steps.
     Prevents gradient explosion early in training when the network
     is far from its optimal parameters.
  2. BF16 mixed precision — more stable than FP16, same throughput.
  3. Gradient clipping + skip on NaN/Inf loss.
  4. Detailed metrics logging (policy entropy, value MSE).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.policy_value_net import PolicyValueNet
from replay.buffer import ReplayBuffer


@dataclass
class TrainConfig:
    batch_size:      int   = 1024
    learning_rate:   float = 3e-4
    momentum:        float = 0.9
    weight_decay:    float = 1e-4
    warmup_steps:    int   = 1_000     # IMPROVEMENT: LR warmup
    cosine_steps:    int   = 100_000
    grad_clip:       float = 1.0
    l2_reg:          float = 1e-4
    device:          str   = "cuda"
    dtype:           str   = "bfloat16"   # IMPROVEMENT: BF16
    log_every:       int   = 100


class TrainLoop:
    def __init__(
        self,
        model:   PolicyValueNet,
        buffer:  ReplayBuffer,
        config:  TrainConfig = TrainConfig(),
    ) -> None:
        self.model   = model
        self.buffer  = buffer
        self.cfg     = config
        self.device  = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype   = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        # IMPROVEMENT: warmup + cosine schedule
        def lr_lambda(step: int) -> float:
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(1, config.cosine_steps - config.warmup_steps)
            return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler    = torch.amp.GradScaler('cuda', enabled=(self.dtype == torch.float16))

        self.step = 0
        self.metrics_history: List[Dict] = []

    # ------------------------------------------------------------------
    def train_step(self) -> Dict[str, float]:
        obs_np, policy_np, value_np = self.buffer.sample(self.cfg.batch_size)

        obs    = torch.from_numpy(obs_np   ).to(self.device)
        policy = torch.from_numpy(policy_np).to(self.device)
        value  = torch.from_numpy(value_np ).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            logits, v_pred = self.model(obs)

            # Policy loss: cross-entropy against MCTS visit counts
            policy_loss = F.cross_entropy(logits, policy)

            # Value loss: MSE
            value_loss = F.mse_loss(v_pred.squeeze(-1), value)
            loss_no_l2 = policy_loss + value_loss

            # L2 regularisation (weight decay in AdamW also does this,
            # but explicit reg makes the ablation clear)
            l2 = self.cfg.l2_reg * sum(
                p.pow(2).sum() for p in self.model.parameters() if p.requires_grad
            )

            loss = loss_no_l2 + l2

        if not torch.isfinite(loss):
            return {"skipped": 1.0}

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.step += 1

        # Policy entropy (higher = more exploration in network prior)
        with torch.no_grad():
            probs   = torch.softmax(logits, dim=-1).clamp(min=1e-8)
            entropy = (-probs * probs.log()).sum(dim=-1).mean().item()
            target_policy = policy.clamp(min=1e-8)
            target_entropy = (-target_policy * target_policy.log()).sum(dim=-1).mean().item()
            target_top_prob = policy.max(dim=-1).values.mean().item()

        metrics = {
            "loss":         loss.item(),
            "loss_no_l2":   loss_no_l2.item(),
            "l2_loss":      l2.item(),
            "policy_loss":  policy_loss.item(),
            "value_loss":   value_loss.item(),
            "entropy":      entropy,
            "target_entropy": target_entropy,
            "target_top_prob": target_top_prob,
            "target_value_mean": value.mean().item(),
            "target_value_std": value.std(unbiased=False).item(),
            "lr":           self.scheduler.get_last_lr()[0],
            "step":         self.step,
        }
        self.metrics_history.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    def run(self, total_steps: int, min_buffer_size: int = 10_000) -> None:
        """Block until min_buffer_size is ready, then train."""
        print(f"Waiting for {min_buffer_size} samples in buffer …")
        while not self.buffer.is_ready(min_buffer_size):
            time.sleep(1.0)

        print(f"Training for {total_steps} steps …")
        t0 = time.time()
        for _ in range(total_steps):
            metrics = self.train_step()
            if self.step % self.cfg.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"[{self.step:>7}] "
                    f"loss={metrics.get('loss', 0):.4f}  "
                    f"core={metrics.get('loss_no_l2', 0):.4f}  "
                    f"p={metrics.get('policy_loss', 0):.4f}  "
                    f"v={metrics.get('value_loss', 0):.4f}  "
                    f"H={metrics.get('entropy', 0):.3f}  "
                    f"lr={metrics.get('lr', 0):.2e}  "
                    f"t={elapsed:.1f}s"
                )
