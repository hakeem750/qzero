"""
PolicyValueNet — full AlphaZero-style network for Quoridor.

Improvements over blueprint:
  1. Squeeze-and-Excitation residual blocks (see residual_block.py).
  2. Weight initialisation: Kaiming normal for convolutions,
     constant 0 for biases — improves early training stability.
  3. Policy head uses 4 filter channels instead of 2 — gives
     the policy head more representational capacity for the large
     action space (140 actions over a 9×9 board).
  4. torch.compile() wrapper applied externally for flexibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_block import ResidualBlock

NUM_INPUT_CHANNELS = 13
BOARD_H = BOARD_W = 9
NUM_ACTIONS = 140
BACKBONE_CHANNELS = 256
NUM_RES_BLOCKS = 20


class PolicyValueNet(nn.Module):

    def __init__(
        self,
        in_channels: int = NUM_INPUT_CHANNELS,
        channels: int = BACKBONE_CHANNELS,
        num_blocks: int = NUM_RES_BLOCKS,
        num_actions: int = NUM_ACTIONS,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()

        # --- Stem -----------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # --- Residual tower -------------------------------------------------
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, se_reduction) for _ in range(num_blocks)]
        )

        # --- Policy head (IMPROVEMENT: 4 filter channels instead of 2) -----
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * BOARD_H * BOARD_W, num_actions),
        )

        # --- Value head -----------------------------------------------------
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(BOARD_H * BOARD_W, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self._init_weights()

    # -----------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming normal init for all conv layers; zero-bias everywhere.
        Special handling for policy head final layer to ensure uniform
        initial policy distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Special: policy head output layer should have small weights
        # to ensure uniform initial distribution (logits ≈ 0)
        if isinstance(self.policy_head[-1], nn.Linear):
            nn.init.uniform_(self.policy_head[-1].weight, -0.01, 0.01)

    # -----------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 13, 9, 9) float tensor

        Returns:
            policy_logits: (B, 140)  — raw logits, not softmaxed
            value:         (B, 1)    — in [-1, 1]
        """
        x = self.stem(x)
        x = self.blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    # -----------------------------------------------------------------------
    def predict(
        self, obs: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masked softmax inference for MCTS leaf evaluation.

        Args:
            obs:        (B, 13, 9, 9)
            legal_mask: (B, 140) bool — True for legal actions

        Returns:
            policy: (B, 140) masked & normalised probabilities
            value:  (B, 1)
        """
        logits, value = self.forward(obs)
        # Mask illegal actions with large negative value
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        policy = torch.softmax(logits, dim=-1)
        
        # Handle NaN from softmax of all -inf (shouldn't happen in valid games)
        # Replace with uniform distribution over legal actions
        nan_mask = torch.isnan(policy)
        if nan_mask.any():
            batch_size = policy.shape[0]
            for b in range(batch_size):
                if nan_mask[b].any():
                    num_legal = legal_mask[b].sum().float()
                    if num_legal > 0:
                        policy[b] = 0.0
                        policy[b, legal_mask[b]] = 1.0 / num_legal
                    else:
                        # Fallback: uniform over all actions (should never happen)
                        policy[b] = 1.0 / policy.shape[1]
        
        return policy, value


def build_net(
    compile_model: bool = True,
    device: str | torch.device = "cpu",
    **kwargs,
) -> PolicyValueNet:
    """Factory with optional torch.compile()."""
    net = PolicyValueNet(**kwargs).to(device)
    if compile_model and torch.cuda.is_available():
        net = torch.compile(net)
    return net
