import pytest
import torch

from network.policy_value_net import PolicyValueNet
from replay.buffer import ReplayBuffer
from trainer.train_loop import TrainConfig, TrainLoop


def _loop(l2_reg: float = 1e-4) -> TrainLoop:
    model = PolicyValueNet(channels=8, num_blocks=1)
    buffer = ReplayBuffer(capacity=1)
    cfg = TrainConfig(device="cpu", batch_size=1, l2_reg=l2_reg)
    return TrainLoop(model, buffer, cfg)


def test_optimizer_weight_decay_defaults_to_zero():
    loop = _loop()

    assert {group["weight_decay"] for group in loop.optimizer.param_groups} == {0.0}


def test_l2_penalty_excludes_bias_and_norm_parameters():
    loop = _loop(l2_reg=0.25)

    with torch.no_grad():
        for param in loop.model.parameters():
            if param.ndim > 1:
                param.zero_()
            else:
                param.fill_(1.0)

    assert loop._l2_penalty().item() == pytest.approx(0.0)

    with torch.no_grad():
        kernel = next(param for param in loop.model.parameters() if param.ndim > 1)
        kernel.fill_(2.0)

    expected = 0.25 * kernel.numel() * 4.0
    assert loop._l2_penalty().item() == pytest.approx(expected)
