import numpy as np
import torch

import scripts.train as train_script
from network.policy_value_net import PolicyValueNet
from replay.buffer import ReplayBuffer
from trainer.train_loop import TrainConfig, TrainLoop


def _filled_buffer(size: int = 64) -> ReplayBuffer:
    rng = np.random.default_rng(123)
    buffer = ReplayBuffer(capacity=size)
    policy = np.ones(140, dtype=np.float32) / 140
    for _ in range(size):
        obs = rng.random((20, 9, 9), dtype=np.float32)
        value = float(rng.choice([-1.0, 0.0, 1.0]))
        buffer.push(obs, policy, value)
    return buffer


def _small_loop(model: PolicyValueNet, buffer: ReplayBuffer) -> TrainLoop:
    cfg = TrainConfig(
        device="cpu",
        batch_size=8,
        dtype="bfloat16",
        warmup_steps=10,
        cosine_steps=100,
    )
    return TrainLoop(model, buffer, cfg)


def test_checkpoint_round_trip_restores_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(train_script, "CKPT_DIR", tmp_path)
    buffer = _filled_buffer()

    model = PolicyValueNet(channels=8, num_blocks=1)
    loop = _small_loop(model, buffer)
    loop.train_step()

    path = train_script.save_checkpoint(model, loop.step, loop)
    saved_params = {name: param.detach().clone() for name, param in model.state_dict().items()}
    saved_scheduler_epoch = loop.scheduler.last_epoch
    saved_lr = loop.scheduler.get_last_lr()[0]

    restored_model = PolicyValueNet(channels=8, num_blocks=1)
    restored_loop = _small_loop(restored_model, buffer)
    step = train_script._load_checkpoint_file(path, restored_model, restored_loop)

    assert step == loop.step
    assert restored_loop.step == loop.step
    assert restored_loop.scheduler.last_epoch == saved_scheduler_epoch
    assert restored_loop.scheduler.get_last_lr()[0] == saved_lr
    assert restored_loop.optimizer.state_dict()["state"]

    for name, param in restored_model.state_dict().items():
        assert torch.equal(param, saved_params[name])
