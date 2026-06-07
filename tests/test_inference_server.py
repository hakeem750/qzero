import torch

from network.inference_server import InferenceServer
from network.policy_value_net import PolicyValueNet


def test_inference_server_keeps_private_eval_model():
    model = PolicyValueNet(channels=8, num_blocks=1)
    server = InferenceServer(model, device="cpu", batch_size=1, dtype=torch.bfloat16)

    assert server.model is not model
    assert not server.model.training

    model.train()
    assert model.training
    assert not server.model.training

    server.update_model(model)
    assert not server.model.training
