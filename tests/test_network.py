"""
Network tests: shapes, determinism, masking.
Run with: pytest tests/test_network.py -v
"""
import pytest
import numpy as np
import torch

from network.policy_value_net import PolicyValueNet, build_net
from network.residual_block import ResidualBlock, SEBlock


class TestSEBlock:
    def test_output_shape(self):
        se = SEBlock(channels=64)
        x = torch.randn(4, 64, 9, 9)
        out = se(x)
        assert out.shape == x.shape

    def test_gate_range(self):
        """SE gate should produce values via sigmoid, so output ∈ (0, max_input)."""
        se = SEBlock(channels=16)
        x = torch.ones(2, 16, 3, 3)
        out = se(x)
        assert out.min() >= 0  # sigmoid * positive input


class TestResidualBlock:
    def test_output_shape(self):
        blk = ResidualBlock(channels=64)
        x = torch.randn(2, 64, 9, 9)
        out = blk(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from identity (SE weights aren't all 1)."""
        blk = ResidualBlock(channels=32)
        x = torch.randn(1, 32, 9, 9)
        out = blk(x)
        assert not torch.allclose(out, x)


class TestPolicyValueNet:
    @pytest.fixture
    def net(self):
        return PolicyValueNet(channels=64, num_blocks=2)

    def test_policy_shape(self, net):
        x = torch.randn(4, 17, 9, 9)
        logits, value = net(x)
        assert logits.shape == (4, 140)

    def test_value_shape(self, net):
        x = torch.randn(4, 17, 9, 9)
        _, value = net(x)
        assert value.shape == (4, 1)

    def test_value_in_range(self, net):
        x = torch.randn(8, 17, 9, 9)
        _, value = net(x)
        assert value.min() >= -1 - 1e-4
        assert value.max() <=  1 + 1e-4

    def test_masked_policy_sums_to_one(self, net):
        x = torch.randn(2, 17, 9, 9)
        mask = torch.zeros(2, 140, dtype=torch.bool)
        mask[:, :12] = True   # only pawn moves legal
        policy, _ = net.predict(x, mask)
        sums = policy.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-4)

    def test_deterministic_forward(self, net):
        net.eval()
        x = torch.randn(1, 17, 9, 9)
        with torch.no_grad():
            l1, v1 = net(x)
            l2, v2 = net(x)
        assert torch.allclose(l1, l2)
        assert torch.allclose(v1, v2)

    def test_illegal_actions_masked(self, net):
        net.eval()
        x = torch.randn(1, 17, 9, 9)
        mask = torch.zeros(1, 140, dtype=torch.bool)
        mask[0, 0] = True   # only action 0 is legal
        with torch.no_grad():
            policy, _ = net.predict(x, mask)
        # All illegal actions should be 0 (or near-zero)
        assert abs(policy[0, 0].item() - 1.0) < 1e-4
        assert policy[0, 1:].sum().item() < 1e-4
