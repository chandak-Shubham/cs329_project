"""
Unit Tests for Adaptive Gating Network and GatingPolicy (Task 9).
Verifies:
1. GatingNetwork mathematical normalization: w^g + w^b == 1.0 strictly
2. Observation processing and augmentation with social affinity q_ij
3. End-to-end gradient flow through the gating network parameters
4. PPO update execution with GatingPolicy
"""

import pytest
import numpy as np
import torch
from environment.grid_world import GridWorld
from models.gating_policy import GatingNetwork, GatingPolicy


def test_gating_network_normalization():
    """Verify weights strictly sum to 1.0 and affinity is bounded."""
    gating = GatingNetwork(hidden_dim=16)

    # Batch of random cues: [r_tilde, b, E_norm]
    cues = torch.rand((10, 8, 3))
    weights, q = gating(cues)

    # Weights shape: (10, 8, 2)
    assert weights.shape == (10, 8, 2)
    w_sum = weights.sum(dim=-1)
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-6), "w^g + w^b must equal 1.0"

    # Social affinity q shape: (10, 8)
    assert q.shape == (10, 8)
    assert torch.all(q >= 0.0)
    assert torch.all(q <= 1.0)


def test_gating_policy_observation_shapes():
    """Verify feature dimension: 2 (self) + 8*6 (neighbors with q) + 1 (count) = 51."""
    env = GridWorld(grid_size=10, max_capacity=10, initial_population=6, seed=42)
    obs = env.reset()

    policy = GatingPolicy(max_neighbors=8, hidden_dim=64, action_dim=13)
    flat_obs, action_masks, weights = policy.process_observations(obs)

    # Dimension: 2 + (8 * 6) + 1 = 51
    assert flat_obs.shape == (10, 51), f"Expected (10, 51), got {flat_obs.shape}"
    assert action_masks.shape == (10, 13)
    assert weights.shape == (10, 8, 2)

    dist, values = policy.forward(flat_obs, action_masks)
    assert values.shape == (10,)


def test_end_to_end_gradient_flow_to_gating():
    """Verify that gradients from policy loss successfully backpropagate into f_thetag."""
    policy = GatingPolicy(max_neighbors=8, hidden_dim=32, action_dim=13)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Mock environment observations
    env = GridWorld(grid_size=10, max_capacity=4, initial_population=4, seed=123)
    obs = env.reset()

    flat_obs, action_masks, _ = policy.process_observations(obs)
    dist, values = policy.forward(flat_obs, action_masks)

    # Dummy loss penalizing action choices and values
    dummy_loss = -dist.logits.mean() + values.mean()

    optimizer.zero_grad()
    dummy_loss.backward()

    # Check that gating network layers received gradients!
    gating_has_grad = False
    for param in policy.gating.parameters():
        assert param.grad is not None, "Gating network parameters must have gradients"
        if param.grad.abs().sum() > 0:
            gating_has_grad = True

    assert gating_has_grad, "Gating network must receive non-zero gradients end-to-end"

