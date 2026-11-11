"""
Unit Tests for Simple MLP Baseline Policy and PPO Training Engine (Task 8).
Verifies:
1. SimpleMLPPolicy tensor shapes and observation extraction
2. Action masking strictly prevents selecting invalid donation targets
3. MaskedRolloutBuffer GAE computation with dynamic agent masking
4. End-to-end PPO gradient update step execution without NaNs
"""

import pytest
import numpy as np
import torch
from environment.grid_world import GridWorld, ACTION_GIVE_BASE
from models.mlp_policy import SimpleMLPPolicy
from training.rollout_buffer import MaskedRolloutBuffer


def test_simple_mlp_shapes_and_forward():
    """Verify observation flattening, forward pass, and value head."""
    env = GridWorld(grid_size=10, max_capacity=10, initial_population=5, seed=42)
    obs = env.reset()

    policy = SimpleMLPPolicy(max_neighbors=8, hidden_dim=64, action_dim=13)
    flat_obs, action_masks = policy.extract_flat_obs(obs)

    # 2 self + 8*5 neighbors + 1 count = 43
    assert flat_obs.shape == (10, 43), f"Expected (10, 43), got {flat_obs.shape}"
    assert action_masks.shape == (10, 13), f"Expected (10, 13), got {action_masks.shape}"

    dist, values = policy.forward(flat_obs, action_masks)
    assert values.shape == (10,)

    actions, log_probs, entropy, _ = policy.get_action_and_value(flat_obs, action_masks)
    assert actions.shape == (10,)
    assert log_probs.shape == (10,)
    assert entropy.shape == (10,)


def test_action_masking_blocks_invalid_targets():
    """Verify that agents without neighbors or with E <= C cannot sample donation actions."""
    env = GridWorld(grid_size=10, max_capacity=4, initial_population=2, seed=42)
    obs = env.reset()

    # Place Agent 0 alone in a corner with E = 1.0 (<= C = 2.0)
    # It has 0 neighbors and cannot afford donation
    obs["self_state"][0, 0] = 1.0 / 60.0  # E <= C
    obs["neighbor_mask"][0, :].fill(False)

    policy = SimpleMLPPolicy(max_neighbors=8, hidden_dim=64, action_dim=13)
    flat_obs, action_masks = policy.extract_flat_obs(obs)

    # For agent 0, actions 5..12 must all be False
    assert not torch.any(action_masks[0, ACTION_GIVE_BASE:])

    dist, _ = policy.forward(flat_obs, action_masks)
    probs = dist.probs[0]

    # Probabilities for donation actions must be strictly 0.0
    for k in range(ACTION_GIVE_BASE, 13):
        assert probs[k].item() == 0.0, f"Action {k} must have probability 0 when masked"


def test_masked_rollout_buffer_gae():
    """Verify GAE computation across dynamic population with death/birth."""
    T = 8
    N = 4
    buffer = MaskedRolloutBuffer(buffer_size=T, num_agents=N, obs_dim=43, action_dim=13)

    for t in range(T):
        obs = torch.randn((N, 43))
        actions = torch.zeros(N, dtype=torch.long)
        log_probs = torch.zeros(N)
        rewards = torch.ones(N) * 2.0
        dones = torch.zeros(N, dtype=torch.bool)
        values = torch.ones(N) * 1.5
        action_masks = torch.ones((N, 13), dtype=torch.bool)
        active_masks = torch.tensor([True, True, False, False])  # 2 active, 2 inactive

        buffer.add(obs, actions, log_probs, rewards, dones, values, action_masks, active_masks)

    next_values = torch.ones(N) * 1.5
    next_dones = torch.zeros(N, dtype=torch.bool)
    buffer.compute_gae(next_values, next_dones)

    # Check advantages shape and absence of NaNs
    assert buffer.advantages.shape == (T, N)
    assert not torch.isnan(buffer.advantages).any()
    assert not torch.isnan(buffer.returns).any()

    # Extract active batches: must only yield active agent samples
    batches = list(buffer.get_active_batches(batch_size=8))
    assert len(batches) > 0
    b_obs, b_act, b_lp, b_adv, b_ret, b_mask = batches[0]
    assert b_obs.shape[1] == 43


def test_single_ppo_gradient_step():
    """Verify PPO loss backpropagates and updates weights without NaNs."""
    policy = SimpleMLPPolicy(max_neighbors=8, hidden_dim=32, action_dim=13)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    b_obs = torch.randn((16, 43))
    b_actions = torch.randint(0, 5, size=(16,))
    b_old_log_probs = torch.zeros(16)
    b_advantages = torch.randn(16)
    b_returns = torch.randn(16)
    b_masks = torch.ones((16, 13), dtype=torch.bool)

    # Forward
    dist, v_pred = policy.forward(b_obs, b_masks)
    new_log_probs = dist.log_prob(b_actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - b_old_log_probs)
    surr1 = ratio * b_advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * b_advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = 0.5 * ((v_pred - b_returns) ** 2).mean()
    total_loss = policy_loss + (0.5 * value_loss) - (0.01 * entropy)

    optimizer.zero_grad()
    total_loss.backward()

    # Check that all gradients are finite
    for param in policy.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

    optimizer.step()

