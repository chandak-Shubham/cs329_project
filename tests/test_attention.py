"""
Comprehensive Unit Tests for Gated Multi-Head Spatial Attention Policy (Task 10).
Strictly validates:
1. No-neighbor case (0 neighbors): handles safely with zero spatial context Z_i and no NaNs
2. One-neighbor case (1 neighbor): attends cleanly to the single neighbor
3. Different neighbor counts in the same batch: simultaneously processes agents with 0, 2, 5, 8 neighbors
4. Invariance to padding: arbitrary values in padded neighbor slots do NOT affect outputs or active logits
5. End-to-end gradient flow through attention, gating, encoders, and heads
"""

import pytest
import numpy as np
import torch
from environment.grid_world import GridWorld, ACTION_GIVE_BASE
from networks.attention_policy import GatedAttentionPolicy


def test_no_neighbour_case_zero_neighbors():
    """Verify agent with 0 neighbors produces finite values and 0 donation probability."""
    policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
    policy.eval()

    N = 3
    self_state = torch.rand((N, 2))
    n_feats = torch.zeros((N, 8, 5))
    n_mask = torch.zeros((N, 8), dtype=torch.bool)  # Exactly 0 neighbors
    act_masks = torch.zeros((N, 13), dtype=torch.bool)
    act_masks[:, :5] = True  # Movement and stay are valid

    dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)

    # Check finite numbers
    assert not torch.isnan(values).any(), "Values must not be NaN for 0 neighbors"
    assert not torch.isnan(dist.probs).any(), "Action distribution must not be NaN"

    # All donation actions (5..12) must have exactly 0 probability
    assert torch.all(dist.probs[:, ACTION_GIVE_BASE:] == 0.0)
    # Movement actions (0..4) must sum to 1.0
    move_probs_sum = dist.probs[:, :5].sum(dim=-1)
    assert torch.allclose(move_probs_sum, torch.ones(N), atol=1e-6)


def test_one_neighbour_case():
    """Verify agent with exactly 1 neighbor attends cleanly and can donate to slot 0."""
    policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
    policy.eval()

    N = 2
    self_state = torch.tensor([[0.5, 0.1], [0.6, 0.2]], dtype=torch.float32)
    n_feats = torch.randn((N, 8, 5))
    n_mask = torch.zeros((N, 8), dtype=torch.bool)
    n_mask[:, 0] = True  # Only neighbor slot 0 is valid!

    act_masks = torch.zeros((N, 13), dtype=torch.bool)
    act_masks[:, :5] = True
    act_masks[:, ACTION_GIVE_BASE + 0] = True  # Donation to neighbor 0 allowed

    dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)

    assert not torch.isnan(values).any()
    assert not torch.isnan(dist.probs).any()
    # Slot 0 donation should be non-zero
    assert torch.all(dist.probs[:, ACTION_GIVE_BASE + 0] > 0.0)
    # Slots 1..7 donation must be strictly 0.0
    assert torch.all(dist.probs[:, ACTION_GIVE_BASE + 1:] == 0.0)


def test_different_neighbour_counts_in_same_batch():
    """Verify a single batch with agents having 0, 2, 5, and 8 neighbors simultaneously."""
    policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
    policy.eval()

    # 4 agents in the batch:
    # Agent 0: 0 neighbors
    # Agent 1: 2 neighbors
    # Agent 2: 5 neighbors
    # Agent 3: 8 neighbors (full)
    N = 4
    self_state = torch.rand((N, 2))
    n_feats = torch.randn((N, 8, 5))
    n_mask = torch.zeros((N, 8), dtype=torch.bool)

    n_mask[0, :] = False          # 0 neighbors
    n_mask[1, :2] = True          # 2 neighbors
    n_mask[2, :5] = True          # 5 neighbors
    n_mask[3, :8] = True          # 8 neighbors

    act_masks = torch.zeros((N, 13), dtype=torch.bool)
    act_masks[:, :5] = True
    for i, count in enumerate([0, 2, 5, 8]):
        for k in range(count):
            act_masks[i, ACTION_GIVE_BASE + k] = True

    dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)

    assert not torch.isnan(values).any()
    assert not torch.isnan(dist.probs).any()

    # Agent 0: 0 donations allowed
    assert torch.all(dist.probs[0, ACTION_GIVE_BASE:] == 0.0)
    # Agent 1: exactly 2 donation slots allowed
    assert torch.all(dist.probs[1, ACTION_GIVE_BASE:ACTION_GIVE_BASE + 2] > 0.0)
    assert torch.all(dist.probs[1, ACTION_GIVE_BASE + 2:] == 0.0)
    # Agent 2: exactly 5 donation slots allowed
    assert torch.all(dist.probs[2, ACTION_GIVE_BASE:ACTION_GIVE_BASE + 5] > 0.0)
    assert torch.all(dist.probs[2, ACTION_GIVE_BASE + 5:] == 0.0)
    # Agent 3: all 8 donation slots allowed
    assert torch.all(dist.probs[3, ACTION_GIVE_BASE:] > 0.0)


def test_padded_neighbours_not_affecting_output():
    """
    CRITICAL INVARIANCE TEST:
    Compare Agent A and Agent B with identical valid neighbors (slots 0, 1),
    but Agent B has arbitrary garbage / random noise in its padded slots (slots 2..7).
    Asserts:
    - Movement logits A == Movement logits B
    - Value estimate A == Value estimate B
    - Valid donation logits A == Valid donation logits B
    """
    policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
    policy.eval()

    self_state = torch.tensor([[0.7, 0.3]], dtype=torch.float32)

    # Identical valid neighbors for slot 0 and slot 1
    valid_neighbor_0 = [0.2, -0.3, 0.5, 0.8, 0.6]
    valid_neighbor_1 = [-0.1, 0.4, 0.4, 0.2, 0.9]

    # Setup Agent A: padded slots filled with zeros
    feats_A = torch.zeros((1, 8, 5), dtype=torch.float32)
    feats_A[0, 0] = torch.tensor(valid_neighbor_0)
    feats_A[0, 1] = torch.tensor(valid_neighbor_1)

    # Setup Agent B: identical slots 0 and 1, BUT padded slots 2..7 filled with extreme garbage noise
    feats_B = torch.zeros((1, 8, 5), dtype=torch.float32)
    feats_B[0, 0] = torch.tensor(valid_neighbor_0)
    feats_B[0, 1] = torch.tensor(valid_neighbor_1)
    feats_B[0, 2:] = torch.randn((6, 5)) * 100.0 + 50.0  # Massive random numbers

    mask = torch.zeros((1, 8), dtype=torch.bool)
    mask[0, :2] = True  # Only slots 0 and 1 are valid

    act_masks = torch.zeros((1, 13), dtype=torch.bool)
    act_masks[0, :5] = True
    act_masks[0, ACTION_GIVE_BASE : ACTION_GIVE_BASE + 2] = True

    with torch.no_grad():
        dist_A, val_A, _ = policy.forward(self_state, feats_A, mask, act_masks)
        dist_B, val_B, _ = policy.forward(self_state, feats_B, mask, act_masks)

    # Value estimate must be strictly identical
    assert torch.isclose(val_A, val_B, atol=1e-5), f"Values differ! A={val_A}, B={val_B}"

    # Movement probabilities (actions 0..4) must be strictly identical
    assert torch.allclose(dist_A.probs[:, :5], dist_B.probs[:, :5], atol=1e-5), "Movement probs differ!"

    # Valid donation probabilities (slots 0 and 1) must be strictly identical
    assert torch.allclose(
        dist_A.probs[:, ACTION_GIVE_BASE : ACTION_GIVE_BASE + 2],
        dist_B.probs[:, ACTION_GIVE_BASE : ACTION_GIVE_BASE + 2],
        atol=1e-5,
    ), "Valid donation probs differ!"


def test_gradient_flow_through_attention():
    """Verify gradients propagate cleanly from loss through MHA, gating, and encoders."""
    policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    env = GridWorld(grid_size=10, max_capacity=5, initial_population=5, seed=123)
    obs = env.reset()

    self_state, n_feats, n_mask, act_masks = policy.extract_tensors(obs)
    dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)

    # Combined loss
    loss = -dist.logits.mean() + values.mean()**2

    optimizer.zero_grad()
    loss.backward()

    # Check non-zero gradients in attention and gating
    assert policy.mha.in_proj_weight.grad is not None
    assert policy.mha.in_proj_weight.grad.abs().sum() > 0, "MHA in_proj_weight must receive gradients"

    assert policy.mha.out_proj.weight.grad is not None
    assert policy.mha.out_proj.weight.grad.abs().sum() > 0, "MHA out_proj must receive gradients"

    for param in policy.gating.parameters():
        assert param.grad is not None
        assert param.grad.abs().sum() > 0, "Gating network must receive gradients"
