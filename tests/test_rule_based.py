"""
Unit Tests for Rule-Based Baseline Policies (Task 5).
Verifies:
1. AlwaysDefect never executes donation actions
2. KinOnly correctly targets highest r_tilde above threshold
3. Reciprocal correctly targets highest b_j above threshold
4. FixedWeight correctly targets weighted sum q_ij
5. Complete step execution in GridWorld across all policies
"""

import pytest
import numpy as np
from environment.grid_world import GridWorld, ACTION_GIVE_BASE
from networks.rule_based import (
    AlwaysDefectPolicy,
    KinOnlyPolicy,
    ReciprocalPolicy,
    FixedWeightPolicy,
    RandomCooperatePolicy,
)


def test_always_defect_never_donates():
    """Verify AlwaysDefect never produces actions >= ACTION_GIVE_BASE."""
    env = GridWorld(grid_size=10, num_agents=8, seed=42)
    obs = env.reset()
    policy = AlwaysDefectPolicy(allow_movement=True, rng=np.random.default_rng(42))

    for _ in range(5):
        actions = policy.act(obs)
        assert np.all(actions < ACTION_GIVE_BASE), "AlwaysDefect must never donate"
        obs, rewards, dones, info = env.step(actions)
        assert info["donations_count"] == 0


def test_kin_only_targeting():
    """Verify KinOnly selects neighbor with highest genetic relatedness."""
    policy = KinOnlyPolicy(threshold=0.2)
    
    # Mock observation: Agent 0 has 2 neighbors
    # Neighbor 0: r_tilde = 0.1 (below threshold)
    # Neighbor 1: r_tilde = 0.8 (above threshold)
    mock_obs = {
        "active_mask": np.array([True]),
        "neighbor_mask": np.array([[True, True]]),
        "neighbor_features": np.zeros((1, 2, 5), dtype=np.float32),
    }
    mock_obs["neighbor_features"][0, 0, 3] = 0.1
    mock_obs["neighbor_features"][0, 1, 3] = 0.8

    actions = policy.act(mock_obs)
    # Should donate to neighbor 1: ACTION_GIVE_BASE + 1 = 6
    assert actions[0] == ACTION_GIVE_BASE + 1


def test_reciprocal_targeting():
    """Verify ReciprocalPolicy selects neighbor with highest behavioral history."""
    policy = ReciprocalPolicy(threshold=0.5)

    # Mock observation:
    # Neighbor 0: b_score = 0.2 (defector)
    # Neighbor 1: b_score = 0.9 (cooperator)
    mock_obs = {
        "active_mask": np.array([True]),
        "neighbor_mask": np.array([[True, True]]),
        "neighbor_features": np.zeros((1, 2, 5), dtype=np.float32),
    }
    mock_obs["neighbor_features"][0, 0, 4] = 0.2
    mock_obs["neighbor_features"][0, 1, 4] = 0.9

    actions = policy.act(mock_obs)
    # Should donate to neighbor 1: ACTION_GIVE_BASE + 1 = 6
    assert actions[0] == ACTION_GIVE_BASE + 1


def test_fixed_weight_combination():
    """Verify FixedWeightPolicy computes 0.5*r + 0.5*b and picks highest."""
    policy = FixedWeightPolicy(w_g=0.5, w_b=0.5, threshold=0.4)

    # Neighbor 0: r=0.9, b=0.1 -> q = 0.5(0.9) + 0.5(0.1) = 0.50
    # Neighbor 1: r=0.2, b=0.9 -> q = 0.5(0.2) + 0.5(0.9) = 0.55 (Winner!)
    mock_obs = {
        "active_mask": np.array([True]),
        "neighbor_mask": np.array([[True, True]]),
        "neighbor_features": np.zeros((1, 2, 5), dtype=np.float32),
    }
    mock_obs["neighbor_features"][0, 0, 3] = 0.9
    mock_obs["neighbor_features"][0, 0, 4] = 0.1
    mock_obs["neighbor_features"][0, 1, 3] = 0.2
    mock_obs["neighbor_features"][0, 1, 4] = 0.9

    actions = policy.act(mock_obs)
    assert actions[0] == ACTION_GIVE_BASE + 1


def test_all_policies_step_cleanly_in_gridworld():
    """Verify each policy can step through a full GridWorld episode without errors."""
    policies = [
        AlwaysDefectPolicy(),
        KinOnlyPolicy(),
        ReciprocalPolicy(),
        FixedWeightPolicy(),
        RandomCooperatePolicy(p_cooperate=0.4),
    ]

    for p in policies:
        env = GridWorld(grid_size=12, num_agents=6, seed=123)
        obs = env.reset()
        for _ in range(10):
            actions = p.act(obs)
            obs, rewards, dones, info = env.step(actions)
            if np.all(dones):
                break

