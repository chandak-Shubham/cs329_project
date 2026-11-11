"""
Unit Test Suite for Minimal GridWorld Environment (Task 3).
Verifies:
1. Toroidal coordinate wrapping and shortest displacement vectors
2. Movement collision handling and strict 1-agent-per-cell occupancy
3. Chebyshev neighbor detection within radius rho
4. Single donation energy transfer (-C to donor, +B to recipient)
5. Additive multi-donation reception (+k*B)
6. Precondition for donation (fails if E <= C)
7. Terminal starvation culling when E <= 0
"""

import pytest
import numpy as np
from environment.grid_world import (
    GridWorld,
    ACTION_STAY,
    ACTION_NORTH,
    ACTION_SOUTH,
    ACTION_EAST,
    ACTION_WEST,
    ACTION_GIVE_BASE,
)


def create_deterministic_env(grid_size=8, num_agents=2, initial_energy=30.0, **kwargs):
    """Helper to create a small deterministic environment for testing."""
    env = GridWorld(
        grid_size=grid_size,
        num_agents=num_agents,
        initial_energy=initial_energy,
        seed=123,
        **kwargs,
    )
    env.reset()
    return env


def test_toroidal_wrapping():
    """Test displacement vectors and wrapping across toroidal boundaries."""
    env = create_deterministic_env(grid_size=10, num_agents=2)
    
    # Distance from x=1 to x=9 on grid of size 10 should wrap:
    # 9 - 1 = 8 -> shortest delta is -2 (1 -> 0 -> 9)
    p1 = np.array([1, 5], dtype=np.int32)
    p2 = np.array([9, 5], dtype=np.int32)
    delta = env.toroidal_delta(p1, p2)
    assert delta[0] == -2, f"Expected dx=-2, got {delta[0]}"
    assert delta[1] == 0, f"Expected dy=0, got {delta[1]}"
    assert env.chebyshev_distance(p1, p2) == 2

    # Agent moving West from x=0 wraps to x=9
    env.positions[0] = np.array([0, 5], dtype=np.int32)
    env.positions[1] = np.array([5, 5], dtype=np.int32)
    env.grid.fill(-1)
    env.grid[0, 5] = 0
    env.grid[5, 5] = 1
    
    actions = np.array([ACTION_WEST, ACTION_STAY])
    env.step(actions)
    assert env.positions[0][0] == 9, f"Expected x=9 after West from 0, got {env.positions[0][0]}"
    assert env.grid[9, 5] == 0
    assert env.grid[0, 5] == -1


def test_collision_handling():
    """Two agents attempting to enter the same cell must never overlap."""
    env = create_deterministic_env(grid_size=8, num_agents=2)
    
    # Place agent 0 at (2, 3) and agent 1 at (4, 3)
    env.grid.fill(-1)
    env.positions[0] = np.array([2, 3], dtype=np.int32)
    env.positions[1] = np.array([4, 3], dtype=np.int32)
    env.grid[2, 3] = 0
    env.grid[4, 3] = 1

    # Agent 0 moves East (towards 3, 3), Agent 1 moves West (towards 3, 3)
    actions = np.array([ACTION_EAST, ACTION_WEST])
    env.step(actions)

    # Exactly one agent should occupy (3, 3); the other must remain at origin
    occ = env.grid[3, 3]
    assert occ in [0, 1], "One agent must have won the cell"
    other = 1 if occ == 0 else 0
    origin_cell = (2, 3) if other == 0 else (4, 3)
    assert tuple(env.positions[other]) == origin_cell, "Losing agent must stay at origin"
    assert env.grid[origin_cell[0], origin_cell[1]] == other
    # No duplicate cell occupancy
    assert not np.array_equal(env.positions[0], env.positions[1]), "Agents must not share a cell"


def test_neighbor_detection():
    """Verify Chebyshev neighborhood detection within radius rho."""
    env = create_deterministic_env(grid_size=16, num_agents=3, obs_radius=2)
    env.grid.fill(-1)
    
    # Agent 0 at (5, 5)
    env.positions[0] = np.array([5, 5], dtype=np.int32)
    env.grid[5, 5] = 0
    # Agent 1 at (7, 6) -> Chebyshev distance = max(|2|, |1|) = 2 (in radius)
    env.positions[1] = np.array([7, 6], dtype=np.int32)
    env.grid[7, 6] = 1
    # Agent 2 at (8, 5) -> Chebyshev distance = 3 (outside radius 2)
    env.positions[2] = np.array([8, 5], dtype=np.int32)
    env.grid[8, 5] = 2

    neighbors_0 = env.get_neighbors(0)
    assert 1 in neighbors_0, "Agent 1 must be detected within radius 2"
    assert 2 not in neighbors_0, "Agent 2 must NOT be detected (distance 3 > radius 2)"


def test_energy_conservation_and_transfer():
    """Single donation: donor loses C, recipient gains B, net delta = B - C."""
    C = 2.0
    B = 6.0
    c_base = 0.5
    c_density = 0.1
    env = create_deterministic_env(
        grid_size=8,
        num_agents=2,
        initial_energy=20.0,
        donation_cost=C,
        donation_benefit=B,
        c_base=c_base,
        c_density=c_density,
        obs_radius=3,
    )
    env.grid.fill(-1)
    # Place agents next to each other
    env.positions[0] = np.array([3, 3], dtype=np.int32)
    env.positions[1] = np.array([3, 4], dtype=np.int32)
    env.grid[3, 3] = 0
    env.grid[3, 4] = 1

    # Agent 0 donates to neighbor 0 (Agent 1)
    # Action 5 = ACTION_GIVE_BASE + 0
    actions = np.array([ACTION_GIVE_BASE, ACTION_STAY])
    obs, rewards, dones, info = env.step(actions)

    # Agent 0: -c_base - c_density*1 - C = -0.5 - 0.1 - 2.0 = -2.6
    # Agent 1: -c_base - c_density*1 + B = -0.5 - 0.1 + 6.0 = +5.4
    expected_e0 = 20.0 - 2.6
    expected_e1 = 20.0 + 5.4

    assert np.isclose(env.energy[0], expected_e0), f"Expected {expected_e0}, got {env.energy[0]}"
    assert np.isclose(env.energy[1], expected_e1), f"Expected {expected_e1}, got {env.energy[1]}"
    assert info["donations_count"] == 1


def test_multi_donation_additive():
    """Agents 0 and 1 both donate to Agent 2: Agent 2 receives 2*B."""
    C = 2.0
    B = 5.0
    env = create_deterministic_env(
        grid_size=8,
        num_agents=3,
        initial_energy=20.0,
        donation_cost=C,
        donation_benefit=B,
        c_base=0.0,
        c_density=0.0,
        obs_radius=3,
    )
    env.grid.fill(-1)
    # Target agent 2 at center (3, 3)
    env.positions[0] = np.array([3, 2], dtype=np.int32)
    env.positions[1] = np.array([3, 4], dtype=np.int32)
    env.positions[2] = np.array([3, 3], dtype=np.int32)
    env.grid[3, 2] = 0
    env.grid[3, 4] = 1
    env.grid[3, 3] = 2

    # For Agent 0: Agent 2 is neighbor at distance 1
    neighbors_0 = env.get_neighbors(0)
    target_idx_0 = neighbors_0.index(2)

    # For Agent 1: Agent 2 is neighbor at distance 1
    neighbors_1 = env.get_neighbors(1)
    target_idx_1 = neighbors_1.index(2)

    actions = np.array([
        ACTION_GIVE_BASE + target_idx_0,
        ACTION_GIVE_BASE + target_idx_1,
        ACTION_STAY
    ])
    env.step(actions)

    # Agent 0: 20 - C = 18.0
    # Agent 1: 20 - C = 18.0
    # Agent 2: 20 + 2*B = 30.0
    assert np.isclose(env.energy[0], 18.0)
    assert np.isclose(env.energy[1], 18.0)
    assert np.isclose(env.energy[2], 30.0)


def test_donation_precondition_insufficient_energy():
    """Agent with E <= C cannot donate; action reverts to Stay with no C deduction."""
    C = 2.0
    env = create_deterministic_env(
        grid_size=8,
        num_agents=2,
        initial_energy=1.5,  # Less than C = 2.0
        donation_cost=C,
        c_base=0.0,
        c_density=0.0,
    )
    env.grid.fill(-1)
    env.positions[0] = np.array([2, 2], dtype=np.int32)
    env.positions[1] = np.array([2, 3], dtype=np.int32)
    env.grid[2, 2] = 0
    env.grid[2, 3] = 1

    actions = np.array([ACTION_GIVE_BASE, ACTION_STAY])
    obs, rewards, dones, info = env.step(actions)

    assert info["donations_count"] == 0, "Donation should fail when E <= C"
    assert np.isclose(env.energy[0], 1.5), "Energy must not be deducted if donation fails"


def test_starvation_death():
    """Agent with energy <= 0 starves, mask becomes False, and cell is cleared."""
    env = create_deterministic_env(
        grid_size=8,
        num_agents=2,
        initial_energy=0.4,
        c_base=0.5,  # After 1 step, energy will drop to -0.1
        c_density=0.0,
    )
    env.grid.fill(-1)
    env.positions[0] = np.array([1, 1], dtype=np.int32)
    env.positions[1] = np.array([5, 5], dtype=np.int32)
    env.grid[1, 1] = 0
    env.grid[5, 5] = 1

    actions = np.array([ACTION_STAY, ACTION_STAY])
    obs, rewards, dones, info = env.step(actions)

    assert not env.mask[0], "Agent 0 must be culled due to starvation"
    assert env.grid[1, 1] == -1, "Grid cell must be cleared"
    assert info["deaths_this_step"] == 2
    assert info["active_population"] == 0
    assert dones[0] and dones[1], "Episode should terminate on extinction"

