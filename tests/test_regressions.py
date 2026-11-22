import pytest
import numpy as np
import torch
from environment.grid_world import (
    GridWorld,
    ACTION_STAY,
    ACTION_NORTH,
    ACTION_SOUTH,
    ACTION_EAST,
    ACTION_WEST,
    ACTION_GIVE_BASE,
)

def test_cell_swap_movement_integrity():
    """
    Regression Test: Two agents swapping positions simultaneously must both survive
    and remain accurately indexed in the spatial grid (no overwriting/erasure).
    """
    env = GridWorld(grid_size=10, initial_population=2, max_capacity=10, seed=42)
    env.reset()

    # Manually place Agent 0 at (3, 3) and Agent 1 at (3, 4)
    env.grid.fill(-1)
    env.positions[0] = np.array([3, 3])
    env.positions[1] = np.array([3, 4])
    env.grid[3, 3] = 0
    env.grid[3, 4] = 1

    # Action 3 is East (+x), Action 4 is West (-x)
    # Agent 0 moves East to (4, 3); Agent 1 moves West from (3, 4)...
    # Let's test exact adjacent cell swap:
    # Agent 0 at (3, 3), moves North to (3, 4)
    # Agent 1 at (3, 4), moves South to (3, 3)
    actions = np.zeros(env.N_max, dtype=int)
    actions[0] = ACTION_NORTH  # (0, 1) -> (3, 4)
    actions[1] = ACTION_SOUTH  # (0, -1) -> (3, 3)

    obs, rewards, dones, info = env.step(actions)

    # Verify both are alive and active
    assert env.mask[0] and env.mask[1]
    # Verify their positions swapped cleanly
    assert np.array_equal(env.positions[0], [3, 4])
    assert np.array_equal(env.positions[1], [3, 3])
    # Verify grid contains both agent indices at their new positions
    assert env.grid[3, 4] == 0
    assert env.grid[3, 3] == 1
    assert (env.grid == 0).sum() == 1
    assert (env.grid == 1).sum() == 1

def test_same_step_death_and_mitosis_dones():
    """
    Regression Test: If an agent starves and another agent reproduces into its slot
    in the same timestep, dones[slot] must be True to prevent GAE value leakage across identities.
    """
    env = GridWorld(grid_size=10, initial_population=2, max_capacity=5, seed=123)
    env.reset()

    # Setup: Agent 0 has zero energy and will starve after baseline metabolism.
    # Agent 1 has surplus energy (>= e_mitosis) and will reproduce.
    env.energy[0] = 0.001
    env.energy[1] = 70.0  # above e_mitosis (60.0)
    old_id_0 = env.agent_unique_ids[0]

    # Idle actions
    actions = np.zeros(env.N_max, dtype=int)
    obs, rewards, dones, info = env.step(actions)

    # Agent 0 starved, so dones[0] must be True!
    assert dones[0] == True
    # If slot 0 was recycled by mitosis, it should now have a new unique agent id and parent id == 1
    if env.mask[0]:
        assert env.agent_unique_ids[0] != old_id_0
        assert env.parent_unique_ids[0] == env.agent_unique_ids[1]

def test_action_mask_in_observations():
    """
    Regression Test: env._get_observations() must include 'action_mask'
    reflecting donation affordability based on energy > donation_cost.
    """
    env = GridWorld(grid_size=10, initial_population=2, max_capacity=5, seed=7)
    env.reset()
    env.energy[0] = 1.0   # <= C (2.0)
    env.energy[1] = 10.0  # > C (2.0)

    obs = env._get_observations()
    assert "action_mask" in obs
    mask = obs["action_mask"]

    # Agent 0 cannot afford donation: actions 5..12 must be False
    assert not np.any(mask[0, ACTION_GIVE_BASE:])
    # Standard actions 0..4 (Stay, N, S, E, W) should be True for active agents
    for a in range(5):
        assert mask[0, a] == True
        assert mask[1, a] == True

def test_mimic_unique_ids_and_ancestry():
    """
    Regression Test: Injected mimics must receive valid non-negative unique IDs,
    have parent_unique_ids = -1, and be recorded in ancestry_records with is_parasite=True.
    """
    env = GridWorld(grid_size=10, initial_population=2, max_capacity=10, seed=42)
    env.reset()
    assert env.agent_unique_ids[0] == 0
    assert env.agent_unique_ids[1] == 1

    spawned = env.inject_mimics(count=2)
    assert spawned == 2

    mimic_slots = np.where(env.is_parasite & env.mask)[0]
    assert len(mimic_slots) == 2
    for s in mimic_slots:
        uid = env.agent_unique_ids[s]
        assert uid >= 2
        assert env.parent_unique_ids[s] == -1
        assert uid in env.ancestry_records
        assert env.ancestry_records[uid]["is_parasite"] == True


