"""
Unit Tests for Dynamic Demographics, Mitosis, and Agent Pooling (Task 7 & Algorithm 1).
Verifies:
1. Strict energy conservation during mitosis (E_parent + E_offspring == E_before)
2. Offspring placement strictly in adjacent unoccupied cell
3. Genomic inheritance, point mutation rate mu, and lineage tracking
4. Tensor slot recycling after starvation
5. Capacity cap N_max enforcement
"""

import pytest
import numpy as np
from environment.grid_world import GridWorld, ACTION_STAY


def test_mitosis_strict_energy_conservation():
    """Verify exact energy halving and zero energy creation during mitosis."""
    env = GridWorld(
        grid_size=10,
        max_capacity=5,
        initial_population=1,
        initial_energy=70.0,  # Exceeds e_mitosis = 60.0
        e_mitosis=60.0,
        enable_mitosis=True,
        c_base=0.0,  # Zero metabolic cost to isolate mitosis energy arithmetic
        c_density=0.0,
        seed=42,
    )
    env.reset()

    # Pre-condition check
    assert env.active_count == 1
    assert env.energy[0] == 70.0

    # Execute step with Stay action
    actions = np.array([ACTION_STAY] * env.N_max)
    obs, rewards, dones, info = env.step(actions)

    assert info["births_this_step"] == 1, "Mitosis must trigger when E >= 60"
    assert env.active_count == 2, "Active count must increase to 2"

    # Parent is slot 0, offspring is slot 1
    parent_energy = env.energy[0]
    offspring_energy = env.energy[1]

    assert np.isclose(parent_energy, 35.0), f"Expected parent E=35.0, got {parent_energy}"
    assert np.isclose(offspring_energy, 35.0), f"Expected offspring E=35.0, got {offspring_energy}"
    assert np.isclose(parent_energy + offspring_energy, 70.0), "Total energy must be strictly conserved"


def test_offspring_adjacent_placement():
    """Verify offspring is placed strictly in a Von Neumann adjacent cell."""
    env = GridWorld(
        grid_size=10,
        max_capacity=5,
        initial_population=1,
        initial_energy=65.0,
        e_mitosis=60.0,
        enable_mitosis=True,
        c_base=0.0,
        c_density=0.0,
        seed=101,
    )
    env.reset()
    parent_pos = env.positions[0].copy()

    actions = np.array([ACTION_STAY] * env.N_max)
    env.step(actions)

    offspring_pos = env.positions[1]
    delta = env.toroidal_delta(parent_pos, offspring_pos)

    # Manhattan distance must be exactly 1
    manhattan_dist = abs(delta[0]) + abs(delta[1])
    assert manhattan_dist == 1, f"Offspring must spawn in adjacent cell, got delta={delta}"
    assert env.grid[offspring_pos[0], offspring_pos[1]] == 1


def test_inheritance_and_lineage_tracking():
    """Verify chromosome inheritance, generation increment, and parent_id tracking."""
    env = GridWorld(
        grid_size=10,
        max_capacity=5,
        initial_population=1,
        initial_energy=65.0,
        e_mitosis=60.0,
        enable_mitosis=True,
        mutation_rate=0.0,  # Zero mutation to check exact inheritance
        c_base=0.0,
        c_density=0.0,
        seed=42,
    )
    env.reset()
    parent_chrom = env.chromosomes[0].copy()

    actions = np.array([ACTION_STAY] * env.N_max)
    env.step(actions)

    offspring_chrom = env.chromosomes[1]
    assert np.array_equal(parent_chrom, offspring_chrom), "Offspring must inherit parent chromosome"
    assert env.parent_ids[1] == 0, "Offspring parent_id must point to parent index"
    assert env.generations[1] == env.generations[0] + 1, "Generation must increment by 1"
    # Offspring history score should be initialized to neutral prior 0.5
    assert env.history.get_behavioral_score(1) == 0.5


def test_slot_recycling_after_starvation():
    """Verify tensor slots from starved agents are reused cleanly upon mitosis."""
    env = GridWorld(
        grid_size=10,
        max_capacity=2,  # Maximum 2 agents
        initial_population=2,
        initial_energy=5.0,
        e_mitosis=60.0,
        enable_mitosis=True,
        c_base=0.0,
        c_density=0.0,
        seed=42,
    )
    env.reset()

    # Agent 0 will divide (E = 80), Agent 1 will starve (E = 0)
    env.energy[0] = 80.0
    env.energy[1] = 0.0  # Will starve

    actions = np.array([ACTION_STAY] * env.N_max)
    obs, rewards, dones, info = env.step(actions)

    assert info["deaths_this_step"] == 1
    assert info["births_this_step"] == 1
    assert env.active_count == 2
    # Slot 1 must have been recycled for the offspring
    assert bool(env.mask[1]) is True
    assert env.parent_ids[1] == 0
    assert np.isclose(env.energy[0], 40.0)
    assert np.isclose(env.energy[1], 40.0)


def test_population_cap_n_max():
    """Verify reproduction is blocked once population reaches N_max."""
    N_max = 3
    env = GridWorld(
        grid_size=10,
        max_capacity=N_max,
        initial_population=N_max,  # Pool already completely full
        initial_energy=80.0,       # All exceed e_mitosis
        e_mitosis=60.0,
        enable_mitosis=True,
        c_base=0.0,
        c_density=0.0,
        seed=42,
    )
    env.reset()
    assert env.active_count == N_max

    actions = np.array([ACTION_STAY] * env.N_max)
    obs, rewards, dones, info = env.step(actions)

    assert info["births_this_step"] == 0, "Mitosis must not occur when active_count == N_max"
    assert env.active_count == N_max
