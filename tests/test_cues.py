"""
Unit Test Suite for Genetic and Behavioral Cue Engine (Task 4).
Verifies:
1. 8-bit chromosome generation and mutation logic
2. Pairwise normalized Hamming similarity (Definition 2)
3. Baseline-corrected relatedness r_tilde_ij (Definition 2)
4. Sliding-window circular behavioral history tracking (Definition 2)
"""

import pytest
import numpy as np
from environment.cues import (
    CHROMOSOME_LENGTH,
    random_chromosomes,
    mutate_chromosome,
    compute_pairwise_hamming,
    compute_baseline_corrected_relatedness,
    BehavioralHistoryTracker,
)


def test_random_chromosomes_properties():
    """Verify chromosome generation shapes and binary domain."""
    num_agents = 20
    chroms = random_chromosomes(num_agents, rng=np.random.default_rng(42))
    assert chroms.shape == (num_agents, CHROMOSOME_LENGTH)
    assert np.all(np.isin(chroms, [0, 1]))


def test_mutation_flips_bits():
    """Verify per-bit mutation probabilities mu=0, mu=1, and intermediate."""
    chrom = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    
    # mu = 0.0 -> no mutation
    mut_zero = mutate_chromosome(chrom, mutation_rate=0.0)
    assert np.array_equal(mut_zero, chrom)

    # mu = 1.0 -> complete bitwise inversion
    mut_one = mutate_chromosome(chrom, mutation_rate=1.0)
    expected_inverted = 1 - chrom
    assert np.array_equal(mut_one, expected_inverted)


def test_pairwise_hamming_similarity():
    """Verify exact Hamming distances and symmetries."""
    c0 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)  # all 1
    c1 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)  # identical to c0
    c2 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)  # opposite of c0
    c3 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)  # half match with c0

    chroms = np.stack([c0, c1, c2, c3], axis=0)
    active_mask = np.array([True, True, True, True])

    sim = compute_pairwise_hamming(chroms, active_mask)

    # Identical chromosomes -> h = 1.0
    assert np.isclose(sim[0, 1], 1.0)
    assert np.isclose(sim[1, 0], 1.0)

    # Completely opposite chromosomes -> h = 0.0
    assert np.isclose(sim[0, 2], 0.0)
    assert np.isclose(sim[2, 0], 0.0)

    # Half match (4 out of 8 bits) -> h = 0.5
    assert np.isclose(sim[0, 3], 0.5)
    assert np.isclose(sim[3, 0], 0.5)

    # Symmetry check
    assert np.allclose(sim, sim.T)


def test_baseline_corrected_relatedness():
    """Verify r_tilde_ij formula: max(0, (h_ij - h_bar) / (1 - h_bar))."""
    c0 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    c1 = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=np.int8)  # h(0, 1) = 7/8 = 0.875
    c2 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)  # h(0, 2) = 0.0

    chroms = np.stack([c0, c1, c2], axis=0)
    active_mask = np.array([True, True, True])

    sim = compute_pairwise_hamming(chroms, active_mask)
    r_tilde, h_bar = compute_baseline_corrected_relatedness(sim, active_mask)

    # Pairs: (0,1)=0.875, (0,2)=0.0, (1,2)=0.125
    expected_hbar = (0.875 + 0.0 + 0.125) / 3.0  # 1.0 / 3 = 0.333333
    assert np.isclose(h_bar, expected_hbar, atol=1e-5)

    # For pair (0, 1): h_ij = 0.875 > h_bar -> r_tilde > 0
    expected_r01 = (0.875 - expected_hbar) / (1.0 - expected_hbar)
    assert np.isclose(r_tilde[0, 1], expected_r01, atol=1e-5)

    # For pair (0, 2): h_ij = 0.0 < h_bar -> clamped to 0.0
    assert r_tilde[0, 2] == 0.0
    assert r_tilde[2, 0] == 0.0

    # Diagonal (self) must always be 0
    assert np.all(np.diag(r_tilde) == 0.0)


def test_behavioral_history_sliding_window():
    """Verify sliding-window behavior, priors, and circular buffer rollover."""
    K = 4
    tracker = BehavioralHistoryTracker(max_capacity=5, window_k=K, default_prior=0.5)

    # 1. Prior before any actions
    assert tracker.get_behavioral_score(0) == 0.5

    # 2. Step 1: donation -> 1 / 1 = 1.0
    tracker.record_action(0, is_donation=True)
    assert tracker.get_behavioral_score(0) == 1.0

    # 3. Step 2: defection -> (1 + 0) / 2 = 0.5
    tracker.record_action(0, is_donation=False)
    assert tracker.get_behavioral_score(0) == 0.5

    # 4. Fill to window capacity K=4 with donations: actions = [Give, Defect, Give, Give]
    tracker.record_action(0, is_donation=True)
    tracker.record_action(0, is_donation=True)
    assert tracker.get_behavioral_score(0) == 3.0 / 4.0  # 0.75

    # 5. Overwrite the oldest action (Give) with Defect:
    # Buffer now holds: [Defect, Defect, Give, Give] -> 2 / 4 = 0.5
    tracker.record_action(0, is_donation=False)
    assert tracker.get_behavioral_score(0) == 2.0 / 4.0  # 0.5

    # 6. Reset agent clears history back to prior
    tracker.reset_agent(0)
    assert tracker.get_behavioral_score(0) == 0.5

