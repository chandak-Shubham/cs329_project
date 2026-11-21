"""
Genetic Similarity, Baseline-Corrected Relatedness, and Behavioral History Tracking (Task 4).
Implements Definition 2 and Section 3 of the paper:
1. 8-bit chromosome normalized Hamming similarity:
   h_ij = 1 - (1/8) * sum(|c_i^(k) - c_j^(k)|)
2. Baseline-corrected genetic cue:
   r_tilde_ij = max(0, (h_ij - h_bar) / (1 - h_bar + eps))
3. Behavioral history b_j^(t):
   Fraction of j's actions over sliding window K that were donations.
"""

from typing import Tuple, Optional
import numpy as np

CHROMOSOME_LENGTH = 8


def random_chromosomes(num_agents: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate random 8-bit binary chromosomes of shape (num_agents, 8)."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=(num_agents, CHROMOSOME_LENGTH), dtype=np.int8)


def mutate_chromosome(
    chromosome: np.ndarray,
    mutation_rate: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Mutate chromosome with per-bit mutation probability mu.
    Can operate on a single chromosome (8,) or a batch (N, 8).
    """
    if rng is None:
        rng = np.random.default_rng()
    mutated = chromosome.copy()
    flips = rng.random(size=chromosome.shape) < mutation_rate
    mutated[flips] = 1 - mutated[flips]
    return mutated


def compute_pairwise_hamming(chromosomes: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """
    Compute pairwise normalized Hamming similarity h_ij in [0, 1] for all active agents.
    h_ij = 1 - (1/8) * sum(|c_i^(k) - c_j^(k)|)
    Returns: (N, N) matrix with 0 for inactive pairs.
    """
    N = chromosomes.shape[0]
    # Difference matrix: (N, 1, 8) - (1, N, 8) -> (N, N, 8)
    diff = np.abs(chromosomes[:, np.newaxis, :] - chromosomes[np.newaxis, :, :])
    hamming_dist = np.sum(diff, axis=-1)  # (N, N)
    similarity = 1.0 - (hamming_dist / float(CHROMOSOME_LENGTH))

    # Mask out inactive pairs
    valid_pairs = active_mask[:, np.newaxis] & active_mask[np.newaxis, :]
    similarity = np.where(valid_pairs, similarity, 0.0)
    return similarity


def compute_baseline_corrected_relatedness(
    similarity_matrix: np.ndarray,
    active_mask: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Computes baseline-corrected relatedness:
    r_tilde_ij = max(0, (h_ij - h_bar) / (1 - h_bar + eps))
    where h_bar is the mean pairwise similarity across all distinct active pairs.
    Self-comparisons (i == j) and inactive pairs are set to 0.0.
    """
    active_indices = np.where(active_mask)[0]
    num_active = len(active_indices)

    if num_active <= 1:
        return np.zeros_like(similarity_matrix, dtype=np.float32), 0.5

    # Extract distinct pairs (i < j)
    i_idx, j_idx = np.triu_indices(num_active, k=1)
    act_i, act_j = active_indices[i_idx], active_indices[j_idx]
    pair_sims = similarity_matrix[act_i, act_j]

    h_bar = float(np.mean(pair_sims)) if len(pair_sims) > 0 else 0.5

    # Denominator (1 - h_bar)
    denom = max(1.0 - h_bar, eps)
    r_tilde = np.maximum(0.0, (similarity_matrix - h_bar) / denom)

    # Valid mask: active pairs excluding self
    valid_pairs = active_mask[:, np.newaxis] & active_mask[np.newaxis, :]
    np.fill_diagonal(valid_pairs, False)

    r_tilde = np.where(valid_pairs, r_tilde, 0.0).astype(np.float32)
    return r_tilde, h_bar


class BehavioralHistoryTracker:
    """
    Circular sliding-window buffer of length K tracking past donation actions.
    b_j^(t) is the fraction of agent j's actions over the past K steps that were donations.
    """
    def __init__(self, max_capacity: int, window_k: int = 10, default_prior: float = 0.5):
        self.max_capacity = max_capacity
        self.window_k = window_k
        self.default_prior = default_prior

        # Buffer: (max_capacity, window_k) binary actions (1 = donation, 0 = otherwise)
        self.buffer = np.zeros((max_capacity, window_k), dtype=np.int8)
        self.counts = np.zeros(max_capacity, dtype=np.int32)    # Number of valid records (<= window_k)
        self.pointers = np.zeros(max_capacity, dtype=np.int32)  # Next write index in circular buffer

    def reset_agent(self, agent_idx: int):
        """Reset an agent slot (on birth or reset) with neutral prior."""
        self.buffer[agent_idx, :].fill(0)
        self.counts[agent_idx] = 0
        self.pointers[agent_idx] = 0

    def record_action(self, agent_idx: int, is_donation: bool):
        """Record an action taken by agent_idx in the current step."""
        ptr = self.pointers[agent_idx]
        self.buffer[agent_idx, ptr] = 1 if is_donation else 0
        self.pointers[agent_idx] = (ptr + 1) % self.window_k
        if self.counts[agent_idx] < self.window_k:
            self.counts[agent_idx] += 1

    def get_behavioral_score(self, agent_idx: int) -> float:
        """
        Return the fraction of donations over the available window.
        If age/count is 0, returns the default prior b_0.
        """
        c = self.counts[agent_idx]
        if c == 0:
            return float(self.default_prior)
        donations = np.sum(self.buffer[agent_idx, :c])
        return float(donations / c)

    def get_all_scores(self, active_mask: np.ndarray) -> np.ndarray:
        """Vectorized return of behavioral scores for all agents."""
        scores = np.full(self.max_capacity, self.default_prior, dtype=np.float32)
        active_indices = np.where(active_mask)[0]
        for idx in active_indices:
            scores[idx] = self.get_behavioral_score(idx)
        return scores

