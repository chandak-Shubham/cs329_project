"""
Rule-Based Baseline Strategies (Task 5).
Implements classical game-theoretic baseline agents:
1. AlwaysDefectPolicy (Selfish baseline): Never donates (stays or wanders)
2. KinOnlyPolicy (Naive Greenbeard): Donates if relatedness r_tilde_ij > tau_kin
3. ReciprocalPolicy (Tit-for-Tat): Donates if neighbor cooperation history b_j > tau_recip
4. FixedWeightPolicy: Donates if 0.5*r_tilde + 0.5*b_j > tau_combined
5. RandomCooperatePolicy: Donates with probability p to a random neighbor
"""

from typing import Dict, Any, Optional
import numpy as np
from environment.grid_world import (
    ACTION_STAY,
    ACTION_NORTH,
    ACTION_SOUTH,
    ACTION_EAST,
    ACTION_WEST,
    ACTION_GIVE_BASE,
)


class BasePolicy:
    """Abstract base class for environment policies."""
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class AlwaysDefectPolicy(BasePolicy):
    """
    Pure Selfish Baseline.
    Never executes donation. Can either stay still or move randomly.
    """
    def __init__(self, allow_movement: bool = True, rng: Optional[np.random.Generator] = None):
        self.allow_movement = allow_movement
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        active_mask = obs["active_mask"]
        N = len(active_mask)
        actions = np.zeros(N, dtype=np.int32)  # Default: ACTION_STAY

        if self.allow_movement:
            # Random movement between Stay, N, S, E, W (actions 0..4)
            move_actions = self.rng.integers(0, 5, size=N)
            actions[active_mask] = move_actions[active_mask]
        return actions


class KinOnlyPolicy(BasePolicy):
    """
    Naive Greenbeard Strategy.
    Rely solely on genetic relatedness r_tilde_ij.
    Donates to the neighbor with highest r_tilde_ij if r_tilde_ij > threshold.
    """
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        active_mask = obs["active_mask"]
        neighbor_features = obs["neighbor_features"]  # (N, max_neighbors, 5)
        neighbor_mask = obs["neighbor_mask"]          # (N, max_neighbors)
        N = len(active_mask)
        actions = np.zeros(N, dtype=np.int32)

        for i in range(N):
            if not active_mask[i]:
                continue

            valid_k = np.where(neighbor_mask[i])[0]
            if len(valid_k) == 0:
                actions[i] = ACTION_STAY
                continue

            # Feature index 3 is r_tilde_ij
            kin_scores = neighbor_features[i, valid_k, 3]
            best_k_idx = np.argmax(kin_scores)
            best_k = valid_k[best_k_idx]

            if kin_scores[best_k_idx] > self.threshold:
                actions[i] = ACTION_GIVE_BASE + best_k
            else:
                actions[i] = ACTION_STAY

        return actions


class ReciprocalPolicy(BasePolicy):
    """
    Tit-for-Tat / Direct Reciprocity Strategy.
    Rely solely on neighbor behavioral history b_j.
    Donates to the neighbor with highest b_j if b_j >= threshold.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        active_mask = obs["active_mask"]
        neighbor_features = obs["neighbor_features"]  # (N, max_neighbors, 5)
        neighbor_mask = obs["neighbor_mask"]          # (N, max_neighbors)
        N = len(active_mask)
        actions = np.zeros(N, dtype=np.int32)

        for i in range(N):
            if not active_mask[i]:
                continue

            valid_k = np.where(neighbor_mask[i])[0]
            if len(valid_k) == 0:
                actions[i] = ACTION_STAY
                continue

            # Feature index 4 is b_j
            behavioral_scores = neighbor_features[i, valid_k, 4]
            best_k_idx = np.argmax(behavioral_scores)
            best_k = valid_k[best_k_idx]

            if behavioral_scores[best_k_idx] >= self.threshold:
                actions[i] = ACTION_GIVE_BASE + best_k
            else:
                actions[i] = ACTION_STAY

        return actions


class FixedWeightPolicy(BasePolicy):
    """
    Fixed-Weight Combination Strategy (w_g = 0.5, w_b = 0.5).
    Computes social affinity q_ij = 0.5 * r_tilde + 0.5 * b_j.
    Donates to the neighbor with highest q_ij if q_ij > threshold.
    """
    def __init__(self, w_g: float = 0.5, w_b: float = 0.5, threshold: float = 0.4):
        self.w_g = w_g
        self.w_b = w_b
        self.threshold = threshold

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        active_mask = obs["active_mask"]
        neighbor_features = obs["neighbor_features"]
        neighbor_mask = obs["neighbor_mask"]
        N = len(active_mask)
        actions = np.zeros(N, dtype=np.int32)

        for i in range(N):
            if not active_mask[i]:
                continue

            valid_k = np.where(neighbor_mask[i])[0]
            if len(valid_k) == 0:
                actions[i] = ACTION_STAY
                continue

            r_tilde = neighbor_features[i, valid_k, 3]
            b_score = neighbor_features[i, valid_k, 4]
            q_scores = (self.w_g * r_tilde) + (self.w_b * b_score)

            best_k_idx = np.argmax(q_scores)
            best_k = valid_k[best_k_idx]

            if q_scores[best_k_idx] > self.threshold:
                actions[i] = ACTION_GIVE_BASE + best_k
            else:
                actions[i] = ACTION_STAY

        return actions


class RandomCooperatePolicy(BasePolicy):
    """
    Random Altruist Strategy.
    Donates to a random valid neighbor with probability p.
    """
    def __init__(self, p_cooperate: float = 0.5, rng: Optional[np.random.Generator] = None):
        self.p_cooperate = p_cooperate
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        active_mask = obs["active_mask"]
        neighbor_mask = obs["neighbor_mask"]
        N = len(active_mask)
        actions = np.zeros(N, dtype=np.int32)

        for i in range(N):
            if not active_mask[i]:
                continue

            valid_k = np.where(neighbor_mask[i])[0]
            if len(valid_k) > 0 and (self.rng.random() < self.p_cooperate):
                chosen_k = self.rng.choice(valid_k)
                actions[i] = ACTION_GIVE_BASE + chosen_k
            else:
                actions[i] = ACTION_STAY

        return actions

