"""
Adaptive Gating Policy Architecture (Task 9 - Ablation 2).
Implements the core neural gating mechanism from Equation 3 of the paper:
1. Gating Network:
   (w^g_ij, w^b_ij) = Softmax(f_thetag([r_tilde_ij || b_j || E_j / E_mitosis]))
2. Social Affinity Score:
   q_ij = w^g_ij * r_tilde_ij + w^b_ij * b_j
3. Augmented Neighbor Features:
   u_ij = [dx/rho, dy/rho, E_j/E_mitosis, r_tilde_ij, b_j, q_ij]
4. End-to-end trained via PPO with purely individual energy reward R_i = Delta E_i.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


class GatingNetwork(nn.Module):
    """
    Evaluates cue reliability (Equation 3):
    Inputs: [r_tilde_ij, b_j, E_j / E_mitosis] (3 dims)
    Outputs: Normalized cue weights (w^g_ij, w^b_ij) with sum = 1.0
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 logits: one for genetic, one for behavioral
        )

    def forward(self, cue_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cue_inputs: Tensor of shape (..., 3) -> [r_tilde, b, E_norm]
        Returns:
            weights: Tensor of shape (..., 2) -> [w^g, w^b] strictly normalized via Softmax
            q: Social affinity score = w^g * r_tilde + w^b * b
        """
        logits = self.net(cue_inputs)
        weights = torch.softmax(logits, dim=-1)  # (..., 2)
        w_g = weights[..., 0]
        w_b = weights[..., 1]

        r_tilde = cue_inputs[..., 0]
        b = cue_inputs[..., 1]
        q = (w_g * r_tilde) + (w_b * b)
        return weights, q


class GatingPolicy(nn.Module):
    def __init__(
        self,
        max_neighbors: int = 8,
        hidden_dim: int = 128,
        action_dim: int = 13,
    ):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.action_dim = action_dim

        # 1. Gating Network
        self.gating = GatingNetwork(hidden_dim=32)

        # 2. Augmented Input Dimension:
        # self_state (2) + per-neighbor features (6 dims each: [dx, dy, E, r, b, q]) * K_max + neighbor_count (1)
        self.input_dim = 2 + (max_neighbors * 6) + 1

        # 3. Shared Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # 4. Policy (Actor) and Value (Critic) Heads
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)
        self.critic_head = nn.Linear(hidden_dim // 2, 1)

    def extract_tensors(
        self, obs: Dict[str, np.ndarray], device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Utility to convert environment observation dictionary into device tensors."""
        self_state = torch.as_tensor(obs["self_state"], dtype=torch.float32, device=device)
        neighbor_features = torch.as_tensor(obs["neighbor_features"], dtype=torch.float32, device=device)
        neighbor_mask = torch.as_tensor(obs["neighbor_mask"], dtype=torch.bool, device=device)

        if "action_mask" in obs:
            action_masks = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=device).clone()
            for k in range(self.max_neighbors):
                action_masks[:, 5 + k] &= neighbor_mask[:, k]
        else:
            N = self_state.shape[0]
            action_masks = torch.zeros((N, self.action_dim), dtype=torch.bool, device=device)
            action_masks[:, :5] = True
            can_afford = self_state[:, 0] > (2.0 / 60.0)
            for k in range(self.max_neighbors):
                action_masks[:, 5 + k] = neighbor_mask[:, k] & can_afford

        return self_state, neighbor_features, neighbor_mask, action_masks

    def forward_from_tensors(
        self,
        self_state: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        """Computes gating and forward pass dynamically with full gradient flow."""
        N = self_state.shape[0]
        r_tilde = neighbor_features[..., 3:4]
        b_score = neighbor_features[..., 4:5]
        e_norm = neighbor_features[..., 2:3]
        cue_inputs = torch.cat([r_tilde, b_score, e_norm], dim=-1)

        weights, q = self.gating(cue_inputs)
        q_masked = torch.where(neighbor_mask, q, torch.zeros_like(q)).unsqueeze(-1)
        augmented_neighbors = torch.cat([neighbor_features, q_masked], dim=-1)
        flat_neighbors = augmented_neighbors.reshape(N, -1)
        neighbor_counts = neighbor_mask.sum(dim=-1, keepdim=True).float() / float(self.max_neighbors)
        flat_obs = torch.cat([self_state, flat_neighbors, neighbor_counts], dim=-1)

        dist, values = self.forward(flat_obs, action_masks)
        return dist, values, weights

    def get_action_and_value_from_tensors(
        self,
        self_state: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO action sampling helper with tensor inputs."""
        dist, value, weights = self.forward_from_tensors(
            self_state, neighbor_features, neighbor_mask, action_masks
        )
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, weights

    def process_observations(
        self, obs: Dict[str, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes environment observations through the Gating Network:
        Returns:
            flat_obs: (N, input_dim) tensor
            action_masks: (N, action_dim) boolean tensor
            gating_weights: (N, max_neighbors, 2) tensor of [w^g, w^b]
        """
        self_state = torch.as_tensor(obs["self_state"], dtype=torch.float32)  # (N, 2)
        neighbor_features = torch.as_tensor(obs["neighbor_features"], dtype=torch.float32)  # (N, K, 5)
        neighbor_mask = torch.as_tensor(obs["neighbor_mask"], dtype=torch.bool)  # (N, K)

        N = self_state.shape[0]

        # Extract cue inputs for gating: [r_tilde, b, E_j] -> indices [3, 4, 2]
        r_tilde = neighbor_features[..., 3:4]
        b_score = neighbor_features[..., 4:5]
        e_norm = neighbor_features[..., 2:3]
        cue_inputs = torch.cat([r_tilde, b_score, e_norm], dim=-1)  # (N, K, 3)

        # Pass through gating network to obtain adaptive weights and affinity q
        weights, q = self.gating(cue_inputs)  # weights: (N, K, 2), q: (N, K)

        # Mask inactive neighbor affinities to 0
        q_masked = torch.where(neighbor_mask, q, torch.zeros_like(q)).unsqueeze(-1)  # (N, K, 1)

        # Augmented neighbor features: [dx, dy, E_j, r_tilde, b, q] -> (N, K, 6)
        augmented_neighbors = torch.cat([neighbor_features, q_masked], dim=-1)
        flat_neighbors = augmented_neighbors.reshape(N, -1)  # (N, K * 6)

        neighbor_counts = neighbor_mask.sum(dim=-1, keepdim=True).float() / float(self.max_neighbors)

        flat_obs = torch.cat([self_state, flat_neighbors, neighbor_counts], dim=-1)

        # Action masks
        if "action_mask" in obs:
            action_masks = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
        else:
            action_masks = torch.zeros((N, self.action_dim), dtype=torch.bool)
            action_masks[:, :5] = True  # Movement and Stay always valid
            can_afford = self_state[:, 0] > (2.0 / 60.0)
            for k in range(self.max_neighbors):
                action_masks[:, 5 + k] = neighbor_mask[:, k] & can_afford

        return flat_obs, action_masks, weights

    def forward(
        self,
        flat_obs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        """Forward pass through encoder, actor, and critic heads."""
        features = self.encoder(flat_obs)
        logits = self.actor_head(features)

        if action_masks is not None:
            logits = torch.where(action_masks, logits, torch.full_like(logits, -1e8))

        dist = Categorical(logits=logits)
        values = self.critic_head(features).squeeze(-1)
        return dist, values

    def get_action_and_value(
        self,
        flat_obs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO action sampling helper."""
        dist, value = self.forward(flat_obs, action_masks)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

