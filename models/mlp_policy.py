"""
Shared-Parameter Actor-Critic MLP Policy (Task 8 - Ablation 1).
Standard MLP without attention or adaptive gating:
- Observation vector: self_state (2) + neighbor_features (K_max * 5 = 40) + neighbor_count (1) = 43 dims
- Actor head: Discrete distribution over 13 actions (Stay, N, S, E, W, Give 0..7)
- Critic head: Scalar state value V(s)
- Action masking: Disallows invalid neighbor targets and donations when E <= C
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


class SimpleMLPPolicy(nn.Module):
    def __init__(
        self,
        max_neighbors: int = 8,
        hidden_dim: int = 128,
        action_dim: int = 13,  # 5 movement/stay + 8 donation targets
    ):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.action_dim = action_dim
        
        # Input: self_state(2) + neighbor_features(max_neighbors * 5) + neighbor_count(1)
        self.input_dim = 2 + (max_neighbors * 5) + 1

        # Shared Feature Extractor
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Policy Head (Actor)
        self.actor_head = nn.Linear(hidden_dim // 2, action_dim)

        # Value Head (Critic)
        self.critic_head = nn.Linear(hidden_dim // 2, 1)

    def extract_flat_obs(self, obs: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts environment observation dictionary into flat PyTorch tensors:
        Returns:
            flat_obs: (N, input_dim) tensor
            action_masks: (N, action_dim) boolean tensor (True = valid action)
        """
        self_state = torch.as_tensor(obs["self_state"], dtype=torch.float32)  # (N, 2)
        neighbor_features = torch.as_tensor(obs["neighbor_features"], dtype=torch.float32)  # (N, K, 5)
        neighbor_mask = torch.as_tensor(obs["neighbor_mask"], dtype=torch.bool)  # (N, K)
        active_mask = torch.as_tensor(obs["active_mask"], dtype=torch.bool)  # (N,)

        N = self_state.shape[0]
        flat_neighbors = neighbor_features.reshape(N, -1)  # (N, K * 5)
        neighbor_counts = neighbor_mask.sum(dim=-1, keepdim=True).float() / float(self.max_neighbors)  # (N, 1)

        flat_obs = torch.cat([self_state, flat_neighbors, neighbor_counts], dim=-1)  # (N, input_dim)

        # Build Action Masks:
        if "action_mask" in obs:
            action_masks = torch.as_tensor(obs["action_mask"], dtype=torch.bool).clone()
            for k in range(self.max_neighbors):
                action_masks[:, 5 + k] &= neighbor_mask[:, k]
        else:
            action_masks = torch.zeros((N, self.action_dim), dtype=torch.bool)
            action_masks[:, :5] = True
            can_afford_donation = self_state[:, 0] > (2.0 / 60.0)
            for k in range(self.max_neighbors):
                action_masks[:, 5 + k] = neighbor_mask[:, k] & can_afford_donation

        return flat_obs, action_masks

    def forward(
        self,
        flat_obs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Computes action distribution and state value.
        Applies action mask by setting invalid action logits to -1e8.
        """
        features = self.encoder(flat_obs)
        logits = self.actor_head(features)

        if action_masks is not None:
            # Set invalid actions to large negative value
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
        """
        Helper for PPO: returns sampled action, log_prob, entropy, and value.
        """
        dist, value = self.forward(flat_obs, action_masks)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

