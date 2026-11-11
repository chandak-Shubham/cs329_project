"""
Full Gated Multi-Head Spatial Attention Policy Architecture (Task 10 - Full Model).
Strictly implements Section 3.3 and Equations 3 & 4 of the paper:
1. Gating Network: (w^g_ij, w^b_ij) = Softmax(f_thetag([r_tilde_ij || b_j || E_j / E_mitosis]))
2. Social Affinity: q_ij = w^g_ij * r_tilde_ij + w^b_ij * b_j
3. Encoded representations:
   - Self representation: z_ii = MLP_self(v_self) in R^d_model
   - Neighbor representation: z_ij = MLP_enc([Delta p_ij || E_j || r_tilde_ij || b_j || q_ij]) in R^d_model
4. Multi-Head Attention: Spatial context Z_i = MHA(Q=z_ii, K=z_ij, V=z_ij) with key-padding mask
5. Targeted Donation Policy (Eq. 4):
   pi(give(j) | o_i) = Softmax_j(f_pi(Z_i || z_ij || q_ij))
6. Movement Policy & Critic:
   logits_move = MLP_move(Z_i || z_ii)
   V(o_i) = MLP_critic(Z_i || z_ii)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from models.gating_policy import GatingNetwork


class GatedAttentionPolicy(nn.Module):
    def __init__(
        self,
        max_neighbors: int = 8,
        d_model: int = 64,
        num_heads: int = 4,
        action_dim: int = 13,  # 5 movement/stay + 8 targeted donation slots
    ):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.d_model = d_model
        self.num_heads = num_heads
        self.action_dim = action_dim

        # 1. Gating Network (Equation 3)
        self.gating = GatingNetwork(hidden_dim=32)

        # 2. Self and Neighbor Encoders
        # Self input: [E_i / 60.0, age_i / 1000.0] -> 2 dims
        self.self_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # Neighbor input: [dx, dy, E_j, r_tilde, b_j, q_ij] -> 6 dims
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(6, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # 3. Multi-Head Attention Module
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        # 4. Movement / Idling Action Head: actions 0..4 (Stay, N, S, E, W)
        self.move_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 5),
        )

        # 5. Targeted Donation Head (Equation 4): for each neighbor j in 0..7
        # Input: [Z_i || z_ij || q_ij] -> (d_model + d_model + 1)
        self.donation_head = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # 6. Critic Value Head
        self.critic_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        self_state: torch.Tensor,         # (N, 2)
        neighbor_features: torch.Tensor,  # (N, K, 5)
        neighbor_mask: torch.Tensor,      # (N, K)
        action_masks: Optional[torch.Tensor] = None,  # (N, 13)
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        """
        Full Forward Pass:
        Returns:
            dist: Categorical action distribution over 13 discrete actions
            value: (N,) state value estimate
            weights: (N, K, 2) learned gating weights (w^g, w^b)
        """
        N = self_state.shape[0]
        K = self.max_neighbors

        # 1. Gating Network: Evaluate [r_tilde, b, E_j]
        r_tilde = neighbor_features[..., 3:4]
        b_score = neighbor_features[..., 4:5]
        e_norm = neighbor_features[..., 2:3]
        cue_inputs = torch.cat([r_tilde, b_score, e_norm], dim=-1)  # (N, K, 3)

        weights, q = self.gating(cue_inputs)  # weights: (N, K, 2), q: (N, K)
        q_masked = torch.where(neighbor_mask, q, torch.zeros_like(q)).unsqueeze(-1)  # (N, K, 1)

        # 2. Encoders
        z_ii = self.self_encoder(self_state)  # (N, d_model)
        augmented_neighbors = torch.cat([neighbor_features, q_masked], dim=-1)  # (N, K, 6)
        z_ij_raw = self.neighbor_encoder(augmented_neighbors)  # (N, K, d_model)

        # Invariance guarantee: Zero-out neighbor representations for all padded/inactive slots
        z_ij = torch.where(neighbor_mask.unsqueeze(-1), z_ij_raw, torch.zeros_like(z_ij_raw))

        # 3. Multi-Head Attention over Neighbors
        # Query: (N, 1, d_model) from z_ii
        # Key / Value: (N, K, d_model) from z_ij
        query = z_ii.unsqueeze(1)

        # Key padding mask: True means IGNORE in PyTorch MHA
        # If an agent has 0 neighbors, to prevent NaN we allow attention to dummy zero key
        all_masked = ~neighbor_mask.any(dim=-1, keepdim=True)  # (N, 1)
        safe_key_mask = (~neighbor_mask) & (~all_masked)  # (N, K)

        attn_out, _ = self.mha(
            query=query,
            key=z_ij,
            value=z_ij,
            key_padding_mask=safe_key_mask,
        )  # (N, 1, d_model)
        Z_i = attn_out.squeeze(1)  # (N, d_model)

        # If agent has 0 neighbors, strictly zero out spatial context
        Z_i = torch.where(neighbor_mask.any(dim=-1, keepdim=True), Z_i, torch.zeros_like(Z_i))

        # 4. Movement Action Logits (actions 0..4)
        combined_state = torch.cat([Z_i, z_ii], dim=-1)  # (N, d_model * 2)
        move_logits = self.move_head(combined_state)      # (N, 5)

        # 5. Targeted Donation Logits (actions 5..12) (Equation 4)
        # Expand Z_i across all neighbors: (N, K, d_model)
        Z_i_expanded = Z_i.unsqueeze(1).expand(-1, K, -1)
        # Input: [Z_i || z_ij || q_ij] -> (N, K, 2 * d_model + 1)
        donation_inputs = torch.cat([Z_i_expanded, z_ij, q_masked], dim=-1)
        donation_logits = self.donation_head(donation_inputs).squeeze(-1)  # (N, K)

        # Combine all 13 logits: [Move/Stay (5) || Give(0..7) (8)]
        total_logits = torch.cat([move_logits, donation_logits], dim=-1)  # (N, 13)

        # Apply Action Masking
        if action_masks is not None:
            total_logits = torch.where(action_masks, total_logits, torch.full_like(total_logits, -1e8))

        dist = Categorical(logits=total_logits)
        values = self.critic_head(combined_state).squeeze(-1)  # (N,)

        return dist, values, weights

    def extract_tensors(
        self, obs: Dict[str, np.ndarray], device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Utility to convert environment observation dictionary into device tensors."""
        self_state = torch.as_tensor(obs["self_state"], dtype=torch.float32, device=device)
        neighbor_features = torch.as_tensor(obs["neighbor_features"], dtype=torch.float32, device=device)
        neighbor_mask = torch.as_tensor(obs["neighbor_mask"], dtype=torch.bool, device=device)

        N = self_state.shape[0]
        action_masks = torch.zeros((N, self.action_dim), dtype=torch.bool, device=device)
        action_masks[:, :5] = True  # Movement and stay are always valid for active agents

        can_afford = self_state[:, 0] > (2.0 / 60.0)
        for k in range(self.max_neighbors):
            action_masks[:, 5 + k] = neighbor_mask[:, k] & can_afford

        return self_state, neighbor_features, neighbor_mask, action_masks

    def get_action_and_value(
        self,
        self_state: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO action sampling helper."""
        dist, value, weights = self.forward(self_state, neighbor_features, neighbor_mask, action_masks)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, weights

