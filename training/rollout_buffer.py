"""
Masked Rollout Buffer for Multi-Agent PPO with Dynamic Populations (Task 8).
Stores trajectories for all N_max slots and computes GAE.
Filters transitions so that only active agents (active_mask == True) receive gradient updates.
"""

from typing import Dict, Generator, Tuple
import torch
import numpy as np


class MaskedRolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Storage Tensors
        self.obs = torch.zeros((buffer_size, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, num_agents), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_agents), dtype=torch.bool, device=device)
        self.values = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.action_masks = torch.zeros((buffer_size, num_agents, action_dim), dtype=torch.bool, device=device)
        self.active_masks = torch.zeros((buffer_size, num_agents), dtype=torch.bool, device=device)

        # Advantage & Return Tensors
        self.advantages = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)

        self.step_idx = 0

    def reset(self):
        self.step_idx = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        action_masks: torch.Tensor,
        active_masks: torch.Tensor,
    ):
        """Insert single timestep transition for all N_max agent slots."""
        idx = self.step_idx
        self.obs[idx].copy_(obs)
        self.actions[idx].copy_(actions)
        self.log_probs[idx].copy_(log_probs)
        self.rewards[idx].copy_(rewards)
        self.dones[idx].copy_(dones)
        self.values[idx].copy_(values)
        self.action_masks[idx].copy_(action_masks)
        self.active_masks[idx].copy_(active_masks)
        self.step_idx += 1

    def compute_gae(self, next_values: torch.Tensor, next_dones: torch.Tensor):
        """
        Computes Generalized Advantage Estimation (GAE) across the rollout.
        delta_t = r_t + gamma * V(s_t+1) * (1 - d_t) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - d_t) * A_t+1
        """
        last_gae = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = (~next_dones).float()
                next_val = next_values
            else:
                next_non_terminal = (~self.dones[t]).float()
                next_val = self.values[t + 1]

            delta = self.rewards[t] + (self.gamma * next_val * next_non_terminal) - self.values[t]
            last_gae = delta + (self.gamma * self.gae_lambda * next_non_terminal * last_gae)

            # Store advantage and returns
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def get_active_batches(
        self,
        batch_size: int = 64,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Extracts and shuffles only active agent transitions for PPO optimization.
        Yields batches of: (obs, actions, old_log_probs, advantages, returns, action_masks)
        """
        # Boolean mask of all active agent transitions across all timesteps
        valid_flat = self.active_masks.reshape(-1)  # (T * N,)
        valid_indices = torch.where(valid_flat)[0]

        if len(valid_indices) == 0:
            return

        # Flatten storage
        flat_obs = self.obs.reshape(-1, self.obs_dim)[valid_indices]
        flat_actions = self.actions.reshape(-1)[valid_indices]
        flat_log_probs = self.log_probs.reshape(-1)[valid_indices]
        flat_advantages = self.advantages.reshape(-1)[valid_indices]
        flat_returns = self.returns.reshape(-1)[valid_indices]
        flat_action_masks = self.action_masks.reshape(-1, self.action_dim)[valid_indices]

        # Normalize advantages over active sample population
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        total_samples = len(valid_indices)
        permutation = torch.randperm(total_samples, device=self.device)

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = permutation[start_idx : start_idx + batch_size]
            yield (
                flat_obs[batch_indices],
                flat_actions[batch_indices],
                flat_log_probs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices],
                flat_action_masks[batch_indices],
            )

