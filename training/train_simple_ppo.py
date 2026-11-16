"""
Vectorized PPO Training Loop with Simple MLP Policy (Task 8 - Ablation 1).
Implements:
- Shared policy parameters across the population
- Dynamic active-agent masking in GAE and PPO optimization
- Purely individualistic reward R_i = Delta E_i (no artificial cooperation bonus)
- Action masking for invalid neighbors
- Comprehensive metric logging (policy loss, value loss, entropy, population, donations)
"""

import os
import sys
import time
import math
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.grid_world import GridWorld
from models.mlp_policy import SimpleMLPPolicy
from training.rollout_buffer import MaskedRolloutBuffer


def train_simple_ppo(
    total_timesteps: int = 15000,
    rollout_length: int = 128,
    num_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    grid_size: int = 20,
    max_capacity: int = 50,
    initial_population: int = 25,
    seed: int = 42,
    log_interval: int = 5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Trains a shared-parameter SimpleMLPPolicy using PPO.
    Returns dictionary of training histories.
    """
    # Set seeds
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    # Initialize environment
    env = GridWorld(
        grid_size=grid_size,
        max_capacity=max_capacity,
        initial_population=initial_population,
        initial_energy=30.0,
        e_mitosis=60.0,
        enable_mitosis=True,
        mutation_rate=0.05,
        c_base=0.5,
        c_density=0.05,
        donation_cost=2.0,
        donation_benefit=6.0,
        obs_radius=3,
        max_neighbors=8,
        max_steps=1000,
        seed=seed,
    )

    # Initialize policy and optimizer
    policy = SimpleMLPPolicy(max_neighbors=8, hidden_dim=128, action_dim=13).to(dev)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    # Initialize rollout buffer
    buffer = MaskedRolloutBuffer(
        buffer_size=rollout_length,
        num_agents=max_capacity,
        obs_dim=policy.input_dim,
        action_dim=policy.action_dim,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=dev,
    )

    # Logging containers
    history = {
        "iteration": [],
        "timesteps": [],
        "mean_active_pop": [],
        "mean_donations_per_step": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "mean_energy": [],
    }

    obs = env.reset(seed=seed)
    global_step = 0
    num_iterations = math.ceil(total_timesteps / (rollout_length * initial_population))
    if num_iterations < 1:
        raise ValueError("Training budget must be positive")

    print("=" * 80)
    print(f"STARTING PPO TRAINING: Simple MLP Baseline (Ablation 1, Seed={seed})")
    print(f"Iterations: {num_iterations} | Rollout Length: {rollout_length} | Batch Size: {batch_size}")
    print("=" * 80)

    for iteration in range(1, num_iterations + 1):
        buffer.reset()
        active_counts = []
        donations_counts = []
        energies = []

        # 1. Rollout Collection Phase
        for step in range(rollout_length):
            global_step += 1
            flat_obs, action_masks = policy.extract_flat_obs(obs)
            flat_obs = flat_obs.to(dev)
            action_masks = action_masks.to(dev)

            with torch.no_grad():
                actions, log_probs, entropy, values = policy.get_action_and_value(flat_obs, action_masks)

            # Step environment
            next_obs, rewards, dones, info = env.step(actions.cpu().numpy())

            active_counts.append(info["active_population"])
            donations_counts.append(info["donations_count"])
            energies.append(info["mean_energy"])

            # Store transition in buffer
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=dev)
            dones_t = torch.as_tensor(dones, dtype=torch.bool, device=dev)
            active_t = torch.as_tensor(obs["active_mask"], dtype=torch.bool, device=dev)

            buffer.add(
                obs=flat_obs,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards_t,
                dones=dones_t,
                values=values,
                action_masks=action_masks,
                active_masks=active_t,
            )

            obs = next_obs

            # Reset environment if all agents went extinct or horizon reached
            if np.sum(obs["active_mask"]) == 0 or np.all(dones):
                obs = env.reset()

        # Compute next state values for GAE
        with torch.no_grad():
            next_flat_obs, next_masks = policy.extract_flat_obs(obs)
            _, next_values = policy.forward(next_flat_obs.to(dev), next_masks.to(dev))
            next_dones = torch.as_tensor(dones, dtype=torch.bool, device=dev)
            buffer.compute_gae(next_values, next_dones)

        # 2. PPO Optimization Phase
        epoch_pi_losses = []
        epoch_v_losses = []
        epoch_entropies = []

        for epoch in range(num_epochs):
            for batch in buffer.get_active_batches(batch_size=batch_size):
                b_obs, b_actions, b_old_log_probs, b_advantages, b_returns, b_masks = batch

                # Forward pass on active batch
                dist, v_pred = policy.forward(b_obs, b_masks)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Policy Loss (Clipped Objective)
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = 0.5 * ((v_pred - b_returns) ** 2).mean()

                # Total Loss
                loss = policy_loss + (vf_coef * value_loss) - (ent_coef * entropy)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                epoch_pi_losses.append(policy_loss.item())
                epoch_v_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())

        # Log iteration metrics
        mean_pi = float(np.mean(epoch_pi_losses)) if epoch_pi_losses else 0.0
        mean_vf = float(np.mean(epoch_v_losses)) if epoch_v_losses else 0.0
        mean_ent = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0
        mean_pop = float(np.mean(active_counts))
        mean_don = float(np.mean(donations_counts))
        mean_en = float(np.mean(energies))

        history["iteration"].append(iteration)
        history["timesteps"].append(global_step * initial_population)
        history["mean_active_pop"].append(mean_pop)
        history["mean_donations_per_step"].append(mean_don)
        history["policy_loss"].append(mean_pi)
        history["value_loss"].append(mean_vf)
        history["entropy"].append(mean_ent)
        history["mean_energy"].append(mean_en)

        if iteration % log_interval == 0 or iteration == num_iterations:
            print(
                f"Iter {iteration:>2}/{num_iterations} | "
                f"Active Pop: {mean_pop:>4.1f} | "
                f"Donations/Step: {mean_don:>4.2f} | "
                f"Pi Loss: {mean_pi:>7.4f} | "
                f"V Loss: {mean_vf:>7.4f} | "
                f"Entropy: {mean_ent:>5.3f} | "
                f"Mean Energy: {mean_en:>5.1f}"
            )

    print("=" * 80)
    print("PPO TRAINING COMPLETE.")
    
    # Save trained checkpoint and history
    os.makedirs("results/checkpoints", exist_ok=True)
    # Isolate the legacy hidden_dim=128 model from the unified hidden_dim=64 suite.
    path = Path(f"results/checkpoints/legacy/simple_mlp_seed_{seed}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path)
    path.with_suffix('.json').write_text(json.dumps({"seed": seed, "hidden_dim": 128,
        "environment_steps": num_iterations * rollout_length,
        "requested_nominal_agent_budget": total_timesteps, "learning_curve": history,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
    
    return history


if __name__ == "__main__":
    train_simple_ppo(total_timesteps=15000, rollout_length=64, num_epochs=3, seed=42)

