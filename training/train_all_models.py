"""
Unified PPO Training Engine for All Neural Ablations (Audit Step 1).
Trains:
1. SimpleMLPPolicy (Ablation 1)
2. GatingPolicy (Ablation 2)
3. GatedAttentionPolicy (Full Model)
under identical training budgets, seeds, and environments.
Saves verified model checkpoints to results/checkpoints/.
"""

import os
import sys
import time
import json
import hashlib
import math
from pathlib import Path
from typing import Dict, Any, Type
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.grid_world import GridWorld
from models.mlp_policy import SimpleMLPPolicy
from models.gating_policy import GatingPolicy
from models.attention_policy import GatedAttentionPolicy
from training.rollout_buffer import MaskedRolloutBuffer

CHECKPOINTS_DIR = "results/checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def train_model(
    model_type: str,  # "mlp", "gating", "attention"
    total_timesteps: int = 12000,
    rollout_length: int = 64,
    num_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    seed: int = 42,
    save_path: str = "",
    bc_ratio: float = 2.5,
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # Environment setup
    env = GridWorld(
        grid_size=18,
        max_capacity=45,
        initial_population=24,
        initial_energy=30.0,
        e_mitosis=60.0,
        enable_mitosis=True,
        mutation_rate=0.05,
        c_base=0.6,
        c_density=0.08,
        donation_cost=2.0,
        donation_benefit=2.0 * bc_ratio,
        seed=seed,
    )

    # Initialize model
    if model_type == "mlp":
        policy = SimpleMLPPolicy(max_neighbors=8, hidden_dim=64, action_dim=13).to(device)
        obs_dim = policy.input_dim
    elif model_type == "gating":
        policy = GatingPolicy(max_neighbors=8, hidden_dim=64, action_dim=13).to(device)
        obs_dim = policy.input_dim
    elif model_type in ("attention", "fixed_attention"):
        policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13,
                                      fixed_gate=0.5 if model_type == "fixed_attention" else None).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Unified buffer representation:
    # Obs: self_state (2) + neighbor_features (40) + neighbor_mask (8) + n_count (1) = 51 dims
    obs_dim = 51
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    buffer = MaskedRolloutBuffer(
        buffer_size=rollout_length,
        num_agents=env.N_max,
        obs_dim=obs_dim,
        action_dim=13,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device,
    )

    obs = env.reset(seed=seed)
    num_iterations = math.ceil(total_timesteps / (rollout_length * 24))
    if num_iterations < 1:
        raise ValueError("Training budget must be positive")
    history = []
    active_transitions = 0
    started = time.time()
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ("environment", "models", "training")
                     for p in sorted(Path(folder).glob("*.py"))}

    print(f"\n--- Training {model_type.upper()} Policy ({num_iterations} iterations, Seed={seed}) ---")

    for iteration in range(1, num_iterations + 1):
        buffer.reset()
        active_counts = []
        donations_counts = []

        # 1. Rollout Collection
        for step in range(rollout_length):
            s_st = torch.as_tensor(obs["self_state"], dtype=torch.float32)
            n_feats = torch.as_tensor(obs["neighbor_features"], dtype=torch.float32)
            n_mask = torch.as_tensor(obs["neighbor_mask"], dtype=torch.bool)
            n_count = n_mask.sum(dim=-1, keepdim=True).float() / 8.0
            flat_nfeats = n_feats.reshape(env.N_max, 40)

            if "action_mask" in obs:
                act_masks = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
            else:
                act_masks = torch.zeros((env.N_max, 13), dtype=torch.bool)
                act_masks[:, :5] = True
                can_afford = s_st[:, 0] > (2.0 / 60.0)
                for k in range(8):
                    act_masks[:, 5 + k] = n_mask[:, k] & can_afford

            buf_obs = torch.cat([s_st, flat_nfeats, n_mask.float(), n_count], dim=-1).detach()

            with torch.no_grad():
                if model_type == "mlp":
                    mlp_in = torch.cat([s_st, flat_nfeats, n_count], dim=-1)
                    actions, log_probs, _, values = policy.get_action_and_value(mlp_in, act_masks)
                elif model_type == "gating":
                    actions, log_probs, _, values, _ = policy.get_action_and_value_from_tensors(
                        s_st, n_feats, n_mask, act_masks
                    )
                elif model_type in ("attention", "fixed_attention"):
                    actions, log_probs, _, values, _ = policy.get_action_and_value(
                        s_st, n_feats, n_mask, act_masks
                    )

            # Step environment
            next_obs, rewards, dones, info = env.step(actions.numpy())

            active_counts.append(info["active_population"])
            donations_counts.append(info["donations_count"])

            rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
            dones_t = torch.as_tensor(dones, dtype=torch.bool)
            active_t = torch.as_tensor(obs["active_mask"], dtype=torch.bool)
            active_transitions += int(active_t.sum())

            buffer.add(
                obs=buf_obs,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards_t,
                dones=dones_t,
                values=values,
                action_masks=act_masks,
                active_masks=active_t,
            )

            obs = next_obs
            if np.sum(obs["active_mask"]) == 0 or np.all(dones):
                obs = env.reset()

        # Compute GAE
        with torch.no_grad():
            next_s = torch.as_tensor(obs["self_state"], dtype=torch.float32)
            next_nf = torch.as_tensor(obs["neighbor_features"], dtype=torch.float32)
            next_nm = torch.as_tensor(obs["neighbor_mask"], dtype=torch.bool)
            next_nc = next_nm.sum(dim=-1, keepdim=True).float() / 8.0
            if "action_mask" in obs:
                next_act_masks = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
            else:
                next_act_masks = torch.zeros((env.N_max, 13), dtype=torch.bool)
                next_act_masks[:, :5] = True
                next_can_afford = next_s[:, 0] > (2.0 / 60.0)
                for k in range(8):
                    next_act_masks[:, 5 + k] = next_nm[:, k] & next_can_afford

            if model_type == "mlp":
                next_mlp_in = torch.cat([next_s, next_nf.reshape(env.N_max, 40), next_nc], dim=-1)
                _, next_values = policy.forward(next_mlp_in, next_act_masks)
            elif model_type == "gating":
                _, next_values, _ = policy.forward_from_tensors(next_s, next_nf, next_nm, next_act_masks)
            elif model_type in ("attention", "fixed_attention"):
                _, next_values, _ = policy.forward(next_s, next_nf, next_nm, next_act_masks)

            next_dones = torch.as_tensor(dones, dtype=torch.bool)
            buffer.compute_gae(next_values, next_dones)

        # 2. PPO Optimization
        epoch_pi_losses = []
        epoch_v_losses = []

        for epoch in range(num_epochs):
            for batch in buffer.get_active_batches(batch_size=batch_size):
                b_obs, b_actions, b_old_log_probs, b_advantages, b_returns, b_masks = batch

                b_s = b_obs[:, :2]
                b_nf = b_obs[:, 2:42].reshape(-1, 8, 5)
                b_nm = b_obs[:, 42:50] > 0.5
                b_nc = b_obs[:, 50:51]

                if model_type == "mlp":
                    b_mlp_in = torch.cat([b_s, b_nf.reshape(-1, 40), b_nc], dim=-1)
                    dist, v_pred = policy.forward(b_mlp_in, b_masks)
                elif model_type == "gating":
                    dist, v_pred, _ = policy.forward_from_tensors(b_s, b_nf, b_nm, b_masks)
                elif model_type in ("attention", "fixed_attention"):
                    dist, v_pred, _ = policy.forward(b_s, b_nf, b_nm, b_masks)

                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((v_pred - b_returns) ** 2).mean()

                loss = policy_loss + (vf_coef * value_loss) - (ent_coef * entropy)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

                epoch_pi_losses.append(policy_loss.item())
                epoch_v_losses.append(value_loss.item())

        history.append({"iteration": iteration, "environment_steps": iteration * rollout_length,
                        "active_agent_transitions": active_transitions,
                        "mean_population": float(np.mean(active_counts)),
                        "donations_per_step": float(np.mean(donations_counts)),
                        "policy_loss": float(np.mean(epoch_pi_losses)),
                        "value_loss": float(np.mean(epoch_v_losses))})
        print(
            f"  Iter {iteration:>2}/{num_iterations} | "
            f"Pop: {np.mean(active_counts):>4.1f} | "
            f"Donations/step: {np.mean(donations_counts):>4.1f} | "
            f"Pi Loss: {np.mean(epoch_pi_losses):>7.4f} | "
            f"V Loss: {np.mean(epoch_v_losses):>7.4f}"
        )

    # Save trained checkpoint
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), save_path)
        provenance = {"model_type": model_type, "seed": seed, "bc_ratio": bc_ratio,
                      "nominal_agent_budget_requested": total_timesteps,
                      "environment_steps": num_iterations * rollout_length,
                      "active_agent_transitions": active_transitions,
                      "rollout_length": rollout_length, "num_epochs": num_epochs,
                      "batch_size": batch_size, "lr": lr, "gamma": gamma,
                      "gae_lambda": gae_lambda, "clip_eps": clip_eps,
                      "vf_coef": vf_coef, "ent_coef": ent_coef,
                      "environment": {"grid_size": 18, "max_capacity": 45, "initial_population": 24,
                                      "initial_energy": 30, "e_mitosis": 60, "mutation_rate": .05,
                                      "c_base": .6, "c_density": .08, "donation_cost": 2,
                                      "donation_benefit": 2 * bc_ratio, "max_steps": env.max_steps},
                      "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                      "source_sha256": source_hashes, "elapsed_seconds": time.time() - started,
                      "checkpoint_sha256": hashlib.sha256(Path(save_path).read_bytes()).hexdigest(),
                      "learning_curve": history}
        Path(save_path).with_suffix(".json").write_text(json.dumps(provenance, indent=2))
        print(f"  Saved trained checkpoint to {save_path}")

    return {"model_type": model_type, "final_pop": float(np.mean(active_counts))}


def train_all():
    """Compatibility entry point for the complete replicated study."""
    return train_replicates()


def train_replicates(seeds=(42, 101, 202, 303, 404), environment_steps=1024):
    """Equal environment-transition budgets; independent network, action and world seeds."""
    import shutil
    aliases = {"mlp": "simple_mlp", "gating": "gating_policy", "attention": "gated_attention",
               "fixed_attention": "fixed_attention"}
    for seed in seeds:
        for kind, alias in aliases.items():
            path = f"{CHECKPOINTS_DIR}/{alias}_seed_{seed}.pt"
            train_model(kind, total_timesteps=environment_steps * 24, seed=seed, save_path=path)
            if seed == seeds[0]:
                shutil.copyfile(path, f"{CHECKPOINTS_DIR}/{alias}.pt")
                shutil.copyfile(Path(path).with_suffix('.json'), f"{CHECKPOINTS_DIR}/{alias}.json")
        for ratio in (1.5, 3.0, 4.5):
            train_model("attention", total_timesteps=environment_steps * 24, seed=seed, bc_ratio=ratio,
                        save_path=f"{CHECKPOINTS_DIR}/incentive_{ratio}_seed_{seed}.pt")


if __name__ == "__main__":
    import argparse
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-steps", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-kind", choices=["mlp", "gating", "attention", "fixed_attention"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc-ratio", type=float, default=2.5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker_kind:
        train_model(args.worker_kind, total_timesteps=args.environment_steps*24,
                    seed=args.seed, bc_ratio=args.bc_ratio, save_path=args.output)
    else:
        import shutil
        aliases = {"mlp": "simple_mlp", "gating": "gating_policy", "attention": "gated_attention", "fixed_attention": "fixed_attention"}
        jobs = []
        for seed in (42, 101, 202, 303, 404):
            for kind, alias in aliases.items():
                jobs.append((kind, seed, 2.5, f"{CHECKPOINTS_DIR}/{alias}_seed_{seed}.pt"))
            for ratio in (1.5, 3., 4.5):
                jobs.append(("attention", seed, ratio, f"{CHECKPOINTS_DIR}/incentive_{ratio}_seed_{seed}.pt"))
        def worker(job):
            kind, seed, ratio, path = job
            with open(Path(path).with_suffix('.log'), 'w') as log:
                subprocess.run([sys.executable, '-u', __file__, '--worker-kind', kind, '--seed', str(seed),
                                '--bc-ratio', str(ratio), '--environment-steps', str(args.environment_steps),
                                '--output', path], stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Completed {path}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, jobs))
        for alias in aliases.values():
            for suffix in ('.pt', '.json'):
                shutil.copyfile(f"{CHECKPOINTS_DIR}/{alias}_seed_42{suffix}", f"{CHECKPOINTS_DIR}/{alias}{suffix}")
