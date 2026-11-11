"""
Cooperation Feasibility Study (Task 6).
Solves and validates the fundamental cooperation dilemma before applying RL:
1. Runs 5 random seeds per condition
2. Compares Pure Selfish vs. Pure Cooperative vs. Mixed Competitions
3. Quantifies survival lifespans, mean energy trajectories, and donation frequencies
4. Validates that (B > C) enables cooperative strategies to outperform defection
"""

import os
import sys
import json
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Any
from environment.grid_world import GridWorld, ACTION_STAY
from models.rule_based import (
    AlwaysDefectPolicy,
    KinOnlyPolicy,
    ReciprocalPolicy,
    FixedWeightPolicy,
)


def run_single_simulation(
    env: GridWorld,
    policy_assignment: List[str],  # policy name for each agent index
    max_steps: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs a single simulation episode with heterogeneous or homogeneous policies."""
    obs = env.reset(seed=seed)
    
    # Initialize policy instances
    policies = {
        "defect": AlwaysDefectPolicy(allow_movement=True, rng=np.random.default_rng(seed)),
        "kin": KinOnlyPolicy(threshold=0.3),
        "recip": ReciprocalPolicy(threshold=0.5),
        "fixed": FixedWeightPolicy(w_g=0.5, w_b=0.5, threshold=0.35),
    }

    # Tracking metrics
    lifespans = np.zeros(env.N, dtype=np.int32)
    cumulative_rewards = np.zeros(env.N, dtype=np.float32)
    history_energy = []
    total_donations = 0

    for step in range(1, max_steps + 1):
        # Determine action for each agent according to its assigned policy
        actions = np.zeros(env.N, dtype=np.int32)
        active_indices = np.where(obs["active_mask"])[0]

        if len(active_indices) == 0:
            break

        # Group agents by assigned policy to call policy.act in batch
        for pol_name in set(policy_assignment):
            agents_with_pol = [i for i in active_indices if policy_assignment[i] == pol_name]
            if not agents_with_pol:
                continue
            # Get actions from policy
            pol_actions = policies[pol_name].act(obs)
            for a_idx in agents_with_pol:
                actions[a_idx] = pol_actions[a_idx]

        obs, rewards, dones, info = env.step(actions)

        # Update metrics
        for i in active_indices:
            lifespans[i] = step
            cumulative_rewards[i] += rewards[i]

        total_donations += info["donations_count"]
        history_energy.append(info["mean_energy"])

        if np.all(dones):
            break

    return {
        "final_step": step,
        "mean_lifespan": float(np.mean(lifespans)),
        "lifespans": lifespans.tolist(),
        "total_donations": total_donations,
        "mean_energy_trajectory": history_energy,
        "cumulative_rewards": cumulative_rewards.tolist(),
    }


def evaluate_feasibility(seeds: List[int] = [101, 102, 103, 104, 105]):
    """
    Evaluates 5 experimental setups across 5 random seeds:
    Condition 1: Pure Selfish (100% AlwaysDefect)
    Condition 2: Pure Kin Altruists (100% KinOnly)
    Condition 3: Pure Reciprocal Cooperators (100% Reciprocal)
    Condition 4: Pure Fixed-Weight Cooperators (100% FixedWeight)
    Condition 5: Mixed Arena (50% Defectors vs. 50% Reciprocal Cooperators)
    """
    N = 30
    grid_size = 20
    max_steps = 400

    conditions = {
        "Pure Selfish": ["defect"] * N,
        "Pure Kin": ["kin"] * N,
        "Pure Reciprocal": ["recip"] * N,
        "Pure Fixed-Weight": ["fixed"] * N,
        "Mixed (50% Defect / 50% Recip)": ["defect"] * (N // 2) + ["recip"] * (N - N // 2),
    }

    results = {}

    print("=" * 80)
    print(f"RUNNING COOPERATION FEASIBILITY STUDY ({len(seeds)} Seeds per Condition)")
    print(f"Environment: N={N}, Grid={grid_size}x{grid_size}, B=6.0, C=2.0 (B/C=3.0), c_base=0.5")
    print("=" * 80)

    for cond_name, assignment in conditions.items():
        cond_lifespans = []
        cond_donations = []
        cond_final_steps = []
        defector_lifespans = []
        cooperator_lifespans = []

        for s in seeds:
            env = GridWorld(
                grid_size=grid_size,
                num_agents=N,
                initial_energy=25.0,
                c_base=0.5,
                c_density=0.05,
                donation_cost=2.0,
                donation_benefit=6.0,
                obs_radius=3,
                max_steps=max_steps,
                seed=s,
            )
            sim_res = run_single_simulation(env, assignment, max_steps=max_steps, seed=s)
            
            cond_lifespans.append(sim_res["mean_lifespan"])
            cond_donations.append(sim_res["total_donations"])
            cond_final_steps.append(sim_res["final_step"])

            # If mixed condition, compare defectors vs cooperators
            if "Mixed" in cond_name:
                for idx, pol in enumerate(assignment):
                    if pol == "defect":
                        defector_lifespans.append(sim_res["lifespans"][idx])
                    else:
                        cooperator_lifespans.append(sim_res["lifespans"][idx])

        results[cond_name] = {
            "mean_lifespan": float(np.mean(cond_lifespans)),
            "std_lifespan": float(np.std(cond_lifespans)),
            "mean_donations": float(np.mean(cond_donations)),
            "mean_final_step": float(np.mean(cond_final_steps)),
        }

        if "Mixed" in cond_name:
            results[cond_name]["defector_mean_lifespan"] = float(np.mean(defector_lifespans))
            results[cond_name]["cooperator_mean_lifespan"] = float(np.mean(cooperator_lifespans))

        print(
            f"[{cond_name:<30}] "
            f"Lifespan: {np.mean(cond_lifespans):>6.1f} ± {np.std(cond_lifespans):>4.1f} steps | "
            f"Donations: {np.mean(cond_donations):>6.1f} | "
            f"Colony Extinction Step: {np.mean(cond_final_steps):>5.1f}"
        )
        if "Mixed" in cond_name:
            print(
                f"   -> Subgroup Analysis: Defectors Survived = {np.mean(defector_lifespans):.1f} steps vs. "
                f"Cooperators Survived = {np.mean(cooperator_lifespans):.1f} steps"
            )

    print("=" * 80)
    
    # Save results to json for reporting
    os.makedirs("results", exist_ok=True)
    with open("results/cooperation_feasibility.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/cooperation_feasibility.json")
    return results


if __name__ == "__main__":
    evaluate_feasibility()
