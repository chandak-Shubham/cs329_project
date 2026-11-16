"""
Cooperation Feasibility Study (Task 6).
Evaluates the cooperation dilemma using rule-based controls before applying RL:
1. Runs 5 random seeds per condition
2. Compares Pure Selfish vs. Pure Cooperative vs. Mixed Competitions
3. Quantifies survival lifespans, mean energy trajectories, and donation frequencies
4. Tests whether productive donations improve outcomes relative to defection
"""

import os
import sys
import json
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Any, Optional
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
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs a single simulation episode with heterogeneous or homogeneous policies."""
    if seed is not None:
        obs = env.reset(seed=seed)
    else:
        obs = env._get_observations()
    
    # Initialize policy instances
    policies = {
        "defect": AlwaysDefectPolicy(allow_movement=True, rng=np.random.default_rng(seed)),
        "kin": KinOnlyPolicy(threshold=0.3),
        "recip": ReciprocalPolicy(threshold=0.5),
        "fixed": FixedWeightPolicy(w_g=0.5, w_b=0.5, threshold=0.35),
    }

    # Map individual unique agent IDs to their assigned / inherited policy
    uid_policy_map: Dict[int, str] = {}
    founder_uids: List[int] = []
    founder_lifespans: List[int] = [0] * len(policy_assignment)
    founder_uid_to_idx: Dict[int, int] = {}

    for i in range(len(policy_assignment)):
        uid = int(env.agent_unique_ids[i])
        pol = policy_assignment[i]
        uid_policy_map[uid] = pol
        founder_uids.append(uid)
        founder_uid_to_idx[uid] = i

    # Tracking metrics
    completed_lifespans: List[int] = []
    survivor_lifespans: List[int] = []
    history_energy: List[float] = []
    cumulative_rewards: np.ndarray = np.zeros(env.N_max, dtype=np.float32)
    individual_rewards = {}
    total_donations = 0

    for step in range(1, max_steps + 1):
        active_indices = np.where(obs["active_mask"])[0]
        if len(active_indices) == 0:
            break

        # Determine action for each active agent by looking up its policy via unique ID
        actions = np.zeros(env.N_max, dtype=np.int32)
        unique_policies_active = set()
        agent_pol_list = []
        for a_idx in active_indices:
            uid = int(env.agent_unique_ids[a_idx])
            pol = uid_policy_map.get(uid, "defect")
            unique_policies_active.add(pol)
            agent_pol_list.append((a_idx, pol))

        for pol_name in unique_policies_active:
            sub_indices = [a_idx for a_idx, p in agent_pol_list if p == pol_name]
            if not sub_indices:
                continue
            pol_actions = policies[pol_name].act(obs)
            for a_idx in sub_indices:
                actions[a_idx] = pol_actions[a_idx]

        # Step environment
        acting_uids = {int(i): int(env.agent_unique_ids[i]) for i in active_indices}
        obs, rewards, dones, info = env.step(actions)

        # 1. Accumulate rewards for all agents that were active during this step
        for a_idx in active_indices:
            cumulative_rewards[a_idx] += rewards[a_idx]
            uid = acting_uids[int(a_idx)]
            individual_rewards[uid] = individual_rewards.get(uid, 0.0) + float(rewards[a_idx])

        # 2. Process explicit birth records to inherit parent's policy by child UID
        birth_records = info.get("birth_records", [])
        for b_rec in birth_records:
            c_uid = b_rec["uid"]
            p_uid = b_rec["parent_uid"]
            uid_policy_map[c_uid] = uid_policy_map.get(p_uid, "defect")

        # 3. Process explicit death records (captures exact individual age before slot reuse)
        death_records = info.get("death_records", [])
        for d_rec in death_records:
            completed_lifespans.append(d_rec["age"])
            d_uid = d_rec["uid"]
            if d_uid in founder_uid_to_idx:
                f_idx = founder_uid_to_idx[d_uid]
                founder_lifespans[f_idx] = d_rec["age"]

        total_donations += info["donations_count"]
        history_energy.append(info["mean_energy"])

        if np.all(dones) or info["active_population"] == 0:
            break

    # Record surviving agents at end of simulation (right-censored)
    final_active = np.where(obs["active_mask"])[0]
    for a_idx in final_active:
        surv_age = int(env.age[a_idx])
        survivor_lifespans.append(surv_age)
        uid = int(env.agent_unique_ids[a_idx])
        if uid in founder_uid_to_idx:
            f_idx = founder_uid_to_idx[uid]
            founder_lifespans[f_idx] = surv_age

    all_lifespans = completed_lifespans + survivor_lifespans
    mean_lifespan = float(np.mean(all_lifespans)) if all_lifespans else 0.0
    mean_completed = float(np.mean(completed_lifespans)) if completed_lifespans else 0.0

    return {
        "final_step": step,
        "mean_lifespan": mean_lifespan,
        "mean_completed_lifespan": mean_completed,
        "completed_lifespans": completed_lifespans,
        "survivor_lifespans": survivor_lifespans,
        "founder_lifespans": founder_lifespans,
        "total_donations": total_donations,
        "mean_energy_trajectory": history_energy,
        "cumulative_rewards": cumulative_rewards.tolist(),
        "cumulative_rewards_unit": "recyclable tensor slot; use individual_rewards for individual returns",
        "individual_rewards": individual_rewards,
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
                        defector_lifespans.append(sim_res["founder_lifespans"][idx])
                    else:
                        cooperator_lifespans.append(sim_res["founder_lifespans"][idx])

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
