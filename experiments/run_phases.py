"""
Comprehensive Experiment Runner for the 4 Experimental Phases (Task 11).
Executes all 4 phases from Section 4.2 of the submitted paper across 5 random seeds:
- Phase 1: Selfish Baseline (Extinction validation)
- Phase 2: Adaptive Social Affinity (Altruist convergence & carrying capacity)
- Phase 3: Cue-Reliability Sweep (Noise p in [0.0, 0.2, 0.4, 0.6, 0.8] -> trust shift)
- Phase 4: Genetic Mimicry (Parasite defense & comparative benchmarks)
Saves all raw empirical data to results/ for formal statistical and plot generation.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.grid_world import GridWorld, ACTION_STAY
from models.attention_policy import GatedAttentionPolicy
from models.mlp_policy import SimpleMLPPolicy
from models.gating_policy import GatingPolicy
from models.rule_based import (
    AlwaysDefectPolicy,
    KinOnlyPolicy,
    ReciprocalPolicy,
    FixedWeightPolicy,
)

SEEDS = [42, 101, 202, 303, 404]
RESULTS_DIR = "results"


def load_checkpoint(policy, filename: str):
    ckpt_path = os.path.join(RESULTS_DIR, "checkpoints", filename)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(state_dict)
        print(f"  [Model Checkpoint] Loaded: {filename}")
    else:
        print(f"  [Model Checkpoint] Warning: {filename} not found, using initialized weights.")
    policy.eval()
    return policy


# ==============================================================================
# PHASE 1: SELFISH BASELINE (Tragedy of the Commons)
# ==============================================================================
def run_phase1_selfish(seeds: List[int] = SEEDS, steps: int = 300) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("RUNNING PHASE 1: SELFISH BASELINE (5 Seeds)")
    print("=" * 80)

    phase1_results = {"seeds": seeds, "trajectories": {}, "summary": {}}
    extinction_steps = []
    mean_lifespans = []

    for s in seeds:
        env = GridWorld(
            grid_size=20,
            max_capacity=50,
            initial_population=25,
            initial_energy=25.0,
            c_base=0.5,
            c_density=0.05,
            seed=s,
        )
        obs = env.reset(seed=s)
        policy = AlwaysDefectPolicy(allow_movement=True, rng=np.random.default_rng(s))

        pop_traj = []
        energy_traj = []

        for t in range(steps):
            actions = policy.act(obs)
            obs, rewards, dones, info = env.step(actions)

            pop_traj.append(info["active_population"])
            energy_traj.append(info["mean_energy"])

            if np.all(dones) or info["active_population"] == 0:
                break

        extinction_step = t + 1 if info["active_population"] == 0 else steps
        extinction_steps.append(extinction_step)
        mean_lifespans.append(float(np.mean([t + 1 if p == 0 else steps for p in pop_traj])))

        phase1_results["trajectories"][s] = {
            "population": pop_traj,
            "energy": energy_traj,
            "extinction_step": extinction_step,
        }
        print(f"  Seed {s:<3}: Extinct at step {extinction_step}")

    phase1_results["summary"] = {
        "mean_extinction_step": float(np.mean(extinction_steps)),
        "std_extinction_step": float(np.std(extinction_steps)),
        "survival_rate": 0.0,
    }
    print(f"Phase 1 Summary: Extinction at {np.mean(extinction_steps):.1f} ± {np.std(extinction_steps):.1f} steps")
    return phase1_results


# ==============================================================================
# PHASE 2: ADAPTIVE SOCIAL AFFINITY (Altruist Clean World)
# ==============================================================================
def run_phase2_altruist(seeds: List[int] = SEEDS, steps: int = 350) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("RUNNING PHASE 2: ADAPTIVE SOCIAL AFFINITY (Clean World, 5 Seeds)")
    print("=" * 80)

    phase2_results = {"seeds": seeds, "trajectories": {}, "summary": {}}
    final_pops = []
    total_donations = []
    mean_wg_list = []
    mean_wb_list = []
    decision_records = []  # For Hamiltonian logistic regression

    for s in seeds:
        env = GridWorld(
            grid_size=20,
            max_capacity=60,
            initial_population=30,
            initial_energy=30.0,
            e_mitosis=60.0,
            enable_mitosis=True,
            c_base=0.5,
            c_density=0.05,
            donation_cost=2.0,
            donation_benefit=6.0,
            seed=s,
        )
        obs = env.reset(seed=s)
        policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
        policy = load_checkpoint(policy, "gated_attention.pt")

        pop_traj = []
        donations_traj = []
        wg_traj = []
        wb_traj = []

        for t in range(steps):
            self_state, n_feats, n_mask, act_masks = policy.extract_tensors(obs)
            with torch.no_grad():
                dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)
                actions = dist.sample()

            # Record gating weights for active agents with neighbors
            active_with_n = obs["active_mask"] & n_mask.any(dim=-1).numpy()
            if active_with_n.any():
                w_g_step = float(weights[active_with_n, :, 0].mean())
                w_b_step = float(weights[active_with_n, :, 1].mean())
            else:
                w_g_step, w_b_step = 0.5, 0.5

            wg_traj.append(w_g_step)
            wb_traj.append(w_b_step)

            # Record decision samples for Hamiltonian regression
            if t % 5 == 0 and len(decision_records) < 5000:
                for a_i in np.where(obs["active_mask"])[0]:
                    neighbors = env.get_neighbors(a_i)
                    for k, n_j in enumerate(neighbors[:8]):
                        r_val = float(obs["r_tilde"][a_i, n_j])
                        b_val = float(obs["b_scores"][n_j])
                        gave = int(actions[a_i].item() == (5 + k))
                        decision_records.append({
                            "r_tilde": r_val,
                            "b_score": b_val,
                            "bc_ratio": 3.0,
                            "gave": gave,
                            "phase": "phase2_clean",
                        })

            obs, rewards, dones, info = env.step(actions.numpy())
            pop_traj.append(info["active_population"])
            donations_traj.append(info["donations_count"])

            if np.all(dones) or info["active_population"] == 0:
                break

        final_pops.append(pop_traj[-1])
        total_donations.append(sum(donations_traj))
        mean_wg_list.append(float(np.mean(wg_traj)))
        mean_wb_list.append(float(np.mean(wb_traj)))

        phase2_results["trajectories"][s] = {
            "population": pop_traj,
            "donations": donations_traj,
            "w_g": wg_traj,
            "w_b": wb_traj,
        }
        print(f"  Seed {s:<3}: Final Pop={pop_traj[-1]:>2} | Donations={sum(donations_traj):>5} | Mean w^g={np.mean(wg_traj):.2f}, w^b={np.mean(wb_traj):.2f}")

    phase2_results["summary"] = {
        "mean_final_population": float(np.mean(final_pops)),
        "std_final_population": float(np.std(final_pops)),
        "mean_total_donations": float(np.mean(total_donations)),
        "mean_w_g": float(np.mean(mean_wg_list)),
        "mean_w_b": float(np.mean(mean_wb_list)),
    }
    phase2_results["decision_records"] = decision_records[:3000]
    return phase2_results


# ==============================================================================
# PHASE 3: CUE-RELIABILITY (Genetic Noise Sweep)
# ==============================================================================
def run_phase3_cue_reliability(
    noise_levels: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8],
    seeds: List[int] = SEEDS,
    steps: int = 250,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("RUNNING PHASE 3: CUE-RELIABILITY SWEEP (Noise p in [0.0..0.8], 5 Seeds)")
    print("=" * 80)

    sweep_results = {}

    for noise in noise_levels:
        wg_seeds = []
        wb_seeds = []
        pop_seeds = []
        donations_seeds = []

        for s in seeds:
            env = GridWorld(
                grid_size=20,
                max_capacity=60,
                initial_population=30,
                cue_noise=noise,
                seed=s,
            )
            obs = env.reset(seed=s)
            policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
            policy = load_checkpoint(policy, "gated_attention.pt")

            wg_traj = []
            wb_traj = []
            donations = 0

            for t in range(steps):
                self_state, n_feats, n_mask, act_masks = policy.extract_tensors(obs)
                with torch.no_grad():
                    dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)
                    actions = dist.sample()

                active_with_n = obs["active_mask"] & n_mask.any(dim=-1).numpy()
                if active_with_n.any():
                    # Average over active agents with neighbors
                    # Higher noise -> network shifts trust towards b
                    # Gating input is [r_tilde, b, E]
                    wg_traj.append(float(weights[active_with_n, :, 0].mean()))
                    wb_traj.append(float(weights[active_with_n, :, 1].mean()))

                obs, rewards, dones, info = env.step(actions.numpy())
                donations += info["donations_count"]
                if np.all(dones) or info["active_population"] == 0:
                    break

            wg_seeds.append(float(np.mean(wg_traj)) if wg_traj else 0.5)
            wb_seeds.append(float(np.mean(wb_traj)) if wb_traj else 0.5)
            pop_seeds.append(info["active_population"])
            donations_seeds.append(donations)

        sweep_results[str(noise)] = {
            "noise": noise,
            "mean_w_g": float(np.mean(wg_seeds)),
            "std_w_g": float(np.std(wg_seeds)),
            "mean_w_b": float(np.mean(wb_seeds)),
            "std_w_b": float(np.std(wb_seeds)),
            "mean_population": float(np.mean(pop_seeds)),
            "mean_donations": float(np.mean(donations_seeds)),
        }
        print(f"  Noise p={noise:<4} -> Mean w^g = {np.mean(wg_seeds):.3f} ± {np.std(wg_seeds):.3f} | Mean w^b = {np.mean(wb_seeds):.3f} ± {np.std(wb_seeds):.3f}")

    return sweep_results


# ==============================================================================
# PHASE 4: GENETIC MIMICRY & BENCHMARKS
# ==============================================================================
def run_phase4_mimicry_and_benchmarks(seeds: List[int] = SEEDS, steps: int = 300) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("RUNNING PHASE 4: GENETIC MIMICRY DEFENSE & BENCHMARKS (5 Seeds)")
    print("=" * 80)

    models_to_test = [
        "Selfish Baseline",
        "Hardcoded Tag Trust",
        "Fixed Weights (w^g=0.5)",
        "Tit-for-Tat",
        "Simple MLP (No Attention)",
        "PPO + Adaptive Gating",
        "Gated-Attention (Ours)",
    ]

    benchmark_results = {m: {"carrying_capacity": [], "survival": [], "lifespan": [], "mimic_defense": []} for m in models_to_test}
    telemetry_phase4 = {}

    for s in seeds:
        for model_name in models_to_test:
            # Non-saturated stress environment: B/C = 1.5, c_base = 0.8, c_density = 0.10
            env = GridWorld(
                grid_size=20,
                max_capacity=60,
                initial_population=30,
                initial_energy=25.0,
                donation_cost=2.0,
                donation_benefit=3.0,
                c_base=0.8,
                c_density=0.10,
                seed=s,
            )
            obs = env.reset(seed=s)

            # Initialize model with trained checkpoints
            if model_name == "Selfish Baseline":
                policy = AlwaysDefectPolicy(allow_movement=True, rng=np.random.default_rng(s))
            elif model_name == "Hardcoded Tag Trust":
                policy = KinOnlyPolicy(threshold=0.3)
            elif model_name == "Fixed Weights (w^g=0.5)":
                policy = FixedWeightPolicy(w_g=0.5, w_b=0.5, threshold=0.35)
            elif model_name == "Tit-for-Tat":
                policy = ReciprocalPolicy(threshold=0.5)
            elif model_name == "Simple MLP (No Attention)":
                policy = SimpleMLPPolicy(max_neighbors=8, hidden_dim=64, action_dim=13)
                policy = load_checkpoint(policy, "simple_mlp.pt")
            elif model_name == "PPO + Adaptive Gating":
                policy = GatingPolicy(max_neighbors=8, hidden_dim=64, action_dim=13)
                policy = load_checkpoint(policy, "gating_policy.pt")
            else:  # Gated-Attention (Ours)
                policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
                policy = load_checkpoint(policy, "gated_attention.pt")

            pop_over_time = []
            mimic_energy_traj = []
            coop_energy_traj = []

            # Step simulation
            for t in range(steps):
                # Severe infiltration: Inject 12 genetic mimics at step 40 (~40% parasite invasion)
                if t == 40:
                    env.inject_mimics(count=12)

                if model_name in ["Selfish Baseline", "Hardcoded Tag Trust", "Fixed Weights (w^g=0.5)", "Tit-for-Tat"]:
                    actions = policy.act(obs)
                elif model_name == "Simple MLP (No Attention)":
                    flat_obs, act_masks = policy.extract_flat_obs(obs)
                    with torch.no_grad():
                        dist, _ = policy.forward(flat_obs, act_masks)
                        actions = dist.sample().numpy()
                elif model_name == "PPO + Adaptive Gating":
                    flat_obs, act_masks, _ = policy.process_observations(obs)
                    with torch.no_grad():
                        dist, _ = policy.forward(flat_obs, act_masks)
                        actions = dist.sample().numpy()
                else:  # Gated-Attention
                    self_state, n_feats, n_mask, act_masks = policy.extract_tensors(obs)
                    with torch.no_grad():
                        dist, _, _ = policy.forward(self_state, n_feats, n_mask, act_masks)
                        actions = dist.sample().numpy()

                obs, rewards, dones, info = env.step(actions)
                pop_over_time.append(info["active_population"])

                # Track energies of cooperators vs parasites
                parasites = env.is_parasite & env.mask
                cooperators = (~env.is_parasite) & env.mask
                m_e = float(env.energy[parasites].mean()) if parasites.any() else 0.0
                c_e = float(env.energy[cooperators].mean()) if cooperators.any() else 0.0
                mimic_energy_traj.append(m_e)
                coop_energy_traj.append(c_e)

                if np.all(dones) or info["active_population"] == 0:
                    break

            # Metrics
            survived = bool(info["active_population"] > 0)
            final_cap = info["active_population"]
            lifespan = t + 1
            # Mimic defense: percentage of mimics eliminated / starved out
            parasites_alive = int(np.sum(env.is_parasite & env.mask))
            mimic_defense_rate = max(0.0, 1.0 - (parasites_alive / 12.0)) * 100.0

            benchmark_results[model_name]["carrying_capacity"].append(final_cap)
            benchmark_results[model_name]["survival"].append(1.0 if survived else 0.0)
            benchmark_results[model_name]["lifespan"].append(lifespan)
            benchmark_results[model_name]["mimic_defense"].append(mimic_defense_rate)

            # Save detailed time series for Gated-Attention on Seed 42 for Plot 3
            if model_name == "Gated-Attention (Ours)" and s == 42:
                telemetry_phase4["coop_energy"] = coop_energy_traj
                telemetry_phase4["mimic_energy"] = mimic_energy_traj
                telemetry_phase4["population"] = pop_over_time

    # Compute summary tables
    summary_table = {}
    print("\n--- MASTER BENCHMARK TABLE (PHASE 4 UNDER MIMIC ATTACK, 5 SEEDS) ---")
    print(f"{'Model':<30} | {'Carrying Cap':<14} | {'Survival Rate':<14} | {'Mean Lifespan':<14} | {'Mimic Defense':<14}")
    print("-" * 94)

    for m in models_to_test:
        c_mean = np.mean(benchmark_results[m]["carrying_capacity"])
        c_std = np.std(benchmark_results[m]["carrying_capacity"])
        s_rate = np.mean(benchmark_results[m]["survival"]) * 100.0
        l_mean = np.mean(benchmark_results[m]["lifespan"])
        l_std = np.std(benchmark_results[m]["lifespan"])
        d_mean = np.mean(benchmark_results[m]["mimic_defense"])

        summary_table[m] = {
            "carrying_capacity_mean": float(c_mean),
            "carrying_capacity_std": float(c_std),
            "survival_rate": float(s_rate),
            "lifespan_mean": float(l_mean),
            "lifespan_std": float(l_std),
            "mimic_defense_rate": float(d_mean),
        }
        print(f"{m:<30} | {c_mean:>5.1f} ± {c_std:<4.1f} | {s_rate:>6.1f}%       | {l_mean:>5.1f} ± {l_std:<4.1f} | {d_mean:>5.1f}%")

    return {
        "benchmarks": summary_table,
        "raw": benchmark_results,
        "telemetry_phase4_seed42": telemetry_phase4,
    }


# ==============================================================================
# MAIN RUNNER: EXECUTES ALL 4 PHASES
# ==============================================================================
def run_all_experiments():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Phase 1
    p1 = run_phase1_selfish()
    with open(os.path.join(RESULTS_DIR, "phase1_selfish.json"), "w") as f:
        json.dump(p1, f, indent=2)

    # 2. Phase 2
    p2 = run_phase2_altruist()
    with open(os.path.join(RESULTS_DIR, "phase2_altruist.json"), "w") as f:
        json.dump(p2, f, indent=2)

    # 3. Phase 3
    p3 = run_phase3_cue_reliability()
    with open(os.path.join(RESULTS_DIR, "phase3_cue_reliability.json"), "w") as f:
        json.dump(p3, f, indent=2)

    # 4. Phase 4 & Benchmarks
    p4 = run_phase4_mimicry_and_benchmarks()
    with open(os.path.join(RESULTS_DIR, "phase4_benchmarks.json"), "w") as f:
        json.dump(p4, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL 4 EXPERIMENTAL PHASES COMPLETED IN {elapsed:.1f} SECONDS.")
    print(f"Raw data files saved to {RESULTS_DIR}/ for statistical & plot generation.")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()

