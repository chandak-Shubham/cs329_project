"""
Comprehensive Experiment Runner for the 4 Experimental Phases (Task 11).
Executes all 4 phases from Section 4.2 of the submitted paper across 5 random seeds:
- Phase 1: Selfish Baseline (Extinction validation)
- Phase 2: Adaptive Social Affinity (Altruist convergence & carrying capacity)
- Phase 3: Historical frozen-policy robustness sweep (noise p from 0.0 to 0.5)
- Phase 4: Genetic Mimicry (Parasite defense & comparative benchmarks)
Saves all raw empirical data to results/ for formal statistical and plot generation.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
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
from evaluation.uncertainty import mean_ci, proportion_ci

SEEDS = [42, 101, 202, 303, 404]
RESULTS_DIR = "results"


def load_checkpoint(policy, filename: str, training_seed=42):
    filename = filename.replace(".pt", f"_seed_{training_seed}.pt")
    ckpt_path = os.path.join(RESULTS_DIR, "checkpoints", filename)
    if os.path.exists(ckpt_path):
        metadata_path = Path(ckpt_path).with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        if metadata["checkpoint_sha256"] != hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest():
            raise ValueError(f"Checkpoint hash differs from its training record: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(state_dict)
        print(f"  [Model Checkpoint] Loaded: {filename}")
    else:
        raise FileNotFoundError(f"Required trained checkpoint missing: {ckpt_path}")
    torch.manual_seed(training_seed + 10000)
    policy.eval()
    return policy


# ==============================================================================
# PHASE 1: SELFISH BASELINE (Tragedy of the Commons)
# ==============================================================================
def run_phase1_selfish(seeds: List[int] = SEEDS, steps: int = 300) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING PHASE 1: SELFISH BASELINE ({len(seeds)} Seeds)")
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
            seed=s + 10000,
        )
        obs = env.reset(seed=s + 10000)
        policy = AlwaysDefectPolicy(allow_movement=True, rng=np.random.default_rng(s + 10000))

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
            "evaluation_seed": s + 10000,
            "population": pop_traj,
            "energy": energy_traj,
            "extinction_step": extinction_step,
        }
        print(f"  Seed {s:<3}: Extinct at step {extinction_step}")

    phase1_results["summary"] = {
        "mean_extinction_step": float(np.mean(extinction_steps)),
        "std_extinction_step": float(np.std(extinction_steps)),
        "survival_rate": float(np.mean([v["population"][-1] > 0 for v in phase1_results["trajectories"].values()])),
        "extinction_step_uncertainty": mean_ci(extinction_steps),
    }
    print(f"Phase 1 Summary: Extinction at {np.mean(extinction_steps):.1f} ± {np.std(extinction_steps):.1f} steps")
    return phase1_results


# ==============================================================================
# PHASE 2: ADAPTIVE SOCIAL AFFINITY (Altruist Clean World)
# ==============================================================================
def run_phase2_altruist(seeds: List[int] = SEEDS, steps: int = 350) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING PHASE 2: ADAPTIVE SOCIAL AFFINITY (Clean World, {len(seeds)} Seeds)")
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
            seed=s + 10000,
        )
        obs = env.reset(seed=s + 10000)
        policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
        policy = load_checkpoint(policy, "gated_attention.pt", s)

        pop_traj = []
        donations_traj = []
        wg_traj = []
        wb_traj = []

        for t in range(steps):
            self_state, n_feats, n_mask, act_masks = policy.extract_tensors(obs)
            with torch.no_grad():
                dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)
                actions = dist.sample()

            # Record gating weights ONLY for actual active neighbors (not padded slots)
            valid_neighbor_mask = obs["active_mask"][:, None] & n_mask.numpy()
            if valid_neighbor_mask.any():
                w_g_step = float(weights[:, :, 0].numpy()[valid_neighbor_mask].mean())
                w_b_step = float(weights[:, :, 1].numpy()[valid_neighbor_mask].mean())
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
                            "bc_ratio": env.B / env.C,
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
            "training_seed": s,
            "evaluation_seed": s + 10000,
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
        "uncertainty": {"final_population": mean_ci(final_pops), "donations": mean_ci(total_donations),
                        "w_g": mean_ci(mean_wg_list), "w_b": mean_ci(mean_wb_list)},
    }
    phase2_results["decision_records"] = decision_records[:3000]
    return phase2_results


# ==============================================================================
# PHASE 3: CUE-RELIABILITY (Genetic Noise Sweep)
# ==============================================================================
def run_phase3_cue_reliability(
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    seeds: List[int] = SEEDS,
    steps: int = 250,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING PHASE 3: CUE-RELIABILITY SWEEP ({noise_levels}, {len(seeds)} Seeds)")
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
                seed=s + 10000,
            )
            obs = env.reset(seed=s + 10000)
            policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
            policy = load_checkpoint(policy, "gated_attention.pt", s)

            wg_traj = []
            wb_traj = []
            donations = 0

            for t in range(steps):
                self_state, n_feats, n_mask, act_masks = policy.extract_tensors(obs)
                with torch.no_grad():
                    dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)
                    actions = dist.sample()

                # Average gating weights ONLY over actual active neighbors (not padded slots)
                valid_neighbor_mask = obs["active_mask"][:, None] & n_mask.numpy()
                if valid_neighbor_mask.any():
                    wg_traj.append(float(weights[:, :, 0].numpy()[valid_neighbor_mask].mean()))
                    wb_traj.append(float(weights[:, :, 1].numpy()[valid_neighbor_mask].mean()))

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
            "uncertainty": {"w_g": mean_ci(wg_seeds), "w_b": mean_ci(wb_seeds),
                            "population": mean_ci(pop_seeds), "donations": mean_ci(donations_seeds)},
            "raw": {"w_g": wg_seeds, "w_b": wb_seeds, "population": pop_seeds, "donations": donations_seeds},
        }
        print(f"  Noise p={noise:<4} -> Mean w^g = {np.mean(wg_seeds):.3f} ± {np.std(wg_seeds):.3f} | Mean w^b = {np.mean(wb_seeds):.3f} ± {np.std(wb_seeds):.3f}")

    return sweep_results


# ==============================================================================
# PHASE 4: GENETIC MIMICRY & BENCHMARKS
# ==============================================================================
def run_phase4_mimicry_and_benchmarks(seeds: List[int] = SEEDS, steps: int = 300, invade: bool = True) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING PHASE 4: GENETIC MIMICRY DEFENSE & BENCHMARKS ({len(seeds)} Seeds)")
    print("=" * 80)

    models_to_test = [
        "Selfish Baseline",
        "Hardcoded Tag Trust",
        "Fixed cue threshold (rule-based)",
        "Reputation Threshold",
        "Simple MLP (No Attention)",
        "PPO + Adaptive Gating",
        "Gated-Attention (Ours)",
    ]

    benchmark_results = {
        m: {
            "carrying_capacity": [],
            "survival": [],
            "lifespan": [],
            "faced_invasion": [],
            "mimics_injected": [],
            "residents_survived": [],
            "parasites_eliminated": [],
            "mimic_defense": [],
            "successful_defense": [],
            "donations_to_mimics": [],
            "post_invasion_donations": [],
            "mimic_donation_share": [],
            "energy_to_mimics": [],
        }
        for m in models_to_test
    }
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
                seed=s + 10000,
            )
            obs = env.reset(seed=s + 10000)

            # Initialize model with trained checkpoints
            if model_name == "Selfish Baseline":
                policy = AlwaysDefectPolicy(allow_movement=True, rng=np.random.default_rng(s + 10000))
            elif model_name == "Hardcoded Tag Trust":
                policy = KinOnlyPolicy(threshold=0.3)
            elif model_name == "Fixed cue threshold (rule-based)":
                policy = FixedWeightPolicy(w_g=0.5, w_b=0.5, threshold=0.35)
            elif model_name == "Reputation Threshold":
                policy = ReciprocalPolicy(threshold=0.5)
            elif model_name == "Simple MLP (No Attention)":
                policy = SimpleMLPPolicy(max_neighbors=8, hidden_dim=64, action_dim=13)
                policy = load_checkpoint(policy, "simple_mlp.pt", s)
            elif model_name == "PPO + Adaptive Gating":
                policy = GatingPolicy(max_neighbors=8, hidden_dim=64, action_dim=13)
                policy = load_checkpoint(policy, "gating_policy.pt", s)
            else:  # Gated-Attention (Ours)
                policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
                policy = load_checkpoint(policy, "gated_attention.pt", s)

            pop_over_time = []
            mimic_energy_traj = []
            coop_energy_traj = []
            faced_invasion = False
            mimics_injected = 0
            mimic_donations = post_invasion_donations = 0
            mimic_energy = 0.0

            # Step simulation
            for t in range(steps):
                # Severe infiltration: Inject 12 genetic mimics at step 40 (~40% parasite invasion)
                if invade and t == 40:
                    mimics_injected = env.inject_mimics(count=12)
                    faced_invasion = (mimics_injected > 0)
                    obs = env._get_observations()  # Refresh observations immediately after invasion!

                if model_name in ["Selfish Baseline", "Hardcoded Tag Trust", "Fixed cue threshold (rule-based)", "Reputation Threshold"]:
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
                if faced_invasion:
                    post_invasion_donations += info["donations_count"]
                    mimic_donations += info["donations_to_mimics"]
                    mimic_energy += info["energy_to_mimics"]
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
            residents_alive = int(np.sum((~env.is_parasite) & env.mask))
            parasites_alive = int(np.sum(env.is_parasite & env.mask))
            residents_survived = bool(residents_alive > 0)
            final_cap = info["active_population"]
            lifespan = t + 1

            if faced_invasion and mimics_injected > 0:
                parasites_eliminated = bool(parasites_alive == 0)
                parasites_killed = mimics_injected - parasites_alive
                elimination_rate = max(0.0, float(parasites_killed) / float(mimics_injected)) * 100.0
                # Colony must have surviving residents to be credited with successful defense:
                successful_defense = bool(residents_survived and parasites_eliminated)
                defense_rate = elimination_rate if residents_survived else 0.0
            else:
                parasites_eliminated = None
                successful_defense = None
                defense_rate = None  # Colony went extinct before invasion could occur (not evaluated)

            benchmark_results[model_name]["carrying_capacity"].append(final_cap)
            benchmark_results[model_name]["survival"].append(1.0 if residents_survived else 0.0)
            benchmark_results[model_name]["lifespan"].append(lifespan)
            benchmark_results[model_name]["faced_invasion"].append(faced_invasion)
            benchmark_results[model_name]["mimics_injected"].append(mimics_injected)
            benchmark_results[model_name]["residents_survived"].append(residents_survived)
            benchmark_results[model_name]["parasites_eliminated"].append(parasites_eliminated)
            benchmark_results[model_name]["mimic_defense"].append(defense_rate)
            benchmark_results[model_name]["successful_defense"].append(successful_defense)
            benchmark_results[model_name]["donations_to_mimics"].append(mimic_donations)
            benchmark_results[model_name]["post_invasion_donations"].append(post_invasion_donations)
            benchmark_results[model_name]["energy_to_mimics"].append(mimic_energy)
            benchmark_results[model_name]["mimic_donation_share"].append(
                mimic_donations / post_invasion_donations if faced_invasion and post_invasion_donations else None)

            # Save detailed time series for actual invasion defense analysis
            run_telemetry = {
                "model_name": model_name,
                "training_seed": s,
                "evaluation_seed": s + 10000,
                "coop_energy": coop_energy_traj,
                "mimic_energy": mimic_energy_traj,
                "population": pop_over_time,
                "faced_invasion": faced_invasion,
                "mimics_injected": mimics_injected,
                "invasion_step": 40,
                "step_indexing": "state-time boundary: after 40 transitions, before transition 41",
            }
            if model_name == "Reputation Threshold" and s == 42:
                telemetry_phase4["tit_for_tat"] = run_telemetry
            if model_name == "Gated-Attention (Ours)" and s == 42:
                telemetry_phase4["gated_attention"] = run_telemetry

    # Compute summary tables
    summary_table = {}
    print(f"\n--- MASTER BENCHMARK TABLE (PHASE 4 UNDER MIMIC ATTACK, {len(seeds)} SEEDS) ---")
    print(f"{'Model':<30} | {'Final Pop':<14} | {'Resident Survival':<18} | {'Episode Duration':<18} | {'Mimic Defense':<18}")
    print("-" * 98)

    for m in models_to_test:
        c_mean = np.mean(benchmark_results[m]["carrying_capacity"])
        c_std = np.std(benchmark_results[m]["carrying_capacity"])
        s_rate = np.mean(benchmark_results[m]["survival"]) * 100.0
        l_mean = np.mean(benchmark_results[m]["lifespan"])
        l_std = np.std(benchmark_results[m]["lifespan"])

        # Compute mimic defense only over runs that actually experienced invasion
        inv_eval_runs = [d for d in benchmark_results[m]["mimic_defense"] if d is not None]
        inv_count = sum(1 for f in benchmark_results[m]["faced_invasion"] if f)

        if inv_eval_runs:
            d_mean = float(np.mean(inv_eval_runs))
            d_succ_rate = float(np.mean([1.0 if s else 0.0 for s in benchmark_results[m]["successful_defense"] if s is not None])) * 100.0
            d_str = f"{d_mean:>5.1f}% ({len(inv_eval_runs)}/{len(seeds)} inv)"
        else:
            d_mean = None
            d_succ_rate = None
            d_str = f"N/A (0/{len(seeds)} invaded)"

        summary_table[m] = {
            "carrying_capacity_mean": float(c_mean),
            "carrying_capacity_std": float(c_std),
            "survival_rate": float(s_rate),
            "lifespan_mean": float(l_mean),
            "lifespan_std": float(l_std),
            "faced_invasion_count": inv_count,
            "mimic_defense_rate": d_mean,
            "successful_defense_rate": d_succ_rate,
            "mimic_donation_share": mean_ci(benchmark_results[m]["mimic_donation_share"]),
            "energy_to_mimics": mean_ci(benchmark_results[m]["energy_to_mimics"]),
            "uncertainty": {"final_population": mean_ci(benchmark_results[m]["carrying_capacity"]),
                            "survival": proportion_ci(benchmark_results[m]["survival"]),
                            "duration": mean_ci(benchmark_results[m]["lifespan"]),
                            "successful_defense": proportion_ci(benchmark_results[m]["successful_defense"])},
        }
        print(f"{m:<30} | {c_mean:>5.1f} ± {c_std:<4.1f} | {s_rate:>6.1f}%       | {l_mean:>5.1f} ± {l_std:<4.1f} | {d_str}")

    chosen_telemetry = telemetry_phase4.get("tit_for_tat", telemetry_phase4.get("gated_attention", {}))
    return {
        "benchmarks": summary_table,
        "raw": benchmark_results,
        "telemetry_phase4_seed42": chosen_telemetry,
        "telemetry_all": telemetry_phase4,
        "invasion_enabled": invade,
        "training_seeds": list(seeds),
        "evaluation_seeds": [s+10000 for s in seeds],
        "interpretation": "Historical stress test (out of training distribution). Mimic extinction with resident survival is a secondary outcome, not proof of discrimination; use donation share and energy to mimics.",
    }


# ==============================================================================
# MAIN RUNNER: EXECUTES ALL 4 PHASES
# ==============================================================================
def run_all_experiments():
    torch.set_num_threads(1)
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
    p4_clean = run_phase4_mimicry_and_benchmarks(invade=False)
    with open(os.path.join(RESULTS_DIR, "phase4_no_invasion.json"), "w") as f:
        json.dump(p4_clean, f, indent=2)

    elapsed = time.time() - start_time
    provenance = {"training_seeds": SEEDS, "evaluation_seeds": [s+10000 for s in SEEDS],
                  "evaluation_action_seed": "training_seed + 10000, reset after each model load",
                  "maximum_steps": {"phase1": 300, "phase2": 350, "phase3": 250, "phase4": 300},
                  "elapsed_seconds": elapsed,
                  "checkpoints_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in Path(RESULTS_DIR, "checkpoints").glob("*_seed_*.pt")},
                  "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for folder in ("environment", "models", "experiments")
                                    for p in Path(folder).glob("*.py")}}
    Path(RESULTS_DIR, "evaluation_provenance.json").write_text(json.dumps(provenance, indent=2))
    print("\n" + "=" * 80)
    print(f"ALL 4 EXPERIMENTAL PHASES COMPLETED IN {elapsed:.1f} SECONDS.")
    print(f"Raw data files saved to {RESULTS_DIR}/ for statistical & plot generation.")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()

