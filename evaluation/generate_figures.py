"""
Publication-Quality Figure & Table Generator (Task 13).
Reads actual logged data from results/ and generates Figures 1-6 and Tables 1-3
directly for the final CS329 project report and presentation.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.uncertainty import mean_ci

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


# ==============================================================================
# FIGURE 1: POPULATION DYNAMICS (Tragedy of Commons vs Altruism)
# ==============================================================================
def plot_figure_1():
    print("Generating Figure 1: Population Dynamics...")
    with open(os.path.join(RESULTS_DIR, "phase1_selfish.json"), "r") as f:
        p1 = json.load(f)
    with open(os.path.join(RESULTS_DIR, "phase2_altruist.json"), "r") as f:
        p2 = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    # Phase 1: Selfish trajectories
    p1_trajs = [data["population"] for data in p1["trajectories"].values()]
    max_len_p1 = max(len(t) for t in p1_trajs)
    p1_padded = np.zeros((len(p1_trajs), max_len_p1))
    for i, t in enumerate(p1_trajs):
        p1_padded[i, :len(t)] = t

    p1_mean = np.mean(p1_padded, axis=0)
    p1_std = np.array([(mean_ci(col)["ci95"][1]-mean_ci(col)["mean"]) for col in p1_padded.T])
    steps_p1 = np.arange(1, max_len_p1 + 1)

    ax.plot(steps_p1, p1_mean, color="#d9534f", linewidth=2.5, label="Phase 1: Pure Selfish (Always Defect)")
    ax.fill_between(steps_p1, np.maximum(0, p1_mean - p1_std), p1_mean + p1_std, color="#d9534f", alpha=0.25)

    # Phase 2: Altruist trajectories
    p2_trajs = [data["population"] for data in p2["trajectories"].values()]
    min_len_p2 = max(len(t) for t in p2_trajs)
    p2_arr = np.array([t + [t[-1]] * (min_len_p2-len(t)) for t in p2_trajs])
    p2_mean = np.mean(p2_arr, axis=0)
    p2_std = np.array([(mean_ci(col)["ci95"][1]-mean_ci(col)["mean"]) for col in p2_arr.T])
    steps_p2 = np.arange(1, min_len_p2 + 1)

    ax.plot(steps_p2, p2_mean, color="#2e7d32", linewidth=2.5, label="Phase 2: Adaptive Social Affinity (Ours)")
    ax.fill_between(steps_p2, p2_mean - p2_std, p2_mean + p2_std, color="#2e7d32", alpha=0.25)

    # Annotation
    extinction = p1["summary"]["mean_extinction_step"]
    ax.axvline(x=extinction, color="#d9534f", linestyle="--", alpha=.7)
    ax.text(extinction + 4, 15, f"Selfish mean endpoint: {extinction:.1f}", color="#d9534f", fontsize=9)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Active Population Size ($N_t$)")
    ax.set_title("Figure 1: Population Dynamics (mean and 95% CI)")
    ax.set_ylim(-2, max(65, float(np.max(p1_mean+p1_std))+3, float(np.max(p2_mean+p2_std))+3))
    ax.set_xlim(0, 350)
    ax.legend(loc="center right", frameon=True)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig1_population_dynamics.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ==============================================================================
# FIGURE 2: CUE-RELIABILITY (Genetic Noise Sweep)
# ==============================================================================
def plot_figure_2():
    print("Generating Figure 2: Cue-Reliability Weights...")
    with open(os.path.join(RESULTS_DIR, "phase3_cue_reliability.json"), "r") as f:
        p3 = json.load(f)

    noises = [float(k) for k in p3.keys()]
    noises.sort()

    w_g_means = [p3[str(n)]["mean_w_g"] for n in noises]
    w_g_stds = [p3[str(n)]["uncertainty"]["w_g"]["ci95"][1] - p3[str(n)]["mean_w_g"] for n in noises]
    w_b_means = [p3[str(n)]["mean_w_b"] for n in noises]
    w_b_stds = [p3[str(n)]["uncertainty"]["w_b"]["ci95"][1] - p3[str(n)]["mean_w_b"] for n in noises]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.errorbar(noises, w_g_means, yerr=w_g_stds, fmt="-o", color="#1976d2", linewidth=2.2, capsize=4, label="Genetic Cue Weight ($w^g$)")
    ax.errorbar(noises, w_b_means, yerr=w_b_stds, fmt="-s", color="#e65100", linewidth=2.2, capsize=4, label="Behavioral Cue Weight ($w^b$)")

    ax.axhline(0.5, color="gray", linestyle=":", label="Unweighted Neutral Prior ($w=0.5$)")
    ax.set_xlabel("Genetic Cue Noise Level ($p$)")
    ax.set_ylabel("Learned Gating Weight")
    ax.set_title("Figure 2: Cue Weights across Noise Levels (95% CI)")
    ax.set_ylim(0, 1)
    ax.set_xlim(min(noises)-.05, max(noises)+.05)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig2_cue_reliability_shift.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ==============================================================================
# FIGURE 3: GENETIC MIMICRY DEFENSE (Impostor Energy Collapse)
# ==============================================================================
def plot_figure_3():
    print("Generating Figure 3: Mimic Defense Trajectory...")
    json_path = os.path.join(RESULTS_DIR, "phase4_benchmarks.json")
    if not os.path.exists(json_path):
        print(f"  Skipping Figure 3: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        p4 = json.load(f)

    # Check for telemetry from available runs
    telemetry = None
    if "telemetry_all" in p4 and "tit_for_tat" in p4["telemetry_all"]:
        telemetry = p4["telemetry_all"]["tit_for_tat"]
    elif "telemetry_phase4_seed42" in p4:
        telemetry = p4["telemetry_phase4_seed42"]

    if not telemetry or "coop_energy" not in telemetry:
        print("  Skipping Figure 3: telemetry data missing.")
        return

    coop_e = telemetry["coop_energy"]
    mimic_e = telemetry.get("mimic_energy", [0.0] * len(coop_e))
    model_name = telemetry["model_name"]
    faced_invasion = telemetry.get("faced_invasion", False)
    inj_step = telemetry["invasion_step"]
    mimics_inj = telemetry["mimics_injected"]


    steps = np.arange(1, len(coop_e) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(steps, coop_e, color="#2e7d32", linewidth=2.2, label=f"Resident Population ({model_name}) Mean Energy")

    if faced_invasion and len(steps) >= inj_step:
        ax.plot(steps, mimic_e, color="#d32f2f", linewidth=2.2, linestyle="-.", label="Genetic Mimics (Parasites) Mean Energy")
        ax.axvline(x=inj_step, color="#d32f2f", linestyle="--", alpha=0.8)
        text_y = max(max(coop_e, default=40) * 0.7, 20)
        ax.text(inj_step + 3, text_y, f"Parasite Infiltration\n({mimics_inj} Mimics Injected after {inj_step} Transitions)", color="#d32f2f", fontweight="bold", fontsize=9.5)
        ax.set_title(f"Figure 3: Genetic Mimicry Defense Dynamics ({model_name})")
    else:
        # Honest representation: colony collapsed before scheduled invasion occurred
        ax.set_title(f"Figure 3: Colony Dynamics Prior to Scheduled Invasion ({model_name})")
        ax.text(len(steps) * 0.5, max(coop_e, default=10) * 0.5, "Colony went extinct before scheduled invasion\n(No Mimic Defense Evaluated)", color="#555555", ha="center", fontsize=10, style="italic")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Metabolic Energy ($E$)")
    ax.set_xlim(0, max(200, len(steps) + 10))
    ax.set_ylim(0, max(max(coop_e, default=50) + 10, 50))
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig3_mimic_defense_dynamics.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ==============================================================================
# FIGURE 4: HAMMING-BASED SPATIAL ASSORTMENT
# ==============================================================================
def plot_figure_4():
    print("Generating Figure 4: Spatial Assortment Bar Chart...")
    with open(os.path.join(RESULTS_DIR, "statistical_analysis.json"), "r") as f:
        stats_data = json.load(f)

    assort = stats_data["spatial_assortment"]
    mu_keys = sorted(assort.keys(), key=lambda x: float(x))

    mus = [float(k) for k in mu_keys]
    h_adj = [assort[k]["mean_h_adjacent"] for k in mu_keys]
    h_adj_err = [assort[k]["adjacent_uncertainty"]["ci95"][1] - assort[k]["mean_h_adjacent"] for k in mu_keys]
    h_rand = [assort[k]["mean_h_random"] for k in mu_keys]
    h_rand_err = [assort[k]["random_uncertainty"]["ci95"][1] - assort[k]["mean_h_random"] for k in mu_keys]

    x = np.arange(len(mus))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    rects1 = ax.bar(x - width/2, h_adj, width, yerr=h_adj_err, capsize=4, label=r"Adjacent Neighbors ($\bar{h}_{\mathrm{adj}}$)", color="#1976d2")
    rects2 = ax.bar(x + width/2, h_rand, width, yerr=h_rand_err, capsize=4, label=r"Random Baseline ($\bar{h}_{\mathrm{rand}}$)", color="#9e9e9e")

    ax.set_xlabel(r"Mutation Rate ($\mu$)")
    ax.set_ylabel(r"Pairwise Hamming Similarity ($h_{ij}$)")
    ax.set_title("Figure 4: Strict Adjacency vs Random Pairs (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\mu = {m}$" for m in mus])
    upper = max(max(np.array(h_adj)+h_adj_err), max(np.array(h_rand)+h_rand_err))
    ax.set_ylim(0, max(1.05, upper+.1))
    ax.legend(loc="upper right", frameon=True)

    # Note on trend for mu = 0.01
    for index, key in enumerate(mu_keys):
        p_value = assort[key]["p_value"]
        label = f"paired p = {p_value:.3f}" if p_value is not None else "test unavailable"
        ax.text(index, max(h_adj[index]+h_adj_err[index], h_rand[index]+h_rand_err[index])+.025, label, ha="center", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig4_spatial_assortment.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ==============================================================================
# FIGURE 5: EXPLORATORY HAMILTON PHASE DIAGRAM
# ==============================================================================
def plot_figure_5():
    print("Generating Figure 5: Exploratory Hamilton Phase Diagram...")
    with open(os.path.join(RESULTS_DIR, "statistical_analysis.json"), "r") as f:
        stats_data = json.load(f)

    phase_data = stats_data["phase_diagram"]
    r_vals = np.array(phase_data["r_vals"])
    bc_vals = np.array(phase_data["bc_vals"])
    P_grid = np.array(phase_data["P_give_grid"])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    cax = ax.pcolormesh(
        r_vals, bc_vals, P_grid,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.15, np.max(P_grid)),
    )
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(r"Donation Probability $P(a_i = \mathrm{give}(j))$")

    # Plot theoretical Hamilton line: r * (B/C) = 1.0 -> B/C = 1 / r
    r_curve = np.linspace(0.22, 0.95, 100)
    bc_curve = 1.0 / r_curve
    valid_pts = bc_curve <= bc_vals[-1]
    ax.plot(r_curve[valid_pts], bc_curve[valid_pts], color="white", linestyle="--", linewidth=2.5, label="Hamilton Boundary ($rB = C$) [Exploratory Reference]")

    ax.set_xlabel(r"Genetic Relatedness ($\tilde{r}_{ij}$)")
    ax.set_ylabel(r"Incentive Ratio ($B/C$)")
    ax.set_yticks(bc_vals)
    ax.set_title("Figure 5: Separately Trained Incentive Conditions")
    ax.text(0.5, 1.3, "Policies trained at each B/C; fixed-input relatedness probe", color="white", fontsize=8.5, ha="center")
    ax.legend(loc="upper right", frameon=True, facecolor="black", labelcolor="white")
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig5_hamilton_phase_diagram.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ==============================================================================
# FIGURE 6: LOGISTIC REGRESSION COEFFICIENTS
# ==============================================================================
def plot_figure_6():
    print("Generating Figure 6: Logistic Regression Coefficients...")
    with open(os.path.join(RESULTS_DIR, "statistical_analysis.json"), "r") as f:
        stats_data = json.load(f)

    reg = stats_data["hamiltonian_regression"]
    features = reg["features"][1:]  # Exclude intercept
    coeffs = reg["coefficients"][1:]
    intervals = np.array(reg["ci95"])[1:]
    stds = np.vstack([np.array(coeffs)-intervals[:, 0], intervals[:, 1]-np.array(coeffs)])

    fig, ax = plt.subplots(figsize=(7, 4.2))

    y_pos = np.arange(len(features))
    colors = ["#1976d2" if c > 0 else "#d32f2f" for c in coeffs]

    ax.barh(y_pos, coeffs, xerr=stds, align="center", color=colors, alpha=0.85, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel(r"Unstandardized Logistic Regression Coefficient ($\beta$)")
    ax.set_title("Figure 6: Predictors (training-seed cluster 95% CI)")
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig6_regression_coefficients.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ==============================================================================
# TABLES GENERATOR: MARKDOWN & LATEX TABLES
# ==============================================================================
def generate_tables():
    print("Generating Report Tables...")
    with open(os.path.join(RESULTS_DIR, "phase4_benchmarks.json"), "r") as f:
        p4 = json.load(f)
    with open(os.path.join(RESULTS_DIR, "phase3_cue_reliability.json"), "r") as f:
        p3 = json.load(f)
    with open(os.path.join(RESULTS_DIR, "statistical_analysis.json"), "r") as f:
        stats_data = json.load(f)

    # Table 1: Master Benchmarks
    t1_md = [
        "# Table 1: Master Benchmark Evaluation (Phase 4 under Genetic Mimicry, 5 Training Seeds)",
        "| Model / Strategy | Final Population | Resident Survival (%) | Episode Duration (Steps) | Mimic Defense Rate (%) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for m, vals in p4["benchmarks"].items():
        if vals.get('mimic_defense_rate') is not None:
            def_str = f"${vals['mimic_defense_rate']:.1f}\\%$"
        else:
            def_str = "N/A (attack not performed)"
        t1_md.append(
            f"| **{m}** | ${vals['carrying_capacity_mean']:.1f} \\pm {vals['carrying_capacity_std']:.1f}$ | "
            f"${vals['survival_rate']:.1f}\\%$ | ${vals['lifespan_mean']:.1f} \\pm {vals['lifespan_std']:.1f}$ | "
            f"{def_str} |"
        )
    t1_md.append("\n95% intervals across training-seed units (population: Student t; survival: Wilson):")
    for model, vals in p4["benchmarks"].items():
        u = vals["uncertainty"]
        t1_md.append(f"- {model}: final population {u['final_population']['ci95']}; survival proportion {u['survival']['ci95']}; defense success {u['successful_defense']['ci95']}")
    t1_md.append("\n*Evaluation regime: Non-saturated stress environment (B/C = 1.5, c_base = 0.8, 12 genetic mimics). Reported over 5 independent training seeds paired with separate evaluation seeds with 300 maximum steps per run.*")
    with open(os.path.join(TAB_DIR, "table1_benchmarks.md"), "w") as f:
        f.write("\n".join(t1_md))

    # Table 2: Cue Corruption Sweep
    t2_md = [
        "# Table 2: Parameter Sweep Matrix Across Genetic Cue Noise (5 Training Seeds)",
        "| Noise Level ($p$) | Mean $w^g$ (Genetic) | Mean $w^b$ (Behavioral) | Final Population | Total Donations |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for n in sorted([float(k) for k in p3.keys()]):
        vals = p3[str(n)]
        t2_md.append(
            f"| ${n:.1f}$ | ${vals['mean_w_g']:.3f} \\pm {vals['std_w_g']:.3f}$ | "
            f"${vals['mean_w_b']:.3f} \\pm {vals['std_w_b']:.3f}$ | "
            f"${vals['mean_population']:.1f}$ | ${vals['mean_donations']:.0f}$ |"
        )
    t2_md.append(f"\n*Mean weights across noise conditions: genetic {np.mean([v['mean_w_g'] for v in p3.values()]):.3f}, behavioral {np.mean([v['mean_w_b'] for v in p3.values()]):.3f}. Frozen policies; these sweeps do not measure online learning.*")
    t2_md.append("\n95% Student t intervals for genetic and behavioral weights:")
    for noise, vals in p3.items():
        t2_md.append(f"- Noise {noise}: genetic {vals['uncertainty']['w_g']['ci95']}; behavioral {vals['uncertainty']['w_b']['ci95']}")
    with open(os.path.join(TAB_DIR, "table2_cue_sweeps.md"), "w") as f:
        f.write("\n".join(t2_md))

    # Table 3: Logistic Regression
    reg = stats_data["hamiltonian_regression"]
    t3_md = [
        "# Table 3: Hamiltonian Consistency Logistic Regression Analysis",
        "Model: $\\text{logit}(P(\\text{give})) = \\beta_0 + \\beta_1 \\tilde{r}_{ij} + \\beta_2 b_j + \\beta_3 (B/C)$",
        "",
        "| Predictor Variable | Coefficient ($\\beta$) | Cluster Standard Error | $t$-Statistic (df=4) | $p$-Value |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for feat, b, se, z, p in zip(reg["features"], reg["coefficients"], reg["std_errors"], reg["z_scores"], reg["p_values"]):
        p_str = "$< 0.001$" if p < 0.001 else f"${p:.4f}$"
        t3_md.append(f"| **{feat}** | ${b:.4f}$ | ${se:.4f}$ | ${z:.3f}$ | {p_str} |")
    t3_md.append(f"\n*McFadden's Pseudo-$R^2 = {reg['pseudo_r2']:.4f}$ (Sample Size: $N = {reg['num_samples']:,}$)*")
    p_r = reg['p_values'][1]
    p_b = reg['p_values'][2]
    p_bc = reg['p_values'][3]
    significant = [name for name, p in zip(reg['features'][1:], reg['p_values'][1:]) if p < .05]
    t3_md.append(f"\n*Exploratory effects at unadjusted p < .05: {', '.join(significant) or 'none'}. {reg['inference']}; {reg['num_clusters']} independent clusters. Coefficients are unstandardized. B/C is unobserved; this regression alone does not establish Hamilton's rule.*")

    t3_md.append("\nTraining-seed cluster 95% intervals:")
    for feature, ci in zip(reg["features"], reg["ci95"]):
        t3_md.append(f"- {feature}: [{ci[0]:.4f}, {ci[1]:.4f}]")
    with open(os.path.join(TAB_DIR, "table3_regression.md"), "w") as f:
        f.write("\n".join(t3_md))

    print("Tables saved to results/tables/")


if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    generate_tables()
    print("ALL PUBLICATION FIGURES & TABLES GENERATED SUCCESSFULLY!")
