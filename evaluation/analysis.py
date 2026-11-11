"""
Statistical, Hamiltonian & Spatial Assortment Analysis (Task 12).
Directly answers Research Questions 2 and 3 from the paper:
1. Hamiltonian Consistency Logistic Regression:
   logit(P(give(j))) = beta_0 + beta_1 * r_tilde_ij + beta_2 * b_j + beta_3 * (B/C)
   Computes MLE coefficients, standard errors, z-values, p-values, and McFadden's pseudo-R^2.
2. Empirical Hamilton's Rule Boundary Verification (rB > C):
   Grid sweep over (r, B/C) plane evaluating exploratory geometric correspondence with theoretical r*B = C.
3. Hamming-Based Spatial Assortment:
   Computes adjacent pair Hamming similarity vs. randomized baseline across mutation rates mu in [0.01..0.20].
   Performs two-sample t-test evaluating spatial clustering trends.
Saves all analytics to results/statistical_analysis.json.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import scipy.stats as stats
from scipy.optimize import minimize

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.grid_world import GridWorld
from environment.cues import compute_pairwise_hamming
from models.attention_policy import GatedAttentionPolicy

SEEDS = [42, 101, 202, 303, 404]
RESULTS_DIR = "results"


def load_trained_model(checkpoint_name: str = "gated_attention.pt") -> GatedAttentionPolicy:
    policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
    ckpt_path = os.path.join(RESULTS_DIR, "checkpoints", checkpoint_name)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(state_dict)
        print(f"Loaded trained checkpoint: {ckpt_path}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Running with initialized weights.")
    policy.eval()
    return policy


# ==============================================================================
# 1. HAMILTONIAN CONSISTENCY LOGISTIC REGRESSION
# ==============================================================================
def fit_logistic_regression(
    X: np.ndarray,  # Shape (N_samples, n_features)
    y: np.ndarray,  # Shape (N_samples,) binary {0, 1}
    feature_names: List[str],
    l2_reg: float = 1e-4,
) -> Dict[str, Any]:
    """
    Fits logistic regression with mild L2 regularization for numerical stability.
    Computes exact standard errors via Fisher Information (Hessian inversion),
    Wald z-statistics, and two-tailed p-values.
    """
    N, D = X.shape

    # Negative log-likelihood with L2 ridge regularization
    def objective(beta):
        logits = np.dot(X, beta)
        log_loss = np.sum(np.logaddexp(0.0, logits) - y * logits)
        ridge = 0.5 * l2_reg * np.sum(beta[1:] ** 2)  # Don't penalize intercept
        return log_loss + ridge

    def gradient(beta):
        p = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, beta), -20, 20)))
        grad = np.dot(X.T, (p - y))
        grad[1:] += l2_reg * beta[1:]
        return grad

    beta_init = np.zeros(D)
    res = minimize(objective, beta_init, jac=gradient, method="BFGS")
    beta_hat = res.x

    # Compute Hessian for standard errors: H = X^T * W * X + l2_reg * I
    p = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, beta_hat), -20, 20)))
    w = p * (1.0 - p)
    Hessian = np.dot(X.T * w, X) + np.eye(D) * max(l2_reg, 1e-5)

    try:
        cov_matrix = np.linalg.pinv(Hessian)
        std_errors = np.sqrt(np.abs(np.diag(cov_matrix)))
    except np.linalg.LinAlgError:
        std_errors = np.ones(D) * 0.1

    z_scores = np.where(std_errors > 0, beta_hat / std_errors, 0.0)
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))

    # McFadden's pseudo-R^2
    logits_pred = np.dot(X, beta_hat)
    ll_model = -np.sum(np.logaddexp(0.0, logits_pred) - y * logits_pred)
    y_mean = np.mean(y)
    ll_null = np.sum(y * np.log(y_mean + 1e-12) + (1 - y) * np.log(1 - y_mean + 1e-12))
    pseudo_r2 = max(0.0, 1.0 - (ll_model / ll_null)) if ll_null < 0 else 0.0

    summary = {
        "features": feature_names,
        "coefficients": beta_hat.tolist(),
        "std_errors": std_errors.tolist(),
        "z_scores": z_scores.tolist(),
        "p_values": p_values.tolist(),
        "pseudo_r2": float(pseudo_r2),
        "log_likelihood": float(ll_model),
        "num_samples": N,
    }
    return summary


def run_hamiltonian_regression_analysis() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("1. RUNNING HAMILTONIAN CONSISTENCY LOGISTIC REGRESSION")
    print("   Model: logit(P(give)) = beta_0 + beta_1*r_tilde + beta_2*b + beta_3*(B/C)")
    print("=" * 80)

    # Collect multi-condition decision samples across varying B/C ratios
    records = []
    print("Sampling agent decisions across diverse B/C ratios [1.5, 3.0, 4.5]...")
    for bc_ratio in [1.5, 3.0, 4.5]:
        for s in [42, 101]:
            env = GridWorld(
                grid_size=18,
                max_capacity=40,
                initial_population=24,
                donation_cost=2.0,
                donation_benefit=2.0 * bc_ratio,
                seed=s,
            )
            obs = env.reset(seed=s)
            policy = load_trained_model("gated_attention.pt")

            for step in range(30):
                self_state, n_feats, n_mask, act_masks = policy.extract_tensors(obs)
                with torch.no_grad():
                    dist, _, _ = policy.forward(self_state, n_feats, n_mask, act_masks)
                    actions = dist.sample()

                for a_i in np.where(obs["active_mask"])[0]:
                    neighbors = env.get_neighbors(a_i)
                    for k, n_j in enumerate(neighbors[:8]):
                        r_val = float(obs["r_tilde"][a_i, n_j])
                        b_val = float(obs["b_scores"][n_j])
                        gave = int(actions[a_i].item() == (5 + k))
                        records.append({
                            "r_tilde": r_val,
                            "b_score": b_val,
                            "bc_ratio": bc_ratio,
                            "gave": gave,
                        })
                obs, _, dones, _ = env.step(actions.numpy())
                if np.all(dones):
                    break

    # Build design matrix X and target y
    N_samples = len(records)
    X = np.ones((N_samples, 4), dtype=np.float64)  # Col 0: Intercept
    y = np.zeros(N_samples, dtype=np.float64)

    for i, r in enumerate(records):
        X[i, 1] = r["r_tilde"]
        X[i, 2] = r["b_score"]
        X[i, 3] = r["bc_ratio"]
        y[i] = r["gave"]

    feature_names = ["Intercept", "Relatedness (r_tilde)", "Behavior (b)", "Incentive (B/C)"]
    reg_results = fit_logistic_regression(X, y, feature_names)

    print(f"{'Feature':<26} | {'Coeff':<10} | {'Std Err':<10} | {'z-value':<10} | {'p-value':<10}")
    print("-" * 75)
    for name, b, se, z, p in zip(
        feature_names,
        reg_results["coefficients"],
        reg_results["std_errors"],
        reg_results["z_scores"],
        reg_results["p_values"],
    ):
        p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
        print(f"{name:<26} | {b:>9.4f}  | {se:>9.4f}  | {z:>9.3f}  | {p_str:>9}")
    print(f"McFadden's Pseudo-R^2: {reg_results['pseudo_r2']:.4f} (Samples: {N_samples})")

    return reg_results


# ==============================================================================
# 2. EXPLORATORY PHASE-SPACE GEOMETRIC CORRESPONDENCE ANALYSIS
# ==============================================================================
def run_hamilton_phase_diagram_analysis(grid_resolution: int = 15) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("2. EXPLORATORY PHASE-SPACE CORRESPONDENCE SWEEP (rB > C)")
    print("=" * 80)

    # Grid of Relatedness r in [0, 1] and B/C in [1, 5]
    r_vals = np.linspace(0.05, 0.95, grid_resolution)
    bc_vals = np.linspace(1.1, 4.9, grid_resolution)

    R_grid, BC_grid = np.meshgrid(r_vals, bc_vals)
    P_give_grid = np.zeros_like(R_grid)
    theoretical_boundary = 1.0 / bc_vals  # Hamilton condition: r = C / B -> r * (B/C) = 1

    policy = load_trained_model("gated_attention.pt")

    # Query learned policy on synthesized neighbor pairs at each (r, B/C) point
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            r = R_grid[i, j]
            bc = BC_grid[i, j]

            # Construct mock observation with relatedness r
            self_state = torch.tensor([[0.5, 0.1]], dtype=torch.float32)
            n_feats = torch.zeros((1, 8, 5), dtype=torch.float32)
            n_feats[0, 0, 0] = 0.33  # dx/rho
            n_feats[0, 0, 1] = 0.0   # dy/rho
            n_feats[0, 0, 2] = 0.5   # E_j / 60
            n_feats[0, 0, 3] = r     # r_tilde
            n_feats[0, 0, 4] = 0.5   # neutral b

            n_mask = torch.zeros((1, 8), dtype=torch.bool)
            n_mask[0, 0] = True

            act_masks = torch.zeros((1, 13), dtype=torch.bool)
            act_masks[0, :5] = True
            act_masks[0, 5] = True  # Donation to neighbor 0 valid

            with torch.no_grad():
                dist, _, _ = policy.forward(self_state, n_feats, n_mask, act_masks)
                p_give = dist.probs[0, 5].item()

            P_give_grid[i, j] = p_give

    # Compute correlation between Hamilton condition (r*B/C > 1) and actual donation probability
    hamilton_indicator = (R_grid * BC_grid > 1.0).astype(float)
    corr = np.corrcoef(hamilton_indicator.flatten(), P_give_grid.flatten())[0, 1]
    r_squared = float(corr ** 2)

    print(f"Exploratory Geometric Correspondence with Hamilton Boundary (r*B > C): R^2 = {r_squared:.4f}")

    return {
        "r_vals": r_vals.tolist(),
        "bc_vals": bc_vals.tolist(),
        "P_give_grid": P_give_grid.tolist(),
        "theoretical_boundary": theoretical_boundary.tolist(),
        "r_squared_alignment": r_squared,
    }


# ==============================================================================
# 3. HAMMING-BASED SPATIAL ASSORTMENT ANALYSIS
# ==============================================================================
def run_spatial_assortment_analysis(
    mutation_rates: List[float] = [0.01, 0.05, 0.10, 0.20],
    seeds: List[int] = SEEDS,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("3. RUNNING HAMMING-BASED SPATIAL ASSORTMENT ANALYSIS")
    print("   Hypothesis: Adjacent neighbor similarity (h_adj) > Random baseline (h_rand)")
    print("=" * 80)

    assortment_results = {}

    for mu in mutation_rates:
        adj_similarities = []
        rand_similarities = []

        for s in seeds:
            env = GridWorld(
                grid_size=20,
                max_capacity=60,
                initial_population=25,
                initial_energy=45.0,
                e_mitosis=50.0,
                enable_mitosis=True,
                donation_cost=2.0,
                donation_benefit=5.0,
                mutation_rate=mu,
                seed=s,
            )
            env.reset(seed=s)

            # Step environment to equilibrium (allowing reproduction and kin clustering)
            policy = load_trained_model("gated_attention.pt")

            for _ in range(25):
                self_state, n_feats, n_mask, act_masks = policy.extract_tensors(env._get_observations())
                with torch.no_grad():
                    dist, _, _ = policy.forward(self_state, n_feats, n_mask, act_masks)
                    actions = dist.sample()
                env.step(actions.numpy())
                if env.active_count == 0:
                    break

            active_indices = np.where(env.mask)[0]
            if len(active_indices) < 2:
                continue

            # 1. Compute mean adjacent neighbor Hamming similarity
            h_adj_pairs = []
            for i in active_indices:
                neighbors = env.get_neighbors(i)
                for n_j in neighbors:
                    # Normalized Hamming: 1 - sum(|c_i - c_j|) / 8
                    diff = np.abs(env.chromosomes[i] - env.chromosomes[n_j])
                    h_sim = 1.0 - (np.sum(diff) / 8.0)
                    h_adj_pairs.append(h_sim)

            # 2. Compute randomized baseline: pairs placed uniformly at random
            h_rand_pairs = []
            N_active = len(active_indices)
            for _ in range(len(h_adj_pairs) * 2):
                i_rand, j_rand = env.rng.choice(active_indices, size=2, replace=False)
                diff = np.abs(env.chromosomes[i_rand] - env.chromosomes[j_rand])
                h_rand_pairs.append(1.0 - (np.sum(diff) / 8.0))

            if h_adj_pairs and h_rand_pairs:
                adj_similarities.append(float(np.mean(h_adj_pairs)))
                rand_similarities.append(float(np.mean(h_rand_pairs)))

        # Two-sample t-test
        t_stat, p_val = stats.ttest_ind(adj_similarities, rand_similarities, equal_var=False)
        mean_adj = float(np.mean(adj_similarities))
        mean_rand = float(np.mean(rand_similarities))
        assortment_gap = mean_adj - mean_rand

        assortment_results[str(mu)] = {
            "mutation_rate": mu,
            "mean_h_adjacent": mean_adj,
            "std_h_adjacent": float(np.std(adj_similarities)),
            "mean_h_random": mean_rand,
            "std_h_random": float(np.std(rand_similarities)),
            "assortment_gap": float(assortment_gap),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
        }

        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        print(
            f"  Mutation mu={mu:<4} | "
            f"h_adjacent = {mean_adj:.3f} | "
            f"h_random = {mean_rand:.3f} | "
            f"Gap = +{assortment_gap:.3f} | "
            f"t={t_stat:>5.2f}, p-val={p_str}"
        )

    return assortment_results


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def run_all_statistical_analyses():
    reg_summary = run_hamiltonian_regression_analysis()
    phase_summary = run_hamilton_phase_diagram_analysis()
    assort_summary = run_spatial_assortment_analysis()

    full_analysis = {
        "hamiltonian_regression": reg_summary,
        "phase_diagram": phase_summary,
        "spatial_assortment": assort_summary,
    }

    output_path = os.path.join(RESULTS_DIR, "statistical_analysis.json")
    with open(output_path, "w") as f:
        json.dump(full_analysis, f, indent=2)

    print("\n" + "=" * 80)
    print(f"TASK 12 COMPLETE: Full statistical analysis saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_statistical_analyses()
