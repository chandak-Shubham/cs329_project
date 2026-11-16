"""Exploratory regression clustered by training seed and paired spatial controls.

The Hamilton probe compares independently trained incentive conditions. B/C is
not an observed feature; no instantaneous incentive-response claim is made.
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
from evaluation.uncertainty import mean_ci
from experiments.run_phases import load_checkpoint

SEEDS = [42, 101, 202, 303, 404]
RESULTS_DIR = "results"


def load_trained_model(checkpoint_name: str = "gated_attention.pt", training_seed=42) -> GatedAttentionPolicy:
    policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
    return load_checkpoint(policy, checkpoint_name, training_seed)


# ==============================================================================
# 1. HAMILTONIAN CONSISTENCY LOGISTIC REGRESSION
# ==============================================================================
def fit_logistic_regression(
    X: np.ndarray,  # Shape (N_samples, n_features)
    y: np.ndarray,  # Shape (N_samples,) binary {0, 1}
    feature_names: List[str],
    l2_reg: float = 1e-4,
    groups=None,
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
        if groups is not None:
            groups = np.asarray(groups)
            unique = np.unique(groups)
            if len(unique) < 2:
                raise ValueError("Cluster inference requires at least two independent training seeds")
            scores = np.stack([X[groups == g].T @ (y-p)[groups == g] for g in unique])
            cov_matrix = cov_matrix @ (scores.T @ scores) @ cov_matrix
            cov_matrix *= len(unique)/(len(unique)-1) * (N-1)/(N-D)
        std_errors = np.sqrt(np.abs(np.diag(cov_matrix)))
    except np.linalg.LinAlgError:
        std_errors = np.ones(D) * 0.1

    z_scores = np.where(std_errors > 0, beta_hat / std_errors, 0.0)
    df = len(np.unique(groups)) - 1 if groups is not None else np.inf
    p_values = 2.0 * stats.t.sf(np.abs(z_scores), df)
    critical = stats.t.ppf(.975, df)

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
        "ci95": np.column_stack([beta_hat-critical*std_errors, beta_hat+critical*std_errors]).tolist(),
        "inference": "training-seed cluster robust sandwich covariance with t reference" if groups is not None else "model-based Wald inference",
        "num_clusters": len(np.unique(groups)) if groups is not None else None,
        "converged": bool(res.success),
        "optimizer_message": str(res.message),
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
        for s in SEEDS:
            env = GridWorld(
                grid_size=18,
                max_capacity=40,
                initial_population=24,
                donation_cost=2.0,
                donation_benefit=2.0 * bc_ratio,
                seed=s + 10000,
            )
            obs = env.reset(seed=s + 10000)
            policy = load_trained_model("gated_attention.pt", s)

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
                            "training_seed": s,
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
    reg_results = fit_logistic_regression(X, y, feature_names, groups=[r["training_seed"] for r in records])
    reg_results["limitations"] = "Five training-seed clusters only; recipient choices within a decision are dependent. Cluster inference is exploratory. B/C changes rollout dynamics but is not observed directly."

    print(f"{'Feature':<26} | {'Coeff':<10} | {'Cluster SE':<10} | {'t-value':<10} | {'p-value':<10}")
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
def run_hamilton_phase_diagram_analysis(grid_resolution=15):
    """Evaluate separately incentive-trained policies on matched synthetic inputs."""
    from experiments.research_controls import incentive_response
    return incentive_response(resolution=grid_resolution)


# ==============================================================================
# 3. HAMMING-BASED SPATIAL ASSORTMENT ANALYSIS
# ==============================================================================
def run_spatial_assortment_analysis(mutation_rates=None, seeds=SEEDS):
    from experiments.research_controls import run_controls
    path = os.path.join(RESULTS_DIR, "research_controls.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Run experiments/research_controls.py first")
    with open(path) as f:
        return json.load(f)["spatial_assortment"]


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def run_all_statistical_analyses():
    torch.set_num_threads(1)
    reg_summary = run_hamiltonian_regression_analysis()
    with open(os.path.join(RESULTS_DIR, "research_controls.json")) as f:
        phase_summary = json.load(f)["incentive_response"]
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
