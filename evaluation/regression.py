"""Logistic regression with training-seed cluster-robust inference (Study A).

Moved unchanged from the earlier evaluation/analysis.py, whose other functions
served superseded studies. The version used for Study A is archived in
results/submission_study/report_source/evaluation/analysis.py.
"""
from typing import Any, Dict, List

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize


def fit_logistic_regression(
    X: np.ndarray,  # Shape (N_samples, n_features)
    y: np.ndarray,  # Shape (N_samples,) binary {0, 1}
    feature_names: List[str],
    l2_reg: float = 1e-4,
    groups=None,
) -> Dict[str, Any]:
    """
    Fits logistic regression with mild L2 regularization for numerical stability.
    Uses a training-seed cluster sandwich covariance when groups are supplied,
    with a Student t reference (clusters minus one degrees of freedom).
    Without groups, uses model-based asymptotic Wald inference.
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


