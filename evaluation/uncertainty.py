"""Uncertainty across independent training runs, never across recipient decisions."""
import numpy as np
from scipy import stats


def mean_ci(values):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(x):
        return {"n": 0, "mean": None, "ci95": None}
    mean = float(x.mean())
    half = float(stats.t.ppf(.975, len(x) - 1) * stats.sem(x)) if len(x) > 1 else None
    return {"n": len(x), "mean": mean,
            "ci95": [mean - half, mean + half] if half is not None else None,
            "method": "Student t interval across independent training-seed units"}


def proportion_ci(values):
    x = [v for v in values if v is not None]
    if not x:
        return {"n": 0, "mean": None, "ci95": None}
    n, p, z = len(x), float(np.mean(x)), stats.norm.ppf(.975)
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return {"n": n, "mean": p, "ci95": [float(max(0., center-half)), float(min(1., center+half))], "method": "Wilson 95% interval"}
