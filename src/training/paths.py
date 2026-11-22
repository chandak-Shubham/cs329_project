"""Where each study keeps its files.

models/<study>/       trained checkpoints (.pt) and their training records (.json)
results/<study>/      analysis.json, results.md, CSV tables, raw evaluation logs, archived source
plots/<study>/        figures
"""
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
STUDY_A, STUDY_B, STUDY_C = 'study_a_energy', 'study_b_hamilton', 'study_c_robustness'


def study(name):
    results = ROOT/'results'/name
    return SimpleNamespace(name=name, models=ROOT/'models'/name, results=results,
                           evaluations=results/'evaluations', source=results/'source', plots=ROOT/'plots'/name)
