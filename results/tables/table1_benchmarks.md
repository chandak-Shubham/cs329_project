# Table 1: Master Benchmark Evaluation (Phase 4 under Genetic Mimicry, 5 Training Seeds)
| Model / Strategy | Final Population | Resident Survival (%) | Episode Duration (Steps) | Mimic Defense Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Selfish Baseline** | $0.0 \pm 0.0$ | $0.0\%$ | $27.4 \pm 1.0$ | N/A (attack not performed) |
| **Hardcoded Tag Trust** | $6.0 \pm 3.6$ | $80.0\%$ | $253.8 \pm 92.4$ | $80.0\%$ |
| **Fixed Weights (w^g=0.5)** | $16.4 \pm 2.8$ | $100.0\%$ | $300.0 \pm 0.0$ | $100.0\%$ |
| **Reputation Threshold** | $17.2 \pm 1.9$ | $100.0\%$ | $300.0 \pm 0.0$ | $100.0\%$ |
| **Simple MLP (No Attention)** | $0.0 \pm 0.0$ | $0.0\%$ | $41.2 \pm 13.1$ | $0.0\%$ |
| **PPO + Adaptive Gating** | $0.0 \pm 0.0$ | $0.0\%$ | $33.0 \pm 2.5$ | N/A (attack not performed) |
| **Gated-Attention (Ours)** | $0.0 \pm 0.0$ | $0.0\%$ | $41.8 \pm 13.2$ | $0.0\%$ |

95% intervals across training-seed units (population: Student t; survival: Wilson):
- Selfish Baseline: final population [0.0, 0.0]; survival proportion [0.0, 0.43448246478317476]; defense success None
- Hardcoded Tag Trust: final population [0.9563369964693829, 11.043663003530618]; survival proportion [0.37553462976252533, 0.9637758913675698]; defense success [0.37553462976252533, 0.9637758913675698]
- Fixed Weights (w^g=0.5): final population [12.512976852723082, 20.287023147276916]; survival proportion [0.5655175352168251, 1.0]; defense success [0.5655175352168251, 1.0]
- Reputation Threshold: final population [14.508136597662094, 19.891863402337904]; survival proportion [0.5655175352168251, 1.0]; defense success [0.5655175352168251, 1.0]
- Simple MLP (No Attention): final population [0.0, 0.0]; survival proportion [0.0, 0.43448246478317476]; defense success [0.0, 0.7934506856227626]
- PPO + Adaptive Gating: final population [0.0, 0.0]; survival proportion [0.0, 0.43448246478317476]; defense success None
- Gated-Attention (Ours): final population [0.0, 0.0]; survival proportion [0.0, 0.43448246478317476]; defense success [0.0, 0.7934506856227626]

*Evaluation regime: Non-saturated stress environment (B/C = 1.5, c_base = 0.8, 12 genetic mimics). Reported over 5 independent training seeds paired with separate evaluation seeds with 300 maximum steps per run.*