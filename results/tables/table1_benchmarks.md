# Table 1: Master Benchmark Evaluation (Phase 4 under Genetic Mimicry, 5 Seeds)
| Model / Strategy | Equilibrium Pop ($N^*$) | Survival Rate (%) | Mean Lifespan (Steps) | Mimic Defense Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Selfish Baseline** | $0.0 \pm 0.0$ | $0.0\%$ | $28.4 \pm 1.0$ | $100.0\%$ |
| **Hardcoded Tag Trust** | $5.2 \pm 4.3$ | $60.0\%$ | $235.6 \pm 79.2$ | $100.0\%$ |
| **Fixed Weights (w^g=0.5)** | $14.6 \pm 2.6$ | $100.0\%$ | $300.0 \pm 0.0$ | $100.0\%$ |
| **Tit-for-Tat** | $17.0 \pm 3.4$ | $100.0\%$ | $300.0 \pm 0.0$ | $100.0\%$ |
| **Simple MLP (No Attention)** | $0.0 \pm 0.0$ | $0.0\%$ | $67.8 \pm 2.6$ | $100.0\%$ |
| **PPO + Adaptive Gating** | $0.0 \pm 0.0$ | $0.0\%$ | $68.6 \pm 1.4$ | $100.0\%$ |
| **Gated-Attention (Ours)** | $0.0 \pm 0.0$ | $0.0\%$ | $36.0 \pm 1.5$ | $100.0\%$ |

*Evaluation regime: Non-saturated stress environment (B/C = 1.5, c_base = 0.8, 12 genetic mimics). Reported over 5 independent seeds with 300 maximum steps per run.*