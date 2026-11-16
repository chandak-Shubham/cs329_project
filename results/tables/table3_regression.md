# Table 3: Hamiltonian Consistency Logistic Regression Analysis
Model: $\text{logit}(P(\text{give})) = \beta_0 + \beta_1 \tilde{r}_{ij} + \beta_2 b_j + \beta_3 (B/C)$

| Predictor Variable | Coefficient ($\beta$) | Cluster Standard Error | $t$-Statistic (df=4) | $p$-Value |
| :--- | :---: | :---: | :---: | :---: |
| **Intercept** | $-3.7712$ | $0.1067$ | $-35.331$ | $< 0.001$ |
| **Relatedness (r_tilde)** | $-0.5561$ | $0.1976$ | $-2.815$ | $0.0481$ |
| **Behavior (b)** | $-0.3359$ | $0.6418$ | $-0.523$ | $0.6284$ |
| **Incentive (B/C)** | $-0.0684$ | $0.0173$ | $-3.957$ | $0.0167$ |

*McFadden's Pseudo-$R^2 = 0.0020$ (Sample Size: $N = 39,202$)*

*Exploratory effects at unadjusted p < .05: Relatedness (r_tilde), Incentive (B/C). training-seed cluster robust sandwich covariance with t reference; 5 independent clusters. Coefficients are unstandardized. B/C is unobserved; this regression alone does not establish Hamilton's rule.*

Training-seed cluster 95% intervals:
- Intercept: [-4.0675, -3.4748]
- Relatedness (r_tilde): [-1.1047, -0.0076]
- Behavior (b): [-2.1178, 1.4461]
- Incentive (B/C): [-0.1164, -0.0204]