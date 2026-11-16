# Completion evidence - 16 November 2026

This report supersedes the outstanding-work status in the earlier dated reviews. Those reviews remain historical records.

Test evidence: 49 passed in 41.51s

Validated 35 newly trained checkpoint files against their SHA256 sidecars. Training budgets: [1024] environment transitions per run, five independent training seeds per condition. Active-agent transitions and full per-update curves are saved per run. These are short-budget experiments, not a convergence demonstration.

Implemented: replicated corrected-environment training; independent evaluation/action seeds; missing-checkpoint errors; 95% intervals; matched invasion/no-invasion controls; trained neural fixed-gating ablation; random offspring-placement control; strict adjacency and seed-paired assortment inference; training-seed cluster regression; before/after mimic trust comparison; separate incentive-conditioned training; regenerated static figures, tables and replay; README/dependencies; bytecode untracking.

## Regenerated observations

Phase 2 final populations: [0, 0, 0, 0, 0]

| Phase 4 model | Resident survival (%) | Tested invasions | Successful defense (%) |
| --- | ---: | ---: | ---: |
| Selfish Baseline | 0.0 | 0 | not tested |
| Hardcoded Tag Trust | 80.0 | 5 | 80.0 |
| Fixed Weights (w^g=0.5) | 100.0 | 5 | 100.0 |
| Reputation Threshold | 100.0 | 5 | 100.0 |
| Simple MLP (No Attention) | 0.0 | 1 | 0.0 |
| PPO + Adaptive Gating | 0.0 | 0 | not tested |
| Gated-Attention (Ours) | 0.0 | 1 | 0.0 |

## Paired controls

Intervals use independent training-seed differences. Tests are exploratory and unadjusted; zero-width mean intervals for identical outcomes are not population-level certainty.

Genetic-weight pre/post difference in differences: -0.0082 (95% CI -0.0100 to -0.0064; n=5).

| Contrast | Metric | Mean difference and 95% CI |
| --- | --- | --- |
| fixed_attack_minus_clean | resident_survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| fixed_attack_minus_clean | final_residents | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| fixed_attack_minus_clean | duration | -6.6000 (95% CI -14.8175 to 1.6175; n=5) |
| fixed_attack_minus_clean | donations | -4.2000 (95% CI -21.8997 to 13.4997; n=5) |
| adaptive_attack_minus_clean | resident_survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| adaptive_attack_minus_clean | final_residents | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| adaptive_attack_minus_clean | duration | -6.6000 (95% CI -21.0429 to 7.8429; n=5) |
| adaptive_attack_minus_clean | donations | -1.0000 (95% CI -20.0950 to 18.0950; n=5) |
| adaptive_minus_fixed_clean | resident_survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| adaptive_minus_fixed_clean | duration | 4.8000 (95% CI -22.5645 to 32.1645; n=5) |
| adaptive_minus_fixed_clean | donations | 18.4000 (95% CI -40.0809 to 76.8809; n=5) |
| trust_shift_difference_in_differences | genetic weight | -0.0082 (95% CI -0.0100 to -0.0064; n=5) |
| phase4_attack_minus_clean_Selfish Baseline | survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Selfish Baseline | carrying_capacity | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Selfish Baseline | lifespan | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Hardcoded Tag Trust | survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Hardcoded Tag Trust | carrying_capacity | -2.8000 (95% CI -6.5662 to 0.9662; n=5) |
| phase4_attack_minus_clean_Hardcoded Tag Trust | lifespan | -2.8000 (95% CI -10.5740 to 4.9740; n=5) |
| phase4_attack_minus_clean_Fixed Weights (w^g=0.5) | survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Fixed Weights (w^g=0.5) | carrying_capacity | 0.4000 (95% CI -1.0157 to 1.8157; n=5) |
| phase4_attack_minus_clean_Fixed Weights (w^g=0.5) | lifespan | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Reputation Threshold | survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Reputation Threshold | carrying_capacity | 0.6000 (95% CI -0.8157 to 2.0157; n=5) |
| phase4_attack_minus_clean_Reputation Threshold | lifespan | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Simple MLP (No Attention) | survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Simple MLP (No Attention) | carrying_capacity | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Simple MLP (No Attention) | lifespan | 4.6000 (95% CI -8.1716 to 17.3716; n=5) |
| phase4_attack_minus_clean_PPO + Adaptive Gating | survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_PPO + Adaptive Gating | carrying_capacity | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_PPO + Adaptive Gating | lifespan | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Gated-Attention (Ours) | survival | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Gated-Attention (Ours) | carrying_capacity | 0.0000 (95% CI 0.0000 to 0.0000; n=5) |
| phase4_attack_minus_clean_Gated-Attention (Ours) | lifespan | 4.6000 (95% CI -8.1716 to 17.3716; n=5) |

## Incentive response

High-minus-low training incentive effect on fixed-input donation probabilities: 0.0843 (95% CI -0.0368 to 0.2053; n=5).

Above-minus-below the theoretical rB=C threshold: -0.0013 (95% CI -0.0028 to 0.0002; n=5).

Independent policies trained at each B/C, paired by seed; fixed synthetic inputs; B/C absent from observations. Training-condition response, not instantaneous incentive sensitivity or proof of Hamilton's rule.

## Spatial controls

| Mutation | Adjacent-minus-random pairs | Paired p | Local-minus-random offspring placement |
| --- | --- | --- | --- |
| 0.01 | 0.0370 (95% CI 0.0002 to 0.0739; n=5) | 0.049260277620193034 | 0.0256 (95% CI 0.0091 to 0.0421; n=5) |
| 0.05 | 0.0662 (95% CI 0.0448 to 0.0876; n=5) | 0.0010124640695394539 | 0.0412 (95% CI 0.0161 to 0.0663; n=5) |
| 0.1 | 0.0439 (95% CI 0.0052 to 0.0826; n=5) | 0.03442622450463244 | 0.0415 (95% CI -0.0256 to 0.1085; n=5) |
| 0.2 | 0.0292 (95% CI -0.0004 to 0.0587; n=5) | 0.05205893025478512 | 0.0145 (95% CI -0.0412 to 0.0702; n=5) |

Matched high reserve exercises reproduction; low-reserve exploratory check had only 0-1 births per seed. This is a diagnostic regime, not a confirmatory test.

The initial low-reserve data are retained in `results/placement_low_reserve_exploratory.json`.

Placement birth counts (local / global placement, by training seed):

- Mutation 0.01: [30, 30, 30, 31, 30] / [30, 30, 30, 30, 30]
- Mutation 0.05: [30, 30, 30, 31, 30] / [30, 30, 30, 30, 30]
- Mutation 0.1: [30, 30, 30, 30, 30] / [30, 30, 30, 30, 30]
- Mutation 0.2: [30, 30, 30, 30, 30] / [30, 30, 30, 30, 30]

## Limits on conclusions

Neural residents share a single policy per run. No competition among separately evolving neural lineage policies is implemented; that proposed claim has been removed from the current scope. Baseline-corrected chromosome similarity is not a validated pedigree relatedness estimate. B/C is unobserved, so incentive-conditioned training effects are not instantaneous incentive perception. Frozen-policy population weight changes can reflect changing recipients and population composition.

Regression uses 5 independent training-seed clusters, with 39202 dependent recipient observations; coefficient estimates (relatedness, behavior, B/C) are [-0.5561486280389478, -0.33586508101566453, -0.06837888813233363] and p-values are [0.04808848318436561, 0.6284048064694769, 0.016720703415198976]. These are rollout associations, not direct incentive perception. Small-cluster inference is exploratory. Assortment averages living states, not equilibrium populations. One evaluation environment per trained policy does not separate training variation from evaluation-world variation.

The results above determine whether effects are positive, negative or uncertain. Donation counts or a narrow synthetic-probe association do not by themselves establish learned cooperation or Hamilton's rule. Longer training, more independent seeds and a prespecified confirmatory design remain follow-up research, not completed positive findings.

## Reproduction and artifacts

See README.md for exact execution order and budgets. `results/artifact_manifest.json` records checksums and installed package versions. Checkpoint sidecars retain source hashes as captured at each run's start; non-training tooling may subsequently change without invalidating trained transitions.
