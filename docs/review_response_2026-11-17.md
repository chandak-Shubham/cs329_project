> Historical 17 November response. The [18 November follow-up](final_review_response_2026-11-18.md) supersedes its browser-model and future-work status.

# Response to the CS329 Group 21 review of 17 November 2026

Source: the five-page review supplied by the user, “CS329 Group 21 – Project Review Emergent Kin Selection in MARL.pdf”, attributed to @Preet. The document was read in full. Its suggestions are research feedback; the user's request authorizes the repository changes. The original review PDF is preserved.

## Research decision

We select review option **(b), individual survival and reproduction rewards**, and retain energy-only training as a control. The revised reward is

`R_i = 1 × [the individual survives this transition] + 5 × [births to that individual in this transition]`.

The parent gets the reproduction bonus. A dying individual's recycled slot gets no survival or newborn bonus. No recipient benefit, chromosome similarity, Hamilton threshold, or direct donation bonus enters this reward. This makes survival and reproduction training objectives, but does not guarantee that PPO learns reciprocity. It changes the original energy-only research question.

All neural residents still share one policy per run. This project therefore tests cooperation and cue-dependent decisions under parameter sharing, **not the selection of competing lineage-specific neural weights**. Implementing an evolutionary outer loop would be a different experiment. A favourable outcome here would not establish that original claim.

## Revised comparison (exploratory)

The new entry point is `experiments/revised_study.py`. Historical phase scripts and artifacts are separate.

| Condition | Reward | Training bit-flip noise | Training invasion | Demographics |
| --- | --- | --- | --- | --- |
| energy_clean | Individual energy change | 0 | None | Enabled |
| demographic_clean | Survival + reproduction | 0 | None | Enabled |
| demographic_noise025 | Survival + reproduction | 0.25 | None | Enabled |
| demographic_noise050 | Survival + reproduction | 0.5 | None | Enabled |
| demographic_noise_mimics | Survival + reproduction | 0.5 | Up to 6 at state 10 | Enabled |
| fixed_noise_mimics | Survival + reproduction, same contextual attention architecture with gate fixed at 0.5 | 0.5 | Up to 6 at state 10 | Enabled |
| fallback | Individual energy change | 0 | None | Starvation and mitosis disabled |

Five training seeds: 42, 101, 202, 303, 404. The executed budget is 8,192 environment transitions per run, 128 rollouts of length 64, up to 128 PPO updates, three epochs per nonempty update, minibatch 64, learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clipping 0.2, value coefficient 0.5, entropy coefficient 0.01. This is eight times the historical budget, still not a claim of convergence. Curves, actual resident-agent samples, checkpoint hashes and configurations are saved. Mimics are scripted agents and excluded from PPO samples. A rollout containing only mimics has no resident training samples: no optimizer step is performed, and loss is null. Two early sidecars originally logged these absent losses as NaN; their documented `telemetry_corrections` replace those entries with null, without changing checkpoints or trajectories.

The world is defined once in `experiments/config.py`: 18 × 18, 24 founders, capacity 45, energy 30, mitosis threshold 60, B/C = 5/2, basal cost 0.6, density cost 0.08, mutation 0.05, 300-transition horizon. Training and primary evaluation share this ecology. Noise, attack, reward and demographic switches vary only in their named conditions. The original Phase 2–4 ecology shifts remain explicitly historical out-of-distribution tests.

Each checkpoint has matched no-attack and attack evaluations at seed `training_seed + 10000`. The attack occurs after ten transitions, before decision eleven, with observations refreshed immediately. Both arms start from the same world and action RNG seed; post-intervention paths can diverge. An untrained network initialized with the same seed is evaluated as an additional baseline. The demographic-clean policy is evaluated at mutation rates 0, 0.05, 0.1 and 0.2; these are frozen-policy mutation robustness tests, not separately mutation-trained policies.

## Reliance and exploitation measurements

The contextual gate observes the focal neighbour's three cues plus the masked neighbourhood means and standard deviations of those cues. Padding never contributes. This is neighbourhood context, not access to the true noise parameter or a validated reliability estimator.

The encoder still sees raw genetic and behavioural cues. Consequently, **gate weights alone are never called reliance**. We permute one cue over valid neighbours and measure the total variation distance between the original and perturbed full action distributions. This includes raw-cue pathways. A separate fixed synthetic probe uses exactly the same input and permutation arrays for every checkpoint, separating input sensitivity from changes in population composition. These interventions can create unrealistic cue combinations; they are exploratory attribution diagnostics, not causal estimates of cue reliability. Zero sensitivity on collapsed near-deterministic policies is not evidence of successful trust adaptation.

Primary exploitation metrics are actual donations and energy transferred to mimic recipients, and the mimic share of all donations following a performed invasion. A zero denominator or unperformed invasion yields null. Attack exposure and injection count are saved. A second share conditions the denominator on transitions with living mimics, preventing later mimic-free donations from diluting exploitation. Resident births are also separated from total births. A low donation share caused by never donating is not successful cooperative discrimination: interpret it alongside resident survival, reproduction and donation frequency. Extinction of unfed mimics is a secondary demographic outcome, not proof of defense.

Means use 95% Student t intervals across training-seed units. Binary survival uses Wilson intervals. Matched effects use per-seed differences. One evaluation world per checkpoint and five independent training seeds provide exploratory precision only; recipient rows and repeated probes are not independent replicates. The fallback's disabled deaths make survival uninterpretable as ecological performance.

## Review checklist

| Review concern | Change / disposition |
| --- | --- |
| Reward removes cooperation; inheritance claim unsupported | Demographic objective with an energy-only control; no lineage-policy evolutionary claim. |
| Tiny budget | 8× longer replicated runs, actual budgets and complete curves, configurable further training. Convergence is not asserted. |
| Frozen clean-trained gate does not establish adaptation | Separate noise-trained and noise/mimic-trained policies; contextual gate; full-policy attribution and common-input probes. |
| Training/evaluation worlds differ | Shared configuration in the revised study. Historical shifts explicitly labelled. |
| Mimic extinction mistaken for defense | Actual recipient-level donation/energy accounting and post-invasion donation share, including null cases and attack exposure. |
| Negative regression | README opens with the unsupported hypothesis, negative coefficient and low explanatory fit. Historical regression is preserved. |
| Ordered mimic placement and conflicting copied genome | Python and browser both sample empty cells randomly and copy the modal resident genome with a deterministic smallest-genome tie break. |
| Hardcoded Phase 2 B/C and outdated noise range | Decision records use `env.B / env.C`; Phase 3 docstring states 0–0.5 and frozen-policy scope. |
| Confounded fixed-weight label | Historical Table 1 identifies the rule-based threshold comparator. Revised study includes the same neural architecture with its gate fixed. |
| Demographic fallback unused | A separately trained and evaluated fallback condition disables starvation and mitosis. |
| Browser forces Hamilton consistency | Enforcement defaults off; visible labelled toggle, clear mode explanation, decision-time CSV mode. |
| Browser model/ecology misleading | Historical MLP explicitly named; default ecology matches its training; new attention models are not claimed to run in JavaScript. |
| Saturated B/C=3 demo | Default B/C=2.5 and matched training density; stress controls remain available. Outcomes are not forced to separate strategies. |
| Mutation sweep absent from main experiments | Revised evaluation sweeps mutation using a fixed trained checkpoint, separately from spatial assortment. |
| PDF energy/history terminology | Corrected definitions below and in README; the ideation PDF was not supplied for editing. |

## Corrected report terminology

- `h_ij` is chromosome Hamming **similarity**, the fraction of matching bits.
- `r_tilde` is population-baseline-corrected similarity, not a validated pedigree kin coefficient.
- `b_j` is the fraction of j's recent actions that were donations **given**, not received.
- A donation subtracts C from the donor and adds B to the recipient: total energy changes by B−C. Describe this as productive energy conversion, not strict conservation.
- Mitosis splits parent energy between parent and child and conserves their combined energy.
- Reputation Threshold uses a neighbour's recent donation frequency; it is not dyadic Tit-for-Tat.
- Offspring inherit chromosomes and persistent ancestry; all neural residents use the same shared policy. No per-lineage neural policy inheritance or evolutionary competition is measured.

## Executed evidence

The generated [results table](../results/revised_study/results.md), [raw analysis](../results/revised_study/analysis.json), checkpoint sidecars, figures and hash manifest contain the executed outcomes. Original 1,024-step results remain historical. Validation evidence and final interpretation are recorded below after the run completes.

Validation: all 59 Python tests and 30 Node engine tests passed. Chrome passed the acceptance suite at six responsive widths; Chrome and Edge passed the robustness suite, including slow playback and recovery from an injected rendering fault. Desktop and mobile screenshots were inspected. The embedded historical MLP still matches its checkpoint hash and exported tensors. The end-to-end revised analysis, figure generation and manifest pipeline also passed a separate one-seed smoke run; these temporary smoke results are not the replicated study.

The neighbour lookup was vectorized to make the longer study practical. A regression test compares its complete distance-then-slot ordering with the original grid scan across five training worlds. Transition and browser/Python parity tests pass. Checkpoint source hashes retain the actual source snapshot used by each run, including runs that began before this equivalent optimization.
