# Strict Evaluation Against the CS329 Project Guidelines

Group 21 — *Emergent Kin Selection and Spatial Assortment in MARL*
Assessed: ideation PDF (4 pages) and repository state on 21 Nov 2026.
Scale: Project Idea Review **10** + Implementation & Report **28** = **38**.

---

## 1. Score summary

| Component | Max | Awarded | Verdict |
| --- | --- | --- | --- |
| Project Idea Review | 10 | **7** | Strong, original idea; formulation errors and an unsound core mechanism |
| Implementation & Report | 28 | **20** | Heavy engineering and honest negative results, but no final report yet and a drifting research question |
| **Total** | **38** | **27** | Reaching 38 requires the fixes in Section 5 |

This is a strict marking. A grader who weights originality and effort heavily could give 30–32. No grader can give 38/38 to the current state, because the final report does not exist yet.

---

## 2. Idea Review (10 marks), item by item

| # | Guideline question | Marks | Strict comment |
| --- | --- | --- | --- |
| 1 | Decision problem and why it is interesting | 1.5 / 1.5 | Clear social dilemma: donation costs C and helps a neighbour by B. Well motivated by Hamilton's rule. |
| 2 | Agents and what each controls | 1.0 / 1.0 | Agents, observations and action space (Eq. 1) are defined. |
| 3 | What is known and unknown | 0.5 / 1.0 | Only one sentence says that agents do not know cue reliability. There is no explicit known/unknown section, although the guidelines ask for one. |
| 4 | Precise question | 1.0 / 1.5 | Three research questions are stated. RQ3 ("under what conditions … consistent with rB > C") is not operationalised: there is no threshold test and no rB − C interaction term. |
| 5 | Algorithm | 0.75 / 1.5 | Gated attention with PPO is specified, but the mechanism that should produce kin selection is wrong (Section 3, error A). Shared versus independent parameters is contradictory. |
| 6 | Evaluation: metrics, baselines, experiments | 1.25 / 1.5 | Good: 5 seeds, 95% CIs, ablations, Tit-for-tat, random-placement control. Missing: every numeric parameter (grid size, B, C, ρ, K, E_mitosis, N_max, μ, b₀, horizon, training budget). |
| 7 | Success criteria and limitations | 0.5 / 1.0 | A fallback is given, which is good. The success criterion ("drop in w^g") is not identifiable, because raw r̃ also enters the encoder (error D). There is no criterion for RQ2 or RQ3. |
| 8 | Uniqueness | 0.5 / 0.5 | Combining kin cues, behavioural cues, a learned gate and mimics is genuinely original. |
| | **Total** | **7 / 10** | |

---

## 3. Mistakes in the ideation PDF (fix all of these in the final report)

### Conceptual errors (these cost the most marks)

**A. The mechanism cannot produce kin selection as written.** The reward is R_i = ΔE_i, so giving always yields −C to the donor. PPO maximises the individual return, which ends when the agent dies. Sharing parameters does **not** make the gradient value a relative's benefit B. The sentence "the shared-parameter policy discovers that feeding valid kin maximizes the proliferation of its own neural network weights" is false: PPO has no term for weight proliferation. Kin selection needs either an inclusive-fitness reward or a selection loop over heritable policies. The repository's own results confirm this. The clean energy-only PPO population went extinct with 0.07 donations per step, whereas untrained ancestors survived with 0.54. Learning destroyed cooperation.

**B. Shared versus independent parameters contradict each other.** The abstract says "independent PPO". Section 3.1 says "agents belonging to the same evolutionary lineage share policy parameters". Section 4.1 says "PPO with shared policy parameters" (one network). These are three different designs.

**C. The gate cannot infer reliability.** Eq. 3 feeds the gate only per-pair [r̃_ij ‖ b_j ‖ E_j]. Cue corruption is a population-level property. A single pair carries no information about whether tags are corrupted, so the gate needs context such as neighbourhood statistics.

**D. The success criterion is not identifiable.** r̃_ij and b_j enter both q_ij and the encoder z_ij (and therefore the policy logits directly). The network can ignore the gate entirely, so a change in w^g does not measure reliance. Use cue-permutation or intervention probes on the full policy.

**E. The regression cannot test Hamilton's rule.** In logit P(give) = β₀ + β₁r̃ + β₂b + β₃(B/C), B/C is constant within a run, so it is identified only across runs. Hamilton's rule is a threshold on the product r·B/C, which requires an r̃ × (B/C) interaction term. "Hamiltonian consistency" is also a misnomer: Hamiltonian is a physics term. Write "Hamilton's-rule consistency".

**F. The Hamilton check is trivial.** Definition 1 assumes B > C. With r = 1, giving is then always favoured. The interesting regime is B/C close to 1/r. State the r̃ values at which rB = C for each B/C tested.

### Formal and notation errors

1. Definition 2 calls h_ij the "normalized Hamming **distance**", but the formula is a **similarity**.
2. Table 1 defines b_ij as "donations **received**", while Definition 2 says the fraction of j's actions that were donations (**given**). The notation also switches between b_ij and b_j.
3. Algorithm 1, line 9: `k ← argmin(M == 0)` returns an **occupied** slot. It should be `argmax(M == 0)` or `argmin(M)`. The code (`np.where(~mask)[0][0]`) is correct; the PDF is wrong.
4. Algorithm 1 does not handle the case where the adjacent spawn cell is occupied.
5. "Energy is strictly conserved" holds only for mitosis. Each donation creates B − C of net energy, so total energy is not conserved. Say so explicitly.
6. Eq. 2 combined with mitosis: if mitosis halves E, then ΔE gives a reward of −E/2 for reproducing, which penalises reproduction. State whether the reward is computed before or after the split. (The code computes it before.)
7. Eq. 4 normalises the softmax over give(j) only. The full action space also contains moves and defect. The way the two heads combine is unspecified.
8. "defect" is never defined (stay put? refuse a request?). In a donation game, defecting simply means not giving.
9. Affordability is unspecified: can an agent with E < C give?
10. The radius ρ, the window K and the prior b₀ appear but are never given values. Ω_i is used both for the observation set and for the density count.
11. Figure 1 has a bidirectional arrow between Action and Policy, and an Observation → Action edge that bypasses the policy.
12. Phase 1 ("selfish baseline") and Phase 2 ("altruist-only environment") do not say how selfish or altruist behaviour is imposed on a learning agent.
13. The "neutral prior b₀" for offspring and the reset of mimic history are unspecified.

### Writing and citation errors

1. "Previous MARL studies … [3]–[8]": [3] is Dawkins (1976), a biology book, not a MARL study.
2. Hamilton [2] is not cited where Hamilton's rule is introduced.
3. Reference [1] is a YouTube video. It is acceptable as motivation but should not appear as a scientific source.
4. The Contribution section uses "Implemented" (past tense) for one member and "Formulate" or "Design" for the others. Use consistent tense.
5. The title claims "Emergent Kin Selection". Within-lifetime learning is not kin selection, which is an evolutionary process over generations. Hedge the title: "Can MARL agents learn kin-biased cooperation?"
6. "Standard centralized critics … are not used because the population size changes" is a weak justification, because attention critics handle variable sets. Give the real reason: individual, decentralised rewards.

---

## 4. Implementation & Report (28 marks)

Basis: the code, tests and results in this repository. The 95 final-study checkpoints exist; the final evaluation and report outputs are not yet generated.

| Guideline item | Max | Awarded | Strict comment |
| --- | --- | --- | --- |
| Clear mathematical/computational formulation | 5 | 3.5 | The environment specification is detailed, but the formulation changed across studies (energy, demographic, descendant rewards; shared, then inherited policies). The final report must state **one** formulation. The PDF errors above are not yet corrected in a submitted document. |
| Implementation of the proposed method | 6 | 5 | Gated attention, PPO, agent pooling, mitosis, mimics, family policy inheritance with survivor selection, and many tests. This is excellent engineering. One deduction: the outer selection loop is a deviation from the proposal and must be justified. |
| Baselines and alternatives | 4 | 3.5 | Selfish, tag trust, fixed threshold, reputation, dyadic Tit-for-tat, fixed gate, MLP, untrained ancestors, evolution-only, no-selection. This is very complete. |
| Experiments that vary meaningful parameters | 5 | 3.5 | 19 one-factor conditions × 5 seeds cover noise, mimics, B/C, μ, placement, LR, entropy and budget. Deductions: the B/C sweep has only 3 levels; the noise and mimic conditions are moot because clean cooperation never formed; the study is one-factor rather than factorial. |
| Quantitative evaluation with suitable metrics | 4 | 2.5 | Seed-level t intervals and Wilson intervals are appropriate. However, the submission study's `results.md` and `analysis.json` **do not exist yet**, and n = 5 with three worlds per seed is thin. |
| Analysis of results and limitations | 4 | 2 | The limitations are honest, but there is **no final report** (`docs/final_report.md` is missing, although the README links to it). Results are spread over 4 separate studies, which a grader will read as scope drift. |
| **Total** | **28** | **20** | |

### Main findings so far (from `results/inheritance_study`)

| Condition | Final residents | Donations / resident step |
| --- | --- | --- |
| Clean energy-only PPO + selection | 0 (extinct in all 5 seeds) | 0.075 |
| No selection (PPO only) | 0 | 0.076 |
| Evolution only (no PPO) | 45 (capacity) | 0.484 |
| Untrained ancestors | 45 | 0.536 |
| Dyadic Tit-for-tat (rule) | 44.4 | 0.648 |
| Public reputation (rule) | 45 | 0.986 |
| Selfish (rule) | 0 | 0 |
| Fallback (no death/birth), trained | 24 (forced) | 0.006 |

This is a strong, clean answer to RQ1. Individually rewarded PPO actively **unlearns** cooperation that selection alone preserves. Make this the headline instead of presenting it as a failure.

---

## 5. What is needed to get as close to 38/38 as possible

1. **Write the final report** (largest deduction). Use one consistent formulation: energy-only reward, inherited family policies, survivor selection. Use the corrected Eq. 2 / Algorithm 1 / Definition 2 from Section 3. Move the demographic and descendant-reward studies to an appendix or remove them.
2. **Reframe the research question** around the result you actually have: *"Does individual-reward PPO preserve, create or destroy cooperation that evolutionary selection on heritable policies sustains, and do cue reliability and B/C change this?"* This is precise, original, answered by data, and fits guideline directions A and C.
3. **Explain why PPO destroys cooperation** (error A). The reward probe already shows that giving changes the donor's reward by −C and the recipient's by +B. Add one figure of donation rate against PPO update for the clean and evolution_only conditions.
4. **Add one inclusive-fitness control.** Train with R_i = ΔE_i + Σ_j r̃_ij · ΔE_j. If it cooperates and the energy-only reward does not, you have directly tested Hamilton's rule, which turns RQ3 from "not answered" into "answered".
5. **Fix the Hamilton regression**: add an r̃ × (B/C) interaction and use at least 5 B/C levels (for example 1.0, 1.5, 2.25, 3.0, 4.5).
6. **Replace gate weight with cue-permutation sensitivity** as the RQ1 reliance measure (already implemented), and state why.
7. **Show assortment** (RQ2): local versus random placement, adjacent Hamming similarity with CI. Run it under evolution_only, where populations survive, so the comparison is not empty.
8. **Fill the "known vs unknown" gap** with one short table: known (B, C, own energy, neighbour cues); unknown (cue reliability, who is a mimic, neighbours' policies).
9. **Repair the README links**: `docs/final_report.md`, `docs/final_study_protocol.md` and the other docs links now point to deleted or missing files.
10. **Finish the submission-study evaluation** so that `results/submission_study/results.md` and `analysis.json` exist, and quote those numbers in the report, not the older inheritance-study numbers.

With items 1–3 done, expect about 32–34/38. With all 10, a 36–38 is realistic. A full 38 still depends on the grader's view of originality and presentation.
