> Follow-up: implementation, replicated retraining, controls, and regenerated evidence are documented in [the completion report](completion_2026-11-16.md). This recheck remains a historical snapshot from before that work.

# Recheck â€” 16 November 2026

**Verdict: the main implementation bugs from the last review are now fixed in the checked cases. All originally requested corrections are not yet complete.** This report updates the 15 November review; do not treat that older document's resolved bug descriptions as current failures.

## Verified today

- **41 tests pass**, one third-party pytest deprecation warning.
- All project Python files parse successfully.
- Independent slot-reuse reproduction now shows the newborn using its parent's policy, founder ages `[1, 3]`, and mean observed age `2.0`, exactly as expected.
- Injected mimics now receive distinct valid IDs `[0, 1, 2]` in the checked example. Persistent ancestry and explicit birth/death records are present.
- Movement/occupancy consistency passes 1,000 random movement steps.
- Both training entry points now reset at the horizon as well as extinction.
- A one-seed, 100-step benchmark check in memory correctly reports unperformed invasions as null defense and attacked resident-extinction runs as zero defense. Resident survival and successful defense are tracked separately.
- Saved result JSON files, six static figures, and tables have been regenerated. The new noise range and invasion metadata are present. The saved Figure 3 now actually shows the Tit-for-Tat baseline's invasion, rather than the old attention-model pre-invasion extinction trace.
- All four checkpoints and the GIF remain byte-for-byte identical to the repository baseline. The regenerated evaluations use the old trained networks. New training after correcting environment transitions is still required to evaluate learning under the corrected implementation.

## Status of the eight specific problems in the 15 November recheck

| Problem | Status now |
| --- | --- |
| 1. Wrong offspring policy on slot reuse | Fixed and independently reproduced as correct. |
| 2. Founder/death age accounting | Fixed in the reproduced case; dead-agent rewards are accumulated again. Completed and survivor ages are returned separately. Reward totals still aggregate by recyclable slot, so those totals are not per-individual returns. |
| 3. False defense credit | Main defects fixed: untested defense is null and resident extinction gets no credit. The summary `successful_defense_rate` still becomes 0 rather than null when no runs were tested. Console invasion denominators are hardcoded to five even when another seed count is supplied. |
| 4. Figure 3 mislabeled or invented invasion | Correct with the regenerated artifact. Legacy fallback still invents a default model name if metadata is absent. Event metadata labels `invasion_step=40` as one-based although injection happens before transition 41, after 40 transitions. Define this as a state-time boundary or correct the one-based label. |
| 5. Hamilton analysis | The misleading Hamilton-boundary alignment R-squared was removed. The code now explicitly describes a relatedness-only diagnostic and states B/C is unobserved. That is a valid limitation disclosure, but it does not implement the proposal's incentive-response or Hamilton-emergence test. |
| 6. Invalid mimic IDs | Fixed; ancestry and reset fields also added. |
| 7. Missing horizon reset in simple PPO | Fixed in source. |
| 8. Old artifacts | Results, tables, and static figures regenerated; checkpoints and animation still old. |

## Remaining work from the original 26-item audit

1. **Retrain models under corrected transitions and keep training provenance.** Current checkpoints predate occupancy and terminal-transition fixes. The training entry point still produces one default-seed checkpoint per architecture; independent training-seed replication, persisted run configuration, and learning curves remain absent. Evaluation action sampling is not explicitly seeded per run. Missing checkpoint loaders still silently evaluate randomly initialized weights.

2. **Complete the research controls and uncertainty reporting, or revise the project claims.** There are still no promised 95% confidence intervals, matched invasion/no-invasion controls, neural fixed-gating ablation, or randomized-offspring-placement control. The assortment analysis still runs 25 steps, uses radius-3 neighbor pairs rather than strictly adjacent cells, compares against random pairs from the same world, and applies an independent t-test to seed-matched summaries. The regression still pools dependent recipient decisions and does not estimate the proposed before/after-mimic trust shift.

3. **Correct remaining output inaccuracies.** The regenerated Figure 4 still prints `Trend (p = 0.085, n.s.)`; its current saved result for mutation 0.01 is **p = 0.277122**, so the annotation is false. Figure 6 still says standardized coefficients even though predictors are unstandardized and its error bars are one SE, not 95% confidence intervals. Table 2's prose still hardcodes weights about 0.596/0.404 while its new rows are about 0.600/0.400. Figure 1 still truncates learned trajectories at the shortest run. Conditional table/regression interpretations should derive from current values rather than assuming every future result is nonsignificant.

4. **Update documentation and project packaging.** The specification still blocks moves into initially occupied cells whereas code permits simultaneous swaps/chains. Donation productivity is still called strict energy conservation. A README, dependency manifest, exact execution order, and documented actual training budgets are missing. Tracked bytecode remains in Git. The reputation strategy is still named Tit-for-Tat; all residents still share one policy rather than competing lineage policies. The advertised starvation-disable fallback is absent.

## What the regenerated evidence supports

- Phase 2 still ends at population zero in all five seeds.
- All three neural models still have 0% resident survival in the saved Phase 4 benchmarks.
- Phase 3 weights stay near genetic 0.400 and behavioral 0.600; all final populations are zero.
- Current relatedness, behavior, and incentive regression p-values are approximately **0.736, 0.893, and 0.275**. Spatial assortment p-values are approximately **0.277, 0.970, 0.890, and 0.545**. These do not establish the proposed positive findings.
- The new relatedness-only diagnostic has correlation approximately **âˆ’0.9978**, meaning donation probability decreases with relatedness in that particular synthetic probe. Squaring it produces a high value while hiding the negative direction; it is not positive evidence for kin preference or Hamilton's rule.

Negative findings are legitimate. A completed, honest project could report them and narrow its claims; code corrections alone do not guarantee the proposed behavior emerges.

No implementation, checkpoints, or saved outputs were modified by this review. Checks used temporary scripts under `.review_tools`; Figure 3 saving was mocked and the short benchmark ran in memory. Only this review report was added to project documentation.
