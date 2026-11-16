# Follow-up review — 15 November 2026

Verdict: the changes are useful, but they are not all complete or fully correct. The updated test suite passes; additional checks still reproduce errors in policy inheritance, lifetime accounting, and mimic identity. The original research-design and reporting gaps largely remain.

This is a review, not an implementation change. Source files, trained models, and experiment outputs were not reverted or regenerated during this check. A temporary verification script was added under `.review_tools`.

## Verification performed

- Compared current changes with the previous review and repository baseline.
- Ran all tests: **38 passed**, with one third-party pytest deprecation warning.
- Parsed every project Python source/test file successfully.
- Ran 1,000 environment steps with random movements, collisions, and full population-grid consistency checks: passed.
- Tested a founder death followed by immediate slot reuse using deterministic policy stubs to isolate bookkeeping.
- Checked IDs of injected mimics.
- Compared saved result files, tables, checkpoints, and figures with the repository baseline, accounting for Windows line endings.

## Remaining correctness problems

### 1. Slot reuse still gives offspring the previous occupant's policy

`experiments/cooperation_feasibility.py:64` assigns inheritance only when `slot_policy_map[i] is None`. A used slot never returns to None. If a reciprocal parent's child occupies a dead defector's slot, it keeps the defector policy.

Reproduction: founder UID 0 used the defect policy and died in step 1; founder UID 1 used the reciprocal policy and produced child UID 2 in slot 0. The runner continued assigning action 0 from the defect stub to child UID 2, rather than action 5 from its parent's reciprocal stub. Policy stubs were used only inside the temporary check, not edited into project code.

Fix assignment by individual UID or explicit birth events, not by whether a tensor slot has ever had a policy.

### 2. Lifespan accounting still reads overwritten state and confuses founders with descendants

`experiments/cooperation_feasibility.py:88` reads `env.age[i]` after reproduction may have reset that slot's age to zero. Lines 89–95 also track a founder using its slot, so descendants can overwrite the founder's lifespan. Final-step rewards for agents that die are skipped because reward accumulation now checks the post-step active mask.

In the same three-step check, founder lifespans were reported as `[3, 3]` instead of `[1, 3]`. Mean lifespan was `1.6667` instead of `2.0` for individuals with observed ages `[1, 2, 3]`. Capture each pre-step individual's ID, age, and reward, and persist death records before reusing slots. Distinguish censored survivor ages from completed lifetimes when reporting biological lifespans.

### 3. Mimic defense remains an incomplete metric

`experiments/run_phases.py:388–393` now avoids assigning 100% when no invasion happened, but substitutes zero rather than an explicit not-evaluated value. That mixes untested cases with tested failures in the mean. If both residents and mimics die after an actual attack, the formula still reports 100% defense because parasite count is zero. The `survived` flag measures any surviving population rather than surviving residents, and injection counts are not persisted for every model/seed.

Record actual attack time/count and resident survival per run. Report parasite elimination separately from successful resident defense. Do not count a colony that died with its invaders as successful defense.

### 4. Figure 3 is still unsafe to regenerate from existing data

`evaluation/generate_figures.py:136–155` still reads `telemetry_phase4_seed42`, but the new experiment code changes the meaning of that key from attention-model telemetry to Tit-for-Tat telemetry. The JSON currently on disk has the old attention-model data, with no `faced_invasion` field. Regenerating now would label an attention-model extinction trace as Tit-for-Tat and assert an invasion occurred.

The invasion count and step remain hardcoded. The loop injects when `t == 40`, after 40 transitions and before transition 41, whereas plotted trajectories are indexed starting at 1. Record and use the chosen timestep convention. Check an explicit model name and actual invasion event in the artifact instead of inferring them from a legacy key. Suppress defense interpretation if invasion did not happen. Existing saved figures also still need regeneration.

### 5. Hamilton analysis has not been fixed

`evaluation/analysis.py:211` still assigns B/C to `bc` without using it in the policy observation or selecting a differently conditioned/trained model. Changing the B/C axis still cannot change the queried policy at a fixed relatedness. The saved diagram's duplicated rows and the associated R-squared interpretation remain unresolved. The regression and assortment analysis implementations are unchanged as well.

### 6. Mimics do not receive unique IDs

The new unique-ID fields are correctly assigned for founders and births, but `environment/grid_world.py:182–213` does not initialize them during mimic injection. A one-founder world followed by two injected mimics had active IDs `[0, -1, -1]`. New entities therefore do not all have unique valid identities, and related ancestry fields may retain stale slot data. Persistent ancestry records are also absent.

### 7. Time-limit handling was fixed in only one training entry point

The unified trainer resets on extinction or all done flags. `training/train_simple_ppo.py:155` still resets only on extinction and can continue stepping past the configured horizon. Explicit truncation handling and the intended bootstrap convention should be consistent across both entry points.

### 8. Saved evidence does not reflect the corrected implementation

Saved noise-sweep keys remain `0.0, 0.2, 0.4, 0.6, 0.8`, whereas the revised runner uses values through 0.5. Phase 4 JSON still uses the old telemetry schema and contains the old 100% defense summaries. Checkpoints and figures are unchanged. The current source edits alone do not establish corrected training behavior or experimental outcomes.

## Status of every original review item

| Original item | Current status |
| --- | --- |
| 1. Movement occupancy | Code fix verified, including 1,000 stress steps. Documentation still forbids entering initially occupied cells while implementation/tests allow swaps. |
| 2. Same-step death and slot reuse in PPO | Fixed at the environment terminal-mask level; the new regression test passes. |
| 3. Mimic defense | Partial; no-attack cases become zero, extinction after attack still looks successful. |
| 4. Figure 3 invented invasion | Partial; constants/title changed, but old data can now be mislabeled as a different model and event validation is absent. |
| 5. Hamilton B/C diagram | Not fixed. |
| 6. Feasibility birth handling/lifespans | Initial capacity indexing fixed; reused-slot inheritance and identity/lifetime accounting remain incorrect. |
| 7. Padded trust telemetry | Fixed in phase runners and visualizer; outputs need regeneration. |
| 8. Nonmonotonic genetic-noise sweep | Default sweep restricted to 0–0.5, addressing the reported range issue. Saved data and older descriptive text still use the old range. |
| 9. Stale invasion observations | Fixed in benchmark runner and visualizer. |
| 10. Training time limits | Fixed in unified trainer only; simple PPO entry point remains unchanged. |
| 11. Hardcoded affordability | Environment-supplied masks now used on normal observation paths. Fixed for the reviewed default neighbor/action configuration; fallback paths still assume 2/60. |
| 12. Lineage identities | Partial; founder/birth IDs added, mimic IDs and persistent ancestry unresolved. |
| 13. Positive outcomes not established | No new validated result set; original negative outcomes cannot be replaced by assumptions about the fixes. |
| 14. Independent training seeds/evaluation sampling | Not fixed; one checkpoint per architecture and unseeded evaluation action sampling remain. |
| 15. Promised 95% confidence intervals | Not implemented; SD/SE remain in outputs. |
| 16. Changed environment conditions/confounded comparisons | Not resolved; matching-condition controls remain absent. |
| 17. Spatial assortment control/equilibrium/statistics | Not fixed; 25 steps, radius-3 pairs, random-pair baseline, and independent t-test remain. |
| 18. Regression dependence and mimic trust-shift comparison | Not fixed; no before/after-mimic coefficient comparison or dependence-aware analysis. |
| 19. Neural fixed-weight ablation and Tit-for-Tat naming | Not fixed; comparator is a handcoded threshold rule, and reciprocal strategy uses overall donation reputation. |
| 20. Lineage-specific policy/evolutionary interpretation | Not resolved; all residents still share one policy. |
| 21. Demographic fallback | Not implemented. |
| 22. Ignore rules/cache tracking | Bare wildcard removed and temporary tools ignored. Existing tracked bytecode remains; .gitignore and new regression test are still untracked. |
| 23. Dependencies/run documentation/checkpoint provenance | Still absent; experiment loaders still silently accept missing checkpoints by using initialized weights. |
| 24. Exact budgets and persisted convergence histories | Not fixed. |
| 25. Hardcoded annotations/trajectory truncation/layout | Mostly unchanged; Figure 3 changes are incomplete. |
| 26. Document terminology and conservation wording | Not corrected in the supplied specification; no revised proposal was provided for this check. |

The passing tests establish that the added regression cases work; they do not validate all 26 findings. Further work is needed before the project can be described as fully corrected. Prioritize identity-aware feasibility accounting, invasion metrics/event metadata, Hamilton analysis, and consistent training termination; then complete the experimental controls, independent seeds, uncertainty reporting, and artifact regeneration.
