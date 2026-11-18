# Final re-review response — 18 November 2026

Both supplied reviews were read in full, including the seven-page final re-review's checklist. They are research feedback; the user's request authorizes the implementation. The review PDFs themselves are preserved. This report supersedes the browser and follow-up status in the 17 November response.

## Pages audit

The initial live audit found that the build for `7e942ed` had already **succeeded**, and the fresh live HTML already contained B=5, 24 founders, the Hamilton checkbox and the post-invasion mimic-share display. Pages is configured for `main`, repository root, with status `built`. The review's stale page observation was not reproducible with a cache-busted fetch. Caching is a possible explanation; no failed build was observed.

Evidence: [the successful build](https://github.com/chandak-Shubham/cs329_project/actions/runs/44151784737) and [the live site](https://chandak-shubham.github.io/cs329_project/). New changes use that existing Pages configuration. Publication validation checks both the pushed commit's build status and the actual live HTML, rather than assuming a successful push updates the site instantly.

## Website changes

- A prominent headline states that the kin-selection hypothesis is not yet supported.
- The initial selected behaviour is **trained PPO contextual attention**. The four scripted strategies are labelled “baseline, not learned”. Hamilton enforcement remains off by default and is visibly labelled as forcing behaviour when enabled.
- The browser evaluates the actual attention actor: contextual cue gate, neighbour/self encoders, two-head attention, movement and targeted-donation heads. It uses the fixed seed-42 clean lineage checkpoint, not a hand-selected best seed. Embedded metadata identifies the source, objective, training budget and SHA256. There is no fabricated donation heuristic or online learning.
- JavaScript action probabilities are compared with PyTorch across zero to eight valid neighbours, including padded and empty neighbourhoods. The tests also verify action masking, replay and checkpoint integrity.
- Static, visible results are generated from saved JSON: the revised survival/donation/mimic table, cue-sensitivity table, both research figures, historical benchmarks labelled 1,024 steps, and historical regression coefficients, intervals and explanatory fit. The lineage follow-up is displayed separately after evaluation. Source artifacts and this report are linked.
- Caveats appear beside their tables: fallback survival is forced; sensitivity of non-donating policies is not successful cooperative adaptation; mimic-share estimates from one or two donation-bearing runs are not findings. Original raw estimates remain linked rather than silently clipped or altered.
- Population capacity, horizon, grid size, founder count and mitosis labels derive from the active engine settings.
- “Compare five fresh runs” runs each behaviour with the same seed and base ecology, showing donation frequency, time to capacity, episode duration and mimic share. This makes differences visible even when three rules all fill the pool. It is explicitly a single-seed demonstration, separate from replicated research.

`python visualizer/export_web_data.py` refreshes both embedded weights and visible tables; `--check` rejects stale content. Research figures are served as repository-relative assets. The complete checkout works offline.

## Research choice: continuing descendant fitness

The new `reward_mode="lineage"` implements

`R_i(t) = 1 * survived_i(t) + 1 * living_children_and_grandchildren_i(t)`.

The two unit coefficients value one individual's continued survival and one descendant's continued survival equally. Depth two includes children and grandchildren, limits distant credit and is an explicit configurable research choice. These are objective hyperparameters, not action rules. The one-off demographic birth bonus is **replaced**, not added again.

Descendants are identified through persistent individual IDs and ancestry records, including through dead intermediate parents. An unrelated agent with an identical chromosome earns no descendant credit. Slot reuse cannot turn a newborn's family into the dead occupant's family. Tests cover these cases. No chromosome similarity, corrected relatedness, spatial distance, donation action, recipient benefit or Hamilton threshold enters this reward.

This explicitly designs a lineage-fitness objective; it does not make that objective emergent. Recipient selection remains learned. There is still one shared neural policy, with no separate inherited lineage networks or evolutionary outer loop. Rewards and value credit end at the individual's terminal transition; later posthumous descendant returns are not propagated. These limitations matter when interpreting a negative result.

## Clean-first experiment

`experiments/lineage_study.py` retains the matched 18×18 ecology, contextual architecture, PPO hyperparameters, five training seeds and 8,192-transition budget. Each run has 128 rollouts; any empty resident rollout skips optimization. Historical and revised artifacts are preserved separately.

Before running noise/mimic conditions, **every** clean training seed must:

1. Retain living residents through the configured 300-transition evaluation.
2. Average at least 0.1 donations per resident-step over its last 100 transitions.
3. Donate in at least half those transitions.

These declared thresholds control which experiments run; they never constrain agent actions or rewards. They are an exploratory feasibility screen, not a statistical test of kin selection. Clean evaluation uses seed `training_seed + 10000`, with a matched untrained-network control. If the screen fails, noisy-cue, mimic and fixed-gate follow-ups are recorded as **not run**, as requested in the review. The code supports those conditional experiments without presenting them as completed evidence.

Executed outcomes are in [the lineage results](../results/lineage_study/results.md), [feasibility records](../results/lineage_study/feasibility.json), [raw analysis](../results/lineage_study/analysis.json) and checkpoint sidecars. The website displays these data directly. Any continued collapse is reported as a negative result under this particular objective, architecture, budget and evaluation regime—not as proof that all lineage-reward approaches must fail.

**Executed result:** all five trained lineages became extinct after 41, 44, 42, 46 and 45 transitions (training seeds 42, 101, 202, 303, 404), with zero donations in clean evaluation. Resident survival was 0% (Wilson 95% interval 0–43.4%). All five matched untrained controls survived at capacity 45; their mean donation rate was 0.5497 per resident-step (95% t interval 0.4966–0.6028). The feasibility screen failed. No noise, mimic or fixed-gate lineage follow-up was run. The five checkpoints each completed exactly 8,192 training transitions; their SHA256 sidecars were verified.

The result is therefore **not a solved cooperation or kin-selection claim**. The requested alternative objective was implemented and tested, and the failure is retained visibly. Future work would need to address lineage-level credit beyond individual death, alternative optimization or evolving lineage policies; those experiments are not claimed here.

## Correct wording for the final report

- Each donation creates net energy B−C. Only mitosis conserves combined parent/child energy. Do not call donation energy strictly conserved.
- Behavioural history `b_j` is the fraction of j's recent actions that were donations **given**, not received.
- Chromosomes and ancestry are inherited; neural policies are shared, not independently inherited and selected per lineage.
- Mimics are **scripted**, non-donating environmental agents, excluded from PPO optimization.
- Demographic coefficients 1 and 5 were exploratory survival and birth incentives, not tuned evidence of cooperation. The new continuing-descendant objective uses unit coefficients and replaces the birth bonus.
- Fixed gating and threshold rules are intentional ablations/baselines. Hamilton filtering is a demonstration constraint. None should be labelled spontaneous learning.

## Reproduction

```powershell
python -m pytest -q
node --test tests/web_simulation.test.cjs
python experiments/lineage_study.py --environment-steps 8192 --workers 3
python evaluation/revised_report.py --study-dir results/lineage_study
python visualizer/export_web_data.py
python visualizer/export_web_data.py --check
python visualizer/check_web_demo.py --browsers chrome msedge
python visualizer/check_web_robustness.py
```

The earlier 35-checkpoint revised study remains under `results/revised_study`; it need not be overwritten or rerun to execute the lineage follow-up. Its regression and rule benchmarks remain clearly historical.

## Final validation

- All **64 Python tests** and **31 JavaScript tests** passed after exporting the lineage checkpoint.
- Chrome and Edge acceptance passed: learned default, research tables and loaded figures, five-strategy comparison, controls, reproducibility, CSV export, extinction, offline use and six responsive widths.
- Chrome and Edge robustness checks passed for Hamilton enforcement, slowed playback, rendering-error recovery and mobile controls.
- All local page links and image paths resolved. The exporter freshness check passed, and every lineage checkpoint's SHA256 and 8,192-transition budget were verified.
- Desktop, results and mobile screenshots were inspected. Screenshots and browser downloads remain in the ignored `.review_tools` directory.
