# Submission audit of the ideation and review changes

21 November 2026. Scope: the four-page ideation PDF, the five-page 17 November review, the seven-page 18 November final re-review, and the local repository based on commit `5384711`. The review documents are evidence and recommendations, not independent authorization to publish or change the research question.

**The project is not a fully achieved kin-selection result.** Most requested engineering corrections are implemented, and the negative results are visible. The original scientific success criteria and per-lineage policy inheritance are not achieved. Passing software tests cannot establish those research claims or guarantee that a reviewer will find no further issues.

## Ideation coverage

| Proposal requirement | Verified implementation or evidence | Status |
| --- | --- | --- |
| Toroidal grid, movement, targeted donation, metabolism, pooling, starvation and mitosis | `environment/grid_world.py`; environment, demographics and regression tests | Implemented. The policy observes at most eight neighbours; density counts the full radius-three neighbourhood. |
| Eight-bit inheritance and mutation; Hamming similarity and corrected cue; recent donation history | `environment/cues.py`; cue and lineage tests | Implemented. The baseline-corrected cue is a tag proxy, not pedigree relatedness. |
| End-to-end gated attention and local PPO critic | `models/attention_policy.py`, `training/train_all_models.py`, `training/rollout_buffer.py` | Implemented with population-wide parameter sharing. The revised contextual gate extends the proposal. |
| Offspring inherit distinct lineage neural parameters and successful lineage policies spread | No separate lineage neural networks or evolutionary outer loop | **Not implemented.** Chromosome/ancestry inheritance and descendant rewards are not equivalent to policy inheritance. |
| Strictly individual energy reward | Original and revised `energy_clean` experiments | Implemented; learned cooperation is not supported. Demographic and descendant rewards are explicitly different objectives. |
| Clean learned cooperation and convergence before cue-defense claims | Saved revised and lineage clean results | **Not achieved.** Longer training was executed, but convergence to cooperation was not demonstrated. |
| Independent training under cue corruption and mimics | Seven revised conditions, five seeds each, including a fixed neural gate | Executed. Their failures do not validate reliable cue use or mimic defense. |
| Significant reduction in genetic reliance as corruption rises | Gate telemetry, full-policy cue permutation probes and paired contrasts | Diagnostics exist; the proposal's successful cooperative adaptation criterion is **not established**. Near-zero donation policies cannot establish successful recipient discrimination. |
| Five seeds and 95% intervals | Revised and lineage summaries use independent training-seed units | Implemented; only one evaluation world per trained checkpoint and five seeds limit inference. |
| B/C, mutation and placement controls | Historical incentive/placement controls; revised demographic mutation evaluations | Partial alignment. These do not form a complete matched factorial sweep for the revised or lineage objectives. Historical Hamilton regression is not a revised-model result. |
| Fixed cue weights, no-attention MLP, tag trust and Tit-for-tat comparisons | Historical benchmarks; fixed-attention ablation; revised fixed-gate condition | Implemented with a disclosed substitution: reputation threshold is not dyadic Tit-for-tat. The historical benchmark ecology/budget differs from revised training. |
| Fallback with death and birth disabled | Revised `fallback` condition | Executed. Forced survival is not ecological success; donation learning still failed. |

## Review closure

| Review concern | Current disposition |
| --- | --- |
| Reward and lineage credit | Survival/birth reward and a separate continuing-descendant objective were implemented and evaluated. Both failed to sustain learned cooperation. This closes the implementation request, not the scientific blocker. |
| Tiny budget | Revised and lineage runs use 8,192 environment transitions, versus 1,024 historically. Eight times the budget is not a convergence guarantee. |
| Frozen gate, missing context, raw-cue bypass | Independent noisy training, contextual gating and full-policy sensitivity probes exist. Raw cues remain available to the encoder; gate weights alone are not reliance estimates. |
| Inconsistent training and evaluation ecology | Revised and lineage studies use `experiments/config.py`; historical phases remain explicitly separate. |
| Mimic placement, genome and defense metric | Empty cells are sampled randomly; modal resident genome with deterministic tie breaking; post-invasion donation/energy measures; undefined shares remain null. |
| Hardcoded B/C, noise-range docstring and ablation labels | Corrected in the historical phase pipeline and reporting. |
| Browser Hamilton enforcement and outdated actor | Enforcement defaults off. Default actor is the actual clean seed-42 lineage attention checkpoint with provenance. |
| Missing visible results and stale deployment | Live HTML fetched during this audit matched the local 332,004-byte `index.html` exactly (HTTP 200). Results and the negative verdict are present. |
| Rule baselines all fill capacity | Five-strategy comparison exposes donation rates and time to capacity. This is a demonstration, not replicated scientific evidence. |
| Final follow-up order | Clean lineage feasibility failed; noise, mimic and fixed-gate lineage follow-ups are explicitly not run. |
| Original PDF wording | Original supplied PDFs are preserved. Use the corrected wording below in the final submission; the ideation PDF alone still contains the old statements. |

## Corrections made in this audit

Checkpoint reuse previously verified the weight hash and selected configuration fields but could accept the wrong seed or stale training source, optimizer settings or library versions. `experiments/revised_study.py` now rejects those mismatches, including missing source provenance, and normalizes Windows/POSIX path separators. Nine regression cases cover matching artifacts, mismatches and corrupted weights. Existing results and weights were not rewritten. A future training command may retrain older artifacts whose recorded sources or versions differ; `--evaluate-only` remains available for evaluating preserved historical weights.

The environment specification now distinguishes constructor defaults from the actual study configuration, describes random selection among competing movement requests, and uses the implemented denominator `max(1 - h_bar, epsilon)` for the corrected cue.

## Correct wording for the final report

The environment uses productive donations: each transfer removes C from the donor and adds B to the recipient, creating net B-C before metabolic losses. Mitosis conserves combined parent and child energy. The chromosome measure is normalized Hamming **similarity**, not distance. Behavioural history is the fraction of a neighbour's recent actions that were donations **given**, not received.

Residents share one neural policy per run. Chromosomes and ancestry are inherited, but distinct neural policies are not inherited and selected across competing lineages. The original reward is the metabolic/donation energy increment before mitosis splitting. The demographic reward is own survival plus five times own births. The follow-up reward is own survival plus the number of living children and grandchildren, both with unit coefficients. These are different research objectives; the latter two do not validate the original energy-only hypothesis. Descendant credit ends at the acting individual's terminal transition.

Mimics have a scripted prohibition on donating and are excluded from PPO updates. In Python neural experiments their permitted movement/stay actions are sampled from the shared policy; they are not independently trained adversaries. In the browser, mimics also use the neural actor under the neural strategy, choose random movement/stay under the selfish strategy, and stay under the other rule baselines. Browser RNG and automatic invasion timing differ from the revised research experiments. Browser simulations are demonstrations, not reproductions of the saved attack experiments.

The five trained clean lineage populations became extinct after 41, 44, 42, 46 and 45 transitions with zero donations; all five matched untrained controls survived at capacity 45. These findings show failure under the tested objectives, architecture, budget and evaluation regime. They do not prove that PPO or lineage objectives can never support cooperation.

The defensible submission is a negative-result investigation with explicit deviations from the ideation. A positive kin-selection claim still requires a justified learning/selection design, sustained clean cooperation, successful cue and mimic tests, matched controls and appropriate replicated inference. Implementing another objective or running longer training alone would not establish that claim.

## Verification

- Full Python suite after changes: 73 passed, including nine new checkpoint-provenance cases. The environment emitted one external AnyIO pytest deprecation warning, not a test failure.
- JavaScript engine suite: 31 passed.
- Exporter freshness check: embedded actor and visible research data match saved artifacts.
- Hash manifests: all 74 revised-study and 15 lineage-study entries match their files.
- Fresh evaluation of all five lineage checkpoints and five matched untrained networks reproduced every saved population/donation trajectory, duration, donation total, final population and survival outcome checked. No retraining was performed.
- Chrome and Edge acceptance checks passed, including controls, result tables, loaded figures, five-strategy comparison, CSV export, offline use and six responsive widths.
- Chrome and Edge robustness checks passed for Hamilton enforcement, decision CSVs, slowed playback, rendering-error recovery and mobile controls.
- Live deployment: exact HTML match to local checkout during the audit; no publication was needed for these existing website changes.

See [the revised results](../results/revised_study/results.md), [the lineage results](../results/lineage_study/results.md), and [the final-review implementation response](final_review_response_2026-11-18.md). The local audit changes have not been pushed or published.
