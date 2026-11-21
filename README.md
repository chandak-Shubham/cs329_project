# Investigating Kin Selection in Multi-Agent Reinforcement Learning

CS329 | Group 21 | IIT Gandhinagar

## Final submission study

The current submission investigates whether individual energy-only PPO, inherited family policies and survivor selection sustain cooperation and reliable recipient choice. The report distinguishes ecological survival, learning, marker preference and evidence for kin selection.

- [Final report](docs/final_report.md), [printable PDF](docs/final_report.pdf), and [readable HTML](docs/final_report.html).
- [Complete experimental protocol](docs/final_study_protocol.md).
- [All conditions and confidence intervals](results/submission_study/results.md), [raw evaluations](results/submission_study/analysis.json), and [source integrity and optimizer diagnostics](results/submission_study/submission_diagnostics.json).

The complete study has 19 conditions and five training seeds per condition, with three fresh evaluation worlds per checkpoint. Matched runs use 2,048 environment transitions; the longer-training control uses 8,192. Conditions cover neural ablations, cue corruption, mimics, B/C, mutation, placement, selection, static demographics, learning rate, entropy and training duration. All conditions execute even if clean feasibility fails. These are finite-budget exploratory experiments, not a convergence guarantee or proof of kin selection.

```powershell
python -m pip install -r requirements.txt
python experiments/inheritance_study.py --complete --steps 2048 --workers 3 --worlds 3 --out results/submission_study
python evaluation/inheritance_report.py --study-dir results/submission_study
python evaluation/final_report.py --study-dir results/submission_study
python visualizer/export_web_data.py
python -m pytest -q
node --test tests/web_simulation.test.cjs
python visualizer/export_web_data.py --check
```

Matching completed checkpoints are verified and reused. Source files, settings, versions or budgets that differ cause an explicit error; use a new `--out` directory for a changed experiment. The exact training source is saved under `results/submission_study/source`. Report-generation code and exact package versions are archived with the results. Independent units are the five training seeds; evaluation worlds and family networks do not inflate that sample size.

Optional HTML/PDF regeneration requires an installed Edge browser:

```powershell
python -m pip install -r requirements-report.txt
python evaluation/export_submission.py
```

The browser's research section links the complete study. Its live actor remains the earlier shared-policy checkpoint; it does not execute the new multi-policy experiment. Earlier experiments below are retained as historical evidence and use distinct objectives or protocols.

## Earlier studies and scientific status

**The kin-selection hypothesis is not yet supported.** In the historical five-seed, 1,024-step study, every learned Phase 2 population became extinct. The historical regression estimated a negative relatedness coefficient (-0.556) and pseudo-R-squared around 0.002. Those observations do not demonstrate learned kin preference or adaptive cue reliability.

The [21 November submission audit](docs/submission_audit_2026-11-21.md) records the gaps found before the latest implementation. The separate [inherited-policy experiment](docs/inherited_policy_protocol.md) now implements independent founder networks, family policy inheritance, energy-only PPO and survivor-based selection. This closes the inheritance implementation gap; it does not by itself validate the hypothesis.

The 17 November review response adds a separate, reproducible study with an individual survival/reproduction objective, contextual gating, independently trained noise conditions, matched ecological settings, mimic donation-share measurements, a fixed neural gate, and the demographic fallback. Changing the reward changes the research question. That study uses a shared policy and does not implement competing, inherited lineage policies.

See [the review response and protocol](docs/review_response_2026-11-17.md), [executed revised results](results/revised_study/results.md), and [raw revised results](results/revised_study/analysis.json). Historical results remain separate; their tables and figures must not be presented as outputs of the revised training.

The [18 November final-review response](docs/final_review_response_2026-11-18.md) adds a continuing descendant-fitness objective and a clean-first feasibility screen. In its [separate results](results/lineage_study/results.md), all five trained populations became extinct after 41–46 transitions with zero donations; all five matched untrained controls survived. The failed screen withheld noise/mimic follow-ups. The website displays these results alongside the earlier evidence, figures and caveats, and defaults to the actual trained attention policy.

To reproduce this follow-up without overwriting the revised study:

```powershell
python experiments/lineage_study.py --environment-steps 8192 --workers 3
python evaluation/revised_report.py --study-dir results/lineage_study
python visualizer/export_web_data.py
```

## Run the inherited-policy study

```powershell
python experiments/inheritance_study.py --steps 8192 --workers 3 --out results/inheritance_reproduction
python evaluation/inheritance_report.py --study-dir results/inheritance_reproduction
```

Each founder has a separate network and optimizer; children inherit their family policy by persistent ID. PPO uses the original energy-only reward. At completed episode horizons, the next population samples actual surviving residents and inherits their genomes, policy and optimizer states. Extinction produces no selection winner. This explicit outer loop is documented in the [protocol](docs/inherited_policy_protocol.md).

Five independent training seeds each receive three fresh evaluation worlds. Worlds are averaged within seeds before inference. The clean screen controls further noise/mimic experiments; selection-only, PPO-without-selection and static-demography controls diagnose failure even when clean cooperation is not sustained. The matched clean benchmarks now include a genuine dyadic Tit-for-tat rule, separately from public reputation. Outputs live in `results/inheritance_study`; earlier results remain separate. The browser simulator still executes the earlier single-policy actor; the multi-policy experiment has its own visible research section.

## Run the revised study

Use Python 3.12 from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
node --test tests/web_simulation.test.cjs
python experiments/revised_study.py --environment-steps 8192 --workers 3
python evaluation/revised_report.py
```

This trains seven conditions with five independent training seeds (35 checkpoints), then evaluates clean and attacked worlds and mutation controls. The budget is eight times the historical 1,024 steps; it is **not a convergence guarantee**. The budget must be a positive multiple of the 64-step rollout length. Each checkpoint records its exact configuration, training curve, actual environment and resident-agent transitions, source hashes, versions, and SHA256. Verified matching completed runs can be reused. Increase `--environment-steps` for a longer study; reduce `--workers` to 1 to limit concurrent CPU/memory load. Evaluation can be rerun with `--evaluate-only`.

The matched default world is 18 x 18, 24 founders, capacity 45, initial energy 30, mitosis threshold 60, cost 2, benefit 5, basal drain 0.6, crowding cost 0.08, mutation 0.05, and a 300-transition horizon. Only named experimental factors vary. Attack controls add up to six mimics after transition 10, refreshing observations before the next decision. No attack or no post-attack donations yields an undefined (null) donation share, never successful defense.

## Browser simulation

Open [index.html](index.html) locally. It runs offline without installation or a server. The [live GitHub Pages site](https://chandak-shubham.github.io/cs329_project/) builds from the repository root on `main`; local changes appear after pushing and a successful Pages build.

**Hamilton enforcement is off by default.** The checkbox explicitly forces the rule when enabled. Results in that mode cannot establish learned Hamilton behaviour. The default executes the **actual trained contextual attention checkpoint**, identified with objective, seed and training budget. The four other strategies are explicitly hand-written baselines. The exporter selects clean lineage seed 42 after evaluation, otherwise revised demographic seed 42. Default ecology matches training; stress presets and manual changes are distribution shifts. Browser and Python use different seeded RNG implementations, so identical numeric seeds do not imply identical trajectories. Action-probability parity is tested separately.

Mimics copy the most common resident genome (smallest genome breaks ties), occupy randomly sampled empty cells, and never donate. The interface and CSV report donations to mimics as a share of post-invasion donations. Affordability and mimic restrictions always apply. Compare five fresh browser runs to inspect donation rates and time to capacity. See [the browser manual](docs/web_simulation.md). Refresh weights and visible research tables with `python visualizer/export_web_data.py`; `--check` verifies both against source artifacts.

## Interpretation

- The revised demographic reward is `1 * survived_transition + 5 * own_births`. It contains no relatedness, recipient benefit, donation bonus, or Hamilton threshold. The energy-only control retains the original reward. Survival/reproduction incentives may help reciprocity, but do not guarantee it.
- The contextual gate sees masked neighbourhood cue means and standard deviations. Encoders retain raw cues, so gate weight is descriptive telemetry. Full-policy cue permutation sensitivity and common-input probes measure action changes, including raw-cue paths; they are exploratory attribution diagnostics, not causal reliability estimates.
- Noise conditions train independently at bit-flip probabilities 0, 0.25 and 0.5. Noise/mimic training is separate from frozen-policy robustness testing. A frozen policy may respond to changed observations; it does not update its parameters online.
- Donation adds net energy B-C; only parent/child mitosis conserves energy. Behavioural history is the fraction of the neighbour's recent actions that were donations **given**, not received. Chromosome similarity is a proxy, not a measured pedigree kin coefficient.
- Confidence intervals use independent training-seed units: Student t for means and Wilson for survival proportions. Paired contrasts use per-seed differences. Five clusters and one evaluation world per checkpoint leave substantial uncertainty; the results are exploratory.
- The fallback actually trains with starvation and mitosis disabled. Negative energy is allowed there, so its survival metric cannot support ecological claims.

## Historical study

The original 35 checkpoints, phase results, regression, spatial controls, figures and animation are retained under `results/`. Their 1,024-step budgets and differing evaluation ecologies are documented in [the historical completion report](docs/completion_2026-11-16.md). `experiments/run_phases.py`, `experiments/research_controls.py` and `evaluation/analysis.py` reproduce that historical design with current bug fixes; they do not train or analyze the revised study. The historical fixed-weight threshold rule is distinct from the fixed-gating attention ablation.

Training data use persistent identities, terminal death transitions even on slot reuse, action masks from the configured environment, and active resident masks. Movement permits swaps and chains with consistent occupancy. The rule-based feasibility arena inherits strategy labels by persistent individual ID; neural residents share one policy per run.
