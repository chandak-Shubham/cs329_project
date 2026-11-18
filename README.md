# Emergent Kin Selection in Multi-Agent Reinforcement Learning

CS329 | Group 21 | IIT Gandhinagar

## Results and scientific status

**The kin-selection hypothesis is not yet supported.** In the historical five-seed, 1,024-step study, every learned Phase 2 population became extinct. The historical regression estimated a negative relatedness coefficient (-0.556) and pseudo-R-squared around 0.002. Those observations do not demonstrate learned kin preference or adaptive cue reliability.

The 17 November review response adds a separate, reproducible study with an individual survival/reproduction objective, contextual gating, independently trained noise conditions, matched ecological settings, mimic donation-share measurements, a fixed neural gate, and the demographic fallback. Changing the reward changes the research question. A shared policy still does not implement competing, inherited lineage policies.

See [the review response and protocol](docs/review_response_2026-11-17.md), [executed revised results](results/revised_study/results.md), and [raw revised results](results/revised_study/analysis.json). Historical results remain separate; their tables and figures must not be presented as outputs of the revised training.

The [18 November final-review response](docs/final_review_response_2026-11-18.md) adds a continuing descendant-fitness objective and a clean-first feasibility screen. In its [separate results](results/lineage_study/results.md), all five trained populations became extinct after 41–46 transitions with zero donations; all five matched untrained controls survived. The failed screen withheld noise/mimic follow-ups. The website displays these results alongside the earlier evidence, figures and caveats, and defaults to the actual trained attention policy.

To reproduce this follow-up without overwriting the revised study:

```powershell
python experiments/lineage_study.py --environment-steps 8192 --workers 3
python evaluation/revised_report.py --study-dir results/lineage_study
python visualizer/export_web_data.py
```

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
