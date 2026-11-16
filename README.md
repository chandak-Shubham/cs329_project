# Learned cooperation in a spatial population

A CPU PyTorch research project evaluating shared neural donation policies in a dynamic grid world. Negative findings are valid outcomes. Donation, survival, kin preference and Hamilton's rule are separate hypotheses; the code does not assume any of them emerge.

## Reproduce

Run from the repository root using Python 3.12:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python training/train_all_models.py --environment-steps 1024 --workers 4
python experiments/run_phases.py
python experiments/research_controls.py
python evaluation/analysis.py
python evaluation/generate_figures.py
python experiments/cooperation_feasibility.py
python visualizer/live_sim.py
python evaluation/build_run_report.py
```

Training must finish before evaluation. A missing checkpoint raises an error. Reduce `--workers` to 1 on memory-constrained machines. All computation uses CPU; no external accounts or paid services are required. This is a short-budget replicated study, not a convergence claim. Increase the training budget and repeat the complete evaluation sequence for stronger evidence.

## Training and provenance

The default suite trains **35 checkpoints**: MLP, gating MLP, attention, and fixed-gating attention at B/C=2.5, plus attention at B/C=1.5, 3.0 and 4.5; each uses seeds 42, 101, 202, 303 and 404. Every run starts from a fresh network, optimizer and world. The fixed neural gate is 0.5/0.5 throughout training and evaluation; it retains the attention architecture and raw cues. It tests the gate contribution, not removal of genetic or behavioral information.

Each run uses 1,024 environment transitions, 64-step rollouts, 16 PPO updates, three epochs per update, batch size 64, learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clipping 0.2, value coefficient 0.5 and entropy coefficient 0.01. The legacy `total_timesteps` function argument is a nominal initial-population agent budget (24 times the environment budget), rounded up to complete rollouts; actual active-agent transitions differ and are saved. No hidden minimum training iterations apply. Each checkpoint has a JSON sidecar with configuration, source and checkpoint SHA256 hashes, package versions, actual budgets, elapsed time and learning curves. Seed-42 aliases support the single-run visualizer; evaluations load each seed-specific checkpoint.

Evaluation worlds and action sampling use seed `training_seed + 10000`; random-pair measurement uses a separate RNG. These are five independent joint training/evaluation units, with one evaluation world per trained policy per condition. Intervals therefore include both training and environment variability and do not separate them. More evaluation worlds per checkpoint would improve precision.

## Experiments and interpretation

- Phases 1-4 retain selfish, clean neural, noise-sweep and stress-invasion evaluations. Phase 4 injection occurs at the state boundary after 40 transitions, before transition 41. Attacks that cannot occur are null, not successful defense. Defense requires resident survival. Final population and episode duration are not equilibrium carrying capacity or individual lifespan.
- `research_controls.json` contains paired attack/no-attack and adaptive/fixed neural comparisons. Every Phase 4 strategy also has a matched no-invasion stress run in `phase4_no_invasion.json`. Additional neural controls inject after 10 transitions so short-lived populations can actually face attack. They use the same per-seed initial worlds and action RNG seeds; stochastic paths can diverge after interventions. Report both this early-invasion control and the original Phase 4 stress setting.
- Before/after genetic-weight changes use ten-transition windows and a matched no-invasion difference in differences. This measures a composition-sensitive population response of a frozen gate, not online learning or individual trust updating.
- Spatial assortment uses unordered pairs at toroidal Manhattan distance exactly one, measured over living states up to 300 transitions. It compares seed-paired random chromosome pairs and independently simulated random global offspring placement. Both placement arms use initial energy 60 and a mitosis threshold of 50 to exercise reproduction in every seed; the initial low-reserve check had only 0-1 births per seed. This supplementary diagnostic was refined after that exposure check and is exploratory; `results/placement_low_reserve_exploratory.json` retains the initial check. Per-run time averages avoid treating all pair observations as independent. Extinct runs contribute their pre-extinction states; the statistic is not an equilibrium estimate. Missing pairs are null and excluded explicitly.
- The incentive-response experiment trains separate attention policies at each B/C and probes identical synthetic inputs over relatedness. It reports a paired high-minus-low incentive response and above-minus-below Hamilton threshold contrast across training seeds. B/C is not in the observation, so these are **training-condition effects**, not instantaneous incentive sensitivity. The curve uses baseline-corrected chromosome similarity, not an independently estimated pedigree kin coefficient. This experiment alone cannot prove Hamilton's rule.
- Regression uses training-seed cluster sandwich covariance and a Student t reference with four degrees of freedom. Five clusters are few: all inference is exploratory. Predictors are unstandardized. Environmental B/C in a rollout regression can reflect state-distribution changes, not direct perception of incentives.

Means use 95% Student t intervals across independent runs; binary survival uses Wilson intervals. A zero observed survival rate still has nonzero uncertainty. Paired controls use per-seed differences. P-values are exploratory, two-sided and unadjusted for multiple comparisons. JSON retains raw seed records; generated figures/tables derive annotations from those results.

## Environment scope

Simultaneous movement permits swaps and chains into vacated cells; competing destination intents are cancelled. Donation subtracts C and adds B, so B>C creates net energy B-C. Only mitosis conserves energy by splitting the parent's reserve. Neural residents share one policy per run; this is not competition among evolving lineage-specific neural policies. The rule-based feasibility arena does inherit strategy labels by persistent individual ID. The reputation baseline is called **Reputation Threshold**, not Tit-for-Tat.

`offspring_placement="random"` enables the global-empty-cell control. `enable_starvation=False` disables energy-based deaths as an explicit diagnostic only: energy can become negative and these runs cannot validate ecological survival. Standard results keep starvation enabled.

Results, small checkpoints, their provenance, figures and the GIF are research artifacts. Python bytecode and caches are ignored. Historical reviews in `docs/project_review.md` and the dated rechecks describe earlier snapshots; use `docs/completion_2026-11-16.md` for the latest executed evidence and remaining limitations.
