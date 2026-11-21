# Final study protocol

The final report is generated from measured outcomes by `evaluation/submission_report.py`. The computational question is whether private energy-reward PPO preserves giving supported by inherited policies and survivor selection, and how reward alignment, incentives and placement change the result. A positive kin-selection result is not assumed.

## Model

The world is an 18 x 18 torus with 24 founders, capacity 45, initial energy 30 and mitosis threshold 60. Donation costs 2 and normally gives 5 to its recipient. Basal drain is 0.6 and crowding costs 0.08 per living neighbour within Chebyshev radius 3. The policy observes at most eight nearest neighbours; ties use storage-slot order. Each episode lasts at most 300 transitions.

An action is stay, a cardinal move, or donation to an observed neighbour. Affordability requires energy strictly greater than the cost. All 13 logits share one masked categorical normalization. Recipient identity is resolved at decision time. Crowding is assessed after movement. A donation produces B-C net energy. Energy reward is measured before the parent/child split; mitosis conserves their combined energy. Reproduction requires both a free tensor slot and a legal unoccupied cell. Death terminates the old identity even if its slot is immediately reused.

Chromosomes have eight bits. Mutation flips each bit independently with probability 0.05. Hamming similarity is one minus normalized distance. The cue subtracts the population's distinct-pair mean and divides by `max(1-h_bar, 1e-6)`, clipping below zero. Under corruption, independently flipped visible chromosomes supply the cue. True chromosomes supply spatial measurements. Behaviour is donations **given** in the last ten actions, with prior 0.5 for founders and newborns. These tags are not pedigree coefficients.

## Networks, rewards and selection

Each founder owns an independent 32-dimensional, two-head contextual gated-attention actor and local critic, with its own Adam optimizer. Within a family, descendants share the policy. The contextual gate sees local masked cue means and standard deviations, but raw cues also enter the encoder. Cue reliance therefore uses full-policy permutation sensitivity rather than gate weights alone.

PPO settings are rollout 64, epochs 3, minibatch 64, learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clip 0.2, value coefficient 0.5, entropy coefficient 0.01, and gradient clipping 0.5. GAE is an individual return, terminated at death. Family sharing pools samples; it does not add a relative's reward to that individual's target.

At a horizon, survivor restocking samples 24 actual living residents with replacement, and copies their genomes, networks and optimizer states into independent new founders. Families are weighted by surviving abundance. Total resident extinction produces no selection winner; the previous founder composition and learned policies are retained on reset. Episode resets introduce energy and are part of the model.

The main objective is `R_i = Delta E_i`. The new diagnostic objective is `R_i = Delta E_i + sum_(j != i) r_tilde_ij Delta E_j`, using only the decision-time active individuals and decision-time cues. The simulator's raw ecology is unchanged. This deliberately privileged, marker-weighted reward is an incentive control, not biological inclusive fitness or evidence of unshaped emergence. For a matched isolated donation its immediate change is `-C + r_tilde B`.

## Original study: complete evaluation of saved training

The 95 supplied checkpoints represent 19 conditions and five independent seed blocks. Base training budget is 2,048 environment transitions; the long-training control uses 8,192. Conditions are clean, no selection, evolution only, fixed gate, MLP, noise 0.25, noise 0.5, noise plus mimics, fixed gate plus the same attack, B/C=1.5, B/C=4.5, mutation 0, 0.1 and 0.2, random placement, static-demography fallback, learning rate 0.00003, entropy 0.1, and long training.

Original training versions are retained in checkpoint metadata. The new frozen evaluation records its own runtime separately; it does not claim to rerun that training. Genomes and initial ancestor composition are matched in the untrained comparison. Clean rules include selfish, tag trust, fixed threshold, public reputation and true directed dyadic Tit-for-tat. These are scripted controls, not learned policies.

Noise corrupts visible tags. Attacks inject up to six stationary non-donating mimics at transition 10; they copy the modal resident chromosome, with a deterministic tie break. Observations are refreshed after injection. The no-attack and attack evaluations use the same checkpoint. A missing donation denominator makes mimic share undefined.

## New study: 65 matched training runs

The reward x incentive factorial has two objectives (energy and marker-weighted inclusive) and five B/C levels (1, 1.5, 2.5, 3, 4.5), with C fixed at 2. Five seeds per cell give 50 new training runs. At B/C=2.5, three additional cells complete the learning x placement comparison: evolution only with local offspring, evolution only with random offspring, and energy PPO with random offspring. Combined with the factorial's energy PPO/local cell, this gives a 2 x 2 comparison and 15 more runs.

Seeds are 42, 101, 202, 303 and 404. Each run has exactly 2,048 environment transitions, with a final partial rollout if needed. This is a finite-budget study, not a convergence claim. More time is informative but must use a separate output directory to preserve provenance.

Each saved policy is evaluated, without optimizer or selection updates, in three fresh worlds with seeds `training_seed + 10000 + 1000*replicate`, where replicate is 0, 1 or 2. Action sampling uses that world seed. Workers are separate processes, avoiding shared PyTorch RNG state across threaded experiments. Different conditions may consume different numbers of RNG draws; matching seed labels does not promise identical trajectories.

## Outcomes and statistical units

The independent unit is the training-seed block. World means are averaged within seeds before inference. Family networks, timesteps and recipient opportunities do not increase n. Main metrics are donations per acting-resident transition, final resident count, full-horizon survival, late donation rate and frequency, births, opportunity-adjusted recipient tag preference and mimic donation share.

The original clean feasibility screen requires every expected seed/world to survive through transition 300, donate at least 0.1 times per resident transition in its final 100 transitions, and give in at least half of those transitions. All original conditions are evaluated even if the screen fails; robustness results then remain exploratory transient measurements.

Late assortment is measured only in surviving worlds during their final 100 transitions. It includes true adjacent Hamming similarity, contemporaneous all-pair similarity, and their difference. Empty adjacent-pair windows or extinction yield undefined late metrics. Paired local/random contrasts use only seed means defined on both sides and report the available n. The evolution-only comparison addresses assortment in a viable population rather than treating extinction as an equilibrium.

Mean intervals use Student t over seed means; all-world survival events use Wilson intervals over seed blocks. Mean intervals are not clipped to physical bounds. Reward donation-rate comparisons use paired seed differences, exact two-sided sign flips and Holm correction over the five B/C levels. With five pairs the smallest attainable two-sided p is 0.0625. Results emphasize effect sizes and exploratory uncertainty; nonsignificance does not establish equivalence.

The logistic association model includes tag cue, history, B/C, tag x B/C, number of available neighbours and own normalized energy. Each objective is fit separately. Outcomes are giving to a legal recipient opportunity. An outcome-blind reservoir retains at most 4,000 opportunities per world; fitting samples at most 12,000 per seed. The five seed blocks determine cluster inference. B/C is a training condition rather than a policy input. These associations cannot validate Hamilton's biological rule.

## Reproduction commands

Run from the repository root with Python and CPU PyTorch:

```powershell
python -m pip install -r requirements.txt
python experiments/mechanism_study.py --steps 2048 --workers 3
python evaluation/complete_study.py --study-dir results/submission_study --workers 3
python evaluation/complete_study.py --study-dir results/mechanism_study --workers 3
python evaluation/submission_report.py
python -m pytest -q
node --test tests/web_simulation.test.cjs
python -m pip install -r requirements-report.txt
python evaluation/export_submission.py
```

The evaluation caches are regenerable and excluded from Git. `analysis.json` holds all world outcomes, `diagnostics.json` holds contrasts and regression results, `hamilton_decisions.npz` holds compressed retained decision data, and manifests record artifact SHA256 values. Checkpoint files, training metadata and the exact training source are retained. Archives preserve exact line endings. The browser actor remains a distinct historical shared-policy model.
