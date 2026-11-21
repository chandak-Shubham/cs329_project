# Inherited neural policy experiment protocol

This experiment addresses the ideation's missing per-lineage neural inheritance while retaining the original energy-only PPO reward. It is separate from the historical shared-policy study, demographic-reward study and descendant-reward study. Neither successful implementation nor survival alone establishes emergent kin selection.

## Networks and inheritance

Each of the 24 founders starts with a separately initialized attention network and Adam optimizer. A newborn inherits its parent's policy membership through persistent individual IDs; family members share that lineage's parameters. Unrelated founders never share parameter storage. The eight-bit chromosome mutates at mitosis as before, while neural parameters are updated by PPO without weight mutation. Genomes, policy membership and pedigree are separate concepts: equal tags do not assign agents to the same policy.

The architecture is the existing contextual gated attention model (32-dimensional embeddings, two attention heads). Each lineage optimizes only its own resident transitions using the individual pre-mitosis energy increment. No donor bonus, relatedness coefficient, Hamilton filter, survival bonus or descendant bonus enters its PPO reward. Advantages are normalized within each lineage's samples. Death terminates the individual's return even on same-step slot reuse. The final short rollout is included so the requested transition budget is exact.

## Explicit evolutionary extension

Within an episode, mitosis increases the number of carriers of the parent's policy. At a completed 300-transition horizon, the next episode's 24 founders are sampled with replacement from living residents. The sampled individual's chromosome, policy weights, ancestral origin and optimizer state are inherited together. Copied networks use independent parameter storage, allowing descendant branches to learn separately in the next episode. This samples a lineage in proportion to its surviving abundance, without scoring donations, tags or attention weights.

After total resident extinction, there is no evolutionary winner: networks and the current generation's founder genomes are retained for another PPO episode, and the failed selection event is recorded. Every reset supplies the standard initial energy and random positions. This periodic restocking is an explicit evolutionary outer loop added to make the proposal's selection mechanism operational; it is not spontaneous energy creation within an episode or an unmentioned natural process.

Selection uses the ecological outcomes collected before the final PPO update of that episode, then inherits the updated weights. At a budget ending mid-episode, the bank is saved without an extra selection event. Evaluation starts a fresh population with that generation's founder genomes and bank, including policies whose carriers died during the incomplete episode. No best checkpoint, best seed or post-hoc winning family is chosen for evaluation.

## Declared experiment

Defaults are 8,192 environment transitions per training run and seeds 42, 101, 202, 303 and 404. Each trained seed is evaluated in three fresh worlds using seeds `training_seed + 10000 + 1000 * replicate`, with replicate 0, 1 and 2. Evaluation freezes both learning and selection. World means are averaged within training seeds before computing exploratory 95% t intervals. A separate Wilson interval describes the independent seed-level event that all three evaluation worlds survive.

Ecology matches `experiments/config.py`: 18 x 18 grid, 24 founders, capacity 45, initial energy 30, mitosis threshold 60, C=2, B=5, basal drain 0.6, density cost 0.08, mutation 0.05 and horizon 300. PPO uses 64-step rollouts, three epochs, minibatches of 64, learning rate 0.0003, discount 0.99, GAE lambda 0.95, clipping 0.2, value coefficient 0.5, entropy coefficient 0.01 and gradient norm limit 0.5.

The clean feasibility screen requires every seed/world to survive the full horizon, give at least 0.1 donations per resident-step over its final 100 transitions, and donate in at least half of those transitions. These thresholds only decide whether to run further experiments. They never alter policy actions, rewards or checkpoint selection. They are not a convergence test or statistical evidence of kin preference.

Regardless of the clean screen, run the mechanism controls (independent PPO without outer selection; outer selection without PPO) and the proposal's fallback (starvation and mitosis disabled). Fallback survival is mechanically guaranteed and has no ecological interpretation. An additional untrained-ancestor evaluation uses initial ancestor weights with the trained population's founder genomes and ancestor composition. This controls the effect of learned weights conditional on evolved ancestry; it does not replace the separately trained evolution-only control.

The clean-world benchmark also evaluates selfish movement, tag trust, fixed cue threshold, public reputation and a new dyadic Tit-for-tat baseline in the same worlds and founder genomes. These are labelled rules and have no training budget. For the dyadic rule, a neighbour cooperates with the focal individual only when it gives to that individual during a directed observable encounter. Unknown pairs start cooperatively; each individual remembers the neighbour's most recent such decision and chooses uniformly among available reciprocating recipients. This operational definition is necessary because the spatial game permits one targeted donation among several neighbours. Pair memory uses persistent IDs and cannot transfer a dead individual's social debt to a newborn occupying its slot. The public-reputation rule is retained separately.

If clean feasibility passes, the runner supports independently trained noise levels 0.25 and 0.5, noise plus mimics, fixed attention gates, MLP without attention, B/C values 1.5 and 4.5, mutation rates 0, 0.1 and 0.2, and randomized offspring placement. Only named factors differ. This is a set of one-factor controls, not a full factorial interaction study. Explicit `--conditions` runs are labelled exploratory and do not silently override the screen.

Mimics in this new protocol are stationary, scripted, non-donating invaders; their descendants inherit that status. They are excluded from every neural optimizer. Attacks introduce up to six mimics after transition 10 in the attack condition. Unlike the historical scripts, no resident network supplies mimic movement. Post-invasion donation shares are undefined when there are no eligible donations. This changed adversary definition is recorded rather than mixed with historical attack results.

## Interpretation and remaining hypothesis tests

Sustained giving is necessary but insufficient for kin-directed cooperation. Opportunity-adjusted tag preference is the donated recipient's corrected genetic cue minus the mean cue of available recipients, averaged only over actual donations. It is a descriptive diagnostic, not pedigree relatedness or causal evidence. True-chromosome adjacent similarity requires a matched random-placement comparison. A mostly homogeneous genome population offers little cue variation and cannot establish learned genetic discrimination.

Any claim of adaptive cue reliability additionally needs the independently trained noise/mimic comparisons, common-input full-policy cue interventions, and a justified statistical comparison across independent training seeds. A Hamilton regression needs nondegenerate donation outcomes and the matched B/C conditions; an absent or unidentifiable regression is not a positive finding. Raw cue paths remain in the encoder, so gate weights alone never establish reliance. Results from a failed clean screen must remain a negative or inconclusive result.

## Reproduction and provenance

```powershell
python -m pytest -q
python experiments/inheritance_study.py --steps 8192 --workers 3
```

Use `--out` for a separate study directory, `--seeds` for explicitly smaller diagnostic runs, and `--evaluate-only` to evaluate saved checkpoints. Existing checkpoints are reused only if hashes, source files, versions, settings, seed and budget match. Otherwise the command refuses to overwrite them. The checkpoint contains all family networks and founder genomes; its sidecar contains per-rollout histories, per-generation selection records, exact active-agent transitions and source/version hashes. Historical artifacts are preserved. Checkpoints are frozen evaluation artifacts, not mid-rollout resume snapshots.
