# Does private-reward learning preserve cooperation?

**CS329 Multi-Agent Systems | Group 21 | IIT Gandhinagar**

We investigate a spatial donation game: does private energy-reward PPO preserve giving sustained by inherited policies and survivor selection? How do reward alignment, benefit/cost ratio, cue reliability and offspring placement change the answer?

## Read the submission

- [Final report](docs/final_report.md), [PDF](docs/final_report.pdf), and [HTML](docs/final_report.html).
- [Five-seed and ten-seed comparison](docs/final_report.md#what-changed-with-longer-training) with [saved evaluation worlds](results/main_comparison).
- [Complete model and experimental protocol](docs/final_study_protocol.md).
- [Original 19-condition results](results/submission_study/results.md) and [raw worlds](results/submission_study/analysis.json).
- [New reward and placement results](results/mechanism_study/results.md), [raw worlds](results/mechanism_study/analysis.json), and [statistical diagnostics](results/mechanism_study/diagnostics.json).

We complete evaluation of **95 original checkpoints** and add **65 matched training runs**: two objectives x five B/C ratios x five seeds, plus three placement/learning controls x five seeds. Each checkpoint receives three fresh evaluation worlds, averaged within seed before inference. The independent sample size is five seed blocks, not fifteen worlds or 24 family networks.

The new marker-weighted reward control adds other residents' energy changes to private reward. It tests incentive alignment; it does not establish that kin selection emerges from an unchanged private objective. Original and new results are separate because their training runtimes differ.

## Reproduce

Use Python 3.12 or newer and CPU PyTorch, from the repository root:

```powershell
python -m pip install -r requirements.txt
python experiments/mechanism_study.py --steps 2048 --workers 3
python evaluation/complete_study.py --study-dir results/submission_study --workers 3
python evaluation/complete_study.py --study-dir results/mechanism_study --workers 3
python evaluation/submission_report.py
python -m pytest -q
node --test tests/web_simulation.test.cjs
```

Optional HTML/PDF export requires Microsoft Edge:

```powershell
python -m pip install -r requirements-report.txt
python evaluation/export_submission.py
```

Lower `--workers` to 1 to reduce memory use. The 2,048-transition matched budget is an exploratory resource constraint, not evidence of convergence; the original long-training control uses 8,192. New training verifies compatible completed runs and refuses to overwrite incompatible files. Use a fresh `--out` for a changed experiment.

## Implementation and evidence

| Component | What is implemented |
| --- | --- |
| Ecology | Toroidal movement, local donations, crowding, starvation and energy-conserving parent/child splits |
| Inheritance | Persistent identities, independent founder networks and optimizers, shared descendant policies, survivor-weighted restocking |
| Neural method | Contextual gated attention, raw genetic/behavioural cues, a local critic, masked categorical actions and family PPO |
| Baselines | No PPO, no selection, untrained ancestors, MLP, fixed gate, selfish, tag trust, threshold, reputation and dyadic Tit-for-tat |
| Reward study | Matched private and marker-weighted rewards over five B/C levels |
| Spatial study | Learning x offspring placement; late adjacent/global similarity in surviving populations |
| Robustness | Independently trained noise/mimic conditions, full-policy cue permutation and mimic donation shares |
| Inference | Seed-level intervals, paired effects, exact sign flips, Holm correction and tag x B/C regression |
| Provenance | Saved weights, exact training sources, separate evaluation runtime, SHA256 verification and artifact manifests |

## Interpretation

- Donation adds B-C net energy; only mitosis conserves parent-plus-child energy.
- Private reward precedes reproduction. Shared weights alone do not credit a relative's benefit to the donor.
- Eight-bit tags are proxies, not pedigree relatedness. The shaped objective is a privileged diagnostic, not biological inclusive fitness.
- Raw cues also enter the encoder. Full-policy permutation sensitivity measures both paths; gate weights alone cannot identify reliance.
- Late assortment is undefined after extinction. We test it under selection without PPO and subtract global similarity to distinguish spatial clustering from population-wide tag similarity.
- No post-invasion giving means an undefined mimic share, not successful defense.
- With five paired seeds, exact two-sided sign flips cannot produce p below 0.0625. We report effects and exploratory uncertainty.
- Negative results can answer the computational question. Survival, cooperation, recipient discrimination and evolutionary kin selection are distinct claims.

## Repository map

```text
environment/                       Spatial game and cue generation
models/                            Attention, gate, MLP and rule policies
training/                          Historical shared-policy PPO
experiments/inherited_policies.py   Family PPO and survivor selection
experiments/mechanism_study.py      Matched reward/incentive and placement study
evaluation/complete_study.py        Verified frozen evaluation
evaluation/submission_report.py     Final report, figures and analyses
results/submission_study/           Original checkpoints and completed evaluation
results/mechanism_study/            New checkpoints, sources and results
tests/                             Scientific invariants and browser parity
index.html                         Offline browser illustration
```

## Browser and historical work

Open [index.html](index.html) for the offline illustration. Its actor is an earlier single shared-policy checkpoint; it does not execute the new family-policy study. Hamilton enforcement is off by default; turning it on imposes the inequality and cannot establish learned behaviour. Browser and Python RNGs differ.

Earlier objective variants remain in separate result directories and are not pooled into the final study. **`evaluation/submission_report.py` is the final submission entry point.** The [earlier strict evaluation](docs/project_evaluation.md) records the state before these changes, not a revised grade.
