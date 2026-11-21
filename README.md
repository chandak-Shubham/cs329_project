# Does Learning Need Kinship?

**Individual and inclusive-fitness reinforcement learning in a spatial donation game**
CS329 Multi-Agent Systems · Group 21 · IIT Gandhinagar
Vansh Barfiwala · Pratham Choksi · Shubham Chandak

📄 **[Final report](docs/final_report.md)** · 📊 **[All results and statistics](results/hamilton_study/results.md)** · 📋 **Spreadsheets: [per trained model](results/hamilton_study/seed_summary.csv), [per evaluation episode](results/hamilton_study/evaluation_runs.csv)** · 🌐 **[Interactive simulation](https://chandak-shubham.github.io/cs329_project/)**

---

## The question

Agents on an 18 × 18 grid lose energy every step. They can donate energy to a neighbour at a cost $C$, and the neighbour receives $B > C$. Donations are the only source of new energy, so without cooperation the population starves, yet each individual is better off not giving.

Founder families own independent gated-attention neural policies trained with PPO. Children inherit their family's policy, and surviving families are selected between episodes. Agents see each neighbour's genetic tag, which may be corrupted, and its recent giving.

**Can self-interested learners discover cooperation, and does what they learn follow Hamilton's rule $rB > C$?**

## Main results

![Survival, donations, lifetime and Hamilton gap across B/C](results/hamilton_study/figures/bc_sweep.png)

| B/C | Inclusive-fitness PPO: donations / step | Energy-only PPO: donations / step | Untrained networks | Seeds surviving (inclusive / energy) |
| --- | --- | --- | --- | --- |
| 1.0 | 0.010 | 0.007 | 0.320 | 0/5 · 0/5 |
| 1.5 | 0.012 | 0.007 | 0.316 | 0/5 · 0/5 |
| 2.5 | 0.333 | 0.005 | 0.514 | 1/5 · 0/5 |
| 4.0 | 0.526 | 0.021 | 0.584 | **5/5** · 0/5 |
| 6.0 | **0.894** | 0.037 | 0.510 | **5/5** · 0/5 |

1. **Individual reward destroys cooperation.** With the energy-only reward from our proposal, PPO learns to stop giving. In all **90/90** robustness runs (18 conditions × 5 seeds), trained populations gave less than untrained ones, and no energy-only population survived the $B/C$ sweep (0/25).
2. **Hamilton's kin term fixes it.** Adding $r_{ij}B$ to the donor's reward makes inclusive learners give more in **24/25** paired runs ($p = 1.6\times10^{-6}$). They survive in **10/10** runs at $B/C \ge 4$, against 0/10 for energy-only ($p = 1.1\times10^{-5}$).
3. **Learned giving follows Hamilton's threshold.** Below it ($B/C \le 1.5$), inclusive learners also abandon giving. Above it ($B/C = 6$), they give more than untrained networks in 5/5 seeds.
4. **Kin recognition is the weak link.** Learners prefer related recipients, and their reliance on a corrupted genetic cue falls in every seed. But they never learn to use behavioural history, and they keep feeding genetic mimics, which a simple reputation rule ignores completely.

The full analysis is in the [report](docs/final_report.md). It includes the conditional-logit recipient model, the stratified Hamilton gap, cue reliability, mimicry, ablations, a corrected version of every error found in the ideation document, and limitations.

## Repository layout

```
docs/final_report.md        Final report (submission)
environment/                Spatial donation game: energy, starvation, mitosis, cues, mimics
models/                     Gated-attention policy, MLP ablation, scripted strategies
experiments/
  hamilton_study.py         Final study (Study B): energy vs inclusive-fitness reward, 20 conditions
  inherited_policies.py     Per-family PPO, policy inheritance, survivor selection
  inheritance_study.py      Energy-only robustness study (Study A), 19 conditions
  dyadic_baseline.py        Tit-for-tat and scripted-strategy adapters
evaluation/
  hamilton_report.py        Statistics, regressions, figures -> results/hamilton_study/
  verify_hamilton.py        Independent verification of the final study (no retraining)
  inheritance_report.py, final_report.py   Study A analysis and integrity audit
results/
  hamilton_study/           Final study: results.md, analysis.json, figures/, 100 trained models, raw evaluations,
                            seed_summary.csv and evaluation_runs.csv (open in Excel / Google Sheets)
  submission_study/         Study A: results, diagnostics, checkpoints, archived source
  revised_study/, lineage_study/, inheritance_study/, ...   Earlier studies (historical evidence)
tests/                      Python and browser-engine test suites
index.html                  Interactive browser simulation (GitHub Pages)
visualizer/                 Exporter that embeds results into index.html, browser checks
```

## Reproduce

Python 3.12, CPU only.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q                                          # Python test suite
node --test tests/web_simulation.test.cjs                    # browser-engine tests

# Verify the committed final-study results without retraining (~2 minutes):
# model integrity, independent recomputation of the headline numbers, and exact re-simulation
python evaluation/verify_hamilton.py            # add --all to re-simulate all 100 models

# Retrain the final study from scratch (~1 hour on 12 cores), then analysis (~5 minutes)
python experiments/hamilton_study.py --steps 8192 --workers 12
python evaluation/hamilton_report.py

# Study A: energy-only robustness
python experiments/inheritance_study.py --complete --steps 2048 --workers 3 --worlds 3 --out results/submission_study
python evaluation/inheritance_report.py --study-dir results/submission_study
python evaluation/final_report.py --study-dir results/submission_study

# Refresh the web page's embedded results
python visualizer/export_web_data.py
python visualizer/export_web_data.py --check
```

All 100 trained models of the final study (`results/hamilton_study/checkpoints/`) and their raw evaluation logs (`results/hamilton_study/evaluations/`) are committed, so the results can be checked **without retraining**. Every checkpoint sidecar records its configuration, learning curve, library versions and SHA-256 hash, and completed runs are reused on re-execution.

## Interactive simulation

Open [`index.html`](index.html) locally (it works offline) or visit the [GitHub Pages site](https://chandak-shubham.github.io/cs329_project/). You can run scripted strategies (never donate, genetic relatives, recent donors, both cues equally) or an earlier learned shared-policy checkpoint, inject mimics, and change $B$, $C$ and other parameters. The page's research section summarises the final study. Hamilton enforcement is off by default; turning it on forces the rule rather than learning it.

## Earlier studies

The folders `results/revised_study`, `results/lineage_study`, `results/inheritance_study` and the top-level historical files in `results/` come from earlier iterations: a shared policy, demographic and descendant rewards, and a 1,024-step pilot. All used individual rewards and all failed to sustain learned cooperation. That failure motivated the final inclusive-fitness study. They are kept as evidence and are not part of the final claims.
