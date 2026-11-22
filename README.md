# Emergent Kin Selection and Spatial Assortment in Multi-Agent Reinforcement Learning

CS329 Multi-Agent Systems · Group 21 · IIT Gandhinagar
Vansh Barfiwala · Pratham Choksi · Shubham Chandak

| | |
| --- | --- |
| 📄 **Report** | [final_report.pdf](report/final_report.pdf) (LaTeX source: [final_report.tex](report/final_report.tex)) |
| 🌐 **Live demo, no installation** | <https://chandak-shubham.github.io/cs329_project/> |
| 📊 **Result tables** | [Study B](results/study_b_hamilton/results.md) · [Study C](results/study_c_robustness/results.md) · [Study A](results/study_a_energy/results.md) |

---

## The project in one paragraph

Agents on an 18 × 18 grid lose energy every step. An agent can donate energy to a neighbour: the donor pays a cost $C$ and the neighbour receives $B > C$. Donations are the only source of new energy, so without cooperation the population starves, yet each gift costs its donor. Each of 24 founder families owns its own gated-attention network trained with PPO; children inherit their family's network, and surviving families are selected between episodes. Agents see each neighbour's genetic tag, which may be corrupted or copied by non-donating mimics, and its recent giving.

**Can self-interested learners sustain cooperation, and does rewarding the recipient's relatedness (Hamilton's rule $rB > C$) change whom they help?**

## Key results

10 independent training seeds per condition, 3 fresh evaluation worlds per seed. All numbers come from the `analysis.json` files in `results/`.

![Survival, donations, lifetime and Hamilton gap across B/C](plots/study_b_hamilton/bc_sweep.png)

| B/C | Seeds surviving: inclusive reward | Seeds surviving: energy-only reward | Seeds surviving: untrained networks | Donations / resident-step: inclusive | Donations / resident-step: energy-only |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 0/10 | 0/10 | 0/10 | 0.009 | 0.006 |
| 1.5 | 0/10 | 0/10 | 0/10 | 0.011 | 0.007 |
| 2.5 | 1/10 | 0/10 | 10/10 | 0.242 | 0.005 |
| 4.0 | **7/10** | 0/10 | 10/10 | 0.405 | 0.016 |
| 6.0 | **10/10** | 0/10 | 10/10 | **0.803** | 0.032 |

1. **Individual reward destroys cooperation.** With the energy-only reward from our proposal, PPO learns to stop giving and no population survives (0/50 runs). Untrained networks with the same founders do survive, so learning removes cooperation that was already there. Study A confirms this in 18 robustness conditions: in all 90/90 runs, trained populations gave less than untrained ones.
2. **An explicit kin-weighted reward preserves it.** Adding $r_{ij}B$ to the donor's reward: 18/50 runs survive versus 0/50 (Fisher $p = 1.2\times10^{-6}$), and 17/20 versus 0/20 at $B/C \ge 4$ ($p = 2.6\times10^{-8}$). This shows that learners respond to a kin-weighted incentive; it does not show Hamilton's rule emerging from purely individual rewards.
3. **Giving tracks the Hamilton threshold at population level.** Learned donation rates were low at the two lowest ratios ($B/C = 1$ makes $rB > C$ impossible; $B/C = 1.5$ requires $r > 2/3$); at $B/C = 6$ inclusive learners gave more than untrained networks in 9/10 seeds. The predicted increase of recipient preference with $B/C$ was not found.
4. **Kin-weighted rewards improve targeting** (Study C, $B/C = 4$). Paying for every gift (prosocial) kept 10/10 populations alive and the kin-weighted reward 7/10; no significant survival difference was detected. Compared with a kin-blind reward of the same expected size, the kin-weighted reward raised the share of gifts with $rB > C$ by 0.10 (8/10 seeds, $p = 0.010$).
5. **Kin recognition is the weak link.** Learners rely less on a corrupted genetic cue but never switch to behavioural history, and they keep feeding genetic mimics, which a simple reputation rule ignores completely.
6. **Limits.** At $\gamma = 0.99$ both rewards survive equally rarely (2/10 each); 4× more training at $B/C = 2.5$ did not help.

## How to check this project

### Option 1: in the browser, no installation (recommended)

Open the **[live demo](https://chandak-shubham.github.io/cs329_project/)**. It downloads our committed trained models, checks each file's SHA-256 against its training record, and runs all 24 family networks in your browser.

| On the page | What you get |
| --- | --- |
| **Trained model** (condition + seed) | Loads any of 160 trained models (16 conditions × 10 seeds). Press **Play** to watch the population cooperate or starve. |
| **Experiment / Agent behaviour** | Switch to the trained model's own ecology, to hand-written rule baselines, or to mimic attacks. |
| **Benefit / Cost sliders, Add mimics, Scarcity shock** | Test a trained model in conditions it was not trained for. |
| **Reproduce a result** | Reruns all 10 seeds of a condition and shows the browser outcome next to the recorded Python result. |
| **Research results** | The study tables, key findings and figures, generated from `results/`. |

### Option 2: verify locally without retraining (about 5 minutes)

```powershell
python -m pip install -r requirements.txt                 # Python 3.12; pins PyTorch 2.5.1 and NumPy 2.3.2
python evaluation/verify_hamilton.py                      # Study B
python evaluation/verify_hamilton.py --study robustness   # Study C
```

What you get: every model's SHA-256 and training settings are checked, the headline numbers are recomputed from the raw evaluation logs (Study B) or every mimic attack is checked (Study C), and saved models are re-simulated in the same seeded worlds, where every recorded metric must match exactly. Each run ends with `VERIFICATION PASSED`. Add `--all` to re-simulate every model. PyTorch is pinned because exact re-simulation depends on its random sampling.

### Option 3: run the website locally

```powershell
python -m http.server 8000        # in the repository folder
```

Then open <http://localhost:8000/frontend/>. Opening `frontend/index.html` directly from disk cannot load the models, because browsers block file access from `file://` pages.

### Option 4: retrain from scratch (several hours, CPU only)

Completed runs are reused, so an interrupted run can be restarted.

```powershell
python src/training/hamilton_study.py --workers 14         # Study B, about 2 hours on 14 cores
python evaluation/hamilton_report.py
python src/training/hamilton_robustness.py --workers 14    # Study C, about 2.5 hours
python evaluation/robustness_report.py
python src/training/inheritance_study.py --complete --steps 2048 --workers 3   # Study A
python evaluation/inheritance_report.py
python evaluation/report_figures.py                        # extra report figure
python frontend/export_web_data.py                         # refresh the website (--check to verify)
```

The report is compiled from `report/final_report.tex` with the PRIMEarxiv template (for example on Overleaf, where `PRIMEarxiv.sty` comes with the template).

## What is in the repository

```
report/          final_report.tex, final_report.pdf and the figures they use
frontend/        index.html (the live demo), its data exporter, site builder and browser checks
src/
  environment/   spatial donation game: energy, starvation, mitosis, genetic cue, mimics
  networks/      gated-attention policy, MLP ablation, scripted strategies
  training/      per-family PPO with inheritance and selection; Studies A, B and C
models/          trained checkpoints (.pt) and their training records (.json), one folder per study
results/         per study: results.md, analysis.json, CSV tables, raw evaluation logs, archived training source
plots/           all figures, one folder per study
evaluation/      analysis, figure and verification scripts
tests/           Python and browser-engine tests
.github/         workflow that publishes the live demo
```

| Study | Folder name | What it tests |
| --- | --- | --- |
| A | `study_a_energy` | Energy-only reward in 19 robustness conditions (5 seeds, 95 models) |
| B | `study_b_hamilton` | Energy-only vs inclusive reward at five B/C levels, plus noise, mimics and ablations (20 conditions, 200 models) |
| C | `study_c_robustness` | Prosocial and kin-blind reward controls and robustness at B/C = 4, strategy sweep (18 conditions, 180 models) |

## Tests

```powershell
python -m pytest -q                                   # Python tests
node --test tests/web_simulation.test.cjs             # browser engine against the Python environment
python frontend/check_web_demo.py --browsers msedge   # optional, needs Playwright: full page check in a real browser
```

The browser engine is tested against the Python environment for identical observations, transitions and action probabilities. The live demo is published by `.github/workflows/pages.yml` (Settings → Pages → Source: GitHub Actions).

## Simulation controls

Hover over or focus the question mark beside a control for an explanation.

| Control | What it does |
| --- | --- |
| Training condition | Chooses the experimental setup used to train the checkpoint. |
| Training seed | Chooses an independently trained checkpoint. |
| Experiment | Loads a preset of agent behaviour and environment settings. |
| Agent behaviour | Chooses a trained population or a hand-written rule. |
| Benefit B | Energy added to the recipient of a gift. |
| Cost C | Energy spent by the donor on a gift. |
| Drain | Base energy consumed by each resident per step, before density costs. |
| Speed | Number of simulation steps played per second. |
| World seed | Sets the random sequence for the simulated world. |
| Enforce Hamilton rule | Blocks gifts unless observed relatedness times B exceeds C. This forces the rule. |
| Select agent | Selects an agent to inspect its current state. |
| Reproduction condition | Chooses the saved experimental condition to replay across its training seeds. |
| Worlds per seed | Sets how many evaluation worlds to replay per training seed. |
