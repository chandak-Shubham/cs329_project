# Emergent Kin Selection and Spatial Assortment in Multi-Agent Reinforcement Learning

CS329 Multi-Agent Systems, Group 21, IIT Gandhinagar
Vansh Barfiwala, Pratham Choksi, Shubham Chandak

| | |
| --- | --- |
| **Report** | [final_report.pdf](report/final_report.pdf) (LaTeX source: [final_report.tex](report/final_report.tex)) |
| **Live demo, no installation** | <https://chandak-shubham.github.io/cs329_project/> |
| **Result tables** | [Study B](results/study_b_hamilton/results.md), [Study C](results/study_c_robustness/results.md), [Study A](results/study_a_energy/results.md) |

---

## The project in one paragraph

Agents on an 18 x 18 grid lose energy every step. An agent can donate energy to a neighbour: the donor pays a cost $C$ and the neighbour receives $B > C$. Donations are the only source of new energy, so without cooperation the population starves, yet each gift costs its donor. Each of 24 founder families owns its own gated-attention network trained with PPO; children inherit their family's network, and surviving families are selected between episodes. Agents see each neighbour's genetic tag, which may be corrupted or copied by non-donating mimics, and its recent giving.

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
6. **Limits.** At $\gamma = 0.99$ both rewards survive equally rarely (2/10 each); 4x more training at $B/C = 2.5$ did not help.

## How to check this project

### Option 1: in the browser, no installation (recommended)

Open the **[live demo](https://chandak-shubham.github.io/cs329_project/)**. It downloads our committed trained models, checks each file's SHA-256 against its training record, and runs all 24 family networks in your browser.

| On the page | What you get |
| --- | --- |
| **Trained model** (condition + seed) | Loads any of 160 trained models (16 conditions x 10 seeds). Press **Play** to watch the population cooperate or starve. |
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

The browser engine is tested against the Python environment for identical observations, transitions and action probabilities. The live demo is published by `.github/workflows/pages.yml` (in the repository settings: Pages, Source: GitHub Actions).

## Using the live demo: every dropdown and control

Each control on the page has a question mark beside it; hover over it or focus it for a short explanation. The tables below explain every option in full.

### Dropdown menus

| Dropdown | Options | What it does |
| --- | --- | --- |
| **Training condition** (sidebar, Trained model) | 16 conditions in 5 groups (next table) | Chooses which trained model to load. The page downloads that model file, checks its SHA-256 against the training record, and runs its 24 family networks. |
| **Training seed** (sidebar, Trained model) | 42, 101, 202, 303, 404, 505, 606, 707, 808, 909 | Every condition was trained 10 times from different random starting points. Choose which of the 10 trained models to load; seeds of the same condition can behave differently, which is why results are reported over all 10. |
| **Experiment** (sidebar, Strategy) | Trained model: its training ecology | Runs the loaded model in exactly the setting it was trained in (its B and C, and mimics if it was trained with them). Use this to see the trained behaviour. |
| | Rule baseline: recent donors | Hand-written rule (not learned) in the standard world (B = 5, C = 2). |
| | Rule baseline: never donate | Selfish rule that only moves; the population always starves. A lower bound. |
| | Rule baseline: genetic matching | Hand-written kin rule in the standard world. |
| | Mimics at step 40 | Recent-donors rule in a harsher world (B = 3, starting energy 25, drain 0.8, crowding cost 0.10); 12 mimics arrive at step 40. A demonstration, not a training condition. |
| | Mimics in stress ecology | The same attack on a larger 20 x 20 grid. A demonstration, not a training condition. |
| | Custom settings | Not selectable; it appears automatically when you change a setting by hand. |
| **Agent behaviour** (sidebar, Strategy) | Trained PPO Population | The loaded trained model: each founder uses its own family network, and offspring inherit it. Weights are frozen. |
| | Recent Donors Baseline | Gives to the neighbour with the best recent giving record, if that record is at least 0.5 (a reputation rule). |
| | Genetic Relatives Baseline | Gives to the most genetically similar neighbour if its relatedness r is above 0.3 (a kin rule). |
| | Fixed Both Cues | Gives if 0.5 x relatedness + 0.5 x giving record is above 0.35. |
| | Never Donate (Selfish) | Never gives; only stays or moves. |
| **Select an agent** (Agent inspector) | Every living agent, by number; mimics are marked | Shows that agent's energy, giving record, age, generation, parent and 8-bit chromosome. Clicking an agent on the grid does the same. |
| **Condition** (Reproduce a result) | The same 16 conditions | Chooses which condition to re-run across all 10 of its trained models. |
| **Worlds** (Reproduce a result) | 1 world, 3 worlds | How many fresh evaluation worlds per trained model. 1 world takes under a minute; 3 worlds matches the study exactly and takes 1 to 2 minutes. |

### Training conditions you can load

| Group | Conditions | What it shows |
| --- | --- | --- |
| Inclusive-fitness reward (Study B) | B/C = 1, 1.5, 2.5, 4, 6 | The reward includes Hamilton's kin term r x B. Populations die at B/C = 1 and 1.5, partly survive at 2.5, mostly survive at 4 and always survive at 6. |
| Energy-only reward (Study B) | B/C = 1, 1.5, 2.5, 4, 6 | The original selfish reward. Learners stop giving and every population dies, at every B/C. |
| Mimic attack (Study B) | Inclusive reward, trained with mimics, B/C = 2.5 | 6 non-donating mimics arrive at step 10, in training and in the demo. |
| Reward controls at B/C = 4 (Study C) | Prosocial reward; Kin-blind reward | Prosocial pays B for every gift to anyone; kin-blind pays the same expected kin credit whoever receives the gift. They test whether kin information matters or any reward for giving. |
| Mimic attack at B/C = 4 (Study C) | Inclusive, prosocial and kin-blind rewards, each trained with mimics | How each reward copes with fake relatives. These worlds keep 6 extra places for mimics, so the population counter shows a capacity of 51. |

### Buttons, sliders and other controls

| Control | Range | What it does |
| --- | --- | --- |
| **Play / Step / Reset** | | Run continuously, advance one step, or restart the current world. Keyboard: Space plays or pauses, the right arrow steps. |
| **Benefit (B)** | 0 to 12 | Energy the recipient gains from a gift. Training used C = 2 and B = 2, 3, 5, 8 or 12. Changing it tests a trained model in conditions it was not trained for. |
| **Cost (C)** | 0.5 to 5 | Energy the donor spends on a gift. |
| **Drain** | 0 to 2 | Energy every agent uses per step (0.6 in training); crowding adds 0.08 per nearby agent. |
| **Speed** | 1 to 30 steps per second | Playback speed only; it does not change results. |
| **World seed** | any whole number | Sets the random starting world. The same seed and settings always give the same run. This is not the training seed. |
| **Add up to 12 mimics** | | Adds non-donating agents that copy the most common family's genetic tag. |
| **Scarcity shock** | | Doubles energy use for 30 steps. |
| **Download results (CSV)** | | Saves the current run, one row per step. |
| **Enforce Hamilton rule** | on or off (off by default) | Blocks every gift unless relatedness x B is greater than C. This forces the rule; it is not learned behaviour. |
| **Compare fresh runs** | | Runs the same world once with each behaviour and lists them side by side. |
| **Run 10 seeds** (Reproduce a result) | | Re-runs all 10 trained models of the chosen condition in your browser and shows the outcome next to the recorded Python result. The browser uses its own random numbers, so single worlds can differ while the totals agree. |
