> Current model and reporting changes are documented in [the 18 November response](final_review_response_2026-11-18.md). Earlier verification entries below are historical snapshots.

# Kinship Lab: browser simulation

Implemented on 16 November 2026; simplified on 17 November after reviewing the proposal PDF. The frontend uses a plain white layout with controls in a left sidebar and the simulation and results on the right. The current Python environment, policies, and recorded artifacts take precedence over illustrative claims in the original blueprint and design specification.

## Open and use

**Current behaviour, review response of 18 November:** Hamilton enforcement is **off by default**. Donation choices remain subject to affordability and the non-mimic donor restriction. The visible checkbox enables an explicitly forced Hamilton filter; counts in that mode are not evidence of learning. Below-threshold donations are permitted and counted in observation mode.

The last-step checks retain decision-time values. Turning enforcement on blocks rB <= C (including equality), without a donation transfer. CSV records the enforcement setting, donations and energy delivered to mimics, and their share of post-invasion donations. The share is null without an invasion or post-invasion donations. Mimic extinction alone is not evidence of defense.

Play/Step/Reset are at the top. Mobile collapses secondary settings initially. Behaviour/energy edits mark the experiment “Custom settings.” Automatic injection reports its outcome. Animation progress is clamped to [0,1] using a consistent clock; an unexpected step/drawing error stops playback visibly and Reset starts a fresh run without losing the frame loop.

Double-click `index.html` in the repository root. Styles, simulation, canvas renderer, live charts, and the trained attention actor are embedded. Research figures load from relative repository paths, so keep the complete checkout for offline viewing. No external service or browser runtime dependency is required.

If you prefer an HTTP preview, run this from the repository root, then open `http://127.0.0.1:8000`:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

- **Play / Pause:** advance at the selected 1–30 simulation steps per second. Rendering is separate from simulation timing. A hidden tab does not accumulate missed simulation steps.
- **Step:** pause and execute exactly one transition.
- **Reset:** restart with the entered seed and current strategy, environmental parameters, and scenario configuration. It clears interventions, history, and agent selection. An invalid seed prevents reset.
- **Sliders and strategy:** apply immediately to the current run without replacing the population. Changing a strategy retains agents' existing donation histories.
- **Inject 12 mimics:** copy the most frequent living resident genome, breaking ties by genome number, and occupy up to 12 uniformly sampled empty cells in free slots. Mimics start with neutral reputation and the scenario's initial energy. They never donate. Full capacity means zero injections; residents are never overwritten.
- **Scarcity shock:** double the current base drain for exactly 30 transitions. Changing the drain during a shock changes its doubled value. Pausing also pauses shock duration; repeated shocks cannot stack.
- **Inspect:** click an agent or use the agent selector to see its persistent identity, parent, energy, generation, genome, and recent donation ratio.
- **Export CSV:** download the entire trajectory and intervention log. The charts also expose their latest values to assistive technologies.

Space toggles playback and the right arrow steps when focus is outside a form control or link. The canvas can receive keyboard focus. Reduced-motion preferences suppress birth ripples and pulsing animations. Runs stop at extinction or 300 transitions.

## Fresh scenarios

All scenarios start paused with 24 founders, capacity 45, cost C = 2, mitosis threshold 60, mutation probability 0.05 per bit, observation radius 3, and the entered seed (initially 42).

| Scenario | Grid | Initial energy | Benefit B | Base drain | Crowding cost | Automatic invasion |
| --- | --- | --- | --- | --- | --- | --- |
| Learned PPO (default) | 18 × 18 | 30 | 5 | 0.6 | 0.08 | No |
| Cooperation | 18 × 18 | 30 | 5 | 0.6 | 0.08 | No |
| Selfish baseline | 18 × 18 | 30 | 5 | 0.6 | 0.08 | No |
| Genetic matching | 18 × 18 | 30 | 5 | 0.6 | 0.08 | No |
| Mimics at step 40 | 18 × 18 | 25 | 3 | 0.8 | 0.10 | After 40 transitions |
| Mimics in stress ecology | 20 × 20 | 25 | 3 | 0.8 | 0.10 | After 40 transitions |

Learned PPO selects the trained attention model. The selfish scenario selects Never donate; Genetic matching selects Help genetic relatives; Cooperation and the invasion presets select Help recent donors. The Experiment selector replaces the current run and restores parameter defaults. The invasion scenarios use a lower benefit and higher drain as an explicit out-of-distribution stress test. An attack remains conditional on resident survival and free slots. The automatic attack occurs at state 40, immediately before transition 41; it is logged at step 40. The stress-ecology scenario is an invasion experiment, not cue corruption. The browser does not implement the proposal's cue-corruption sweep or gated-attention training.

The stress-ecology invasion preset uses the historical Phase 4 benefit and drain settings, with 24 founders and capacity 45. It is a distribution-shift demonstration, not a replay of the historical Python evaluation.

## Scientific fidelity

The JavaScript engine ports `environment/grid_world.py`, `environment/cues.py`, and the configured Phase 4 rule policies:

1. All policy decisions use one pre-step state. Neighbors are ordered by toroidal Chebyshev distance, then slot number; policies observe the nearest eight.
2. Donations require donor energy strictly greater than C. A donation costs C and gives B, and mimics cannot donate. A moving recipient retains its identity. The optional browser Hamilton filter adds a strict rB > C check; it defaults off, matching the Python environment.
3. Simultaneous movement allows swaps, cycles, and chains into vacated cells. Destination conflicts randomly select one winner; stationary blockers cancel dependent movement.
4. Metabolism counts all neighbors within radius 3 after movement, not just the eight observed by a policy.
5. Starvation precedes mitosis. Mitosis places a child in a free orthogonally adjacent toroidal cell and halves the parent's energy without creating energy. Newborns do not reproduce again in the same transition.
6. Children inherit genomes and mimic status, with independent bit mutations. Newborn reputation is 0.5 until the first recorded action. Reputation then measures donation frequency over up to ten actual actions.
7. Individual identities and parent identities remain distinct from reusable pool slots, including when a death and birth reuse a slot in one transition.

Reputation Threshold selects the best recipient with b ≥ 0.5. Fixed-Weight Tag Trust selects the highest 0.5r + 0.5b, requiring a score strictly greater than 0.35. Pure Kin requires r > 0.3. These thresholds match the current Phase 4 configuration, including strict comparisons. Non-donating rule-based residents stay; the selfish policy randomly moves or stays. Mimics cannot donate, but can move under the selfish and neural policies, as in Python; under the stationary rule-based policies they stay.

Relatedness is baseline-corrected Hamming similarity over distinct living pairs, calculated from the population rather than manually assigned. It is not an independently estimated pedigree coefficient. The reference r* = C/B updates with sliders, including B = 0. Policies propose actions; only the optional Hamilton filter blocks proposals failing rB > C. B/C is not directly observed by these policies. Compliance after filtering is not proof of learned Hamilton behaviour. Rule-based outcomes are not precomputed or scripted.

The default neural option evaluates the actual exported contextual attention actor, including neighbourhood-context gating and two-head attention. It uses clean lineage seed 42 after the follow-up evaluation, otherwise revised demographic seed 42. The visible note identifies its objective and 8,192-step training budget. The checkpoint SHA256 is verified before export. JavaScript probabilities match PyTorch across zero to eight neighbours; action masking and seeded replay are tested. Weights are frozen; no training occurs in the browser. The four scripted options are explicitly labelled baselines, not learned policies.

A visible Results section presents saved revised tables, cue sensitivity, learning curves, historical benchmarks and regression, plus the lineage follow-up after evaluation. Source links and interpretation caveats accompany the values. The top headline states the unsupported kin-selection hypothesis. Use Compare five fresh runs to compare rates and time to capacity even when rule baselines all saturate the population pool.

Browser runs use a seeded Mulberry32 generator and JavaScript numbers; Python uses NumPy/PyTorch RNGs and some float32 tensors. Same-seed browser resets reproduce browser trajectories when the intervention schedule is identical. They do not reproduce Python seed trajectories. Both implementations copy the modal resident genome (smallest genome breaks ties), and sample empty cells randomly without replacement.

The Hamilton panel counts actual donations plus proposals blocked by rB ≤ C. Actual donations at/below the threshold must stay zero in enforced mode. Relatedness is captured before movement, starvation, reproduction, or baseline updates. Cumulative counts and detailed checks retain decision-time values even if sliders change later. Mean donor-recipient relatedness is shown for the latest step, or a dash if none occurred. Saved multi-seed research results remain in `results/` and are displayed separately from the live simulation below it.

## Export semantics

CSV also includes `hamiltonBlocked`, `energyBlocked`, `mimicBlocked`, `attempts`, `enforceHamilton`, and `checks`. `checks` is JSON with every latest-step proposed transfer, full numerical values, outcomes, and reasons. Reason counters can overlap: a proposal can fail several conditions. Unique blocked attempts are `attempts - totalDonations`. These are attempted donations, not all available recipient opportunities.

CSV contains one row per state, including state 0. Rows contain population, resident/mimic counts, mean energy, last-step and cumulative donations, cumulative `hamiltonAbove` and `hamiltonBelow` counts, births, deaths, total injections, environmental settings, strategy, seed, grid size, initial energy, crowding, mutation rate, automatic-invasion setting, and events.

An intervention applied while paused updates that state row and appends an event; it does not duplicate the step. Parameter values and `shockRemaining` describe the state after that boundary's interventions, hence the configuration for the next transition. `donations` describes the preceding transition. CSV retains all intervention events. It is a measurement export, not a full restorable world snapshot.

## Verify or update

No build is required to use or host the page. Developer checks:

```powershell
node --test tests/web_simulation.test.cjs
python -m pytest -q
python visualizer/export_web_data.py --check
```

The Node suite covers deterministic replay, toroidal neighbors, swaps/cycles/chains, collision resolution, donation timing and affordability, mitosis, mutation, slot recycling, reputation, relatedness, interventions, extinction, export, and all five strategies across several seeds. Four Python tests compare observations, rule actions, multi-step energy/history/movement, attention action probabilities, and embedded research data against Python sources.

The Python parity fixtures explicitly set ecology and disable enforcement (`enforceHamilton:false`). The UI defaults to this observation mode; separate Node tests explicitly enable enforcement to verify allowed/blocked transfers. Run `python visualizer/check_web_robustness.py` for decision CSV, equality failures, input validation, 6× CPU-slowed playback with births, visible controls, mobile settings, and recovery from a deliberately injected rendering error in Chrome and Edge.

After regenerating the attention checkpoint or research results, refresh the embedded data and visible tables:

```powershell
python visualizer/export_web_data.py
```

Optional browser checks use Playwright as a development dependency only:

```powershell
python -m pip install playwright
python visualizer/check_web_demo.py --browsers chrome msedge
```

Chrome and Edge must already be installed. The checker also supports `--browsers firefox` with a Playwright Firefox installation, or `--firefox-executable` pointing at an existing compatible Playwright Firefox binary. Screenshots and CSV downloads go into the ignored `.review_tools` directory.

Executed verification on 16 November 2026: **21 Node tests and 53 Python tests passed**. Chrome, Edge, and Firefox acceptance checks passed at 1,440, 768, 390, and 320 CSS pixels, including all five strategies, keyboard controls, seeded resets, live parameters, agent selection, injections, automatic invasion, shock expiry, CSV downloads, extinction, no document-level horizontal overflow, and no external runtime requests.

Frontend verification on 17 November 2026: **23 Node tests and all four Python browser-engine parity tests passed**. Chrome and Edge acceptance checks passed at 1,440, 1,024, 768, 620, 390, and 320 CSS pixels. These verify the sidebar layout, Hamilton counts, controls, five policies, replay, keyboard controls, interventions, inspection, CSV export, extinction, and offline operation. Desktop and mobile screenshots were visually reviewed.

After adding enforcement and fixing animation timing: **28 Node tests and all four Python parity tests passed** (parity fixtures explicitly use observation-only mode). Chrome and Edge passed the ordinary acceptance suite and additional regression checks for strict inequality, failure explanations, decision CSV, invalid seeds, laptop control visibility, mobile layout, sustained playback at 30 steps/s with 6× CPU slowdown and births, and Reset recovery after a deliberately injected drawing exception. Embedded research artifacts still match their source files. These checks cover known defects and tested boundaries; they do not guarantee that no future error can occur.

## GitHub Pages

GitHub Pages is configured to deploy the repository root from `main`, with `.nojekyll` present. No npm build, backend, or Sites hosting is required. Keep the research files alongside the HTML for its figures and source links.

Push updates to `main`, then check the **pages build and deployment** workflow and verify the [live page](https://chandak-shubham.github.io/cs329_project/). The 18 November response retains the existing hosting configuration and checks the published HTML against the committed file.
