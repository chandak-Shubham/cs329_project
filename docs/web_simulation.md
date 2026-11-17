# Kinship Lab: browser simulation

Implemented on 16 November 2026; simplified on 17 November after reviewing the proposal PDF. The frontend uses a plain white layout with controls in a left sidebar and the simulation and results on the right. The current Python environment, policies, and recorded artifacts take precedence over illustrative claims in the original blueprint and design specification.

## Open and use

Double-click `index.html` in the repository root. Everything needed to run the app is embedded: styles, simulation, canvas renderer, charts, and the trained MLP actor. There are no network requests or browser runtime dependencies.

If you prefer an HTTP preview, run this from the repository root, then open `http://127.0.0.1:8000`:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

- **Play / Pause:** advance at the selected 1–30 simulation steps per second. Rendering is separate from simulation timing. A hidden tab does not accumulate missed simulation steps.
- **Step:** pause and execute exactly one transition.
- **Reset:** restart with the entered seed and current strategy, environmental parameters, and scenario configuration. It clears interventions, history, and agent selection. An invalid seed prevents reset.
- **Sliders and strategy:** apply immediately to the current run without replacing the population. Changing a strategy retains agents' existing donation histories.
- **Inject 12 mimics:** copy the most frequent living resident genome, breaking ties by genome number, and occupy up to 12 free slots. Mimics start with neutral reputation and the scenario's initial energy. They never donate. Full capacity means zero injections; residents are never overwritten.
- **Scarcity shock:** double the current base drain for exactly 30 transitions. Changing the drain during a shock changes its doubled value. Pausing also pauses shock duration; repeated shocks cannot stack.
- **Inspect:** click an agent or use the agent selector to see its persistent identity, parent, energy, generation, genome, and recent donation ratio.
- **Export CSV:** download the entire trajectory and intervention log. The charts also expose their latest values to assistive technologies.

Space toggles playback and the right arrow steps when focus is outside a form control or link. The canvas can receive keyboard focus. Reduced-motion preferences suppress birth ripples and pulsing animations. Runs stop at extinction or 1,000 transitions.

## Fresh scenarios

All scenarios start paused with 30 founders, capacity 60, cost C = 2, mitosis threshold 60, mutation probability 0.05 per bit, observation radius 3, and the entered seed (initially 42).

| Scenario | Grid | Initial energy | Benefit B | Base drain | Crowding cost | Automatic invasion |
| --- | --- | --- | --- | --- | --- | --- |
| Cooperation | 24 × 24 | 30 | 6 | 0.5 | 0.05 | No |
| Selfish baseline | 24 × 24 | 30 | 6 | 0.5 | 0.05 | No |
| Genetic matching | 24 × 24 | 30 | 6 | 0.5 | 0.05 | No |
| Mimics at step 40 | 24 × 24 | 25 | 3 | 0.8 | 0.10 | After 40 transitions |
| Mimics in a smaller grid | 20 × 20 | 25 | 3 | 0.8 | 0.10 | After 40 transitions |

The selfish scenario selects Never donate; Genetic matching selects Help genetic relatives; the others select Help recent donors. The Experiment selector replaces the current run and restores parameter defaults. The invasion scenarios use a lower benefit and higher drain because the benign default fills the pool quickly. An attack remains conditional on resident survival and free slots. The automatic attack occurs at state 40, immediately before transition 41; it is logged at step 40. The smaller-grid scenario is an invasion experiment, not cue corruption. The browser does not implement the proposal's cue-corruption sweep or gated-attention training.

The smaller-grid invasion preset uses the recorded Phase 4 experiment's ecological parameters, but retains the browser's 1,000-step observation limit. It is not a replay of the 300-step Python evaluation.

## Scientific fidelity

The JavaScript engine ports `environment/grid_world.py`, `environment/cues.py`, and the configured Phase 4 rule policies:

1. All policy decisions use one pre-step state. Neighbors are ordered by toroidal Chebyshev distance, then slot number; policies observe the nearest eight.
2. Donations require donor energy strictly greater than C. A donation costs C and gives B, and mimics cannot donate. A moving recipient retains its identity.
3. Simultaneous movement allows swaps, cycles, and chains into vacated cells. Destination conflicts randomly select one winner; stationary blockers cancel dependent movement.
4. Metabolism counts all neighbors within radius 3 after movement, not just the eight observed by a policy.
5. Starvation precedes mitosis. Mitosis places a child in a free orthogonally adjacent toroidal cell and halves the parent's energy without creating energy. Newborns do not reproduce again in the same transition.
6. Children inherit genomes and mimic status, with independent bit mutations. Newborn reputation is 0.5 until the first recorded action. Reputation then measures donation frequency over up to ten actual actions.
7. Individual identities and parent identities remain distinct from reusable pool slots, including when a death and birth reuse a slot in one transition.

Reputation Threshold selects the best recipient with b ≥ 0.5. Fixed-Weight Tag Trust selects the highest 0.5r + 0.5b, requiring a score strictly greater than 0.35. Pure Kin requires r > 0.3. These thresholds match the current Phase 4 configuration, including strict comparisons. Non-donating rule-based residents stay; the selfish policy randomly moves or stays. Mimics cannot donate, but can move under the selfish and neural policies, as in Python; under the stationary rule-based policies they stay.

Relatedness is baseline-corrected Hamming similarity over distinct living pairs. It is not an independently estimated pedigree coefficient. The Hamilton reference r* = C/B updates with the sliders, including B = 0, but does not dictate policy decisions. These policies do not observe B/C directly. B ≤ C with positive metabolic costs depletes total energy; this is not proof of Hamilton's rule.

The neural option evaluates the actual `simple_mlp_seed_42.pt` actor (43 → 64 → 32 → 13, ReLU, action masking, softmax sampling). The actor is frozen and was trained for 1,024 environment transitions. The checkpoint's SHA256 is verified before export and embedded with its metadata. The browser does not load PyTorch, fake neural behavior with a defection heuristic, or train online.

Browser runs use a seeded Mulberry32 generator and JavaScript numbers; Python uses NumPy/PyTorch RNGs and some float32 tensors. Same-seed browser resets reproduce browser trajectories when the intervention schedule is identical. They do not reproduce Python seed trajectories. As requested by the blueprint, injections copy the dominant resident genome; Python's default injection copies its first living agent's genome. Both place injections in the first available x-major empty cells.

The Hamilton panel counts actual donations where rB > C and where rB ≤ C. Relatedness is captured before that transition's movement, starvation, reproduction, or baseline update. Cumulative counts retain the B/C and relatedness classification at the time of each donation even if the sliders change later. Mean donor-recipient relatedness is shown for the latest step, or a dash if no donation occurred. These descriptive counts are not a statistical consistency test: they do not compare donation rates against all available recipients or establish causation. Saved multi-seed research results remain in `results/`; the compact interface does not display the benchmark table.

## Export semantics

CSV contains one row per state, including state 0. Rows contain population, resident/mimic counts, mean energy, last-step and cumulative donations, cumulative `hamiltonAbove` and `hamiltonBelow` counts, births, deaths, total injections, environmental settings, strategy, seed, grid size, initial energy, crowding, mutation rate, automatic-invasion setting, and events.

An intervention applied while paused updates that state row and appends an event; it does not duplicate the step. Parameter values and `shockRemaining` describe the state after that boundary's interventions, hence the configuration for the next transition. `donations` describes the preceding transition. CSV retains all intervention events. It is a measurement export, not a full restorable world snapshot.

## Verify or update

No build is required to use or host the page. Developer checks:

```powershell
node --test tests/web_simulation.test.cjs
python -m pytest -q
python visualizer/export_web_data.py --check
```

The Node suite covers deterministic replay, toroidal neighbors, swaps/cycles/chains, collision resolution, donation timing and affordability, mitosis, mutation, slot recycling, reputation, relatedness, interventions, extinction, export, and all five strategies across several seeds. Four Python tests compare observations, rule actions, multi-step energy/history/movement, neural logits, and embedded research data against Python sources.

After regenerating the MLP checkpoint or Phase 4 results, refresh the embedded data:

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

## GitHub Pages

The root `index.html` and `.nojekyll` are ready for the branch-based deployment described in the original guide. No npm build, backend, or Sites hosting is required. Keep the research files alongside the HTML if you want its source links to work.

Once the changes are reviewed and pushed to the desired branch, select that branch and `/ (root)` under the repository's **Settings → Pages → Deploy from a branch**. GitHub will show the deployment URL when publication succeeds. This implementation does not push commits or change repository hosting settings.
