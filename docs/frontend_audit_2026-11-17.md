# Frontend audit — 17 November 2026

**Historical snapshot:** the follow-up implementation on 17 November fixes the animation crash, moves playback controls to the top, collapses mobile settings, labels customised presets, announces automatic injections, and adds chart step labels. At the user's explicit request, the browser now blocks donations failing rB > C and shows decision-time values and failure reasons. The observations and sweep below describe the pre-enforcement version. See `web_simulation.md` for current semantics and `visualizer/check_web_robustness.py` for regression coverage. Research gaps about learned behaviour remain applicable; enforcing the rule does not establish learning.

The page is a useful, simple behaviour simulator, but is not yet ready for a reliable professor demonstration on slower hardware, and it does not demonstrate the proposal's full learned-cooperation claim. The highest-priority defect is a reproducible animation crash. The main research gap is that the browser runs the MLP ablation and rule-based policies, not the proposed gated-attention model.

This was an audit. Application code and research results were not changed.

## Scope and evidence

- Opened the local page in automated Chrome and Edge, exercised its actual controls, and reviewed desktop/mobile screenshots.
- Opened the deployed page at https://chandak-shubham.github.io/cs329_project/ in Chrome. HTTP 200; its HTML matches local `index.html` after newline normalization.
- Re-read the proposal's research questions and evaluation requirements.
- Ran the full Python suite: **53 passed**, with one pytest plugin deprecation warning.
- Ran the JavaScript engine suite: **23 passed**.
- Ran Chrome and Edge browser acceptance checks at widths 1440, 1024, 768, 620, 390, and 320 pixels: **passed** for the tested interactions and no horizontal page overflow.
- Verified embedded checkpoint/research artifacts against source files: **passed**.
- Added an exploratory audit covering invalid seeds, seed boundaries, full-capacity injection, automatic invasion, preset changes, the 1,000-step limit, and sustained 30-step/s playback with CPU slowdown.
- Ran all five behaviours across world seeds 42, 101, 202, 303, and 404, to extinction or 300 steps, using default energy settings.

Passing engine and ordinary browser tests did not catch the rendering failure below. This audit therefore supersedes an interpretation of the previous test results as an all-clear.

## Findings, in priority order

### 1. High: animation can crash and silently stop playback

**Reproduction:** open the live site in Chrome, use DevTools CPU throttling at 6× slowdown, set speed to 30 steps/s, and press Play on the default cooperation scenario. In the recorded live run it stopped at step 7; the local run stopped at step 12. The exact failing step is timing-dependent. Normal-speed hardware continued through steps 210–241 in the control run.

The browser reports:

```text
IndexSizeError: Failed to execute 'arc' on 'CanvasRenderingContext2D':
The radius provided (-8.58125) is negative.
at drawWorld (.../:635:13)
at frame (.../:742:9)
```

**Cause:** `advance()` records `visualStart = performance.now()`, but `drawWorld(time)` uses the animation frame's earlier timestamp. The animation phase is capped at 1 but is not bounded below by 0 (`index.html:601`). After sufficient work in one frame, the birth-ripple radius `cell * (0.3 + phase * 1.2)` becomes negative (`index.html:635`). The exception occurs before the next frame is scheduled (`index.html:749`), so rendering/playback stops while the running flag remains true. Rapid programmatic Step clicks reproduced the same issue. After the error, Reset updates numbers but does not restart the lost frame loop.

**Required work:** keep animation phase in [0, 1], use a consistent timestamp, and add a browser regression that checks the step counter continues advancing during sustained playback under CPU slowdown. If rendering fails unexpectedly, stop with a visible error rather than retain a running status. This is a functional fix, not a design change.

### 2. High research gap: the live page does not demonstrate the proposed model

The five options are four rule-based strategies and the saved seed-42 MLP. The default “Help recent donors” is hardcoded reputation thresholding. Genetic matching uses a fixed r > 0.3 threshold. Neither demonstrates an emergent learned decision boundary. The page correctly labels the PPO option as MLP and notes that it does not retrain.

The proposal's central mechanism is gated attention, and its main cue-reliability question requires genetic corruption and genetic/behavioural weight measurements. The frontend has no gated-attention inference, corruption control, trust-weight output, or spatial-assortment comparison. Those are interface/integration gaps; related implementations and experiments already exist in the Python project.

**Required work:** show the actual gated-attention policy or an explicitly labelled recorded replay, alongside one simple rule-based baseline. Add a small cue-noise control and genuine weight measurements if demonstrating the proposal's cue-reliability experiment. There is no need for browser-side training or a more elaborate visual design.

### 3. High interpretation risk: current outcomes do not establish learned cooperation or Hamilton's rule

The Hamilton panel correctly classifies donations using decision-time relatedness and the strict rB > C inequality. Its explanatory caveat is appropriate. However, counts of donations alone lack the denominator of available recipient opportunities. They cannot establish that an agent is more likely to donate above the threshold, and they cannot establish an evolutionary fitness benefit.

B/C is absent from the neural observation. Sliders change donation economics and future world states; they do not give the policy an explicit new B/C input. Changes in behaviour after adjusting them are not by themselves evidence of learned threshold sensitivity.

In the new browser-engine sweep, the saved MLP became extinct in every tested world. Existing Python reports also describe short training budgets and limited evidence: `docs/completion_2026-11-16.md` reports Phase 2 final populations of zero across its five runs and a Hamilton threshold contrast whose interval crosses zero. These are valid negative or inconclusive findings, not frontend defects to hide.

**Required work:** provide a compact controlled comparison using the same initial settings and several seeds. For Hamilton consistency, compare donation rates against eligible recipient opportunities above and below the threshold, with clear definitions and uncertainty, or show the existing controlled policy probes with their limitations. Label rule-based and learned results separately. Keep online behaviour, training-condition effects, and proof claims distinct.

### 4. Medium: key controls and the grid do not fit a typical laptop viewport

At 1366×768 on initial load, Play occupies y=894–933, below the visible screen. The grid extends from y=347.5 to y=862.4. Export is at y=1140.5 and the inspector selector at y=1270.9. The sidebar has its own scrollbar, so controls are reachable, but the professor must scroll before the first run and move between separate scrolling areas.

At 390×844, the grid starts around y=1594.5, after the whole control panel. There is no horizontal overflow, but the layout is inefficient for viewing changes while adjusting settings.

**Required work:** put Play/Pause, Step, and Reset at the top of the sidebar or in a persistent control row; reduce spacing enough for the grid and main results to fit a laptop screen. On narrow screens, use a collapsible control panel. Preserve the requested white sidebar/results layout.

### 5. Medium: experiment label can describe the wrong current behaviour

**Reproduction:** choose “Selfish baseline,” then change Agent behaviour to “Help recent donors.” The experiment selector continues to say “Selfish baseline” even though the active strategy is now reputation-based. Changing energy settings similarly leaves the original preset name selected.

The controls themselves apply the correct values, and the page says changes apply immediately. The issue is ambiguous experiment identity during explanation and comparison.

**Required work:** show “Custom settings” or “Modified from Selfish baseline” after changing preset parameters. Include the actual policy and settings in a saved run summary.

### 6. Medium: automatic invasion has no explicit visible event message

In the invasion preset, advancing to transition 41 injected 12 mimics correctly. The intervention notice remained empty. The population numbers and grid changed, and the event was recorded internally/for CSV, but the user received no “12 mimics added after step 40” notice.

**Required work:** reuse the existing status message for automatic injection, including a clear explanation if no invasion occurs because the pool is full or residents have died. Add a small chart marker if useful.

### 7. Low: charts and wording need small explanatory improvements

The charts draw vertical-axis numbers but no numeric horizontal-axis tick labels (`index.html:687` onward). “Horizontal axis: steps” explains the variable but does not let a reader identify the time of an event precisely. The summary says “After 1 steps” and remains generic rather than explaining the relation between selected behaviour and the observed outcome. The energy drain slider names only base drain; total drain also includes a crowding cost that appears only in the scenario details after reset.

**Required work:** add a few step labels, correct singular/plural text, and state briefly that crowding adds energy cost. Keep any result interpretation descriptive rather than attributing causation from one run.

### 8. Low: README collision wording is ambiguous

`README.md:51` says competing destination intents are cancelled. Both implementations choose one random winner and cancel the other conflicting intents. `docs/web_simulation.md` describes this correctly.

**Required work:** make README wording explicit that one winner moves. Avoid using historical design documents as the current implementation contract.

## What works correctly

- White theme and left-control/right-result structure match the requested direction.
- Play/pause, individual stepping, reset, seed changes, policy selection, benefit/cost/drain controls, and speed controls pass ordinary browser checks.
- Four invalid seed cases (empty, negative, above uint32, fractional) are rejected without resetting the current population; seeds 0 and 4294967295 are accepted.
- Same seed and same intervention schedule reproduce JavaScript trajectories. This does not imply matching Python random trajectories.
- All five behaviour options execute. Neural inference matches the saved PyTorch MLP logits in the parity tests.
- Toroidal neighbourhoods, movement chains/swaps, donor affordability, energy transfers, death, reproduction, inherited genomes, mutation, history, and mimic restrictions pass the current automated tests.
- Hamilton counts sum to total donations and retain decision-time classification through demographic changes. Zero benefit and equality cases are handled.
- Full population capacity is respected. In the default seed-42 world, the pool was full by step 10; clicking injection correctly reported zero injected without replacing residents.
- Invasion timing is correct: after 40 completed transitions, before transition 41.
- Scarcity shock lasts 30 simulation transitions, pauses with the simulation, and returns to the current base drain.
- Agent selection/inspection and CSV downloads pass; CSV includes intervention history and Hamilton counts.
- Extinction disables progression. A separate 1,000-step default run reached the observation limit with 60 living agents and 59,396 donations; all were accounted for by the two Hamilton counts. Reset cleared those counts.
- No external runtime dependencies are required for local operation. The deployed page loads successfully and matches the current local application.

## New exploratory behaviour sweep

Conditions: grid 24×24; 30 founders; capacity 60; initial energy 30; B=6; C=2; base drain 0.5; crowding cost 0.05; mutation 0.05; no intervention; up to 300 steps. World seeds: 42, 101, 202, 303, 404.

| Behaviour | Population at end in each world | Extinction steps | Interpretation |
| --- | --- | --- | --- |
| Never donate | 0, 0, 0, 0, 0 | 54, 58, 53, 55, 56 | Selfish baseline collapses in these settings. |
| Help genetic relatives | 60, 60, 60, 60, 60 | None by step 300 | Hardcoded kin rule sustains the population to the observation limit. |
| Use both cues equally | 60, 60, 60, 60, 60 | None by step 300 | Hardcoded combined-cue rule sustains the population to the observation limit. |
| Help recent donors | 60, 60, 60, 60, 60 | None by step 300 | Hardcoded reputation rule sustains the population to the observation limit. |
| Saved PPO MLP | 0, 0, 0, 0, 0 | 108, 79, 96, 101, 92 | The saved MLP fails to sustain the population in these tested worlds. |

The neural row uses one checkpoint across five world/action seeds, not five independently trained models. These are exploratory simulator checks, not a replacement for the Python study. Reaching capacity is not an estimate of equilibrium carrying capacity, and failure here does not establish that PPO cannot learn cooperation.

## Recommended work order

1. Fix the animation exception and add sustained-playback regression coverage.
2. Bring primary controls and the grid into the initial laptop viewport; distinguish modified presets and announce automatic events.
3. Add one compact, fair baseline-versus-learned comparison with saved settings and seed information.
4. Connect the actual gated-attention results/model and cue-reliability measurements if the presentation must cover the proposal's central claim.
5. Improve training/evaluation evidence as a research task; report unsuccessful or inconclusive results honestly. Do not force an rB > C action rule and present that as learned behaviour.

## Limits and reproduction artifacts

Chrome and Edge were exercised. Firefox was not retested because its Playwright executable is missing; the older Firefox result is not evidence for this audit. No Safari, physical mobile device, or full screen-reader audit was performed. CPU throttling simulates a slower environment; exact failure timing varies with hardware. No claim of exhaustive correctness is made.

Detailed local evidence is in the ignored `.review_tools` directory: `frontend-audit.json`, `animation-repro.json`, `live-audit.json`, `frontend-sweep.json`, `audit-desktop.png`, `audit-mobile.png`, `audit-live.png`, and `audit-live-after-play.png`. The mobile screenshot after the edge-case sequence also shows the stale/incomplete canvas after the render loop failed; it should not be interpreted as a separate mobile-only rendering defect. Reproduction scripts are `frontend_audit.py`, `reproduce_animation.py`, `live_audit.py`, and `frontend_sweep.cjs` in the same directory.
