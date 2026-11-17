> Historical design document. Current behaviour and scientific scope are described in [the browser manual](web_simulation.md) and [review response](review_response_2026-11-17.md).

# Technical Design Specification: Aligned Light-Themed Kinship Simulation

**Project:** Emergent Kin Selection and Spatial Assortment in Multi-Agent Reinforcement Learning  
**Course & Team:** CS329 · Group 21 (Vansh Barfiwala, Pratham Choksi, Shubham Chandak)  
**Document Purpose:** Streamline the web demonstration to align strictly with the 4-page proposal PDF (`cs329_ideation_Group_21 (1).pdf`), eliminate unnecessary non-proposal clutter, transition from the dark cyberpunk theme to an academic light-slate theme, and place the simulation arena on the left side with essential controls on the right.

---

## 1. Motivation & Scope Reduction

### 1.1 What We Are Removing (Non-Proposal Clutter)
The previous web interface contained extraneous sections that made the page overwhelming and departed from the proposal:
- ❌ Dark neon / cyberpunk color palette (`#0b1115` black background, neon green glows).
- ❌ Bloated multi-card marketing headers and "Small decisions. Collective survival" slogans.
- ❌ Bulky "Recorded Research Benchmarks" evaluation tables with statistical confidence intervals.
- ❌ Verbose "Run Notebook" event logs.
- ❌ Large accordion panels with deep neural weight hashing and CSV exports.

### 1.2 What We Are Keeping (Strict Alignment with the PDF)
We retain only the core components explicitly formalized in the proposal:
1. **Toroidal Spatial Grid $G$** with periodic boundary conditions (Definition 1, Section 2).
2. **8-bit Binary Chromosome $c_i \in \{0, 1\}^8$** & Hamming-based relatedness $\tilde{r}_{ij} = \max(0, \frac{h_{ij} - \bar{h}}{1 - \bar{h}})$ (Definition 2).
3. **Behavioral History $b_j$** (reputation score over interaction window $K$).
4. **Energy Dynamics**: Single-step transition $\Delta E_i = -c_{base} - c_{dens}|\Omega_i| - C \cdot \mathbb{I}(a_i = give) + B \sum \mathbb{I}(a_j = give(i))$ (Equation 2).
5. **Mitosis & Starvation**: Energy threshold $E_{mitosis}$, offspring spawning in adjacent empty cells with mutation $\mu$, and death when $E \le 0$ (Section 3.2).
6. **Hamilton's Rule Verification**: Dynamic calculation of $r^* = C/B$ and comparison against observed relatedness $r$ (Definition 1).
7. **Four PDF Experimental Phases** (Section 4.2):
   - **Phase 1 (Selfish Baseline):** No cooperation, resource depletion.
   - **Phase 2 (Adaptive Social Affinity):** Altruistic kin cooperation sustaining population.
   - **Phase 3 (Cue-Reliability):** Genetic noise/corruption shifting trust to behavioral cues.
   - **Phase 4 (Genetic Mimicry):** Non-cooperative parasite invasion with matching tags.

---

## 2. Layout & Aesthetic Architecture

### 2.1 Two-Column Split Layout
The page adopts a focused two-column desktop layout (collapsible on mobile):

```
+-----------------------------------------------------------------------------------+
|  HEADER: CS329 Group 21 · Kin Selection in MARL (IIT Gandhinagar)    [Light Mode]  |
+------------------------------------------+----------------------------------------+
|  LEFT COLUMN: THE SIMULATION ARENA       |  RIGHT COLUMN: CONTROLS & PDF PHASES   |
|                                          |                                        |
|  1. Canvas: Toroidal Grid (24x24 / 20x20)|  1. Primary Controls (Play, Step, Reset|
|     - Clean white/light-gray background  |     Speed slider, Seed input)          |
|     - Color-coded chromosomes (8-bit)    |                                        |
|     - Energy halos & donation arrows     |  2. PDF Experimental Phases (1 to 4)   |
|     - Mimic parasite indicators          |     [Phase 1] Selfish Baseline         |
|                                          |     [Phase 2] Kin Cooperation          |
|  2. Live Agent Inspector                 |     [Phase 3] Cue Corruption (Noise)   |
|     - Click agent or pick from dropdown  |     [Phase 4] Genetic Mimics           |
|     - 8-bit chromosome visualizer        |                                        |
|     - Energy, age, reputation, kinship   |  3. Hamilton & Environmental Knobs     |
|                                          |     - Benefit B, Cost C                |
|  3. Population & Energy Sparkline        |     - Hamilton Threshold r* = C/B      |
|     - Clean light-slate micro-chart      |     - Metabolic Drain c_base           |
|                                          |                                        |
|                                          |  4. Live Population & Trust Telemetry  |
|                                          |     - Active Agents / Residents / Mimics|
|                                          |     - Genetic Weight w^g vs w^b        |
+------------------------------------------+----------------------------------------+
```

### 2.2 Color Palette (Academic Light Theme)
- **Primary Background:** `#ffffff` (Pure White) and `#f8fafc` (Slate 50)
- **Panel / Card Background:** `#ffffff` with subtle borders (`#e2e8f0` Slate 200)
- **Primary Text:** `#0f172a` (Slate 900)
- **Secondary / Caption Text:** `#64748b` (Slate 500)
- **Grid Lines:** `#f1f5f9` (Slate 100) / `#e2e8f0`
- **Accent Primary (Action / Play):** `#2563eb` (Royal Blue) or `#059669` (Emerald Green)
- **Donation Energy Beams:** `#d97706` (Warm Amber)
- **Genetic Mimic Indicators:** `#dc2626` (Crimson Red dashed ring)
- **Agent Genomes:** Distinct pastel/vibrant categorical hues representing bit strings.

---

## 3. Component Details & Behavioral Rules

### 3.1 Simulation Engine (Left Column)
- **Grid Size:** $24 \times 24$ (Phases 1-3) or $20 \times 20$ (Phase 4).
- **Periodic Boundary (Toroid):** Coordinates wrap around edges modulo $L$.
- **Rendering:**
  - Clear, high-DPI Canvas with smooth agent transitions.
  - Ring around each agent displays energy relative to $E_{mitosis} = 60$.
  - Donation events draw an amber impulse ray from donor to recipient.
  - Dashed red ring for detected or injected genetic mimics.
  - Click on any agent to inspect its 8-bit genome and local neighborhood.

### 3.2 PDF Control Deck (Right Column)
- **Phase 1 (Selfish Baseline):** Sets policy to Always Defect. Demonstrates starvation and demographic collapse without altruism.
- **Phase 2 (Adaptive Social Affinity):** Enables kin selection. Agents use genetic relatedness $\tilde{r}_{ij}$ to direct aid, sustaining the population.
- **Phase 3 (Cue Corruption):** Injects genetic bit-flip noise $\mu_{cue}$, testing if agents lower reliance on corrupted genetic tags ($w^g \downarrow$).
- **Phase 4 (Genetic Mimicry):** Injects 12 non-cooperative mimics carrying the dominant resident 8-bit chromosome. Demonstrates behavioral verification ($w^b$).
- **Hamilton Formula Box:**
  $$\text{Hamilton Condition: } r \cdot B > C \iff r > r^* = \frac{C}{B}$$
  Dynamically recalculates $r^*$ whenever $B$ or $C$ sliders are adjusted.

---

## 4. Implementation Steps
1. Create this specification file: `docs/pdf_aligned_simulation_spec.md`.
2. Rewrite `index.html` to remove all dark/cyberpunk clutter and restructure into a clean, modern, light-slate two-column application.
3. Validate locally that all controls, canvas animations, phase switching, and agent inspection work seamlessly.

