# Environment Design Specification (EDS)
## Project: Emergent Kin Selection and Spatial Assortment in Multi-Agent Reinforcement Learning (MARL)
**Authors:** Vansh Barfiwala, Pratham Choksi, Shubham Chandak (IIT Gandhinagar)  
**Document Version:** 1.0 (Formal Specification for Task 1)

---

## 1. System Overview & Formal Markov Game Formulation

The environment models a localized spatial ecosystem formalized as a partially observable Markov Game:
$$\mathcal{M} = \left\langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}_{i=1}^{N_t}, \mathcal{P}, \{\mathcal{R}_i\}_{i=1}^{N_t}, \gamma \right\rangle$$

where:
* $\mathcal{N}_t = \{1, 2, \dots, N_t\}$ is the active population at timestep $t$, bounded by maximum static capacity $N_{\text{max}}$.
* $\mathcal{S}$ is the global state space consisting of spatial coordinates, energy levels, genomes, and history buffers of all active agents.
* $\mathcal{A}_i$ is the discrete action space of agent $i$.
* $\mathcal{P}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ defines the state transition probability distribution (including movement, energy dynamics, reproduction, and death).
* $\mathcal{R}_i$ is the strictly individualistic reward function of agent $i$.
* $\gamma \in [0, 1)$ is the temporal discount factor.

---

## 2. Spatial Grid World & Boundary Physics

### 2.1 Grid Topology
* The physical space is a 2D discrete lattice $\mathcal{G} = \{0, 1, \dots, L-1\} \times \{0, 1, \dots, L-1\}$ of size $L \times L$.
* Default grid size: $L = 32$.
* **Periodic Boundary Conditions (Toroidal Lattice):** Both $x$ and $y$ axes wrap around seamlessly.
* **Toroidal Displacement Formula:** For any two positions $p_i = (x_i, y_i)$ and $p_j = (x_j, y_j)$, the shortest displacement vector $\Delta p_{ij} = (\Delta x_{ij}, \Delta y_{ij})$ is defined by:
  $$\Delta x_{ij} = \left((x_j - x_i + \lfloor L/2 \rfloor) \pmod L\right) - \lfloor L/2 \rfloor$$
  $$\Delta y_{ij} = \left((y_j - y_i + \lfloor L/2 \rfloor) \pmod L\right) - \lfloor L/2 \rfloor$$
* **Spatial Distance Metric:** Chebyshev distance (or $L_\infty$ norm) for the local observation window:
  $$d_{\infty}(p_i, p_j) = \max\left(|\Delta x_{ij}|, |\Delta y_{ij}|\right)$$
* **Observation Neighborhood ($\Omega_i$):**
  $$\Omega_i = \left\{ j \in \mathcal{N}_t \setminus \{i\} \;\middle|\; d_{\infty}(p_i, p_j) \le \rho \right\}$$
  Default observation radius: $\rho = 3$. Maximum possible neighbors in window: $(2\rho + 1)^2 - 1 = 48$.

### 2.2 Exclusion Principle (Cell Occupancy)
* Strictly **at most one agent** per cell: $\forall (x, y) \in \mathcal{G}$, $|\{i \in \mathcal{N}_t \mid p_i = (x, y)\}| \le 1$.
* Collision resolution: If multiple agents attempt to move into the same cell simultaneously, or an agent attempts to move into an already occupied cell:
  - If target cell is already occupied: the movement action fails and the agent remains in its current cell.
  - If multiple agents attempt to enter the same unoccupied cell in the same tick: one agent is selected uniformly at random to occupy the cell; the others remain in their origin cells.

---

## 3. Agent State Space Representation

For each agent slot $i \in \{1, \dots, N_{\text{max}}\}$, the state is a tuple:
$$s_i = \left(p_i, E_i, c_i, \mathbf{h}_i, \text{mask}_i, \text{age}_i, \text{parent}_i\right)$$

| Variable | Type / Domain | Definition | Default Initial Value |
| :--- | :--- | :--- | :--- |
| $p_i = (x_i, y_i)$ | $\mathbb{Z}_L^2 \in \{0, \dots, L-1\}^2$ | Grid coordinates on the toroidal lattice | Uniform random unoccupied cell |
| $E_i$ | $\mathbb{R}_{\ge 0}$ | Internal metabolic energy budget | $E_{\text{init}} = 30.0$ |
| $c_i$ | $\{0, 1\}^8$ | 8-bit inherited binary chromosome | Uniform random binary vector |
| $\mathbf{h}_i$ | $\{0, 1\}^K$ | Circular action history buffer of length $K=10$ | All zeros, count $= 0$ |
| $\text{mask}_i$ | $\{0, 1\}$ (boolean) | Active status (1 = alive, 0 = inactive/free slot) | 1 for $i \le N_{\text{init}}$, 0 otherwise |
| $\text{age}_i$ | $\mathbb{N}_0$ | Timesteps survived since birth/spawn | $0$ |
| $\text{parent}_i$ | $\mathbb{Z}$ (Index $\in [-1, N_{\text{max}}-1]$) | Tensor index of parent agent (-1 for genesis agents)| $-1$ |

---

## 4. Action Space Specification

Each active agent $i$ chooses a discrete action $a_i \in \mathcal{A}_i$:
$$\mathcal{A}_i = \underbrace{\{\text{North}, \text{South}, \text{East}, \text{West}\}}_{\text{Movement Actions}} \;\cup\; \underbrace{\{\text{Stay / Defect}\}}_{\text{Idling Action}} \;\cup\; \underbrace{\{\text{Give}(j) \mid j \in \Omega_i\}}_{\text{Targeted Donation Actions}}$$

* **Action Index Encoding (size $5 + K_{\text{max}}$ where $K_{\text{max}} = 8$ closest neighbor slots):**
  - `0`: Stay / Defect (no movement, no donation)
  - `1`: Move North ($\Delta y = +1$)
  - `2`: Move South ($\Delta y = -1$)
  - `3`: Move East ($\Delta x = +1$)
  - `4`: Move West ($\Delta x = -1$)
  - `5 + k`: Targeted donation to $k$-th neighbor sorted by spatial proximity ($k \in \{0, \dots, K_{\text{max}}-1\}$).

---

## 5. Energy Dynamics, Starvation, and Mitosis

### 5.1 Step-wise Energy Transition
At every timestep $t$, every active agent $i$ ($\text{mask}_i = 1$) updates its energy according to:
$$\Delta E_i^{(t)} = -c_{\text{base}} - c_{\text{density}} |\Omega_i| - C \cdot \mathbb{I}(a_i = \text{Give}(j)) + B \sum_{k \in \Omega_i} \mathbb{I}(a_k = \text{Give}(i))$$

#### Parameter Definitions & Operational Rules:
1. **Basal Metabolic Cost ($c_{\text{base}}$):** Constant energetic drain required for cellular respiration/existence. Default: $c_{\text{base}} = 0.5$.
2. **Crowding / Density Penalty ($c_{\text{density}} |\Omega_i|$):** Metabolic penalty proportional to local spatial competition. Default: $c_{\text{density}} = 0.1$.
3. **Donation Cost ($C$):** Strict energetic cost paid by donor. Default: $C = 2.0$.
   - **Precondition:** An agent can only execute $\text{Give}(j)$ if its current energy $E_i > C$. If $E_i \le C$, the donation action fails and reverts to $\text{Stay}$.
4. **Donation Benefit ($B$):** Energy received by the recipient. Default: $B = 6.0$ (giving a baseline ratio $B/C = 3.0$).
5. **Additive Reception:** If multiple neighbors donate to agent $i$ in the same tick, agent $i$ receives the sum of benefits: $\sum B$.
6. **Strict Conservation:**
   - Donor energy: $E_i \leftarrow E_i - C$.
   - Recipient energy: $E_j \leftarrow E_j + B$.
   - Net ecosystem energy gain: $\Delta E_{\text{ecosystem}} = B - C = +4.0$ per successful donation.

### 5.2 Starvation Mechanics (Terminal Elimination)
* If $E_i^{(t+1)} \le 0.0$:
  - Agent $i$ experiences immediate terminal starvation.
  - $\text{mask}_i \leftarrow 0$.
  - Grid cell $p_i$ is vacated ($\text{occupancy}(p_i) \leftarrow -1$).
  - Slot $i$ in the static tensor pool is flagged as available for recycling.

### 5.3 Mitosis Mechanics (Reproduction & Inheritance)
* Condition: If $E_i^{(t+1)} \ge E_{\text{mitosis}}$ AND total active population $\sum_{m=1}^{N_{\text{max}}} \text{mask}_m < N_{\text{max}}$:
  1. **Adjacent Offspring Placement:** The environment inspects the 4 Von Neumann adjacent cells:
     $$\mathcal{C}_{\text{adj}}(p_i) = \left\{(x_i \pm 1 \pmod L, y_i), (x_i, y_i \pm 1 \pmod L)\right\}$$
     If at least one adjacent cell is unoccupied, one is selected uniformly at random as $p_k$. (If all 4 cells are occupied, mitosis is deferred).
  2. **Strict Energy Conservation (Zero Energy Creation):**
     $$E_i \leftarrow \frac{E_i}{2}, \quad E_k \leftarrow \frac{E_i}{2}$$
  3. **Slot Allocation:** The lowest available inactive index $k = \min \{m \mid \text{mask}_m = 0\}$ is activated ($\text{mask}_k \leftarrow 1$).
  4. **Genomic Inheritance & Mutation:**
     Offspring $k$ inherits chromosome $c_k$ from parent $c_i$ with per-bit point mutation probability $\mu$:
     $$\forall b \in \{1, \dots, 8\}, \quad c_k^{(b)} = \begin{cases} 1 - c_i^{(b)} & \text{with probability } \mu \\ c_i^{(b)} & \text{with probability } 1 - \mu \end{cases}$$
     Default mutation rate: $\mu = 0.05$.
  5. **Behavioral Reset:**
     Offspring behavioral history buffer $\mathbf{h}_k$ is reset to empty, with baseline cooperation prior $b_0 = 0.5$.
  6. **Lineage Tracking:**
     $\text{parent}_k \leftarrow i$, $\text{age}_k \leftarrow 0$.

---

## 6. Information Cues Specification

### 6.1 Genetic Relatedness Cue ($\tilde{r}_{ij}$)
1. **Normalized Hamming Similarity ($h_{ij}$):**
   $$h_{ij} = 1 - \frac{1}{8} \sum_{b=1}^8 \left| c_i^{(b)} - c_j^{(b)} \right| \in [0.0, 1.0]$$
2. **Dynamic Population Baseline ($\bar{h}$):**
   Mean pairwise similarity across all active agent pairs:
   $$\bar{h} = \frac{2}{N_t(N_t - 1)} \sum_{m < n, \; \text{mask}_m=\text{mask}_n=1} h_{mn}$$
3. **Baseline-Corrected Relatedness ($\tilde{r}_{ij}$):**
   $$\tilde{r}_{ij} = \max\left(0, \frac{h_{ij} - \bar{h}}{1 - \bar{h} + \epsilon}\right)$$
   where $\epsilon = 10^{-6}$ prevents division by zero. If $N_t \le 1$, $\tilde{r}_{ij} = 0.0$.

### 6.2 Behavioral History Cue ($b_j$)
* Each agent maintains a sliding FIFO action buffer $\mathbf{h}_j$ of maximum length $K = 10$.
* For every step agent $j$ is alive, record $1$ if $a_j \in \{\text{Give}(k)\}$, and $0$ otherwise.
* Behavioral cooperation score:
  $$b_j = \begin{cases} b_0 = 0.5 & \text{if } \text{age}_j = 0 \\ \frac{1}{\min(\text{age}_j, K)} \sum_{\tau=1}^{\min(\text{age}_j, K)} \mathbf{h}_j[\tau] & \text{if } \text{age}_j > 0 \end{cases}$$

---

## 7. Model Observation & Neural Interface

For an active agent $i$, its local observation $o_i$ consists of:
1. **Self State:** $\mathbf{v}_{\text{self}} = [E_i / E_{\text{mitosis}}, \text{age}_i / 1000] \in \mathbb{R}^2$.
2. **Neighbor Feature Matrix:** For each valid neighbor $j \in \Omega_i$:
   $$\mathbf{u}_{ij} = \left[ \frac{\Delta x_{ij}}{\rho}, \; \frac{\Delta y_{ij}}{\rho}, \; \frac{E_j}{E_{\text{mitosis}}}, \; \tilde{r}_{ij}, \; b_j \right] \in \mathbb{R}^5$$
   Zero-padded up to fixed capacity $K_{\text{max}} = 8$ neighbors with a boolean presence mask $\mathbf{m}_{\Omega} \in \{0, 1\}^{K_{\text{max}}}$.

---

## 8. Summary of Hyperparameters & Configuration Defaults

| Parameter | Symbol | Value | Rationale |
| :--- | :---: | :---: | :--- |
| Grid Dimensions | $L \times L$ | $32 \times 32$ | Standard spatial simulation size |
| Max Population Capacity | $N_{\text{max}}$ | $150$ | Static tensor allocation limit |
| Initial Population | $N_{\text{init}}$ | $40$ | Sufficient density without immediate overcrowding |
| Initial Energy | $E_{\text{init}}$ | $30.0$ | Provides initial survival buffer ($\approx 60$ ticks) |
| Mitosis Threshold | $E_{\text{mitosis}}$ | $60.0$ | Requires accumulating energy through cooperation |
| Basal Metabolic Cost | $c_{\text{base}}$ | $0.5$ | Constant energy drain |
| Crowding Density Cost | $c_{\text{density}}$ | $0.1$ | Local competition pressure |
| Donation Cost | $C$ | $2.0$ | Altruistic cost to donor |
| Donation Benefit | $B$ | $6.0$ | Benefit received by recipient ($B/C = 3.0$) |
| Per-bit Mutation Rate | $\mu$ | $0.05$ | Genomic drift across generations |
| Behavioral Window | $K$ | $10$ | Recency memory window |
| Observation Radius | $\rho$ | $3$ | Chebyshev neighborhood window |
| Episode Length | $T_{\text{max}}$ | $1000$ | Maximum steps per trajectory rollout |
| Individual Reward | $R_i^{(t)}$ | $\Delta E_i^{(t)}$ | Strictly individualistic energy change |

---

## 9. Verification & Acceptance Criteria for Task 1
The environment specification is considered complete and approved when:
1. Toroidal coordinate wrapping and Chebyshev distance metrics are mathematically closed and unambiguous.
2. Energy conservation laws are strictly enforced during both donation transfers and mitosis divisions.
3. Death and reproduction rules do not create or destroy memory allocations (fixed tensor bounds).
4. All observation dimensions and action indices are explicitly mapped for PyTorch tensor compatibility.

