"""
GridWorld Environment with Dynamic Demographics (Task 7 & Algorithm 1).
Implements:
- Static-capacity Agent Pooling: N_max slots with active boolean mask M
- Mitosis: E_i >= E_mitosis triggers reproduction into adjacent empty cell
- Strict Energy Conservation: E_parent <- E_i / 2, E_offspring <- E_i / 2 (0 energy created)
- Genomic inheritance with per-bit mutation probability mu
- History reset with neutral prior b_0 = 0.5
- Starvation culling: E_i <= 0 returns tensor slot to pool
- Lineage / Phylogeny tracking (parent_id, generation, birth_step)
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .cues import (
    CHROMOSOME_LENGTH,
    random_chromosomes,
    mutate_chromosome,
    compute_pairwise_hamming,
    compute_baseline_corrected_relatedness,
    BehavioralHistoryTracker,
)

# Action Constants
ACTION_STAY = 0
ACTION_NORTH = 1
ACTION_SOUTH = 2
ACTION_EAST = 3
ACTION_WEST = 4
ACTION_GIVE_BASE = 5

MOVE_DELTAS = {
    ACTION_STAY: (0, 0),
    ACTION_NORTH: (0, 1),
    ACTION_SOUTH: (0, -1),
    ACTION_EAST: (1, 0),
    ACTION_WEST: (-1, 0),
}


class GridWorld:
    def __init__(
        self,
        grid_size: int = 32,
        max_capacity: int = 150,
        initial_population: Optional[int] = None,
        num_agents: Optional[int] = None,
        initial_energy: float = 30.0,
        e_mitosis: float = 60.0,
        enable_mitosis: bool = True,
        mutation_rate: float = 0.05,
        c_base: float = 0.5,
        c_density: float = 0.1,
        donation_cost: float = 2.0,
        donation_benefit: float = 6.0,
        obs_radius: int = 3,
        max_neighbors: int = 8,
        cue_noise: float = 0.0,
        history_window_k: int = 10,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        if num_agents is not None and initial_population is None:
            initial_population = num_agents
            max_capacity = max(max_capacity, num_agents)
        elif initial_population is None:
            initial_population = 40

        self.L = grid_size
        self.N_max = max_capacity
        self.N_init = min(initial_population, max_capacity)
        self.initial_energy = initial_energy
        self.e_mitosis = e_mitosis
        self.enable_mitosis = enable_mitosis
        self.mu = mutation_rate
        self.c_base = c_base
        self.c_density = c_density
        self.C = donation_cost
        self.B = donation_benefit
        self.rho = obs_radius
        self.max_neighbors = max_neighbors
        self.cue_noise = cue_noise
        self.window_k = history_window_k
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)

        # Grid Occupancy Matrix: (L, L), stores agent_idx or -1 if empty
        self.grid = np.full((self.L, self.L), -1, dtype=np.int32)

        # Static-Capacity State Tensors (Algorithm 1)
        self.positions = np.zeros((self.N_max, 2), dtype=np.int32)
        self.energy = np.zeros(self.N_max, dtype=np.float32)
        self.mask = np.zeros(self.N_max, dtype=bool)  # Active mask M
        self.age = np.zeros(self.N_max, dtype=np.int32)
        self.chromosomes = np.zeros((self.N_max, CHROMOSOME_LENGTH), dtype=np.int8)

        # Lineage & Phylogeny tracking
        self.parent_ids = np.full(self.N_max, -1, dtype=np.int32)
        self.generations = np.zeros(self.N_max, dtype=np.int32)
        self.birth_steps = np.zeros(self.N_max, dtype=np.int32)
        self.is_parasite = np.zeros(self.N_max, dtype=bool)  # For Phase 4 mimicry

        # Behavioral History Tracker
        self.history = BehavioralHistoryTracker(self.N_max, window_k=self.window_k)

        # Cached Cues
        self.hamming_sim = np.zeros((self.N_max, self.N_max), dtype=np.float32)
        self.r_tilde = np.zeros((self.N_max, self.N_max), dtype=np.float32)
        self.h_bar = 0.5

        self.step_count = 0

    @property
    def active_count(self) -> int:
        return int(np.sum(self.mask))

    @property
    def N(self) -> int:
        """Alias for compatibility with static interface."""
        return self.N_max

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resets the environment with N_init active agents and cleared tensor pool."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.grid.fill(-1)
        self.positions.fill(-1)
        self.energy.fill(0.0)
        self.mask.fill(False)
        self.age.fill(0)
        self.parent_ids.fill(-1)
        self.generations.fill(0)
        self.birth_steps.fill(0)
        self.is_parasite.fill(False)
        self.step_count = 0

        # Activate initial N_init slots
        self.mask[:self.N_init] = True
        self.energy[:self.N_init] = self.initial_energy

        # Assign random non-overlapping coordinates
        all_cells = [(x, y) for x in range(self.L) for y in range(self.L)]
        chosen_indices = self.rng.choice(len(all_cells), size=self.N_init, replace=False)

        for i, idx in enumerate(chosen_indices):
            pos = all_cells[idx]
            self.positions[i] = pos
            self.grid[pos[0], pos[1]] = i
            self.history.reset_agent(i)

        # Assign random initial chromosomes
        self.chromosomes[:self.N_init] = random_chromosomes(self.N_init, rng=self.rng)

        self._update_cues()
        return self._get_observations()

    def _update_cues(self):
        """Recompute Hamming similarity matrix and baseline-corrected relatedness."""
        if self.cue_noise > 0.0:
            flips = self.rng.random(size=self.chromosomes.shape) < self.cue_noise
            observed_chroms = self.chromosomes.copy()
            observed_chroms[flips] = 1 - observed_chroms[flips]
        else:
            observed_chroms = self.chromosomes

        self.hamming_sim = compute_pairwise_hamming(observed_chroms, self.mask)
        self.r_tilde, self.h_bar = compute_baseline_corrected_relatedness(
            self.hamming_sim, self.mask
        )

    def inject_mimics(self, count: int = 5, target_chrom: Optional[np.ndarray] = None) -> int:
        """
        Phase 4: Spawns non-cooperative mimics into unoccupied grid cells.
        Mimics carry chromosomes matching the target cooperative lineage,
        but are flagged with is_parasite = True.
        """
        active_indices = np.where(self.mask)[0]
        if target_chrom is None and len(active_indices) > 0:
            target_chrom = self.chromosomes[active_indices[0]].copy()
        elif target_chrom is None:
            target_chrom = self.rng.integers(0, 2, size=CHROMOSOME_LENGTH, dtype=np.int8)

        spawned = 0
        all_empty = [(x, y) for x in range(self.L) for y in range(self.L) if self.grid[x, y] == -1]
        free_slots = np.where(~self.mask)[0]

        for s_idx in range(min(count, len(all_empty), len(free_slots))):
            slot = free_slots[s_idx]
            pos = all_empty[s_idx]

            self.mask[slot] = True
            self.energy[slot] = self.initial_energy
            self.positions[slot] = pos
            self.grid[pos[0], pos[1]] = slot
            self.chromosomes[slot] = target_chrom.copy()
            self.is_parasite[slot] = True
            self.history.reset_agent(slot)
            self.age[slot] = 0
            spawned += 1

        self._update_cues()
        return spawned

    def toroidal_delta(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Shortest displacement vector from p1 to p2 on toroidal lattice."""
        half = self.L // 2
        dx = ((p2[0] - p1[0] + half) % self.L) - half
        dy = ((p2[1] - p1[1] + half) % self.L) - half
        return np.array([dx, dy], dtype=np.int32)

    def chebyshev_distance(self, p1: np.ndarray, p2: np.ndarray) -> int:
        """Chebyshev distance d_inf = max(|dx|, |dy|) on toroidal lattice."""
        delta = self.toroidal_delta(p1, p2)
        return int(max(abs(delta[0]), abs(delta[1])))

    def get_neighbors(self, agent_idx: int) -> List[int]:
        """Return active neighbor indices within Chebyshev radius rho, sorted by distance."""
        if not self.mask[agent_idx]:
            return []

        pos = self.positions[agent_idx]
        neighbors_with_dist = []

        for dx in range(-self.rho, self.rho + 1):
            for dy in range(-self.rho, self.rho + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (pos[0] + dx) % self.L
                ny = (pos[1] + dy) % self.L
                occupant = self.grid[nx, ny]
                if occupant != -1 and self.mask[occupant] and occupant != agent_idx:
                    d = max(abs(dx), abs(dy))
                    neighbors_with_dist.append((d, occupant))

        neighbors_with_dist.sort(key=lambda item: (item[0], item[1]))
        return [item[1] for item in neighbors_with_dist]

    def _resolve_movements(self, move_intents: Dict[int, Tuple[int, int]]):
        """Resolves collisions with strict 1-agent-per-cell occupancy."""
        target_requests: Dict[Tuple[int, int], List[int]] = {}
        for agent_idx, target_cell in move_intents.items():
            target_requests.setdefault(target_cell, []).append(agent_idx)

        has_changes = True
        while has_changes:
            has_changes = False
            for target_cell, candidate_agents in list(target_requests.items()):
                current_occ = self.grid[target_cell[0], target_cell[1]]
                if current_occ != -1 and current_occ not in move_intents:
                    for a in candidate_agents:
                        del move_intents[a]
                        has_changes = True
                    del target_requests[target_cell]
                    continue

                if len(candidate_agents) > 1:
                    winner = self.rng.choice(candidate_agents)
                    for a in candidate_agents:
                        if a != winner:
                            del move_intents[a]
                            has_changes = True
                    target_requests[target_cell] = [winner]

        for agent_idx, target_cell in move_intents.items():
            old_pos = tuple(self.positions[agent_idx])
            self.grid[old_pos[0], old_pos[1]] = -1
            self.grid[target_cell[0], target_cell[1]] = agent_idx
            self.positions[agent_idx] = target_cell

    def _step_demographics(self) -> Tuple[int, int]:
        """
        Executes Starvation and Mitosis (Algorithm 1 lines 5-13).
        Enforces strict energy conservation: E_i, E_k <- E_i / 2.
        Returns: (deaths_count, births_count)
        """
        deaths = 0
        births = 0
        active_indices = np.where(self.mask)[0]

        # 1. Starvation Phase
        for i in active_indices:
            if self.energy[i] <= 0.0:
                self.mask[i] = False
                pos = self.positions[i]
                if self.grid[pos[0], pos[1]] == i:
                    self.grid[pos[0], pos[1]] = -1
                self.positions[i] = -1
                deaths += 1

        # 2. Mitosis Phase
        if self.enable_mitosis:
            active_after_starve = np.where(self.mask)[0]
            for i in active_after_starve:
                if self.energy[i] >= self.e_mitosis and self.active_count < self.N_max:
                    pi = self.positions[i]
                    # Check 4 adjacent Von Neumann neighbors
                    candidates = [
                        ((pi[0] + 1) % self.L, pi[1]),
                        ((pi[0] - 1) % self.L, pi[1]),
                        (pi[0], (pi[1] + 1) % self.L),
                        (pi[0], (pi[1] - 1) % self.L),
                    ]
                    empty_cells = [c for c in candidates if self.grid[c[0], c[1]] == -1]

                    if empty_cells:
                        free_slots = np.where(~self.mask)[0]
                        if len(free_slots) > 0:
                            k = free_slots[0]
                            spawn_pos = empty_cells[self.rng.choice(len(empty_cells))]

                            # Activate slot
                            self.mask[k] = True
                            # Strict energy halving (0 energy created!)
                            half_energy = self.energy[i] / 2.0
                            self.energy[i] = half_energy
                            self.energy[k] = half_energy

                            # Position assignment
                            self.positions[k] = spawn_pos
                            self.grid[spawn_pos[0], spawn_pos[1]] = k

                            # Chromosome inheritance with mutation mu
                            self.chromosomes[k] = mutate_chromosome(
                                self.chromosomes[i], mutation_rate=self.mu, rng=self.rng
                            )
                            self.parent_ids[k] = i
                            self.generations[k] = self.generations[i] + 1
                            self.birth_steps[k] = self.step_count
                            self.is_parasite[k] = self.is_parasite[i]
                            self.age[k] = 0

                            # Reset history to neutral prior
                            self.history.reset_agent(k)
                            births += 1

        return deaths, births

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes one environment step according to Algorithm 1:
        1. Action parsing & donation tracking
        2. Movement collision resolution
        3. Energy transitions (-c_base, -c_density, -C, +B)
        4. Demographics (Starvation & Mitosis)
        5. Cue updating & termination checks
        """
        self.step_count += 1
        rewards = np.zeros(self.N_max, dtype=np.float32)

        donations: List[Tuple[int, int]] = []
        move_intents: Dict[int, Tuple[int, int]] = {}

        active_indices = np.where(self.mask)[0]

        # 1. Action parsing
        for i in active_indices:
            act = int(actions[i])
            self.age[i] += 1
            is_donation = False

            if act in [ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST]:
                delta = MOVE_DELTAS[act]
                target = (
                    (self.positions[i][0] + delta[0]) % self.L,
                    (self.positions[i][1] + delta[1]) % self.L,
                )
                move_intents[i] = target

            elif act >= ACTION_GIVE_BASE:
                target_neighbor_idx = act - ACTION_GIVE_BASE
                neighbors = self.get_neighbors(i)
                # Parasites never execute donations
                if not self.is_parasite[i] and target_neighbor_idx < len(neighbors) and self.energy[i] > self.C:
                    recipient = neighbors[target_neighbor_idx]
                    donations.append((i, recipient))
                    is_donation = True

            self.history.record_action(i, is_donation)

        # 2. Movement resolution
        self._resolve_movements(move_intents)

        # 3. Energy updates
        energy_deltas = np.zeros(self.N_max, dtype=np.float32)
        incoming_donations = np.zeros(self.N_max, dtype=np.int32)
        for donor, recipient in donations:
            energy_deltas[donor] -= self.C
            incoming_donations[recipient] += 1

        for i in active_indices:
            num_neighbors = len(self.get_neighbors(i))
            delta = -self.c_base - (self.c_density * num_neighbors)
            delta += energy_deltas[i]
            delta += incoming_donations[i] * self.B

            self.energy[i] += delta
            rewards[i] = delta

        # 4. Demographics (Starvation & Mitosis - Algorithm 1)
        deaths, births = self._step_demographics()

        # 5. Cue update
        self._update_cues()

        # 6. Dones & Info
        all_extinct = bool(self.active_count == 0)
        time_limit = bool(self.step_count >= self.max_steps)
        dones = np.full(self.N_max, all_extinct or time_limit, dtype=bool)
        dones[~self.mask] = True

        info = {
            "active_population": self.active_count,
            "donations_count": len(donations),
            "deaths_this_step": deaths,
            "births_this_step": births,
            "mean_energy": float(np.mean(self.energy[self.mask])) if self.active_count > 0 else 0.0,
            "mean_hbar": float(self.h_bar),
            "step": self.step_count,
        }

        obs = self._get_observations()
        return obs, rewards, dones, info

    def _get_observations(self) -> Dict[str, Any]:
        """Returns structured dictionary of observations for all N_max slots."""
        self_state = np.zeros((self.N_max, 2), dtype=np.float32)
        neighbor_features = np.zeros((self.N_max, self.max_neighbors, 5), dtype=np.float32)
        neighbor_mask = np.zeros((self.N_max, self.max_neighbors), dtype=bool)
        neighbor_indices = np.full((self.N_max, self.max_neighbors), -1, dtype=np.int32)

        b_scores = self.history.get_all_scores(self.mask)

        for i in range(self.N_max):
            if not self.mask[i]:
                continue

            self_state[i, 0] = self.energy[i] / float(self.e_mitosis)
            self_state[i, 1] = self.age[i] / 1000.0

            neighbors = self.get_neighbors(i)
            for k, n_idx in enumerate(neighbors[:self.max_neighbors]):
                delta = self.toroidal_delta(self.positions[i], self.positions[n_idx])
                neighbor_features[i, k, 0] = delta[0] / float(self.rho)
                neighbor_features[i, k, 1] = delta[1] / float(self.rho)
                neighbor_features[i, k, 2] = self.energy[n_idx] / float(self.e_mitosis)
                neighbor_features[i, k, 3] = self.r_tilde[i, n_idx]
                neighbor_features[i, k, 4] = b_scores[n_idx]
                neighbor_mask[i, k] = True
                neighbor_indices[i, k] = n_idx

        return {
            "self_state": self_state,
            "neighbor_features": neighbor_features,
            "neighbor_mask": neighbor_mask,
            "neighbor_indices": neighbor_indices,
            "active_mask": self.mask.copy(),
            "r_tilde": self.r_tilde.copy(),
            "b_scores": b_scores,
        }
