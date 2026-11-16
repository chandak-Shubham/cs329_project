"""
Interactive Data-Driven Ecosystem Simulation Visualizer (Task 13).
Visualizes the 2D spatial grid ecosystem, emergent clan clusters, targeted donations,
and real-time trust telemetry (w^g vs. w^b).
Can either run interactively or render a high-quality GIF/MP4 replay for presentations.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.grid_world import GridWorld
from models.attention_policy import GatedAttentionPolicy


def chromosome_to_rgb(chrom: np.ndarray) -> np.ndarray:
    """Maps an 8-bit chromosome to an RGB color so related kin share identical clan hues."""
    # Bits 0..2 for R, Bits 3..5 for G, Bits 6..7 for B
    r = (chrom[0] * 4 + chrom[1] * 2 + chrom[2]) / 7.0
    g = (chrom[3] * 4 + chrom[4] * 2 + chrom[5]) / 7.0
    b = (chrom[6] * 2 + chrom[7]) / 3.0
    # Normalize and ensure vibrant color
    rgb = np.array([0.2 + 0.7 * r, 0.2 + 0.7 * g, 0.2 + 0.7 * b])
    return np.clip(rgb, 0.1, 0.95)


def generate_simulation_replay(
    total_steps: int = 150,
    grid_size: int = 24,
    max_capacity: int = 50,
    initial_population: int = 25,
    inject_mimics_at: int = 40,
    seed: int = 42,
    output_gif: str = "results/figures/ecosystem_simulation.gif",
):
    print("=" * 80)
    print(f"RENDERING PRESENTATION SIMULATION REPLAY ({total_steps} steps)...")
    print(f"Parasite Mimics Injected at Step {inject_mimics_at}")
    print("=" * 80)

    torch.set_num_threads(1)
    torch.manual_seed(seed + 10000)
    env = GridWorld(
        grid_size=grid_size,
        max_capacity=max_capacity,
        initial_population=initial_population,
        initial_energy=30.0,
        e_mitosis=60.0,
        enable_mitosis=True,
        c_base=0.5,
        c_density=0.05,
        donation_cost=2.0,
        donation_benefit=6.0,
        seed=seed + 10000,
    )
    obs = env.reset(seed=seed + 10000)

    policy = GatedAttentionPolicy(max_neighbors=8, d_model=32, num_heads=2, action_dim=13)
    ckpt_path = os.path.join("results", "checkpoints", "gated_attention.pt")
    from experiments.run_phases import load_checkpoint
    policy = load_checkpoint(policy, "gated_attention.pt", seed)
    policy.eval()

    # Pre-collect simulation trajectory frames
    frames_data = []
    injected_count = 0

    for step in range(1, total_steps + 1):
        if step - 1 == inject_mimics_at:
            injected_count = env.inject_mimics(count=5)
            obs = env._get_observations()  # Refresh observations immediately after invasion

        self_state, n_feats, n_mask, act_masks = policy.extract_tensors(obs)
        with torch.no_grad():
            dist, values, weights = policy.forward(self_state, n_feats, n_mask, act_masks)
            actions = dist.sample()

        # Telemetry: evaluate trust ONLY over actual active neighbors (not padded slots)
        active_indices = np.where(env.mask)[0]
        valid_neighbor_mask = obs["active_mask"][:, None] & n_mask.numpy()
        if valid_neighbor_mask.any():
            w_g = float(weights[:, :, 0].numpy()[valid_neighbor_mask].mean())
            w_b = float(weights[:, :, 1].numpy()[valid_neighbor_mask].mean())
        else:
            w_g, w_b = 0.5, 0.5

        # Record donations for visualization
        active_donations = []
        for i in active_indices:
            act = actions[i].item()
            if act >= 5:
                k = act - 5
                neighbors = env.get_neighbors(i)
                if k < len(neighbors) and not env.is_parasite[i] and env.energy[i] > env.C:
                    active_donations.append((env.positions[i].copy(), env.positions[neighbors[k]].copy()))

        # Save snapshot
        snapshot = {
            "step": step,
            "positions": env.positions[active_indices].copy(),
            "energies": env.energy[active_indices].copy(),
            "is_parasite": env.is_parasite[active_indices].copy(),
            "chromosomes": env.chromosomes[active_indices].copy(),
            "donations": active_donations,
            "pop_count": len(active_indices),
            "mean_energy": float(np.mean(env.energy[active_indices])) if len(active_indices) > 0 else 0.0,
            "w_g": w_g,
            "w_b": w_b,
        }
        frames_data.append(snapshot)

        obs, rewards, dones, info = env.step(actions.numpy())
        if np.all(dones) or env.active_count == 0:
            break

    # Setup 2-panel figure: Left = 2D Grid, Right = Real-time Telemetry HUD
    fig = plt.figure(figsize=(11, 5.8))
    fig.suptitle("Illustrative Simulation Replay (Trained Gated-Attention Policy)", fontsize=12, fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])

    ax_grid = fig.add_subplot(gs[:, 0])
    ax_pop = fig.add_subplot(gs[0, 1])
    ax_trust = fig.add_subplot(gs[1, 1])

    # Plot initial state
    ax_grid.set_xlim(-0.5, grid_size - 0.5)
    ax_grid.set_ylim(-0.5, grid_size - 0.5)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_facecolor("#1a1a1a")

    time_steps = []
    actual_injection = injected_count > 0
    pop_history = []
    wg_history = []
    wb_history = []

    def update(frame_idx):
        data = frames_data[frame_idx]
        ax_grid.clear()
        ax_grid.set_facecolor("#141414")
        ax_grid.set_xlim(-0.5, grid_size - 0.5)
        ax_grid.set_ylim(-0.5, grid_size - 0.5)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        ax_grid.set_title(f"Spatial Grid ({grid_size} x {grid_size}) — Step {data['step']}", fontsize=11, fontweight="bold")

        # Grid mesh lines
        for x in range(grid_size):
            ax_grid.axvline(x - 0.5, color="#222222", linewidth=0.5)
            ax_grid.axhline(x - 0.5, color="#222222", linewidth=0.5)

        # Draw donation beams
        for p_from, p_to in data["donations"]:
            ax_grid.annotate(
                "",
                xy=(p_to[0], p_to[1]),
                xytext=(p_from[0], p_from[1]),
                arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=1.8, mutation_scale=12, alpha=0.85),
            )

        # Draw agents
        for pos, e, is_p, chrom in zip(data["positions"], data["energies"], data["is_parasite"], data["chromosomes"]):
            if is_p:
                # Parasites drawn as distinct red markers
                ax_grid.scatter(pos[0], pos[1], s=120, c="#ff1744", marker="^", edgecolors="white", linewidth=1.2, zorder=5)
            else:
                # Cooperators colored by 8-bit chromosome clan hue
                color = chromosome_to_rgb(chrom)
                # Size proportional to energy
                size = max(40, min(180, int(e * 2.5)))
                ax_grid.scatter(pos[0], pos[1], s=size, color=color, edgecolors="white", linewidth=0.8, zorder=4)

        # Update telemetry histories
        time_steps[:] = [row["step"] for row in frames_data[:frame_idx+1]]
        pop_history[:] = [row["pop_count"] for row in frames_data[:frame_idx+1]]
        wg_history[:] = [row["w_g"] for row in frames_data[:frame_idx+1]]
        wb_history[:] = [row["w_b"] for row in frames_data[:frame_idx+1]]

        # Top Right: Population Plot
        ax_pop.clear()
        ax_pop.plot(time_steps, pop_history, color="#00e676", linewidth=2.0, label="Population ($N_t$)")
        if actual_injection and data["step"] > inject_mimics_at:
            ax_pop.axvline(x=inject_mimics_at, color="#ff1744", linestyle="--", alpha=0.7, label="Parasites Injected")
        ax_pop.set_xlim(0, total_steps)
        ax_pop.set_ylim(0, max_capacity + 5)
        ax_pop.set_ylabel("Active Population")
        ax_pop.set_title("Colony Demographics", fontsize=10, fontweight="bold")
        ax_pop.legend(loc="lower right", fontsize=8)

        # Bottom Right: Live Trust Dial
        ax_trust.clear()
        ax_trust.plot(time_steps, wg_history, color="#2979ff", linewidth=2.0, label="Genetic Trust ($w^g$)")
        ax_trust.plot(time_steps, wb_history, color="#ff9100", linewidth=2.0, label="Behavioral Trust ($w^b$)")
        ax_trust.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
        ax_trust.set_xlim(0, total_steps)
        ax_trust.set_ylim(0.2, 0.8)
        ax_trust.set_xlabel("Environment Step")
        ax_trust.set_ylabel("Adaptive Weight")
        ax_trust.set_title("Neural Gating Trust Dial (Mind of Agent)", fontsize=10, fontweight="bold")
        ax_trust.legend(loc="lower right", fontsize=8)

        fig.tight_layout()

    print(f"Compiling {len(frames_data)} frames into animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=80, repeat=False)

    try:
        ani.save(output_gif, writer="pillow", fps=12)
        print(f"SUCCESS: Interactive replay saved to {output_gif}")
    except Exception as e:
        print(f"Warning: Could not save GIF directly ({e}); saving static snapshot instead.")
        fig.savefig("results/figures/ecosystem_snapshot.png", dpi=200)

    plt.close(fig)
    Path(output_gif).with_suffix('.json').write_text(json.dumps({
        "training_seed": seed, "evaluation_seed": seed+10000,
        "requested_steps": total_steps, "frames": len(frames_data),
        "invasion_boundary": inject_mimics_at, "mimics_injected": injected_count,
        "checkpoint_sha256": hashlib.sha256(Path(f"results/checkpoints/gated_attention_seed_{seed}.pt").read_bytes()).hexdigest(),
        "description": "Illustrative single-run replay, not a statistical survival estimate"
    }, indent=2))


if __name__ == "__main__":
    generate_simulation_replay(total_steps=80, inject_mimics_at=30)

