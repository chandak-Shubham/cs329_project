"""Matched invasion, neural fixed-gating, placement and incentive-training controls.

The independent unit is a training seed. All condition differences are paired.
These short-budget experiments test responses; they do not establish convergence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import numpy as np
import torch
from scipy import stats
from environment.grid_world import GridWorld
from models.attention_policy import GatedAttentionPolicy
from experiments.run_phases import SEEDS, load_checkpoint
from evaluation.uncertainty import mean_ci


def attention(seed, fixed=False, incentive=None):
    policy = GatedAttentionPolicy(d_model=32, num_heads=2, fixed_gate=.5 if fixed else None)
    name = f"incentive_{incentive}.pt" if incentive is not None else ("fixed_attention.pt" if fixed else "gated_attention.pt")
    return load_checkpoint(policy, name, seed)


def simulate(seed, fixed=False, invade=False, placement="adjacent", mutation=.05,
             steps=300, initial_energy=30., invasion_time=10):
    torch.manual_seed(seed + 10000)
    policy = attention(seed, fixed)
    env = GridWorld(grid_size=20, max_capacity=60, initial_population=30,
                    initial_energy=initial_energy, e_mitosis=50., mutation_rate=mutation,
                    donation_benefit=5., c_base=.5, c_density=.05,
                    offspring_placement=placement, seed=seed + 10000)
    obs = env.reset(seed=seed + 10000)
    adjacent, random_pairs, before, after = [], [], [], []
    donations = births = injected = 0
    # Separate RNG prevents measurement from perturbing simulation dynamics.
    measurement_rng = np.random.default_rng(seed + 20000)
    for t in range(steps):
        if invade and t == invasion_time:
            injected = env.inject_mimics(12)
            obs = env._get_observations()
        tensors = policy.extract_tensors(obs)
        with torch.no_grad():
            dist, _, weights = policy(*tensors)
            actions = dist.sample().numpy()
        valid = obs["active_mask"][:, None] & obs["neighbor_mask"]
        if valid.any():
            w = float(weights[..., 0].numpy()[valid].mean())
            if invasion_time - 10 <= t < invasion_time:
                before.append(w)
            if invasion_time <= t < invasion_time + 10:
                after.append(w)
        ids = np.flatnonzero(env.mask)
        # Each unordered pair once, strictly von Neumann adjacent on the torus.
        pair_h = []
        for offset, i in enumerate(ids):
            for j in ids[offset+1:]:
                delta = np.abs(env.positions[i] - env.positions[j])
                if np.minimum(delta, env.L-delta).sum() == 1:
                    pair_h.append(float(np.mean(env.chromosomes[i] == env.chromosomes[j])))
        if pair_h:
            adjacent.append(float(np.mean(pair_h)))
            samples = [measurement_rng.choice(ids, 2, replace=False) for _ in pair_h]
            random_pairs.append(float(np.mean([np.mean(env.chromosomes[i] == env.chromosomes[j]) for i, j in samples])))
        obs, _, dones, info = env.step(actions)
        donations += info["donations_count"]
        births += info["births_this_step"]
        if np.all(dones):
            break
    residents = int(np.sum(env.mask & ~env.is_parasite))
    return {"training_seed": seed, "evaluation_seed": seed+10000,
            "fixed_gate": fixed, "invasion": invade, "placement": placement,
            "invasion_time": invasion_time, "injected": injected,
            "resident_survival": int(residents > 0), "final_residents": residents,
            "duration": t+1, "donations": donations, "births": births,
            "genetic_weight_before": float(np.mean(before)) if before else None,
            "genetic_weight_after": float(np.mean(after)) if after else None,
            "adjacent_similarity": float(np.mean(adjacent)) if adjacent else None,
            "random_pair_similarity": float(np.mean(random_pairs)) if random_pairs else None,
            "adjacency_observed_steps": len(adjacent)}


def paired_summary(a, b, metric):
    differences = [x[metric] - y[metric] for x, y in zip(a, b)
                   if x[metric] is not None and y[metric] is not None]
    out = mean_ci(differences)
    out["difference"] = "first condition minus second condition"
    if len(differences) > 1 and np.std(differences) > 0:
        out["paired_t_p_value"] = float(stats.ttest_1samp(differences, 0).pvalue)
    else:
        out["paired_t_p_value"] = None
    return out


def run_controls(seeds=SEEDS):
    runs, contrasts = {}, {}
    for fixed in (False, True):
        for invade in (False, True):
            key = f"{'fixed' if fixed else 'adaptive'}_{'attack' if invade else 'clean'}"
            runs[key] = [simulate(s, fixed=fixed, invade=invade) for s in seeds]
    for kind in ("fixed", "adaptive"):
        contrasts[f"{kind}_attack_minus_clean"] = {
            metric: paired_summary(runs[f"{kind}_attack"], runs[f"{kind}_clean"], metric)
            for metric in ("resident_survival", "final_residents", "duration", "donations")}
    contrasts["adaptive_minus_fixed_clean"] = {
        metric: paired_summary(runs["adaptive_clean"], runs["fixed_clean"], metric)
        for metric in ("resident_survival", "duration", "donations")}
    trust_changes = []
    for attack, clean in zip(runs["adaptive_attack"], runs["adaptive_clean"]):
        fields = (attack["genetic_weight_before"], attack["genetic_weight_after"],
                  clean["genetic_weight_before"], clean["genetic_weight_after"])
        if attack["injected"] and all(v is not None for v in fields):
            trust_changes.append((fields[1]-fields[0]) - (fields[3]-fields[2]))
    contrasts["trust_shift_difference_in_differences"] = mean_ci(trust_changes)
    assortment = {}
    for mu in (.01, .05, .1, .2):
        # Matched high reserve deliberately exercises placement in every seed.
        a = [simulate(s, mutation=mu, initial_energy=60.) for s in seeds]
        b = [simulate(s, mutation=mu, initial_energy=60., placement="random") for s in seeds]
        key = str(mu)
        runs[f"placement_adjacent_{key}"] = a
        runs[f"placement_random_{key}"] = b
        valid = [r for r in a if r["adjacent_similarity"] is not None]
        gaps = [r["adjacent_similarity"]-r["random_pair_similarity"] for r in valid]
        test = stats.ttest_1samp(gaps, 0) if len(gaps)>1 and np.std(gaps)>0 else None
        assortment[key] = {
            "mutation_rate": mu, "mean_h_adjacent": float(np.mean([r["adjacent_similarity"] for r in valid])) if valid else None,
            "std_h_adjacent": float(np.std([r["adjacent_similarity"] for r in valid])) if valid else None,
            "mean_h_random": float(np.mean([r["random_pair_similarity"] for r in valid])) if valid else None,
            "std_h_random": float(np.std([r["random_pair_similarity"] for r in valid])) if valid else None,
            "assortment_gap": mean_ci(gaps)["mean"],
            "t_statistic": float(test.statistic) if test else None,
            "p_value": float(test.pvalue) if test else None,
            "gap_uncertainty": mean_ci(gaps),
            "placement_control": paired_summary(a, b, "adjacent_similarity"),
            "adjacent_uncertainty": mean_ci([r["adjacent_similarity"] for r in a]),
            "random_uncertainty": mean_ci([r["random_pair_similarity"] for r in a])}
    return {"runs": runs, "contrasts": contrasts, "spatial_assortment": assortment,
            "design": {"seeds": list(seeds), "maximum_steps": 300, "invasion_boundary": 10,
                       "trust_windows": "10 transitions before and after injection; mean genetic weights over valid neighbors",
                       "assortment": "time-mean strictly adjacent unordered pairs over observed living states; no equilibrium claim",
                       "inference": "paired independent training-seed summaries, exploratory unadjusted tests; five units only",
                       "placement_control": "random global empty-cell placement changes only offspring placement",
                       "placement_initial_energy": 60., "placement_mitosis_threshold": 50.,
                       "placement_design_note": "Matched high reserve exercises reproduction; low-reserve exploratory check had only 0-1 births per seed. This is a diagnostic regime, not a confirmatory test.",
                       "trust_limit": "composition-sensitive population weight change, not within-individual learning"}}


def incentive_response(seeds=SEEDS, resolution=15):
    r_values = np.linspace(.05, .95, resolution)
    ratios = [1.5, 3., 4.5]
    grid, raw = [], {}
    for ratio in ratios:
        curves = []
        for seed in seeds:
            policy = attention(seed, incentive=ratio)
            s = torch.tensor([[.5, .1]]).repeat(resolution, 1)
            nf = torch.zeros(resolution, 8, 5)
            nf[:, 0, 0] = .33
            nf[:, 0, 2] = .5
            nf[:, 0, 3] = torch.tensor(r_values, dtype=torch.float32)
            nf[:, 0, 4] = .5
            mask = torch.zeros(resolution, 8, dtype=torch.bool)
            mask[:, 0] = True
            am = torch.zeros(resolution, 13, dtype=torch.bool)
            am[:, :6] = True
            with torch.no_grad():
                dist, _, _ = policy(s, nf, mask, am)
                curves.append(dist.probs[:, 5].tolist())
        raw[str(ratio)] = curves
        grid.append(np.mean(curves, axis=0).tolist())
    low, high = np.array(raw["1.5"]), np.array(raw["4.5"])
    differences = (high-low).mean(axis=1)
    supra_minus_sub = []
    for seed_idx in range(len(seeds)):
        contrasts = []
        for ratio in ratios:
            curve = np.array(raw[str(ratio)][seed_idx])
            contrasts.append(float(curve[r_values*ratio>1].mean()-curve[r_values*ratio<=1].mean()))
        supra_minus_sub.append(float(np.mean(contrasts)))
    return {"r_vals": r_values.tolist(), "bc_vals": ratios, "P_give_grid": grid, "raw": raw,
            "theoretical_boundary": [1/r for r in ratios],
            "incentive_response": mean_ci(differences.tolist()),
            "supra_minus_sub_hamilton_threshold": mean_ci(supra_minus_sub),
            "note": "Independent policies trained at each B/C, paired by seed; fixed synthetic inputs; B/C absent from observations. Training-condition response, not instantaneous incentive sensitivity or proof of Hamilton's rule."}


if __name__ == "__main__":
    torch.set_num_threads(1)
    out = run_controls()
    attack = json.loads(Path("results/phase4_benchmarks.json").read_text())
    clean = json.loads(Path("results/phase4_no_invasion.json").read_text())
    for model in attack["raw"]:
        out["contrasts"][f"phase4_attack_minus_clean_{model}"] = {
            metric: mean_ci([a-b for a, b in zip(attack["raw"][model][metric], clean["raw"][model][metric])])
            for metric in ("survival", "carrying_capacity", "lifespan")}
    out["incentive_response"] = incentive_response()
    out["checkpoints_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path("results/checkpoints").glob("*.pt")}
    out["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for folder in ("environment", "models", "experiments", "evaluation")
                            for p in Path(folder).glob("*.py")}
    Path("results/research_controls.json").write_text(json.dumps(out, indent=2, allow_nan=False))
