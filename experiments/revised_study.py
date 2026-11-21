"""Review-response study: matched worlds, demographic rewards and cue interventions.

Run from the repository root. Historical artifacts are never overwritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from environment.grid_world import GridWorld
from evaluation.uncertainty import mean_ci, proportion_ci
from experiments.config import SEEDS, environment_config
from models.attention_policy import GatedAttentionPolicy
from training.train_all_models import train_model

OUT = Path("results/revised_study")
CONDITIONS = {
    "energy_clean": dict(reward_mode="energy"),
    "demographic_clean": dict(reward_mode="demographic"),
    "demographic_noise025": dict(reward_mode="demographic", cue_noise=.25),
    "demographic_noise050": dict(reward_mode="demographic", cue_noise=.5),
    "demographic_noise_mimics": dict(reward_mode="demographic", cue_noise=.5),
    "fixed_noise_mimics": dict(reward_mode="demographic", cue_noise=.5),
    "fallback": dict(reward_mode="energy", enable_starvation=False, enable_mitosis=False),
}


def checkpoint_path(condition, seed):
    return OUT / "checkpoints" / f"{condition}_seed_{seed}.pt"


def make_policy(condition):
    return GatedAttentionPolicy(d_model=32, num_heads=2, contextual_gate=True,
                                fixed_gate=.5 if condition == "fixed_noise_mimics" else None)


def load_policy(condition, seed):
    path = checkpoint_path(condition, seed)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    policy = make_policy(condition)
    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    policy.eval()
    return policy


def train(condition, seed, steps):
    path = checkpoint_path(condition, seed)
    lock = path.with_suffix('.lock')
    deadline = time.monotonic() + 3600
    while True:
        try:
            handle = lock.open('x')
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Training lock still held: {lock}")
            time.sleep(1)
    try:
        if reusable(condition, seed, steps):
            return
        train_model("attention", seed=seed, environment_steps=steps,
                    env_config=environment_config(**CONDITIONS[condition]),
                    contextual_gate=True, fixed_gate=.5 if condition == "fixed_noise_mimics" else None,
                    invasion_step=10 if "mimics" in condition else None,
                    save_path=str(path))
    finally:
        handle.close()
        lock.unlink()


def reusable(condition, seed, steps):
    path = checkpoint_path(condition, seed)
    if not path.exists() or not path.with_suffix('.json').exists():
        return False
    metadata = json.loads(path.with_suffix('.json').read_text())
    if (metadata['environment_steps'] != steps or
        metadata.get('seed') != seed or
        metadata.get('model_type') != 'attention' or
        metadata['environment'] != environment_config(**CONDITIONS[condition]) or
        not metadata.get('contextual_gate') or
        metadata.get('fixed_gate') != (.5 if condition == 'fixed_noise_mimics' else None) or
        metadata.get('invasion_step') != (10 if 'mimics' in condition else None)):
        return False
    # File integrity alone does not establish that this is the requested run.
    # Reject checkpoints from changed training code, hyperparameters or libraries.
    defaults = inspect.signature(train_model).parameters
    for key in ('rollout_length', 'num_epochs', 'batch_size', 'lr', 'gamma',
                'gae_lambda', 'clip_eps', 'vf_coef', 'ent_coef', 'invasion_count'):
        if metadata.get(key) != defaults[key].default:
            return False
    if (metadata.get('torch_version') != str(torch.__version__) or
        metadata.get('numpy_version') != np.__version__):
        return False
    root = Path(__file__).resolve().parents[1]
    current_sources = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('environment', 'models', 'training')
        for p in sorted((root / folder).glob('*.py'))
    }
    recorded_sources = {name.replace('\\', '/'): digest
                        for name, digest in metadata.get('source_sha256', {}).items()}
    if recorded_sources != current_sources:
        return False
    load_policy(condition, seed)
    return True


def cue_interventions(policy, tensors, rng):
    """Permutation sensitivity of the entire policy, including raw-cue bypasses.

    Positive values mean changing that cue changes action probabilities. These
    are off-manifold diagnostic interventions, not causal reliability estimates.
    """
    s, nf, nm, am = tensors
    valid = nm.any(dim=1)
    if not valid.any():
        return None
    with torch.no_grad():
        reference = policy(s, nf, nm, am)[0].probs
        results = {}
        for name, column in (("genetic", 3), ("behavior", 4)):
            changed = nf.clone()
            values = nf[..., column][nm]
            order = torch.as_tensor(rng.permutation(len(values)))
            changed[..., column][nm] = values[order]
            probs = policy(s, changed, nm, am)[0].probs
            results[name + "_tv"] = float((.5 * (probs-reference).abs().sum(dim=1))[valid].mean())
        return results


def common_input_probe(policy):
    """Same synthetic states for all seeds and training conditions.

    Separates input sensitivity from changing population/recipient composition;
    these synthetic observations are not an ecological performance evaluation.
    """
    rng = np.random.default_rng(7301)
    state = torch.tensor(np.column_stack([rng.uniform(.1, 1, 64), rng.uniform(0, .3, 64)]), dtype=torch.float32)
    features = torch.tensor(rng.uniform(0, 1, (64, 8, 5)), dtype=torch.float32)
    features[..., :2] = features[..., :2] * 2 - 1
    mask = torch.ones(64, 8, dtype=torch.bool)
    actions = torch.ones(64, 13, dtype=torch.bool)
    return cue_interventions(policy, (state, features, mask, actions), np.random.default_rng(7302))


def evaluate(policy, condition, seed, attack=False, mutation=.05):
    eval_seed = seed + 10000
    torch.manual_seed(eval_seed)
    env = GridWorld(**environment_config(**{**CONDITIONS[condition], "mutation_rate": mutation}), seed=eval_seed)
    obs = env.reset(seed=eval_seed)
    rng = np.random.default_rng(seed + 20000)  # measurement must not change action/world RNG
    total = mimic_total = post_total = births = injected = exposures = 0
    exposure_donations = resident_births = 0
    alive_steps = 0
    probes = []
    curve = []
    for step in range(env.max_steps):
        if attack and step == 10:
            injected = env.inject_mimics(6)
            obs = env._get_observations()
        tensors = policy.extract_tensors(obs)
        with torch.no_grad():
            dist, _, weights = policy(*tensors)
            actions = dist.sample().numpy()
        residents = env.mask & ~env.is_parasite
        alive_steps += int(residents.sum())
        exposed = bool((env.mask & env.is_parasite).any())
        exposures += int(exposed)
        if step % 10 == 0:
            # Never include scripted mimics as policy decisions in attribution.
            resident_tensors = tuple(t[residents] for t in tensors)
            probe = cue_interventions(policy, resident_tensors, rng)
            if probe is not None:
                probe["step"] = step
                present = tensors[2] & torch.as_tensor(residents[:, None])
                probe["genetic_weight"] = float(weights[..., 0][present].mean())
                probes.append(probe)
        obs, _, dones, info = env.step(actions)
        total += info["donations_count"]
        mimic_total += info["donations_to_mimics"]
        if injected and step >= 10:
            post_total += info["donations_count"]
        if exposed:
            exposure_donations += info["donations_count"]
        births += info["births_this_step"]
        resident_births += sum(not env.is_parasite[b['slot']] for b in info['birth_records'])
        curve.append(dict(step=step+1, acting_residents=int(residents.sum()), residents=int((env.mask & ~env.is_parasite).sum()),
                          mimics=int((env.mask & env.is_parasite).sum()), donations=info["donations_count"]))
        if np.all(dones):
            break
    return dict(training_seed=seed, evaluation_seed=eval_seed, condition=condition,
                attack=attack, mutation_rate=mutation, environment=environment_config(**{**CONDITIONS[condition], "mutation_rate": mutation}),
                duration=len(curve), final_residents=int((env.mask & ~env.is_parasite).sum()),
                resident_survival=bool((env.mask & ~env.is_parasite).any()),
                donations=total, donations_per_resident_step=total/max(alive_steps, 1), births=births,
                resident_births=int(resident_births), exposure_donations=exposure_donations,
                exposed_mimic_donation_share=mimic_total/exposure_donations if injected and exposure_donations else None,
                mimics_injected=injected, mimic_exposure_steps=exposures,
                post_invasion_donations=post_total, donations_to_mimics=mimic_total,
                energy_to_mimics=mimic_total*env.B,
                mimic_donation_share=mimic_total/post_total if injected and post_total else None,
                probes=probes, curve=curve,
                **{key: float(np.mean([p[key] for p in probes])) if probes else None
                   for key in ("genetic_tv", "behavior_tv", "genetic_weight")})


def analyze(seeds, conditions, attack_modes=(False, True)):
    runs, summary, common_probes = [], {}, []
    for condition in conditions:
        for seed in seeds:
            policy = load_policy(condition, seed)
            common_probes.append(dict(condition=condition, training_seed=seed, **common_input_probe(policy)))
            for attack in attack_modes:
                runs.append(evaluate(policy, condition, seed, attack=attack))
            if condition == "demographic_clean":
                for mutation in (0., .1, .2):
                    runs.append(evaluate(policy, condition, seed, mutation=mutation))
            torch.manual_seed(seed)
            untrained = make_policy(condition).eval()
            baseline = evaluate(untrained, condition, seed)
            baseline["condition"] = condition + "_untrained"
            runs.append(baseline)
    metrics = ("final_residents", "donations_per_resident_step", "births", "resident_births", "mimic_donation_share", "exposed_mimic_donation_share",
               "energy_to_mimics", "genetic_tv", "behavior_tv", "genetic_weight")
    for condition in sorted({r["condition"] for r in runs}):
        for attack in (False, True):
            selected = [r for r in runs if r["condition"] == condition and r["attack"] == attack and r["mutation_rate"] == .05]
            if not selected:
                continue
            summary[f"{condition}/{'attack' if attack else 'clean'}"] = {
                **{key: mean_ci([r[key] for r in selected]) for key in metrics},
                "resident_survival": proportion_ci([r["resident_survival"] for r in selected]),
                "attacked_runs": sum(r["mimics_injected"] > 0 for r in selected),
            }
    contrasts = {}
    for a, b in (("demographic_clean", "energy_clean"),
                 ("demographic_clean", "demographic_clean_untrained"),
                 ("demographic_noise050", "demographic_clean"),
                 ("demographic_noise_mimics", "fixed_noise_mimics")):
        for metric in ("donations_per_resident_step", "final_residents", "genetic_tv", "behavior_tv"):
            pairs = []
            for seed in seeds:
                values = {r["condition"]: r[metric] for r in runs if r["training_seed"] == seed and not r["attack"] and r["mutation_rate"] == .05}
                if values.get(a) is not None and values.get(b) is not None:
                    pairs.append(values[a]-values[b])
            contrasts[f"{a}_minus_{b}/{metric}"] = mean_ci(pairs)
    common_summary = {c: {metric: mean_ci([r[metric] for r in common_probes if r['condition'] == c])
                          for metric in ('genetic_tv', 'behavior_tv')} for c in conditions}
    common_contrasts = {}
    for a, b in (("demographic_noise050", "demographic_clean"),
                 ("demographic_noise_mimics", "demographic_noise050"),
                 ("demographic_noise_mimics", "fixed_noise_mimics")):
        for metric in ('genetic_tv', 'behavior_tv'):
            differences = []
            for seed in seeds:
                values = {r['condition']: r[metric] for r in common_probes if r['training_seed'] == seed}
                if a in values and b in values:
                    differences.append(values[a]-values[b])
            common_contrasts[f'{a}_minus_{b}/{metric}'] = mean_ci(differences)
    result = dict(protocol="review-response-v1", runs=runs, summary=summary, paired_contrasts=contrasts,
                  common_input_probes=common_probes, common_input_summary=common_summary,
                  common_input_paired_contrasts=common_contrasts,
                  limitations=["Shared policy; no per-lineage evolution or proof of kin selection.",
                               "Demographic reward is a changed objective, not evidence for the original energy-only hypothesis.",
                               "Permutation sensitivity is an exploratory input intervention, not a causal estimate of reliability.",
                               "One matched evaluation world per training seed; small-seed inference is exploratory.",
                               "Fallback disables ecological deaths and reproduction; survival is not interpretable.",
                               "Mimic donation share is null without an attack or post-attack donations; elimination is not defense evidence."])
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Revised study: executed results", "", "All intervals and comparisons are exploratory. See protocol and limitations in docs/review_response_2026-11-17.md.", "",
             "Values are means [95% intervals]. Survival uses Wilson intervals; other metrics use Student t intervals across training seeds.", "",
             "| Condition | Resident survival | Final residents | Donations / resident step | Post-invasion mimic donation share |", "| --- | --- | --- | --- | --- |"]
    def format_interval(row, percentage=False):
        if row['mean'] is None:
            return 'not evaluated'
        spec = '.1%' if percentage else '.4f'
        point = format(row['mean'], spec)
        interval = (' [' + ', '.join(format(v, spec) for v in row['ci95']) + ']') if row['ci95'] else ' [CI unavailable]'
        return f"{point}{interval}; n={row['n']}"
    for name, row in summary.items():
        lines.append(f"| {name} | {format_interval(row['resident_survival'], True)} | {format_interval(row['final_residents'])} | {format_interval(row['donations_per_resident_step'])} | {format_interval(row['mimic_donation_share'], True)} |")
    lines += ["", "Fallback survival is forced by disabling starvation and cannot support ecological conclusions. Small-sample t intervals for bounded quantities may extend outside their physical range; raw values and denominators are retained.", "",
              "## Cue sensitivity on identical inputs", "",
              "| Training condition | Genetic cue TV [95% CI] | Behaviour cue TV [95% CI] |",
              "| --- | --- | --- |"]
    for condition, row in common_summary.items():
        lines.append(f"| {condition} | {format_interval(row['genetic_tv'])} | {format_interval(row['behavior_tv'])} |")
    lines += ["", "Raw seed trajectories, exposure-conditioned mimic shares, cue interventions, mutation sweeps and paired contrasts: [analysis.json](analysis.json).", "",
              "Figures: [training curves](figures/learning_curves.png), [common-input cue sensitivity](figures/cue_sensitivity.png).", ""]
    (OUT / "results.md").write_text('\n'.join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-steps', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--worker', choices=CONDITIONS)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    (OUT / 'checkpoints').mkdir(parents=True, exist_ok=True)
    if args.worker:
        train(args.worker, args.seeds[0], args.environment_steps)
        return
    if not args.evaluate_only:
        def worker(job):
            condition, seed = job
            path = checkpoint_path(condition, seed)
            if reusable(condition, seed, args.environment_steps):
                print(f"Reusing verified {condition}, seed {seed}", flush=True)
                return
            with checkpoint_path(condition, seed).with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, '-u', __file__, '--worker', condition,
                                '--seeds', str(seed), '--environment-steps', str(args.environment_steps)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Completed {condition}, seed {seed}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, [(c, s) for c in args.conditions for s in args.seeds]))
    if not args.train_only:
        analyze(args.seeds, args.conditions)


if __name__ == '__main__':
    main()
