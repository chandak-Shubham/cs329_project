"""Matched reward x incentive experiment, plus learning x placement controls.

python experiments/mechanism_study.py --steps 2048 --workers 3
Historical checkpoints are evaluated separately; they are never relabelled as
new training runs. Each worker is a separate process with its own RNG state.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from environment.grid_world import GridWorld
from experiments.config import SEEDS, environment_config
from experiments.inherited_policies import (
    LineagePopulation, PPO, archive_sources, source_hashes, optimize,
)

RATIOS = (1., 1.5, 2.5, 3., 4.5)
CONDITIONS = {
    f'{objective}_bc{int(ratio*100):03d}': dict(objective=objective, ratio=ratio)
    for objective in ('energy', 'inclusive') for ratio in RATIOS
}
CONDITIONS.update({
    'evolution_local': dict(objective='energy', ratio=2.5, learning=False),
    'evolution_random': dict(objective='energy', ratio=2.5, learning=False, placement='random'),
    'energy_random': dict(objective='energy', ratio=2.5, placement='random'),
})


def shaped_reward(energy_deltas, relatedness, acting, objective):
    """Decision-time marker weights, applied to pre-mitosis energy increments.

    This deliberately privileged reward control changes the objective. It is not
    pedigree fitness, nor evidence that kin selection emerges without shaping.
    Pre-step masks prevent death/newborn slot reuse from redirecting credit.
    """
    if objective not in ('energy', 'inclusive'):
        raise ValueError('objective must be energy or inclusive')
    raw = np.asarray(energy_deltas, dtype=np.float32)
    if objective == 'energy':
        return raw.copy()
    weights = np.asarray(relatedness, dtype=np.float32).copy()
    weights[~acting, :] = 0
    weights[:, ~acting] = 0
    np.fill_diagonal(weights, 0)
    return np.where(acting, raw + weights @ raw, 0).astype(np.float32)


def study_sources():
    return {**source_hashes(), **{
        'experiments/mechanism_study.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def train(path, condition, seed, steps):
    if steps < 1:
        raise ValueError('steps must be positive')
    spec = CONDITIONS[condition]
    sources = study_sources()
    config = environment_config(reward_mode='energy', donation_benefit=2*spec['ratio'],
                                offspring_placement=spec.get('placement', 'adjacent'))
    if path.exists() or path.with_suffix('.json').exists():
        raise ValueError(f'Refusing to overwrite {path}')
    archive_sources(path.parent.parent/'source', sources)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = GridWorld(**config, seed=seed)
    obs = env.reset(seed=seed)
    population = LineagePopulation(env.N_init)
    population.bind_founders(env)
    genomes = env.chromosomes[:env.N_init].copy()
    rng = np.random.default_rng(seed+100000)
    history, selection_history = [], []
    completed = generation = active_transitions = 0
    started = time.monotonic()
    while completed < steps:
        rollout, donations, births, resident_steps = [], 0, 0, 0
        terminal = False
        for _ in range(min(PPO['rollout_length'], steps-completed)):
            tensors, owners, actions, logs, values = population.act(env, obs)
            # Copy before step: step mutates masks, chromosomes and cue caches.
            relatedness, acting = env.r_tilde.copy(), env.mask.copy()
            obs, raw, dones, info = env.step(actions.numpy())
            rewards = shaped_reward(raw, relatedness, acting, spec['objective'])
            population.inherit(info['birth_records'])
            terminal = not (env.mask & ~env.is_parasite).any() or env.step_count >= env.max_steps
            if terminal:
                dones[:] = True
            rollout.append(dict(tensors=tensors, owners=torch.as_tensor(owners), actions=actions,
                                logs=logs, values=values, rewards=torch.as_tensor(rewards),
                                dones=torch.as_tensor(dones)))
            completed += 1
            resident_steps += int((owners >= 0).sum())
            donations += info['donations_count']
            births += len(info['birth_records'])
            if terminal:
                break
        active_transitions += resident_steps
        next_values = torch.zeros(env.N_max) if terminal else population.act(env, obs, sample=False)[-1]
        update = optimize(population, rollout, next_values, PPO) if spec.get('learning', True) else dict(
            samples_by_policy={}, optimizer_minibatches=0, mean_loss=None)
        history.append(dict(environment_steps=completed, generation=generation, duration=len(rollout),
                            final_residents=int((env.mask & ~env.is_parasite).sum()),
                            donations_per_step=donations/len(rollout),
                            donations_per_resident_step=donations/max(1, resident_steps),
                            births=births, active_agent_transitions=active_transitions, **update))
        if terminal:
            selected, event = population.select_survivors(env, rng)
            event.update(generation=generation, environment_steps=completed)
            selection_history.append(event)
            if selected is not None:
                genomes = selected
            generation += 1
            env.reset()
            population.bind_founders(env, genomes)
            obs = env._get_observations()
        if completed == steps or len(history) % 16 == 0:
            print(f'{condition} seed={seed} transitions={completed}/{steps}', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(policies=[p.state_dict() for p in population.policies],
                    founder_genomes=torch.as_tensor(genomes), origins=population.origins), path)
    metadata = dict(protocol='matched-reward-incentive-v1', condition=condition, seed=seed,
                    objective=spec['objective'], environment_steps=completed,
                    active_agent_transitions=active_transitions, architecture='attention',
                    selection=True, learning=spec.get('learning', True), invasion=False,
                    environment=config, ppo=PPO, learning_curve=history, selection_history=selection_history,
                    source_sha256=sources, torch_version=str(torch.__version__), numpy_version=np.__version__,
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    elapsed_seconds=time.monotonic()-started)
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=2048)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--out', type=Path, default=Path('results/mechanism_study'))
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument('--worker', choices=CONDITIONS)
    args = p.parse_args()
    if args.steps < 1 or args.workers < 1 or len(set(args.seeds)) != len(args.seeds):
        p.error('positive budgets/workers and distinct seeds required')
    def path(c, s):
        return args.out/'checkpoints'/f'{c}_seed_{s}.pt'
    if args.worker:
        train(path(args.worker, args.seeds[0]), args.worker, args.seeds[0], args.steps)
        return
    archive_sources(args.out/'source', study_sources())
    def execute(job):
        condition, seed = job
        target = path(condition, seed)
        if target.exists():
            from experiments.inherited_policies import load
            _, _, meta = load(target)
            if (meta['source_sha256'] != study_sources() or meta['environment_steps'] != args.steps
                or meta['seed'] != seed or meta['condition'] != condition
                or meta['torch_version'] != str(torch.__version__) or meta['numpy_version'] != np.__version__):
                raise ValueError(f'Incompatible existing experiment: {target}')
            print(f'Verified and reused {target}', flush=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable, '-u', __file__, '--worker', condition, '--seeds', str(seed),
                            '--steps', str(args.steps), '--out', str(args.out)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Completed {condition}, seed {seed}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(execute, [(c,s) for c in args.conditions for s in args.seeds]))


if __name__ == '__main__':
    main()
