"""Evaluate verified saved policies and a matched mechanism study.

python evaluation/complete_study.py --study-dir results/submission_study
python evaluation/complete_study.py --study-dir results/mechanism_study
Training provenance remains immutable. Evaluation versions and sources are
recorded separately: replay on a newer CPU runtime is not original training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.inherited_policies import load
from experiments.inheritance_study import evaluate_checkpoint, summarize, feasibility


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkpoint(path):
    """Verify original training archive, weights, and current actor/ecology code."""
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if digest(path) != meta['checkpoint_sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    source = path.parent.parent/'source'
    for name, expected in meta['source_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError(f'Archived training source mismatch: {name}')
        # The trainer may evolve; the ecology and architecture used for replay
        # must be byte-identical to the source that produced these weights.
        if name.startswith(('environment/', 'models/')) and digest(ROOT/name) != expected:
            raise ValueError(f'Evaluation implementation differs: {name}')
    return meta


class MeasurementPopulation:
    """Passive observer: uses no policy/environment RNG and changes no actions."""
    def __init__(self, population, seed, collect_decisions=False):
        self.population = population
        self.tail = deque(maxlen=100)
        self.decisions = []
        self.collect_decisions = collect_decisions
        self.rng = np.random.default_rng(seed+918273)
        self.seen = 0

    def __getattr__(self, name):
        return getattr(self.population, name)

    def act(self, env, obs):
        output = self.population.act(env, obs)
        _, owners, actions, _, _ = output
        residents = np.flatnonzero(owners >= 0)
        adjacent = []
        for i in residents:
            for j in env.get_neighbors(i):
                if j > i and not env.is_parasite[j] and env.chebyshev_distance(env.positions[i], env.positions[j]) == 1:
                    adjacent.append(float(np.mean(env.chromosomes[i] == env.chromosomes[j])))
        global_pairs = [float(np.mean(env.chromosomes[i] == env.chromosomes[j]))
                        for k,i in enumerate(residents) for j in residents[k+1:]]
        if adjacent and global_pairs:
            local, global_mean = float(np.mean(adjacent)), float(np.mean(global_pairs))
            self.tail.append((env.step_count, local, global_mean, local-global_mean))
        if self.collect_decisions:
            for i in residents:
                for k in np.flatnonzero(obs['neighbor_mask'][i]):
                    if not obs['action_mask'][i, 5+k]:
                        continue
                    feature = obs['neighbor_features'][i,k]
                    r = float(feature[3])
                    row = [1., r, float(feature[4]), env.B/env.C, r*env.B/env.C,
                           float(obs['neighbor_mask'][i].sum()), float(obs['self_state'][i,0]),
                           int(actions[i] == 5+k)]
                    # Uniform reservoir, independent of outcomes and actions.
                    self.seen += 1
                    if len(self.decisions) < 4000:
                        self.decisions.append(row)
                    else:
                        index = int(self.rng.integers(self.seen))
                        if index < 4000:
                            self.decisions[index] = row
        return output


def evaluate_mechanism(path, worlds):
    from experiments.inheritance_study import evaluate
    population, genomes, meta = load(path)
    runs, decisions = [], []
    for replicate in range(worlds):
        world = meta['seed']+10000+replicate*1000
        measured = MeasurementPopulation(population, world,
                                         meta['condition'].startswith(('energy_bc','inclusive_bc')))
        row = evaluate(measured, genomes, meta['environment'], world)
        # A late window means the actual last 100 environment transitions;
        # absence of adjacent pairs in that window remains undefined.
        tail = [v for v in measured.tail if v[0] >= max(0,row['duration']-100)]
        valid_late = row['resident_survival'] and bool(tail)
        for index,name in enumerate(('late_adjacent_similarity','late_global_similarity','late_excess_similarity'),1):
            row[name] = float(np.mean([v[index] for v in tail])) if valid_late else None
        row['late_pair_observation_steps'] = len(tail) if valid_late else 0
        runs.append(dict(training_seed=meta['seed'],control='trained',**row))
        decisions.extend([*r,meta['seed'],world] for r in measured.decisions)
    return runs, decisions


def evaluation_sources():
    names = ['evaluation/complete_study.py','evaluation/uncertainty.py',
             'experiments/inheritance_study.py','experiments/inherited_policies.py',
             'experiments/dyadic_baseline.py']
    names += [p.relative_to(ROOT).as_posix() for folder in ('environment','models')
              for p in (ROOT/folder).glob('*.py')]
    return {name:digest(ROOT/name) for name in names}


def evaluate_one(path, worlds):
    meta = verify_checkpoint(path)
    signature = dict(checkpoint_sha256=meta['checkpoint_sha256'], worlds=worlds,
                     evaluation_source_sha256=evaluation_sources(),
                     torch_version=str(torch.__version__), numpy_version=np.__version__)
    cache = path.parent.parent/'evaluations'/f'{path.stem}.json'
    if cache.exists():
        result = json.loads(cache.read_text(encoding='utf-8'))
        if result['signature'] != signature:
            raise ValueError(f'Stale evaluation; choose a fresh output or remove {cache}')
        return result
    if meta['protocol'] == 'matched-reward-incentive-v1':
        runs, decisions = evaluate_mechanism(path, worlds)
    else:
        condition = path.stem.rsplit('_seed_',1)[0]
        runs = evaluate_checkpoint(path, worlds, include_rules=condition=='clean')
        decisions = []
    result = dict(signature=signature, runs=runs, decisions=decisions)
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,default=Path('results/submission_study'))
    p.add_argument('--worlds',type=int,default=3)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--worker',type=Path)
    args = p.parse_args()
    if args.worlds < 1 or args.workers < 1:
        p.error('worlds and workers must be positive')
    torch.set_num_threads(1)
    if args.worker:
        evaluate_one(args.worker,args.worlds)
        return
    paths = sorted((args.study_dir/'checkpoints').glob('*.pt'))
    if not paths:
        raise ValueError('No checkpoints found')
    def worker(path):
        subprocess.run([sys.executable,__file__,'--worker',str(path),'--worlds',str(args.worlds)],check=True)
        print(f'Evaluated {path.stem}',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker,paths))
    runs,decisions,budgets,seeds,training_runtimes = [],[],{},set(),set()
    for path in paths:
        meta = verify_checkpoint(path)
        condition = path.stem.rsplit('_seed_',1)[0]
        rows = evaluate_one(path,args.worlds)
        runs.extend(dict(condition=condition,**r) for r in rows['runs'])
        decisions.extend([condition,*r] for r in rows['decisions'])
        budgets[condition] = meta['environment_steps']
        seeds.add(meta['seed'])
        training_runtimes.add((meta['torch_version'],meta['numpy_version']))
    for condition in budgets:
        actual = {(r['training_seed'],r['world_seed']) for r in runs
                  if r['condition']==condition and r['control']=='trained' and not r['attack']}
        expected = {(s,s+10000+k*1000) for s in seeds for k in range(args.worlds)}
        if actual != expected:
            raise ValueError(f'Missing seed/world coverage: {condition}')
    result = dict(protocol='verified-frozen-evaluation-v1',training_seeds=sorted(seeds),
                  environment_steps_per_run=min(budgets.values()),budgets_by_condition=budgets,
                  evaluation_worlds_per_seed=args.worlds,not_run=[],
                  clean_feasibility=feasibility(runs,sorted(seeds),args.worlds),
                  evaluation_source_sha256=evaluation_sources(),
                  training_runtimes=[dict(torch=t,numpy=n) for t,n in sorted(training_runtimes)],
                  evaluation_runtime=dict(python=platform.python_version(),
                                          packages={n:version(n) for n in ('numpy','torch','scipy','matplotlib')}),
                  runs=runs,summary=summarize(runs))
    (args.study_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    if decisions:
        (args.study_dir/'decisions.json').write_text(json.dumps(dict(
            columns=['condition','intercept','tag_cue','history','B_over_C','tag_x_B_over_C',
                     'available_neighbors','own_energy','gave','training_seed','world_seed'],rows=decisions)),encoding='utf-8')
    lines = ['# Verified frozen-policy evaluation','','Worlds are averaged within training seed. '
             'Intervals use independent seed blocks; evaluation runtime can differ from training.', '',
             '| Condition/control/evaluation | Residents (95% CI) | Donations/resident step (95% CI) |',
             '| --- | --- | --- |']
    from evaluation.final_report import interval
    for name,row in result['summary'].items():
        lines.append(f'| {name} | {interval(row["final_residents"])} | {interval(row["donations_per_resident_step"])} |')
    (args.study_dir/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Completed {len(paths)} checkpoints, {len(runs)} worlds in {args.study_dir}',flush=True)


if __name__ == '__main__':
    main()
