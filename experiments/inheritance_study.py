"""Clean-first replicated test of inherited neural policies and energy-only PPO.

python experiments/inheritance_study.py --steps 8192 --workers 3
Use --conditions for explicitly exploratory conditions after a failed screen.
All outputs go to a separate directory and existing completed runs are verified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from environment.grid_world import GridWorld
from evaluation.uncertainty import mean_ci, proportion_ci
from experiments.config import SEEDS, environment_config
from experiments.inherited_policies import LineagePopulation, train, load, source_hashes, PPO
from experiments.dyadic_baseline import RulePopulation

OUT = Path('results/inheritance_study')
CONDITIONS = {
    'clean': {},
    'no_selection': dict(selection=False),
    'evolution_only': dict(learning=False),
    'fixed_gate': dict(architecture='fixed_attention'),
    'mlp': dict(architecture='mlp'),
    'noise025': dict(config=dict(cue_noise=.25)),
    'noise050': dict(config=dict(cue_noise=.5)),
    'noise_mimics': dict(config=dict(cue_noise=.5), invasion=True),
    'fixed_noise_mimics': dict(config=dict(cue_noise=.5), invasion=True, architecture='fixed_attention'),
    'bc150': dict(config=dict(donation_benefit=3.)),
    'bc450': dict(config=dict(donation_benefit=9.)),
    'mutation000': dict(config=dict(mutation_rate=0.)),
    'mutation010': dict(config=dict(mutation_rate=.1)),
    'mutation020': dict(config=dict(mutation_rate=.2)),
    'random_placement': dict(config=dict(offspring_placement='random')),
    'fallback': dict(config=dict(enable_starvation=False, enable_mitosis=False)),
}


def evaluate(population, genomes, config, world_seed, attack=False):
    """Frozen policies, no selection or optimizer updates during evaluation."""
    env = GridWorld(**config, seed=world_seed)
    env.reset(seed=world_seed)
    population.bind_founders(env, genomes)
    obs = env._get_observations()
    torch.manual_seed(world_seed)
    total = resident_steps = mimic_donations = post_donations = injected = births = 0
    curve, kin_excess, assortment = [], [], []
    for step in range(env.max_steps):
        if attack and step == 10:
            injected = env.inject_mimics(6)
            obs = env._get_observations()
        _, owners, actions, _, _ = population.act(env, obs)
        residents = int((owners >= 0).sum())
        # Opportunity-adjusted tag preference, conditional on actual giving.
        for slot in np.flatnonzero(owners >= 0):
            action = int(actions[slot])
            if action >= 5:
                cues = obs['neighbor_features'][slot, obs['neighbor_mask'][slot], 3]
                kin_excess.append(float(cues[action-5]-cues.mean()))
        adjacent = [(i,j) for i in np.flatnonzero(env.mask) for j in env.get_neighbors(i)
                    if j > i and env.chebyshev_distance(env.positions[i], env.positions[j]) == 1]
        if adjacent:
            # Use true chromosomes for assortment, not noisy observation cues.
            similarity = [float(np.mean(env.chromosomes[i] == env.chromosomes[j])) for i,j in adjacent]
            assortment.append(float(np.mean(similarity)))
        obs, _, _, info = env.step(actions.numpy())
        population.inherit(info['birth_records'])
        total += info['donations_count']
        resident_steps += residents
        mimic_donations += info['donations_to_mimics']
        if injected:
            post_donations += info['donations_count']
        births += sum(b['parent_uid'] in population.owners for b in info['birth_records'])
        remaining = int((env.mask & ~env.is_parasite).sum())
        curve.append(dict(step=step+1, acting_residents=residents, residents=remaining,
                          donations=info['donations_count']))
        if not remaining:
            break
    tail = curve[-100:]
    return dict(world_seed=world_seed, attack=attack, duration=len(curve), final_residents=remaining,
                resident_survival=remaining > 0 and len(curve) == env.max_steps,
                donations=total, donations_per_resident_step=total/max(resident_steps, 1),
                late_donation_rate=sum(r['donations'] for r in tail)/max(sum(r['acting_residents'] for r in tail), 1),
                late_donation_frequency=sum(r['donations'] > 0 for r in tail)/len(tail),
                resident_births=births, mimics_injected=injected, post_invasion_donations=post_donations,
                donations_to_mimics=mimic_donations,
                mimic_donation_share=mimic_donations/post_donations if post_donations else None,
                opportunity_adjusted_tag_preference=float(np.mean(kin_excess)) if kin_excess else None,
                adjacent_hamming_similarity=float(np.mean(assortment)) if assortment else None, curve=curve)


def evaluate_checkpoint(path, worlds=3, include_rules=False):
    population, genomes, metadata = load(path)
    seed = metadata['seed']
    runs = []
    for replicate in range(worlds):
        world_seed = seed + 10000 + replicate*1000
        for attack in ((False, True) if metadata['invasion'] else (False,)):
            run = evaluate(population, genomes, metadata['environment'], world_seed, attack)
            runs.append(dict(training_seed=seed, control='trained', **run))
        # Keep evolved genomes AND initial ancestor composition matched. This
        # isolates learned weights from the generation's inherited founder mix.
        torch.manual_seed(seed)
        initial = LineagePopulation(len(population.policies), metadata['architecture'])
        initial.policies = [deepcopy(initial.policies[i]) for i in population.origins]
        run = evaluate(initial, genomes, metadata['environment'], world_seed)
        runs.append(dict(training_seed=seed, control='untrained_ancestors', **run))
        if include_rules:
            for name in ('selfish','tag_trust','fixed_threshold','reputation','dyadic_tit_for_tat'):
                rule = RulePopulation(name,world_seed)
                run = evaluate(rule,genomes,metadata['environment'],world_seed)
                runs.append(dict(training_seed=seed,control='rule_'+name,**run))
    return runs


def summarize(runs):
    """Average evaluation worlds WITHIN each training seed before inference."""
    summary = {}
    for condition, control, attack in sorted({(r['condition'], r['control'], r['attack']) for r in runs}):
        selected = [r for r in runs if (r['condition'],r['control'],r['attack']) == (condition,control,attack)]
        seeds = sorted({r['training_seed'] for r in selected})
        row = {}
        for metric in ('donations_per_resident_step','final_residents','resident_survival',
                       'mimic_donation_share','opportunity_adjusted_tag_preference','adjacent_hamming_similarity'):
            means = []
            for seed in seeds:
                values = [r[metric] for r in selected if r['training_seed'] == seed and r[metric] is not None]
                if values:
                    means.append(float(np.mean(values)))
            row[metric] = mean_ci(means)
        # Wilson interval for a well-defined independent seed-level event.
        row['all_worlds_survive'] = proportion_ci([
            all(r['resident_survival'] for r in selected if r['training_seed'] == seed) for seed in seeds])
        summary[f'{condition}/{control}/{"attack" if attack else "clean"}'] = row
    return summary


def feasibility(runs, seeds, worlds):
    selected = [r for r in runs if r['condition'] == 'clean' and r['control'] == 'trained' and not r['attack']]
    keys = {(r['training_seed'], r['world_seed']) for r in selected}
    expected = {(s,s+10000+k*1000) for s in seeds for k in range(worlds)}
    passed = keys == expected and len(selected) == len(expected) and bool(selected) and all(
        r['resident_survival'] and r['late_donation_rate'] >= .1 and r['late_donation_frequency'] >= .5 for r in selected)
    return dict(passed=bool(passed), evaluated_runs=len(selected),
                criterion='Every clean seed/world survives the full horizon; last 100 transitions have >=0.1 donations/resident-step and donations in >=50% of transitions.')


def write_report(out, result):
    def interval(row):
        if row['mean'] is None:
            return 'undefined'
        bounds = '' if row['ci95'] is None else f" [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]"
        return f"{row['mean']:.4f}{bounds}; n={row['n']}"
    lines = ['# Inherited neural policy study', '',
             'Original energy-only PPO with independent founder networks, inherited family policies and explicit survivor-based restocking. No relatedness or donation bonus enters PPO.', '',
             f"Clean feasibility passed: **{result['clean_feasibility']['passed']}**. This screen is not a kin-selection or convergence test.", '',
             'Evaluation worlds are averaged within independent training seeds. Intervals below are exploratory Student t intervals; bounded metrics can have out-of-range intervals. Null mimic shares mean no post-invasion donations, not successful defense.', '',
             '| Condition / control / evaluation | Final residents | Donations / resident step | Mean fraction of worlds surviving | Mimic donation share |',
             '| --- | --- | --- | --- | --- |']
    for name, row in result['summary'].items():
        lines.append('| '+name+' | '+' | '.join(interval(row[m]) for m in (
            'final_residents','donations_per_resident_step','resident_survival','mimic_donation_share'))+' |')
    lines += ['', 'Not run: '+', '.join(result['not_run'])+'.', '',
              'The untrained-ancestor control uses the same evolved founder chromosomes and ancestor composition with their initial neural weights. It is not an independent evolutionary training run. The evolution-only condition is the separate control for selection without PPO.', '',
              'Fallback survival is forced when starvation is disabled. Adjacent Hamming similarity needs its matched random-placement comparison before it supports assortment; tag preference is descriptive, not Hamiltonian or pedigree evidence.', '',
              'See [raw analysis](analysis.json) and [protocol](../../docs/inherited_policy_protocol.md).']
    (out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=8192)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--worlds', type=int, default=3)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS)
    p.add_argument('--worker', choices=CONDITIONS)
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or args.workers < 1 or args.worlds < 1 or len(set(args.seeds)) != len(args.seeds):
        p.error('positive steps/workers/worlds and distinct seeds are required')
    torch.set_num_threads(1)
    checkpoints = args.out/'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    def path(condition, seed):
        return checkpoints/f'{condition}_seed_{seed}.pt'
    if args.worker:
        train(path(args.worker,args.seeds[0]),seed=args.seeds[0],steps=args.steps,**CONDITIONS[args.worker])
        return
    previous = None
    if (args.out/'analysis.json').exists():
        previous = json.loads((args.out/'analysis.json').read_text(encoding='utf-8'))
        if (previous['training_seeds'] != args.seeds or previous['environment_steps_per_run'] != args.steps
            or previous['evaluation_worlds_per_seed'] != args.worlds):
            raise ValueError('Existing analysis has a different budget/seed/world protocol. Use a separate --out directory.')
    def worker(job):
        condition, seed = job
        target = path(condition,seed)
        if args.evaluate_only:
            return
        if target.exists():
            _, _, meta = load(target)
            expected = CONDITIONS[condition]
            if (meta['environment_steps'] == args.steps and meta['seed'] == seed
                and meta['source_sha256'] == source_hashes()
                and meta['torch_version'] == str(torch.__version__) and meta['numpy_version'] == np.__version__
                and meta['architecture'] == expected.get('architecture','attention')
                and meta['selection'] == expected.get('selection',True)
                and meta['learning'] == expected.get('learning',True)
                and meta['invasion'] == expected.get('invasion',False)
                and meta['environment'] == environment_config(**{'reward_mode':'energy',**expected.get('config',{})})
                and meta['ppo'] == PPO):
                print(f'Reusing {target}', flush=True)
                return
            raise ValueError(f'Existing checkpoint differs from requested protocol: {target}. Use a separate --out directory.')
        with target.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,'-u',__file__,'--worker',condition,'--seeds',str(seed),
                            '--steps',str(args.steps),'--out',str(args.out)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Completed {condition}, seed {seed}', flush=True)
    def execute(conditions):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, [(c,s) for c in conditions for s in args.seeds]))
        rows = []
        for condition in conditions:
            for seed in args.seeds:
                rows.extend(dict(condition=condition,**r) for r in evaluate_checkpoint(path(condition,seed),args.worlds,condition == 'clean'))
                print(f'Evaluated {condition}, seed {seed}',flush=True)
        return rows
    conditions = args.conditions or ['clean']
    runs = execute(conditions)
    if previous is not None:
        runs += [r for r in previous['runs'] if r['condition'] not in conditions]
        conditions = sorted({r['condition'] for r in runs})
    gate = feasibility(runs,args.seeds,args.worlds)
    if args.conditions is None:
        followups = [c for c in CONDITIONS if c != 'clean'] if gate['passed'] else ['no_selection','evolution_only','fallback']
        # Mechanism/fallback controls are informative even when clean training fails.
        runs.extend(execute(followups))
        conditions += followups
    result = dict(protocol='inherited-policy-ppo-v1',training_seeds=args.seeds,
                  environment_steps_per_run=args.steps,evaluation_worlds_per_seed=args.worlds,
                  clean_feasibility=gate, explicit_conditions=args.conditions is not None,
                  runs=runs,summary=summarize(runs),not_run=[c for c in CONDITIONS if c not in conditions])
    (args.out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    write_report(args.out,result)
    manifest = {p.relative_to(args.out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.out.rglob('*') if p.suffix in ('.json','.pt','.md') and p.name != 'manifest.json'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f"Clean feasibility: {gate['passed']}. Results: {args.out/'results.md'}",flush=True)


if __name__ == '__main__':
    main()
