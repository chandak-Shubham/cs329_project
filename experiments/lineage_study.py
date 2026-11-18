"""Lineage-fitness follow-up. Stop after clean training if feasibility fails.

Uses the same architecture, budgets and ecology as the revised study. Historical
results remain unchanged. A feasibility threshold controls experiments, not actions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments import revised_study as study
from experiments.config import SEEDS
from evaluation.uncertainty import mean_ci

OUT = Path('results/lineage_study')
CONDITIONS = {
    'lineage_clean': dict(reward_mode='lineage', descendant_reward=1., descendant_depth=2),
    'lineage_noise025': dict(reward_mode='lineage', descendant_reward=1., descendant_depth=2, cue_noise=.25),
    'lineage_noise050': dict(reward_mode='lineage', descendant_reward=1., descendant_depth=2, cue_noise=.5),
    'lineage_noise_mimics': dict(reward_mode='lineage', descendant_reward=1., descendant_depth=2, cue_noise=.5),
    'fixed_noise_mimics': dict(reward_mode='lineage', descendant_reward=1., descendant_depth=2, cue_noise=.5),
}


def clean_feasibility(runs, seeds):
    selected = [r for r in runs if r['condition'] == 'lineage_clean' and not r['attack']]
    records = []
    for run in selected:
        tail = run['curve'][-100:]
        rate = sum(r['donations'] for r in tail) / max(sum(r['acting_residents'] for r in tail), 1)
        frequency = sum(r['donations'] > 0 for r in tail) / max(len(tail), 1)
        passed = run['duration'] == run['environment']['max_steps'] and run['resident_survival'] and rate >= .1 and frequency >= .5
        records.append(dict(seed=run['training_seed'], final_residents=run['final_residents'],
                            duration=run['duration'], late_donations_per_resident_step=rate,
                            late_steps_with_donations=frequency, passed=bool(passed)))
    return dict(passed=len(records) == len(seeds) and all(r['passed'] for r in records), runs=records,
                criterion='Every training seed: residents survive 300 transitions; last 100 transitions have >=0.1 donations/resident-step and donations in >=50% of transitions.',
                interpretation='Exploratory feasibility screen, not a statistical claim of kin selection.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-steps', type=int, default=8192)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--worker', choices=CONDITIONS)
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    study.OUT, study.CONDITIONS = OUT, CONDITIONS
    (OUT/'checkpoints').mkdir(parents=True, exist_ok=True)
    if args.worker:
        study.train(args.worker, args.seeds[0], args.environment_steps)
        return

    def worker(job):
        condition, seed = job
        if study.reusable(condition, seed, args.environment_steps):
            return
        path = study.checkpoint_path(condition, seed).with_suffix('.log')
        with path.open('w') as log:
            subprocess.run([sys.executable, '-u', __file__, '--worker', condition,
                            '--seeds', str(seed), '--environment-steps', str(args.environment_steps)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Completed {condition}, seed {seed}', flush=True)

    if not args.evaluate_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, [('lineage_clean', s) for s in args.seeds]))
    result = study.analyze(args.seeds, ['lineage_clean'], attack_modes=(False,))
    gate = clean_feasibility(result['runs'], args.seeds)
    (OUT/'feasibility.json').write_text(json.dumps(gate, indent=2, allow_nan=False))
    if gate['passed']:
        if not args.evaluate_only:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(worker, [(c, s) for c in CONDITIONS if c != 'lineage_clean' for s in args.seeds]))
        result = study.analyze(args.seeds, list(CONDITIONS))
    result['protocol'] = 'lineage-fitness-v1'
    result['paired_contrasts'] = {}
    for metric in ('donations_per_resident_step', 'final_residents', 'resident_births'):
        differences = []
        for seed in args.seeds:
            values = {r['condition']: r[metric] for r in result['runs'] if r['training_seed'] == seed and not r['attack']}
            differences.append(values['lineage_clean']-values['lineage_clean_untrained'])
        result['paired_contrasts']['trained_minus_untrained/'+metric] = mean_ci(differences)
    result['common_input_paired_contrasts'] = {k:v for k,v in result['common_input_paired_contrasts'].items() if v['n']}
    result['clean_feasibility'] = gate
    result['not_run'] = [] if gate['passed'] else [c for c in CONDITIONS if c != 'lineage_clean']
    result['limitations'] = [
        'Lineage-fitness objective is designed explicitly; no kin-directed action, relatedness or Hamilton threshold enters the reward.',
        'Reward = own survival + living children/grandchildren; coefficients both 1. No separate birth bonus.',
        'Shared neural policy; chromosomes and ancestry are inherited, not lineage-specific networks.',
        'Credit ends with an individual terminal transition; posthumous descendant returns are not propagated.',
        'Five training seeds, one evaluation world per checkpoint; feasibility thresholds are exploratory.',
        'Noisy/mimic follow-ups are withheld if clean feasibility fails, rather than presented as completed experiments.',
    ]
    result['checkpoints'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'checkpoints').glob('*.pt')}
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    note = ('Clean feasibility passed; conditional follow-up conditions were executed.' if gate['passed']
            else 'Clean feasibility failed. Noise, mimic and fixed-gate follow-ups were not run. This remains a negative result.')
    path = OUT/'results.md'
    body = path.read_text().replace('# Revised study: executed results\n', '')
    body = body.replace('docs/review_response_2026-11-17.md', '../../docs/final_review_response_2026-11-18.md')
    body = body.replace('Fallback survival is forced by disabling starvation and cannot support ecological conclusions. ', '')
    body = body.replace('Raw seed trajectories, exposure-conditioned mimic shares, cue interventions, mutation sweeps and paired contrasts:',
                        'Raw clean trajectories, cue interventions, trained-minus-untrained contrasts and explicitly unperformed follow-ups:')
    path.write_text('# Lineage-fitness follow-up\n\n'+note+'\n\n'+body, encoding='utf-8')
    print(note, flush=True)


if __name__ == '__main__':
    main()
