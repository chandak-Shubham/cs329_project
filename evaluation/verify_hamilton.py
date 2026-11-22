"""Independent verification of the final (Hamilton) study, without retraining.

python evaluation/verify_hamilton.py           # ~3-5 minutes: one seed per condition
python evaluation/verify_hamilton.py --all     # every checkpoint
python evaluation/verify_hamilton.py --study robustness   # Study C (B/C = 4 controls)

1. Integrity: every committed checkpoint matches the SHA-256 in its sidecar and
   was trained with the protocol's budget, discount and B/C.
2. Recomputation: headline numbers are recomputed directly from the raw
   evaluation logs (results/study_b_hamilton/evaluations/*.json), independently
   of evaluation/hamilton_report.py, and compared with analysis.json.
   For Study C, step 2 instead checks that every attacked evaluation episode
   received its 6 mimics.
3. Re-simulation: saved checkpoints are re-evaluated from scratch in the same
   seeded worlds; every recorded metric must match the stored evaluation exactly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch

import training.paths as paths
STUDIES = dict(main=paths.study(paths.STUDY_B), robustness=paths.study(paths.STUDY_C))
STUDY = STUDIES['main']
LEVELS = ('bc100', 'bc150', 'bc250', 'bc400', 'bc600')
METRICS = ('duration', 'final_residents', 'resident_survival', 'donations_per_resident_step', 'late_donation_rate',
           'give_rate', 'hamilton_gap', 'hamilton_consistent_fraction', 'kin_preference', 'adjacent_similarity',
           'resident_births', 'mimics_injected', 'post_invasion_donations', 'mimic_donation_share')


def check_integrity(protocol):
    problems = 0
    for sidecar in sorted(STUDY.models.glob('*.json')):
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        condition = sidecar.stem.rsplit('_seed_', 1)[0]
        spec = protocol['conditions'][condition]
        ok = (hashlib.sha256(sidecar.with_suffix('.pt').read_bytes()).hexdigest() == meta['checkpoint_sha256']
              and meta['environment_steps'] == protocol['environment_steps_per_run']*spec.get('budget_multiplier', 1)
              and meta['ppo']['gamma'] == spec['ppo']['gamma']
              and meta['environment']['donation_benefit'] == spec['config']['donation_benefit']
              and meta['environment']['inclusive_fitness'] == spec['config']['inclusive_fitness']
              and meta['seed'] in protocol['training_seeds'])
        if not ok:
            problems += 1
            print('  INTEGRITY FAILURE', sidecar.name)
    count = len(list(STUDY.models.glob('*.json')))
    expected = len(protocol['conditions'])*len(protocol['training_seeds'])
    print(f'1. Integrity: {count}/{expected} checkpoints present, {problems} problems')
    return problems == 0 and count == expected


def recompute(analysis):
    runs = [r for p in sorted(STUDY.evaluations.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]

    def seed_mean(condition, metric, control='trained'):
        out = {}
        for r in runs:
            if (r['condition'], r['control'], r['attack']) == (condition, control, False):
                out.setdefault(r['training_seed'], []).append(r[metric])
        return {s: float(np.mean(v)) for s, v in out.items()}

    def survivors(condition):
        seeds = {}
        for r in runs:
            if (r['condition'], r['control'], r['attack']) == (condition, 'trained', False):
                seeds.setdefault(r['training_seed'], []).append(r['resident_survival'])
        return sum(all(v) for v in seeds.values())

    ok = True
    print('2. Recomputed from raw evaluation logs (seed means; survival = all 3 worlds survive)')
    print('   B/C   inclusive don/step  energy don/step  surv inc  surv energy')
    positives = pairs = 0
    for k in LEVELS:
        inc, ene = seed_mean(f'inclusive_{k}', 'donations_per_resident_step'), seed_mean(f'energy_{k}', 'donations_per_resident_step')
        m_inc, m_ene = np.mean(list(inc.values())), np.mean(list(ene.values()))
        s_inc, s_ene = survivors(f'inclusive_{k}'), survivors(f'energy_{k}')
        n = len(inc)
        print(f'   {int(k[2:])/100:<5} {m_inc:18.4f} {m_ene:16.4f} {s_inc:7d}/{n} {s_ene:9d}/{n}')
        stored = analysis['summary']
        ok &= abs(m_inc - stored[f'inclusive_{k}/trained/clean']['donations_per_resident_step']['mean']) < 1e-12
        ok &= abs(m_ene - stored[f'energy_{k}/trained/clean']['donations_per_resident_step']['mean']) < 1e-12
        for name, count in ((f'inclusive_{k}/trained', s_inc), (f'energy_{k}/trained', s_ene)):
            row = analysis['survival_by_bc'][name]
            ok &= row['n'] == n and round(row['mean']*row['n']) == count
        positives += sum(inc[s] > ene[s] for s in inc)
        pairs += len(inc)
    stacked = analysis['stacked_sign_test']['donations_per_resident_step']
    ok &= (positives, pairs) == (stacked['positive'], stacked['pairs'])
    print(f'   Inclusive > energy-only in {positives}/{pairs} paired runs (stored: {stacked["positive"]}/{stacked["pairs"]})')
    print(f'   Recomputed values match analysis.json: {"yes" if ok else "NO"}')
    return ok


def check_attacks():
    """Every attacked evaluation episode must actually contain its mimics."""
    runs = [r for p in sorted(STUDY.evaluations.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]
    attacked = [r for r in runs if r['attack']]
    short = [r for r in attacked if r['mimics_injected'] < 6]
    print(f'2. Mimic attacks: {len(attacked)-len(short)}/{len(attacked)} attacked episodes received all 6 mimics')
    return not short


def resimulate(protocol, all_seeds):
    import training.hamilton_study as hs
    import training.inherited_policies as ip
    if STUDY == STUDIES['robustness']:
        import training.hamilton_robustness  # noqa: F401  (installs the reward-control environment)
    torch.set_num_threads(1)
    seeds = protocol['training_seeds'] if all_seeds else protocol['training_seeds'][:1]
    jobs = [(c, s) for c in protocol['conditions'] for s in seeds]
    mismatches, start = 0, time.time()
    for i, (condition, seed) in enumerate(jobs, 1):
        stored = json.loads((STUDY.evaluations/f'{condition}_seed_{seed}.json').read_text(encoding='utf-8'))
        population, genomes, meta = ip.load(STUDY.models/f'{condition}_seed_{seed}.pt')
        world = seed + 10000
        attack = bool(meta['invasion'])
        run, _ = hs.evaluate_world(population, genomes, meta['environment'], world, attack,
                                   np.random.default_rng(world + (7 if attack else 0)))
        saved = next(r for r in stored if r['control'] == 'trained' and r['world_seed'] == world and r['attack'] == attack)
        diff = [m for m in METRICS if run[m] != saved[m]]
        mismatches += bool(diff)
        print(f'   [{i:3d}/{len(jobs)}] {condition}_seed_{seed} ({"attack" if attack else "clean"}): '
              + ('identical' if not diff else f'MISMATCH in {diff}'), flush=True)
    print(f'3. Re-simulation: {len(jobs)-mismatches}/{len(jobs)} checkpoints reproduce their stored evaluation exactly '
          f'({time.time()-start:.0f} s)')
    return mismatches == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all', action='store_true', help='re-simulate every checkpoint, not one seed per condition')
    p.add_argument('--study', choices=STUDIES, default='main')
    args = p.parse_args()
    global STUDY
    STUDY = STUDIES[args.study]
    protocol = json.loads((STUDY.results/'protocol.json').read_text(encoding='utf-8'))
    analysis = json.loads((STUDY.results/'analysis.json').read_text(encoding='utf-8'))
    checks = [check_integrity(protocol)]
    if args.study == 'main':
        checks.append(recompute(analysis))
    else:
        checks.append(check_attacks())
    checks.append(resimulate(protocol, args.all))
    results = checks
    print('\nVERIFICATION ' + ('PASSED' if all(results) else 'FAILED'))
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
