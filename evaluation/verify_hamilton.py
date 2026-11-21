"""Independent verification of the final (Hamilton) study, without retraining.

python evaluation/verify_hamilton.py           # ~3-5 minutes: one seed per condition
python evaluation/verify_hamilton.py --all     # every checkpoint (~30 minutes)

1. Integrity: every committed checkpoint matches the SHA-256 in its sidecar and
   was trained with the protocol's budget, discount and B/C.
2. Recomputation: headline numbers are recomputed directly from the raw
   evaluation logs (results/hamilton_study/evaluations/*.json), independently
   of evaluation/hamilton_report.py, and compared with analysis.json.
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
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

STUDY = ROOT/'results/hamilton_study'
LEVELS = ('bc100', 'bc150', 'bc250', 'bc400', 'bc600')
METRICS = ('duration', 'final_residents', 'resident_survival', 'donations_per_resident_step', 'late_donation_rate',
           'give_rate', 'hamilton_gap', 'hamilton_consistent_fraction', 'kin_preference', 'adjacent_similarity',
           'resident_births', 'mimics_injected', 'post_invasion_donations', 'mimic_donation_share')


def check_integrity(protocol):
    problems = 0
    for sidecar in sorted((STUDY/'checkpoints').glob('*.json')):
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        condition = sidecar.stem.rsplit('_seed_', 1)[0]
        spec = protocol['conditions'][condition]
        ok = (hashlib.sha256(sidecar.with_suffix('.pt').read_bytes()).hexdigest() == meta['checkpoint_sha256']
              and meta['environment_steps'] == protocol['environment_steps_per_run']
              and meta['ppo']['gamma'] == spec['ppo']['gamma']
              and meta['environment']['donation_benefit'] == spec['config']['donation_benefit']
              and meta['environment']['inclusive_fitness'] == spec['config']['inclusive_fitness']
              and meta['seed'] in protocol['training_seeds'])
        if not ok:
            problems += 1
            print('  INTEGRITY FAILURE', sidecar.name)
    count = len(list((STUDY/'checkpoints').glob('*.json')))
    expected = len(protocol['conditions'])*len(protocol['training_seeds'])
    print(f'1. Integrity: {count}/{expected} checkpoints present, {problems} problems')
    return problems == 0 and count == expected


def recompute(analysis):
    runs = [r for p in sorted((STUDY/'evaluations').glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]

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
        print(f'   {int(k[2:])/100:<5} {m_inc:18.4f} {m_ene:16.4f} {s_inc:7d}/5 {s_ene:9d}/5')
        stored = analysis['summary']
        ok &= abs(m_inc - stored[f'inclusive_{k}/trained/clean']['donations_per_resident_step']['mean']) < 1e-12
        ok &= abs(m_ene - stored[f'energy_{k}/trained/clean']['donations_per_resident_step']['mean']) < 1e-12
        ok &= round(analysis['survival_by_bc'][f'inclusive_{k}/trained']['mean']*5) == s_inc
        ok &= round(analysis['survival_by_bc'][f'energy_{k}/trained']['mean']*5) == s_ene
        positives += sum(inc[s] > ene[s] for s in inc)
        pairs += len(inc)
    stacked = analysis['stacked_sign_test']['donations_per_resident_step']
    ok &= (positives, pairs) == (stacked['positive'], stacked['pairs'])
    print(f'   Inclusive > energy-only in {positives}/{pairs} paired runs (stored: {stacked["positive"]}/{stacked["pairs"]})')
    print(f'   Recomputed values match analysis.json: {"yes" if ok else "NO"}')
    return ok


def resimulate(protocol, all_seeds):
    import experiments.hamilton_study as hs
    import experiments.inherited_policies as ip
    torch.set_num_threads(1)
    seeds = protocol['training_seeds'] if all_seeds else protocol['training_seeds'][:1]
    jobs = [(c, s) for c in protocol['conditions'] for s in seeds]
    mismatches, start = 0, time.time()
    for i, (condition, seed) in enumerate(jobs, 1):
        stored = json.loads((STUDY/'evaluations'/f'{condition}_seed_{seed}.json').read_text(encoding='utf-8'))
        population, genomes, meta = ip.load(STUDY/'checkpoints'/f'{condition}_seed_{seed}.pt')
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
    args = p.parse_args()
    protocol = json.loads((STUDY/'protocol.json').read_text(encoding='utf-8'))
    analysis = json.loads((STUDY/'analysis.json').read_text(encoding='utf-8'))
    results = [check_integrity(protocol), recompute(analysis), resimulate(protocol, args.all)]
    print('\nVERIFICATION ' + ('PASSED' if all(results) else 'FAILED'))
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
