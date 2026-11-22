"""Robustness and reward-control study at B/C = 4 (Study C).

Same environment, architecture, per-family PPO, inheritance, selection and
evaluation as experiments/hamilton_study.py (imported, unchanged). It adds:

Reward controls (is KIN information what matters, or any prosocial credit?)
  prosocial  R_i = dE_i + B * #gifts                 (credit for helping anyone)
  kinblind   R_i = dE_i + B * mean_{j in Omega_i} r_ij * #gifts
             (the expected kin credit of a random neighbour, with no
              information about which recipient is related)
  inclusive  R_i = dE_i + sum over gifts of r_ij * B (main study)

Robustness at B/C = 4, where inclusive populations are viable: cue noise,
mimics, learned vs fixed gate, MLP, offspring placement, mutation, discount
for both rewards, evolution without learning; and 4x longer training at
B/C = 2.5. A scripted-strategy sweep covers every B/C level.

Mimic reserve. Mimics can only enter free slots, so a population already at
its resident cap (45) would receive no attack. Every attacked ecology here
therefore adds a reserve of 6 slots that only mimics may use
(max_capacity = 45 + 6, births still stop at 45 residents), so the attack
always delivers 6 mimics. With mimic_reserve = 0 the environment is unchanged.

python experiments/hamilton_robustness.py --workers 12
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

import experiments.hamilton_study as hs
import experiments.inherited_policies as ip
from environment.grid_world import GridWorld
from experiments.config import SEEDS as BASE_SEEDS, environment_config

OUT = ROOT/'results/hamilton_robustness'
SEEDS = tuple(BASE_SEEDS) + (505, 606, 707, 808, 909)
VARIANTS = ('inclusive', 'prosocial', 'kinblind')


class RewardControlGridWorld(hs.InclusiveGridWorld):
    """InclusiveGridWorld plus kin-free reward controls."""

    def __init__(self, *args, reward_variant='inclusive', mimic_reserve=0, **kwargs):
        if reward_variant not in VARIANTS:
            raise ValueError(f'reward_variant must be one of {VARIANTS}')
        self.mimic_reserve = int(mimic_reserve)
        super().__init__(*args, **kwargs)
        self.reward_variant = reward_variant

    @property
    def active_count(self):
        # GridWorld allows a birth while active_count < N_max. With a reserve,
        # residents compete for N_max - reserve slots and mimics never block a
        # birth; the extinction test (active_count == 0) is unchanged.
        n = int(np.sum(self.mask))
        if not self.mimic_reserve:
            return n
        residents = int(np.sum(self.mask & ~self.is_parasite))
        return residents + self.mimic_reserve if residents else n

    def step(self, actions):
        obs, rewards, dones, info = self._step(actions)
        info['active_population'] = int(np.sum(self.mask))
        return obs, rewards, dones, info

    def _step(self, actions):
        if not self.inclusive_fitness or self.reward_variant == 'inclusive':
            return super().step(actions)
        residents = self.mask & ~self.is_parasite
        slot_of = {int(u): int(s) for s, u in enumerate(self.agent_unique_ids) if self.mask[s]}
        r = hs.true_relatedness(self.chromosomes, residents)
        # Snapshot what each agent observed when acting; GridWorld.step recomputes
        # observations for the NEXT decision before returning.
        seen = {i: list(js) for i, js in self._pre_step_neighbours.items()}
        mimic = self._pre_step_mimic.copy()
        obs, rewards, dones, info = GridWorld.step(self, actions)   # plain energy reward
        credits, kin_mean = [], {}
        for record in info['donation_records']:
            donor, recipient = slot_of[record['donor_uid']], slot_of[record['recipient_uid']]
            credits.append(0.0 if record['recipient_is_mimic'] else float(r[donor, recipient]))
            if self.reward_variant == 'prosocial':
                rewards[donor] += self.B
            else:  # kinblind: mean true relatedness of the neighbours the donor saw (mimics count 0)
                if donor not in kin_mean:
                    js = seen.get(donor, [])
                    kin_mean[donor] = float(np.mean([0.0 if mimic[j] else r[donor, j] for j in js])) if js else 0.0
                rewards[donor] += kin_mean[donor] * self.B
        info['donation_true_relatedness'] = credits
        return obs, rewards, dones, info

    def _get_observations(self):
        obs = super()._get_observations()
        # Remember who each agent could see when it chose its action.
        self._pre_step_neighbours = {int(i): [int(j) for j in obs['neighbor_indices'][i] if j >= 0]
                                     for i in np.flatnonzero(self.mask)}
        self._pre_step_mimic = self.is_parasite.copy()
        return obs


# Both the trainer and the shared evaluator construct this environment.
ip.GridWorld = RewardControlGridWorld
hs.InclusiveGridWorld = RewardControlGridWorld

B4 = 8.0   # B/C = 4 with C = 2
GAMMA = dict(gamma=.9)


MIMICS = 6  # injected at step 10 (experiments/inherited_policies.py, hamilton_study.evaluate_world)
RESERVE = dict(max_capacity=environment_config()['max_capacity'] + MIMICS, mimic_reserve=MIMICS)


def cfg(reward='inclusive', benefit=B4, variant='inclusive', **extra):
    return dict(inclusive_fitness=reward == 'inclusive', reward_variant=variant,
                donation_cost=hs.COST, donation_benefit=benefit, **extra)


def attacked(**spec):
    """A mimic condition: invasion on, with the mimic-only slot reserve."""
    return {**spec, 'config': {**spec['config'], **RESERVE}, 'invasion': True}


_CONDITIONS = {
    'b4_prosocial': dict(config=cfg(variant='prosocial')),
    'b4_kinblind': dict(config=cfg(variant='kinblind')),
    'b4_inclusive_mimics': attacked(config=cfg()),
    'b4_prosocial_mimics': attacked(config=cfg(variant='prosocial')),
    'b4_kinblind_mimics': attacked(config=cfg(variant='kinblind')),
    'b4_noise025': dict(config=cfg(cue_noise=.25)),
    'b4_noise050': dict(config=cfg(cue_noise=.5)),
    'b4_noise050_mimics': attacked(config=cfg(cue_noise=.5)),
    'b4_fixedgate_noise050_mimics': attacked(config=cfg(cue_noise=.5), architecture='fixed_attention'),
    'b4_fixed_gate': dict(config=cfg(), architecture='fixed_attention'),
    'b4_mlp': dict(config=cfg(), architecture='mlp'),
    'b4_random_placement': dict(config=cfg(offspring_placement='random')),
    'b4_mutation000': dict(config=cfg(mutation_rate=0.)),
    'b4_mutation020': dict(config=cfg(mutation_rate=.2)),
    'b4_inclusive_gamma099': dict(config=cfg(), ppo=dict(gamma=.99)),
    'b4_energy_gamma099': dict(config=cfg('energy'), ppo=dict(gamma=.99)),
    'b4_evolution_only': dict(config=cfg('energy'), learning=False),
    'b25_inclusive_long': dict(config=cfg(benefit=5.0), budget_multiplier=4),
}
CONDITIONS = {k: {**v, 'ppo': {**GAMMA, **v.get('ppo', {})}} for k, v in _CONDITIONS.items()}
RULE_CONDITIONS = ('b4_noise050_mimics', 'b4_inclusive_mimics')

SWEEP_BENEFITS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0)
ECOLOGIES = {**{f'bc{int(b/hs.COST*100):03d}': (cfg('energy', b), False) for b in SWEEP_BENEFITS},
             'bc400_noise050': (cfg('energy', cue_noise=.5), False),
             'bc400_mimics': (cfg('energy', **RESERVE), True),
             'bc400_noise050_mimics': (cfg('energy', cue_noise=.5, **RESERVE), True)}
SWEEP_WORLDS = tuple(50000 + 1000*k for k in range(10))


def training_spec(condition, steps):
    spec = deepcopy(CONDITIONS[condition])
    return steps*spec.pop('budget_multiplier', 1), spec


def sweep_job(ecology, rule, world):
    config, attack = ECOLOGIES[ecology]
    env_cfg = environment_config(**{'reward_mode': 'energy', **config})
    actor = hs.make_rule(rule, world, env_cfg['donation_benefit'])
    run, _ = hs.evaluate_world(actor, None, env_cfg, world, attack, np.random.default_rng(world), sample_rate=0)
    run.pop('curve')
    return dict(ecology=ecology, b_over_c=env_cfg['donation_benefit']/env_cfg['donation_cost'], rule=rule, **run)


def study_hashes():
    hashes = hs.study_hashes()
    path = ROOT/'experiments/hamilton_robustness.py'
    hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=8192)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--worlds', type=int, default=3)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--worker', choices=CONDITIONS)
    p.add_argument('--evaluate', choices=CONDITIONS)
    args = p.parse_args()
    torch.set_num_threads(1)
    hs.RULE_CONDITIONS = RULE_CONDITIONS
    ckpt = args.out/'checkpoints'
    ckpt.mkdir(parents=True, exist_ok=True)
    if args.worker:
        budget, spec = training_spec(args.worker, args.steps)
        ip.train(ckpt/f'{args.worker}_seed_{args.seeds[0]}.pt', seed=args.seeds[0], steps=budget, **spec)
        return
    if args.evaluate:
        hs.evaluate_checkpoint(args.out, args.evaluate, args.seeds[0], args.worlds)
        return
    hashes = study_hashes()
    ip.archive_sources(args.out/'source', hashes)

    def run(job):
        stage, condition, seed = job
        if stage == 'train':
            target = ckpt/f'{condition}_seed_{seed}.pt'
            if target.exists():
                meta = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
                budget, spec = training_spec(condition, args.steps)
                expected = environment_config(**{'reward_mode': 'energy', **spec['config']})
                if meta['environment_steps'] != budget or meta['environment'] != expected:
                    raise ValueError(f'{target} differs from the requested protocol; use a new --out')
                return
            flag, log = ['--worker', condition], target.with_suffix('.log')
        else:
            if (args.out/'evaluations'/f'{condition}_seed_{seed}.json').exists():
                return
            (args.out/'evaluations').mkdir(parents=True, exist_ok=True)
            flag, log = ['--evaluate', condition], args.out/'evaluations'/f'{condition}_seed_{seed}.log'
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-u', __file__, *flag, '--seeds', str(seed), '--steps', str(args.steps),
                            '--worlds', str(args.worlds), '--out', str(args.out)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'{stage} done: {condition} seed {seed}', flush=True)

    # Longest jobs first so the pool drains evenly.
    order = sorted(CONDITIONS, key=lambda c: -CONDITIONS[c].get('budget_multiplier', 1))
    for stage in ('train', 'evaluate'):
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(run, [(stage, c, s) for c in order for s in args.seeds]))
    sweep_path = args.out/'strategy_sweep.json'
    if not sweep_path.exists():
        jobs = [(e, r, w) for e in ECOLOGIES for r in hs.RULES for w in SWEEP_WORLDS]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(sweep_job, *zip(*jobs), chunksize=4))
        sweep_path.write_text(json.dumps(rows, allow_nan=False), encoding='utf-8')
        print(f'strategy sweep done: {len(rows)} episodes', flush=True)
    (args.out/'protocol.json').write_text(json.dumps(dict(
        protocol='hamilton-robustness-v2', mimic_reserve=RESERVE, environment_steps_per_run=args.steps, training_seeds=args.seeds,
        evaluation_worlds_per_seed=args.worlds, base_environment=environment_config(), donation_cost=hs.COST,
        ppo=ip.PPO, conditions=CONDITIONS, rule_conditions=RULE_CONDITIONS, rules=hs.RULES,
        sweep_ecologies={k: dict(config=v[0], attack=v[1]) for k, v in ECOLOGIES.items()},
        sweep_worlds=SWEEP_WORLDS, row_columns=hs.ROW_COLUMNS, source_sha256=hashes), indent=2), encoding='utf-8')
    print('Robustness study complete', flush=True)


if __name__ == '__main__':
    main()
