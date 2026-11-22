"""Hamilton's-rule study: energy-only versus inclusive-fitness PPO.

Final submission experiment. Every condition uses independent founder networks,
family policy inheritance and survivor selection (experiments/inherited_policies.py).
The only manipulated factors are the reward, B/C, cue noise, mimics, the gate,
the architecture and offspring placement.

Reward modes
  energy     R_i = dE_i                                  (original proposal)
  inclusive  R_i = dE_i + sum_{i gives to j} r_ij * B    (Hamilton 1964)
where r_ij is the baseline-corrected similarity of the TRUE chromosomes among
residents, and r_ij = 0 for mimics (invaders share no ancestry). The agent
never observes r_ij: it sees the possibly corrupted cue r~_ij and history b_j.
The one-step incentive to give to j is therefore exactly r_ij*B - C.

python src/training/hamilton_study.py --steps 8192 --workers 12
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch

import training.inherited_policies as ip
import training.paths as paths
from environment.cues import compute_pairwise_hamming
from environment.grid_world import GridWorld
from training.config import SEEDS, environment_config
from training.dyadic_baseline import RulePopulation
from networks.attention_policy import GatedAttentionPolicy

STUDY = paths.STUDY_B
COST = 2.0
BENEFITS = {'bc100': 2.0, 'bc150': 3.0, 'bc250': 5.0, 'bc400': 8.0, 'bc600': 12.0}


def true_relatedness(chromosomes, residents):
    """Baseline-corrected similarity of true chromosomes among residents only."""
    sim = compute_pairwise_hamming(chromosomes, residents)
    idx = np.flatnonzero(residents)
    if len(idx) < 2:
        return np.zeros_like(sim, dtype=np.float32)
    iu = np.triu_indices(len(idx), 1)
    h_bar = float(sim[idx[iu[0]], idx[iu[1]]].mean())
    r = np.clip((sim - h_bar) / max(1.0 - h_bar, 1e-6), 0.0, 1.0)
    r[:, ~residents] = 0.0
    r[~residents, :] = 0.0
    np.fill_diagonal(r, 0.0)
    return r.astype(np.float32)


class InclusiveGridWorld(GridWorld):
    """GridWorld whose donor reward adds Hamilton's inclusive-fitness credit r*B."""

    def __init__(self, *args, inclusive_fitness=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.inclusive_fitness = bool(inclusive_fitness)

    def step(self, actions):
        residents = self.mask & ~self.is_parasite
        slot_of = {int(u): int(s) for s, u in enumerate(self.agent_unique_ids) if self.mask[s]}
        r = true_relatedness(self.chromosomes, residents)
        obs, rewards, dones, info = super().step(actions)
        credits = []
        for record in info['donation_records']:
            donor, recipient = slot_of[record['donor_uid']], slot_of[record['recipient_uid']]
            rel = 0.0 if record['recipient_is_mimic'] else float(r[donor, recipient])
            credits.append(rel)
            if self.inclusive_fitness:
                rewards[donor] += rel * self.B
        info['donation_true_relatedness'] = credits
        return obs, rewards, dones, info


# Training uses the inherited-policy trainer unchanged, with this environment.
ip.GridWorld = InclusiveGridWorld


def cfg(reward='inclusive', bc='bc250', **extra):
    return dict(inclusive_fitness=reward == 'inclusive', donation_cost=COST,
                donation_benefit=BENEFITS[bc], **extra)


# Discount 0.9 for every condition (pilot-selected, disclosed in the report):
# with gamma 0.99 the inclusive pilot collapsed; the Hamilton incentive r*B - C
# is immediate, and the energy reward telescopes, so long horizons add mostly
# uncontrollable metabolic variance. inclusive_gamma099 reports the sensitivity.
GAMMA = dict(gamma=.9)
_CONDITIONS = {
    **{f'energy_{k}': dict(config=cfg('energy', k)) for k in BENEFITS},
    **{f'inclusive_{k}': dict(config=cfg('inclusive', k)) for k in BENEFITS},
    'evolution_only': dict(config=cfg('energy'), learning=False),
    'inclusive_gamma099': dict(config=cfg(), ppo=dict(gamma=.99)),
    'inclusive_noise025': dict(config=cfg(cue_noise=.25)),
    'inclusive_noise050': dict(config=cfg(cue_noise=.5)),
    'inclusive_mimics': dict(config=cfg(), invasion=True),
    'inclusive_noise050_mimics': dict(config=cfg(cue_noise=.5), invasion=True),
    'inclusive_fixedgate_noise050_mimics': dict(config=cfg(cue_noise=.5), invasion=True, architecture='fixed_attention'),
    'inclusive_fixed_gate': dict(config=cfg(), architecture='fixed_attention'),
    'inclusive_mlp': dict(config=cfg(), architecture='mlp'),
    'inclusive_random_placement': dict(config=cfg(offspring_placement='random')),
}
CONDITIONS = {name: {**spec, 'ppo': {**GAMMA, **spec.get('ppo', {})}} for name, spec in _CONDITIONS.items()}
RULE_CONDITIONS = ('inclusive_bc250', 'inclusive_noise050_mimics')
RULES = ('selfish', 'tag_trust', 'reputation', 'dyadic_tit_for_tat', 'hamilton_rule')


def study_hashes():
    hashes = ip.source_hashes()
    for p in (ROOT/'src/training/hamilton_study.py',):
        hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


class HamiltonRule:
    """Scripted Hamilton's rule on the OBSERVED cue: give to argmax r~ if r~*B > C."""
    def __init__(self, benefit, cost):
        self.B, self.C = benefit, cost

    def act(self, obs):
        actions = np.zeros(len(obs['active_mask']), dtype=np.int64)
        for slot in np.flatnonzero(obs['active_mask']):
            ks = [k for k in np.flatnonzero(obs['neighbor_mask'][slot]) if obs['action_mask'][slot, 5+k]]
            if ks:
                r = obs['neighbor_features'][slot, ks, 3]
                best = int(np.argmax(r))
                if r[best]*self.B > self.C:
                    actions[slot] = 5 + ks[best]
        return actions


def make_rule(name, seed, benefit):
    if name == 'hamilton_rule':
        rule = RulePopulation('selfish', seed)
        rule.policy = HamiltonRule(benefit, COST)
        return rule
    return RulePopulation(name, seed)


@torch.no_grad()
def gate_weights(population, env, obs, owners):
    """Mean genetic gate weight w^g over real neighbours, per family network."""
    values = []
    tensors = ip.tensors_from(obs)
    for index in np.unique(owners[owners >= 0]):
        policy = population.policies[index]
        if not isinstance(policy, GatedAttentionPolicy):
            return None
        group = np.flatnonzero(owners == index)
        _, _, w = policy(*(t[group] for t in tensors))
        m = tensors[2][group]
        if m.any():
            values.append(w[..., 0][m])
    return float(torch.cat(values).mean()) if values else None


def evaluate_world(population, genomes, config, world_seed, attack, row_rng, sample_rate=.25):
    """Frozen policies; logs ecology, Hamilton consistency and decision rows."""
    env = InclusiveGridWorld(**config, seed=world_seed)
    env.reset(seed=world_seed)
    population.bind_founders(env, genomes)
    obs = env._get_observations()
    torch.manual_seed(world_seed)
    B, C = env.B, env.C
    total = resident_steps = injected = post = to_mimics = births = 0
    opportunities = gives = 0
    hz = dict(above_opps=0, above_gives=0, below_opps=0, below_gives=0)
    recip_r, kin_pref, consistent, assort, gates, rows, curve = [], [], [], [], [], [], []
    remaining = int((env.mask & ~env.is_parasite).sum())
    for step in range(env.max_steps):
        if attack and step == 10:
            injected = env.inject_mimics(6)
            obs = env._get_observations()
        _, owners, actions, _, _ = population.act(env, obs)
        acting = np.flatnonzero(owners >= 0)
        residents = env.mask & ~env.is_parasite
        r_true = true_relatedness(env.chromosomes, residents)
        if step % 10 == 0 and hasattr(population, 'policies'):
            g = gate_weights(population, env, obs, owners)
            if g is not None:
                gates.append(g)
        for slot in acting:
            ks = np.flatnonzero(obs['neighbor_mask'][slot])
            if not len(ks) or not obs['action_mask'][slot, 5]:
                continue
            nbrs = obs['neighbor_indices'][slot, ks]
            mimic = env.is_parasite[nbrs]
            rt = np.where(mimic, 0.0, r_true[slot, nbrs])
            a = int(actions[slot])
            gave = a - 5 if a >= 5 else -1
            opportunities += 1
            gives += gave >= 0
            key = 'above' if (rt*B > C).any() else 'below'
            hz[key+'_opps'] += 1
            hz[key+'_gives'] += gave >= 0
            if gave >= 0:
                k = int(np.flatnonzero(ks == gave)[0])
                recip_r.append(float(rt[k]))
                kin_pref.append(float(rt[k] - rt.mean()))
                consistent.append(bool(rt[k]*B > C))
            if row_rng.random() < sample_rate:
                f = obs['neighbor_features'][slot, ks]
                for k in range(len(ks)):
                    rows.append((f[k, 3], rt[k], f[k, 4], f[k, 2], float(mimic[k]),
                                 float(ks[k] == gave), float(gave >= 0), len(ks), B/C))
        adjacent = [(i, j) for i in np.flatnonzero(residents) for j in env.get_neighbors(i)
                    if j > i and residents[j] and env.chebyshev_distance(env.positions[i], env.positions[j]) == 1]
        if adjacent:
            assort.append(float(np.mean([np.mean(env.chromosomes[i] == env.chromosomes[j]) for i, j in adjacent])))
        obs, _, _, info = env.step(actions.numpy())
        population.inherit(info['birth_records'])
        total += info['donations_count']
        resident_steps += len(acting)
        to_mimics += info['donations_to_mimics']
        if injected:
            post += info['donations_count']
        births += sum(b['parent_uid'] in population.owners for b in info['birth_records'])
        remaining = int((env.mask & ~env.is_parasite).sum())
        curve.append(dict(step=step+1, residents=remaining, acting=len(acting), donations=info['donations_count']))
        if not remaining:
            break
    tail = curve[-100:]
    rate = lambda g, o: g/o if o else None
    result = dict(
        world_seed=world_seed, attack=attack, duration=len(curve), final_residents=remaining,
        resident_survival=bool(remaining > 0 and len(curve) == env.max_steps),
        donations_per_resident_step=total/max(resident_steps, 1),
        late_donation_rate=sum(r['donations'] for r in tail)/max(sum(r['acting'] for r in tail), 1),
        give_rate=rate(gives, opportunities),
        give_rate_hamilton_favoured=rate(hz['above_gives'], hz['above_opps']),
        give_rate_hamilton_disfavoured=rate(hz['below_gives'], hz['below_opps']),
        hamilton_gap=(rate(hz['above_gives'], hz['above_opps']) - rate(hz['below_gives'], hz['below_opps'])
                      if hz['above_opps'] and hz['below_opps'] else None),
        hamilton_consistent_fraction=float(np.mean(consistent)) if consistent else None,
        recipient_true_relatedness=float(np.mean(recip_r)) if recip_r else None,
        kin_preference=float(np.mean(kin_pref)) if kin_pref else None,
        adjacent_similarity=float(np.mean(assort)) if assort else None,
        genetic_gate_weight=float(np.mean(gates)) if gates else None,
        resident_births=births, mimics_injected=injected, post_invasion_donations=post,
        mimic_donation_share=to_mimics/post if post else None, curve=curve)
    return result, np.asarray(rows, dtype=np.float32).reshape(-1, 9)


ROW_COLUMNS = ['r_obs', 'r_true', 'b', 'energy_j', 'is_mimic', 'chosen', 'gave_any', 'n_neighbors', 'bc']


def evaluate_checkpoint(out, condition, seed, worlds):
    population, genomes, meta = ip.load(out.models/f'{condition}_seed_{seed}.pt')
    config = meta['environment']
    runs, arrays = [], {}
    for rep in range(worlds):
        world = seed + 10000 + 1000*rep
        attacks = (False, True) if meta['invasion'] else (False,)
        torch.manual_seed(seed)
        initial = ip.LineagePopulation(len(population.policies), meta['architecture'])
        initial.policies = [deepcopy(initial.policies[i]) for i in population.origins]
        controls = [('trained', population), ('untrained_ancestors', initial)]
        if condition in RULE_CONDITIONS:
            controls += [('rule_'+n, make_rule(n, world, config['donation_benefit'])) for n in RULES]
        for name, actor in controls:
            for attack in attacks:
                rng = np.random.default_rng(world + (7 if attack else 0))
                run, rows = evaluate_world(actor, genomes, config, world, attack, rng)
                runs.append(dict(condition=condition, training_seed=seed, control=name, **run))
                arrays[f'{name}|{"attack" if attack else "clean"}|{world}'] = rows
    target = out.evaluations
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target/f'{condition}_seed_{seed}_rows.npz', **arrays)
    (target/f'{condition}_seed_{seed}.json').write_text(json.dumps(runs, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=8192)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--worlds', type=int, default=3)
    p.add_argument('--study', default=STUDY, help='study folder name under models/, results/ and plots/')
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS)
    p.add_argument('--worker', choices=CONDITIONS)
    p.add_argument('--evaluate', choices=CONDITIONS)
    args = p.parse_args()
    S = paths.study(args.study)
    torch.set_num_threads(1)
    ckpt = S.models
    ckpt.mkdir(parents=True, exist_ok=True)
    if args.worker:
        spec = deepcopy(CONDITIONS[args.worker])
        ip.train(ckpt/f'{args.worker}_seed_{args.seeds[0]}.pt', seed=args.seeds[0], steps=args.steps, **spec)
        return
    if args.evaluate:
        evaluate_checkpoint(S, args.evaluate, args.seeds[0], args.worlds)
        return
    hashes = study_hashes()
    ip.archive_sources(S.source, hashes)

    def run(job):
        stage, condition, seed = job
        if stage == 'train':
            target = ckpt/f'{condition}_seed_{seed}.pt'
            if target.exists():
                meta = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
                expected = environment_config(**{'reward_mode': 'energy', **CONDITIONS[condition]['config']})
                if meta['environment_steps'] != args.steps or meta['environment'] != expected:
                    raise ValueError(f'{target} differs from the requested protocol; use a new --study')
                return
            flag = ['--worker', condition]
            log = target.with_suffix('.log')
        else:
            if (S.evaluations/f'{condition}_seed_{seed}.json').exists():
                return
            flag = ['--evaluate', condition]
            log = S.evaluations/f'{condition}_seed_{seed}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-u', __file__, *flag, '--seeds', str(seed), '--steps', str(args.steps),
                            '--worlds', str(args.worlds), '--study', args.study],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'{stage} done: {condition} seed {seed}', flush=True)

    conditions = args.conditions or list(CONDITIONS)
    for stage in ('train', 'evaluate'):
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(run, [(stage, c, s) for c in conditions for s in args.seeds]))
    (S.results/'protocol.json').write_text(json.dumps(dict(
        protocol='hamilton-study-v1', environment_steps_per_run=args.steps, training_seeds=args.seeds,
        evaluation_worlds_per_seed=args.worlds, base_environment=environment_config(),
        donation_cost=COST, benefits=BENEFITS, ppo=ip.PPO,
        conditions={c: CONDITIONS[c] for c in conditions}, rule_conditions=RULE_CONDITIONS, rules=RULES,
        row_columns=ROW_COLUMNS, source_sha256=hashes), indent=2), encoding='utf-8')
    print('Study complete', flush=True)


if __name__ == '__main__':
    main()
