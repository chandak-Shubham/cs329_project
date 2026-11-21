"""Independent lineage PPO with chromosome/policy inheritance and selection.

This is a separate experiment: historical shared-policy artifacts stay unchanged.
PPO receives only the original individual energy increment. Periodic restocking
samples actual surviving residents, never donation, relatedness or gate scores.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from environment.grid_world import GridWorld
from experiments.config import environment_config
from models.attention_policy import GatedAttentionPolicy
from models.mlp_policy import SimpleMLPPolicy

ROOT = Path(__file__).resolve().parents[1]
PPO = dict(rollout_length=64, num_epochs=3, batch_size=64, lr=3e-4,
           gamma=.99, gae_lambda=.95, clip_eps=.2, vf_coef=.5,
           ent_coef=.01, max_grad_norm=.5)


def make_policy(architecture='attention'):
    if architecture == 'mlp':
        return SimpleMLPPolicy(hidden_dim=64)
    if architecture not in ('attention', 'fixed_attention'):
        raise ValueError('Unknown architecture')
    return GatedAttentionPolicy(d_model=32, num_heads=2, contextual_gate=True,
                                fixed_gate=.5 if architecture == 'fixed_attention' else None)


def forward(policy, tensors):
    if isinstance(policy, SimpleMLPPolicy):
        s, nf, nm, am = tensors
        dist, values = policy(torch.cat((s, nf.flatten(1), nm.sum(1, keepdim=True)/8), 1), am)
        return dist, values, None
    return policy(*tensors)


def tensors_from(obs):
    return tuple(torch.as_tensor(obs[key], dtype=dtype) for key, dtype in (
        ('self_state', torch.float32), ('neighbor_features', torch.float32),
        ('neighbor_mask', torch.bool), ('action_mask', torch.bool)))


class LineagePopulation:
    """A founder owns a distinct network; its descendants share that network.

    Ownership is keyed by enduring individual UID, never recyclable tensor slot.
    Other founders' optimizers cannot update this lineage's parameters.
    """
    def __init__(self, count, architecture='attention', lr=PPO['lr']):
        self.architecture, self.lr = architecture, lr
        self.policies = [make_policy(architecture) for _ in range(count)]
        self.optimizers = [torch.optim.Adam(p.parameters(), lr=lr, eps=1e-5)
                           for p in self.policies]
        self.owners = {}
        self.origins = list(range(count))
        self._stacked = None

    def bind_founders(self, env, genomes=None):
        founders = np.flatnonzero(env.mask & ~env.is_parasite)
        if len(founders) != len(self.policies):
            raise ValueError('Each founder must have exactly one network')
        self.owners = {int(env.agent_unique_ids[slot]): i for i, slot in enumerate(founders)}
        if genomes is not None:
            env.chromosomes[founders] = genomes
            env._update_cues()

    def inherit(self, births):
        for birth in births:
            parent = int(birth['parent_uid'])
            if parent in self.owners:  # scripted mimic families never own policies
                self.owners[int(birth['uid'])] = self.owners[parent]

    def labels(self, env):
        return np.array([self.owners.get(int(uid), -1) if active and not mimic else -1
                         for uid, active, mimic in zip(env.agent_unique_ids, env.mask, env.is_parasite)])

    @torch.no_grad()
    def act(self, env, obs, sample=True):
        tensors = tensors_from(obs)
        owners = self.labels(env)
        actions = torch.zeros(env.N_max, dtype=torch.long)
        log_probs, values = torch.zeros(env.N_max), torch.zeros(env.N_max)
        active_policies = np.unique(owners[owners >= 0])
        groups = [np.flatnonzero(owners == i) for i in active_policies]
        width = max(map(len, groups), default=0)
        if width:
            # Evaluate independent networks in one batched functional call.
            # Padding is never sampled, recorded or optimized as an agent.
            if self._stacked is None:
                self._stacked = torch.func.stack_module_state(self.policies)
            indices = torch.zeros((len(groups),width),dtype=torch.long)
            for i, group in enumerate(groups):
                indices[i,:len(group)] = torch.as_tensor(group)
            inputs = tuple(t[indices] for t in tensors)
            template = self.policies[0]
            def functional(parameters, buffers, s, nf, nm, am):
                if isinstance(template,SimpleMLPPolicy):
                    flat = torch.cat((s,nf.flatten(1),nm.sum(1,keepdim=True)/8),1)
                    dist,value = torch.func.functional_call(template,(parameters,buffers),(flat,am))
                else:
                    dist,value,_ = torch.func.functional_call(template,(parameters,buffers),(s,nf,nm,am))
                return dist.logits,value
            # Distribution argument validation uses tensor-dependent Python
            # branches; inputs are already constrained by environment masks.
            validate = torch.distributions.Distribution._validate_args
            fastpath = torch.backends.mha.get_fastpath_enabled()
            try:
                torch.distributions.Distribution.set_default_validate_args(False)
                torch.backends.mha.set_fastpath_enabled(False)
                stacked = tuple({k:v[active_policies] for k,v in state.items()} for state in self._stacked)
                logits,predictions = torch.vmap(functional)(*stacked,*inputs)
            finally:
                torch.distributions.Distribution.set_default_validate_args(validate)
                torch.backends.mha.set_fastpath_enabled(fastpath)
            for index, group in enumerate(groups):
                if not len(group):
                    continue
                dist = torch.distributions.Categorical(logits=logits[index,:len(group)])
                chosen = dist.sample() if sample else dist.probs.argmax(-1)
                actions[group], log_probs[group], values[group] = chosen, dist.log_prob(chosen), predictions[index,:len(group)]
        # Explicit stationary non-donating invaders, identical in every condition.
        return tensors, owners, actions, log_probs, values

    def select_survivors(self, env, rng):
        """Restock from living residents, copying policy AND optimizer state.

        Sampling individuals weights each lineage by its surviving abundance.
        Total extinction supplies no winner: retain networks and record failure.
        No artificial fitness is imputed to extinct individuals.
        """
        survivors = np.flatnonzero(env.mask & ~env.is_parasite)
        if not len(survivors):
            return None, dict(selected=False, reason='total resident extinction')
        selected = rng.choice(survivors, size=len(self.policies), replace=True)
        parents = [self.owners[int(env.agent_unique_ids[s])] for s in selected]
        policies = [deepcopy(self.policies[i]) for i in parents]
        optimizers = []
        for policy, parent in zip(policies, parents):
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5)
            optimizer.load_state_dict(deepcopy(self.optimizers[parent].state_dict()))
            optimizers.append(optimizer)
        self.policies, self.optimizers = policies, optimizers
        self._stacked = None
        self.origins = [self.origins[i] for i in parents]
        return env.chromosomes[selected].copy(), dict(
            selected=True, surviving_residents=len(survivors),
            parent_policies=parents, founder_origins=self.origins.copy())


def advantages(rewards, values, dones, next_values, gamma, gae_lambda):
    """Terminal death blocks credit even if the slot contains a newborn next."""
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(next_values)
    for t in reversed(range(len(rewards))):
        continuation = (~dones[t]).float()
        following = next_values if t == len(rewards)-1 else values[t+1]
        delta = rewards[t] + gamma*following*continuation - values[t]
        carry = delta + gamma*gae_lambda*continuation*carry
        result[t] = carry
    return result, result + values


def optimize(population, rollout, next_values, settings):
    population._stacked = None
    tensors = tuple(torch.stack([row['tensors'][i] for row in rollout]) for i in range(4))
    actions, old_logs, rewards, dones, values, owners = (
        torch.stack([row[key] for row in rollout])
        for key in ('actions', 'logs', 'rewards', 'dones', 'values', 'owners'))
    adv, returns = advantages(rewards, values, dones, next_values,
                              settings['gamma'], settings['gae_lambda'])
    losses, counts = [], {}
    for index in range(len(population.policies)):
        selected = owners == index
        count = int(selected.sum())
        if not count:
            continue
        counts[str(index)] = count
        inputs = tuple(t[selected] for t in tensors)
        a, old, target = actions[selected], old_logs[selected], returns[selected]
        advantage = adv[selected]
        advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        policy, optimizer = population.policies[index], population.optimizers[index]
        for _ in range(settings['num_epochs']):
            order = torch.randperm(count)
            for start in range(0, count, settings['batch_size']):
                batch = order[start:start+settings['batch_size']]
                dist, value, _ = forward(policy, tuple(t[batch] for t in inputs))
                ratio = (dist.log_prob(a[batch])-old[batch]).exp()
                clipped = ratio.clamp(1-settings['clip_eps'], 1+settings['clip_eps'])
                actor = -torch.minimum(ratio*advantage[batch], clipped*advantage[batch]).mean()
                critic = .5*(value-target[batch]).square().mean()
                loss = actor + settings['vf_coef']*critic - settings['ent_coef']*dist.entropy().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite lineage PPO loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), settings['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
    return dict(samples_by_policy=counts, optimizer_minibatches=len(losses),
                mean_loss=float(np.mean(losses)) if losses else None)


def source_hashes():
    paths = [ROOT/'experiments/inherited_policies.py', ROOT/'experiments/inheritance_study.py',
             ROOT/'experiments/dyadic_baseline.py', ROOT/'experiments/config.py',
             ROOT/'evaluation/uncertainty.py']
    paths += [p for folder in ('environment', 'models') for p in (ROOT/folder).glob('*.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def archive_sources(directory, hashes):
    """Keep the actual experiment source, not just hashes of unavailable versions."""
    directory = Path(directory)
    for name, digest in hashes.items():
        content = (ROOT/name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'Source changed while preparing experiment: {name}')
        target = directory/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f'Archive contains a different source: {target}')
            continue
        # Publish a complete file atomically. Concurrent workers must never
        # observe another writer's truncated file or overwrite a different one.
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            stream.write(content)
            temporary = Path(stream.name)
        try:
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != content:
                    raise ValueError(f'Archive contains a different source: {target}')
        finally:
            temporary.unlink()


def train(path, seed=42, steps=8192, config=None, architecture='attention',
          selection=True, learning=True, invasion=False, ppo=None):
    """Train an exact environment-transition budget; final short rollouts are valid."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    settings = {**PPO, **(ppo or {})}
    sources = source_hashes()
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise ValueError(f'Refusing to overwrite an existing experiment: {path}')
    archive_sources(path.parent.parent/'source', sources)
    if settings['rollout_length'] < 1:
        raise ValueError('rollout_length must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    config = environment_config(**{'reward_mode': 'energy', **(config or {})})
    if config['reward_mode'] != 'energy':
        raise ValueError('This protocol requires the original energy-only reward')
    env = GridWorld(**config, seed=seed)
    env.reset(seed=seed)
    population = LineagePopulation(env.N_init, architecture, settings['lr'])
    population.bind_founders(env)
    founder_genomes = env.chromosomes[:env.N_init].copy()
    selection_rng = np.random.default_rng(seed+100000)
    obs = env._get_observations()
    history, selection_history = [], []
    completed, generation, active_transitions = 0, 0, 0
    started = time.monotonic()
    while completed < steps:
        rollout, donations, births = [], 0, 0
        terminal = False
        for _ in range(min(settings['rollout_length'], steps-completed)):
            if invasion and env.step_count == 10:
                env.inject_mimics(6)
                obs = env._get_observations()
            tensors, owners, actions, logs, values = population.act(env, obs)
            next_obs, rewards, dones, info = env.step(actions.numpy())
            population.inherit(info['birth_records'])
            # Mimic-only states contain no learners and end this training episode.
            terminal = not (env.mask & ~env.is_parasite).any() or env.step_count >= env.max_steps
            if terminal:
                dones[:] = True
            rollout.append(dict(tensors=tensors, owners=torch.as_tensor(owners), actions=actions,
                                logs=logs, values=values, rewards=torch.as_tensor(rewards),
                                dones=torch.as_tensor(dones)))
            completed += 1
            active_transitions += int((owners >= 0).sum())
            donations += info['donations_count']
            births += sum(b['parent_uid'] in population.owners for b in info['birth_records'])
            obs = next_obs
            if terminal:
                break
        next_values = torch.zeros(env.N_max) if terminal else population.act(env, obs, sample=False)[-1]
        update = optimize(population, rollout, next_values, settings) if learning else dict(
            samples_by_policy={}, optimizer_minibatches=0, mean_loss=None)
        history.append(dict(environment_steps=completed, generation=generation, duration=len(rollout),
                            final_residents=int((env.mask & ~env.is_parasite).sum()),
                            donations_per_step=donations/len(rollout), births=births,
                            active_agent_transitions=active_transitions, **update))
        if terminal:
            genomes, event = population.select_survivors(env, selection_rng) if selection else (None, dict(selected=False, reason='selection disabled'))
            event.update(generation=generation, environment_steps=completed)
            selection_history.append(event)
            if genomes is not None:
                founder_genomes = genomes
            generation += 1
            env.reset()
            population.bind_founders(env, founder_genomes)
            obs = env._get_observations()
        if len(history) % 16 == 0 or completed == steps:
            print(f'seed={seed} steps={completed}/{steps} generation={generation} residents={history[-1]["final_residents"]} donations={history[-1]["donations_per_step"]:.3f}', flush=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(policies=[p.state_dict() for p in population.policies],
                    founder_genomes=torch.as_tensor(founder_genomes), origins=population.origins), path)
    metadata = dict(protocol='inherited-policy-ppo-v1', seed=seed, environment_steps=completed,
                    active_agent_transitions=active_transitions, architecture=architecture,
                    selection=selection, learning=learning, invasion=invasion, environment=config,
                    ppo=settings, learning_curve=history, selection_history=selection_history,
                    source_sha256=sources, torch_version=str(torch.__version__), numpy_version=np.__version__,
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    elapsed_seconds=time.monotonic()-started)
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    return metadata


def load(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    saved = torch.load(path, weights_only=True, map_location='cpu')
    population = LineagePopulation(len(saved['policies']), metadata['architecture'], metadata['ppo']['lr'])
    for policy, weights in zip(population.policies, saved['policies']):
        policy.load_state_dict(weights)
        policy.eval()
    population.origins = saved['origins']
    return population, saved['founder_genomes'].numpy(), metadata
