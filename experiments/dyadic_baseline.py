"""Explicit pairwise reciprocity baseline, separate from public reputation.

In a many-recipient donation game, Tit-for-tat needs an encounter convention.
When j can observe i, j cooperates with i iff its executed choice gives to i.
i remembers that directed encounter until their next one. Unknown pairs start
cooperatively. Among reciprocating eligible recipients, i chooses uniformly.
"""
import numpy as np
import torch

from experiments.inherited_policies import tensors_from
from models.rule_based import AlwaysDefectPolicy, KinOnlyPolicy, FixedWeightPolicy, ReciprocalPolicy


class DyadicTitForTat:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.memory = {}

    def act(self, obs, uids):
        actions = np.zeros(len(uids), dtype=np.int64)
        for slot in np.flatnonzero(obs['active_mask']):
            candidates = []
            for k in np.flatnonzero(obs['neighbor_mask'][slot]):
                other = obs['neighbor_indices'][slot,k]
                if obs['action_mask'][slot,5+k] and self.memory.get((int(uids[slot]),int(uids[other])),True):
                    candidates.append(5+k)
            if candidates:
                actions[slot] = self.rng.choice(candidates)
        return actions

    def observe(self, obs, uids, actions):
        for donor in np.flatnonzero(obs['active_mask']):
            chosen = int(actions[donor])-5
            for k in np.flatnonzero(obs['neighbor_mask'][donor]):
                recipient = obs['neighbor_indices'][donor,k]
                executed = bool(chosen == k and obs['action_mask'][donor,5+k])
                self.memory[(int(uids[recipient]),int(uids[donor]))] = executed


class RulePopulation:
    """Evaluation adapter; masks physical constraints and scripts mimic actions."""
    def __init__(self, name, seed):
        choices = {
            'selfish': lambda: AlwaysDefectPolicy(rng=np.random.default_rng(seed)),
            'tag_trust': lambda: KinOnlyPolicy(threshold=.3),
            'fixed_threshold': lambda: FixedWeightPolicy(threshold=.35),
            'reputation': lambda: ReciprocalPolicy(threshold=.5),
            'dyadic_tit_for_tat': lambda: DyadicTitForTat(seed),
        }
        self.policy = choices[name]()
        self.owners = {}

    def bind_founders(self, env, genomes=None):
        if genomes is not None:
            env.chromosomes[:env.N_init] = genomes
            env._update_cues()
        self.owners = {int(uid):0 for uid in env.agent_unique_ids[env.mask & ~env.is_parasite]}

    def inherit(self, births):
        for birth in births:
            if birth['parent_uid'] in self.owners:
                self.owners[birth['uid']] = 0

    def act(self, env, obs):
        if isinstance(self.policy,DyadicTitForTat):
            actions = self.policy.act(obs,env.agent_unique_ids)
        else:
            actions = self.policy.act(obs)
        valid = obs['action_mask'][np.arange(env.N_max),actions]
        actions[~valid | env.is_parasite] = 0
        if isinstance(self.policy,DyadicTitForTat):
            self.policy.observe(obs,env.agent_unique_ids,actions)
        owners = np.where(env.mask & ~env.is_parasite,0,-1)
        return tensors_from(obs), owners, torch.as_tensor(actions), None, None
