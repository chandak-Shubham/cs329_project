import numpy as np
import pytest
import torch

from environment.grid_world import GridWorld
from experiments.inherited_policies import LineagePopulation, advantages, train, load, forward, optimize, PPO


def family_world(architecture='attention'):
    env = GridWorld(grid_size=5, initial_population=2, max_capacity=3,
                    c_base=0, c_density=0, seed=9)
    env.reset()
    population = LineagePopulation(2,architecture)
    population.bind_founders(env)
    return env, population


def test_founder_networks_are_independent_and_offspring_inherit_on_slot_reuse():
    env, population = family_world()
    first = next(population.policies[0].parameters())
    second = next(population.policies[1].parameters())
    assert first.data_ptr() != second.data_ptr()
    before = second.detach().clone()
    with torch.no_grad():
        first.add_(1)
    assert torch.equal(second, before)
    env.energy[:2] = [80, 0]
    old_uid = int(env.agent_unique_ids[1])
    _, _, dones, info = env.step(np.zeros(3, dtype=int))
    population.inherit(info['birth_records'])
    assert dones[1]
    assert population.owners[old_uid] == 1
    assert population.labels(env).tolist() == [0, 0, -1]
    assert int(env.agent_unique_ids[1]) != old_uid


def test_selection_copies_surviving_policy_without_aliasing_or_tag_scoring():
    env, population = family_world()
    original_parameter = next(population.policies[0].parameters())
    original_parameter.sum().backward()
    population.optimizers[0].step()
    original_moment = population.optimizers[0].state[original_parameter]['exp_avg'].clone()
    env.energy[:2] = [80, 0]
    _, _, _, info = env.step(np.zeros(3, dtype=int))
    population.inherit(info['birth_records'])
    parent_weights = {k: v.clone() for k, v in population.policies[0].state_dict().items()}
    genomes, event = population.select_survivors(env, np.random.default_rng(5))
    assert event['parent_policies'] == [0, 0]
    assert genomes.shape == (2, 8)
    assert population.origins == [0, 0]
    for policy in population.policies:
        assert all(torch.equal(v, parent_weights[k]) for k, v in policy.state_dict().items())
    assert next(population.policies[0].parameters()).data_ptr() != next(population.policies[1].parameters()).data_ptr()
    moments = [optimizer.state[next(policy.parameters())]['exp_avg']
               for policy,optimizer in zip(population.policies,population.optimizers)]
    assert all(torch.equal(moment,original_moment) for moment in moments)
    assert moments[0].data_ptr() != moments[1].data_ptr()


def test_extinction_cannot_create_an_evolutionary_winner():
    env, population = family_world()
    env.energy[:] = 0
    env.step(np.zeros(3, dtype=int))
    original = population.policies.copy()
    genomes, event = population.select_survivors(env, np.random.default_rng(3))
    assert genomes is None and not event['selected']
    assert population.policies == original


@pytest.mark.parametrize('architecture',['attention','fixed_attention','mlp'])
def test_batched_independent_actors_match_separate_network_outputs(architecture):
    env, population = family_world(architecture)
    env.energy[0] = 80
    obs, _, _, info = env.step(np.zeros(3,dtype=int))
    population.inherit(info['birth_records'])
    tensors, owners, actions, logs, values = population.act(env,obs,sample=False)
    with torch.no_grad():
        for index, policy in enumerate(population.policies):
            mask = torch.as_tensor(owners == index)
            dist, predicted, _ = forward(policy,tuple(t[mask] for t in tensors))
            torch.testing.assert_close(values[mask],predicted,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(logs[mask],dist.log_prob(actions[mask]),atol=1e-6,rtol=1e-5)


def test_ppo_cannot_update_an_unrelated_lineage_without_its_transitions():
    env,population = family_world()
    tensors,owners,actions,logs,values = population.act(env,env._get_observations())
    before = [{k:v.clone() for k,v in p.state_dict().items()} for p in population.policies]
    owners[owners != 0] = -1
    rollout = [dict(tensors=tensors,owners=torch.as_tensor(owners),actions=actions,
                    logs=logs,values=values,rewards=torch.ones(env.N_max)*10,
                    dones=torch.ones(env.N_max,dtype=torch.bool))]
    update = optimize(population,rollout,torch.zeros(env.N_max),{**PPO,'num_epochs':1})
    assert update['samples_by_policy'] == {'0':1}
    assert any(not torch.equal(v,before[0][k]) for k,v in population.policies[0].state_dict().items())
    assert all(torch.equal(v,before[1][k]) for k,v in population.policies[1].state_dict().items())
    assert not population.optimizers[1].state
    assert population._stacked is None


def test_gae_does_not_credit_newborn_to_dead_slot_occupant():
    rewards = torch.tensor([[1.], [100.]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[True], [False]])
    adv, returns = advantages(rewards, values, dones, torch.tensor([10.]), 1., 1.)
    torch.testing.assert_close(returns[:, 0], torch.tensor([1., 110.]))


@pytest.mark.parametrize('architecture',['attention','fixed_attention','mlp'])
def test_short_training_exact_budget_masks_mimics_and_roundtrips(tmp_path,architecture):
    path = tmp_path/'test.pt'
    metadata = train(path, steps=13, seed=7, invasion=True, architecture=architecture,
                     config=dict(grid_size=5, initial_population=2, max_capacity=4, max_steps=20),
                     ppo=dict(rollout_length=8, num_epochs=1))
    assert metadata['environment_steps'] == 13
    assert [row['duration'] for row in metadata['learning_curve']] == [8, 5]
    assert metadata['active_agent_transitions'] == 26  # two residents, no mimic samples
    population, genomes, saved = load(path)
    assert len(population.policies) == 2 and genomes.shape == (2, 8)
    assert saved == metadata
    assert all(np.isfinite(row['mean_loss']) for row in metadata['learning_curve'])
    from experiments.inheritance_study import evaluate
    first = evaluate(population,genomes,saved['environment'],10007)
    replay = evaluate(population,genomes,saved['environment'],10007)
    assert first == replay
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='hash mismatch'):
        load(path)
