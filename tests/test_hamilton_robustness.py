import numpy as np
import pytest

from experiments.hamilton_robustness import RewardControlGridWorld, CONDITIONS, ECOLOGIES, training_spec


def world(variant, recipient_is_mimic=False, inclusive=True):
    env = RewardControlGridWorld(grid_size=5, initial_population=3, max_capacity=4, initial_energy=30,
                                 donation_cost=2, donation_benefit=5, c_base=.6, c_density=.08,
                                 inclusive_fitness=inclusive, reward_variant=variant, seed=1)
    env.reset(seed=1)
    env.positions[:3] = [[1, 1], [1, 2], [3, 3]]
    env.grid.fill(-1)
    for slot, (x, y) in enumerate(env.positions[:3]):
        env.grid[x, y] = slot
    # Donor 0 and recipient 1 are identical (r = 1); agent 2 is the complement (r = 0).
    env.chromosomes[:3] = [[0]*8, [0]*8, [1]*8]
    env.is_parasite[1] = recipient_is_mimic
    env._update_cues()
    env._get_observations()          # the trainer always observes before acting
    return env


def give_minus_stay(variant, mimic=False, inclusive=True):
    out = []
    for give in (False, True):
        env = world(variant, mimic, inclusive)
        actions = np.zeros(env.N_max, dtype=int)
        if give:
            actions[0] = 5 + list(env.get_neighbors(0)).index(1)
        _, rewards, _, _ = env.step(actions)
        out.append(float(rewards[0]))
    return out[1] - out[0]


@pytest.mark.parametrize('variant, expected', [
    ('inclusive', -2 + 1.0*5),        # r_01 = 1
    ('prosocial', -2 + 5),            # credit B for any gift
    ('kinblind', -2 + 0.5*5),         # mean r over the two observed neighbours (1 and 0)
])
def test_reward_controls_change_the_donor_reward_exactly(variant, expected):
    assert np.isclose(give_minus_stay(variant), expected)


def test_energy_reward_ignores_variant():
    assert np.isclose(give_minus_stay('prosocial', inclusive=False), -2)


def test_mimic_recipient_rewarded_only_by_prosocial_control():
    assert np.isclose(give_minus_stay('prosocial', mimic=True), -2 + 5)
    assert np.isclose(give_minus_stay('inclusive', mimic=True), -2)
    # Kin-blind cannot tell the mimic apart, but its own neighbourhood credit counts mimics as unrelated.
    assert np.isclose(give_minus_stay('kinblind', mimic=True), -2)


def test_kinblind_uses_neighbours_seen_before_acting():
    env = world('kinblind')
    before = {i: list(js) for i, js in env._pre_step_neighbours.items()}
    actions = np.zeros(env.N_max, dtype=int)
    actions[0] = 5 + list(env.get_neighbors(0)).index(1)
    env.step(actions)
    assert before[0] == [1, 2]


def test_protocol_definitions():
    assert all(spec['ppo']['gamma'] == (.99 if 'gamma099' in name else .9) for name, spec in CONDITIONS.items())
    assert training_spec('b25_inclusive_long', 8192)[0] == 4*8192
    assert CONDITIONS['b4_prosocial']['config']['donation_benefit'] / CONDITIONS['b4_prosocial']['config']['donation_cost'] == 4
    assert {'bc100', 'bc150', 'bc200', 'bc250', 'bc300', 'bc400', 'bc600'} <= set(ECOLOGIES)


def test_mimic_reserve_guarantees_the_attack_and_keeps_the_resident_cap():
    """A full population still receives all 6 mimics; residents never exceed 45."""
    from experiments.config import environment_config
    for name, spec in CONDITIONS.items():
        assert spec.get('invasion', False) == ('mimic_reserve' in spec['config']), name
    cfg = environment_config(**CONDITIONS['b4_prosocial_mimics']['config'])
    env = RewardControlGridWorld(**cfg, seed=1)
    env.reset(seed=1)
    residents = lambda: int((env.mask & ~env.is_parasite).sum())
    for _ in range(80):
        env.energy[env.mask & ~env.is_parasite] = 100.0     # everyone can reproduce
        env.step(np.zeros(env.N_max, dtype=int))
    assert residents() == 45
    assert env.inject_mimics(6) == 6
    for _ in range(5):
        env.energy[env.mask] = 100.0
        env.step(np.zeros(env.N_max, dtype=int))
        assert residents() <= 45
    assert all(ECOLOGIES[e][0].get('mimic_reserve') == 6 for e, (_, attack) in ECOLOGIES.items() if attack)
