import numpy as np
import pytest
from environment.grid_world import GridWorld


def family():
    env = GridWorld(grid_size=8, initial_population=4, max_capacity=5,
                    c_base=0, c_density=0, enable_mitosis=False, reward_mode='lineage', seed=9)
    env.reset()
    # 0 -> 1 -> 2; agent 3 is unrelated despite having the same chromosome.
    for child, parent in ((1, 0), (2, 1)):
        env.parent_unique_ids[child] = parent
        env.ancestry_records[child]['parent_uid'] = parent
    env.chromosomes[:4] = 0
    return env


def test_lineage_reward_counts_living_children_and_grandchildren_not_tag_matches():
    env = family()
    _, reward, _, _ = env.step(np.zeros(5, dtype=int))
    np.testing.assert_array_equal(reward, [3, 2, 1, 1, 0])
    env.chromosomes[:4] = 1
    _, same, _, _ = env.step(np.zeros(5, dtype=int))
    np.testing.assert_array_equal(same, reward)


def test_dead_intermediate_ancestor_does_not_break_persistent_pedigree():
    env = family()
    env.energy[1] = 0
    _, reward, dones, _ = env.step(np.zeros(5, dtype=int))
    assert dones[1]
    assert env.living_descendant_counts()[0] == 1  # living grandchild through dead child
    assert reward[0] == 2 and reward[1] == 1  # final descendant reward, zero own survival


def test_lineage_reward_does_not_leak_to_newborn_in_reused_slot():
    env = family()
    env.enable_mitosis = True
    env.energy[1] = 0
    env.energy[3] = 80
    _, reward, dones, info = env.step(np.zeros(5, dtype=int))
    assert info['birth_records'][0]['slot'] == 1
    assert env.parent_unique_ids[1] == 3
    assert dones[1] and reward[1] == 1  # old individual's living child
    assert reward[3] == 2  # newborn descendant, no demographic +5 birth bonus
    assert env.living_descendant_counts()[0] == 1


def test_descendant_depth_is_explicit():
    env = family()
    env.descendant_depth = 1
    assert env.living_descendant_counts() == {0: 1, 1: 1}
    with pytest.raises(ValueError):
        GridWorld(descendant_depth=0)
