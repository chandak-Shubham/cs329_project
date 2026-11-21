import numpy as np

from experiments.hamilton_study import InclusiveGridWorld, HamiltonRule, true_relatedness, CONDITIONS
from evaluation.hamilton_report import logistic, sign_flip_p, holm, paired


def world(inclusive, recipient_is_mimic=False):
    env = InclusiveGridWorld(grid_size=5, initial_population=3, max_capacity=4, initial_energy=30,
                             donation_cost=2, donation_benefit=5, c_base=.6, c_density=.08,
                             inclusive_fitness=inclusive, seed=1)
    env.reset(seed=1)
    env.positions[:3] = [[1, 1], [1, 2], [3, 3]]
    env.grid.fill(-1)
    for slot, (x, y) in enumerate(env.positions[:3]):
        env.grid[x, y] = slot
    # Donor 0 and recipient 1 identical; agent 2 is the complement.
    env.chromosomes[:3] = [[0]*8, [0]*8, [1]*8]
    env.is_parasite[1] = recipient_is_mimic
    env._update_cues()
    return env


def donor_reward(inclusive, give, mimic=False):
    env = world(inclusive, mimic)
    actions = np.zeros(env.N_max, dtype=int)
    if give:
        actions[0] = 5 + list(env.get_neighbors(0)).index(1)
    _, rewards, _, info = env.step(actions)
    return float(rewards[0]), info


def test_true_relatedness_baseline_correction():
    chrom = np.array([[0]*8, [0]*8, [1]*8], dtype=np.int8)
    r = true_relatedness(chrom, np.array([True, True, True]))
    assert np.isclose(r[0, 1], 1.0) and r[0, 2] == 0 and r[0, 0] == 0


def test_inclusive_reward_is_minus_c_plus_rb():
    stay, _ = donor_reward(True, False)
    give, info = donor_reward(True, True)
    assert info['donation_true_relatedness'] == [1.0]
    assert np.isclose(give - stay, -2 + 1.0*5)


def test_energy_reward_has_no_kin_credit():
    stay, _ = donor_reward(False, False)
    give, _ = donor_reward(False, True)
    assert np.isclose(give - stay, -2)


def test_mimic_recipient_earns_no_inclusive_credit():
    stay, _ = donor_reward(True, False, mimic=True)
    give, info = donor_reward(True, True, mimic=True)
    assert info['donation_true_relatedness'] == [0.0]
    assert np.isclose(give - stay, -2)


def test_hamilton_rule_threshold():
    obs = dict(active_mask=np.array([True]), neighbor_mask=np.array([[True, True] + [False]*6]),
               action_mask=np.ones((1, 13), dtype=bool), neighbor_features=np.zeros((1, 8, 5), dtype=np.float32))
    obs['neighbor_features'][0, :2, 3] = [.3, .5]
    assert HamiltonRule(5, 2).act(obs)[0] == 6      # .5*5 > 2: give to the more related neighbour
    assert HamiltonRule(3, 2).act(obs)[0] == 0      # .5*3 < 2: do not give


def test_conditions_share_discount_except_sensitivity():
    gammas = {c: spec['ppo']['gamma'] for c, spec in CONDITIONS.items()}
    assert gammas.pop('inclusive_gamma099') == .99
    assert set(gammas.values()) == {.9}


def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(20000), rng.normal(size=20000)])
    y = (rng.random(20000) < 1/(1+np.exp(-(-.5 + 1.5*X[:, 1])))).astype(float)
    beta = logistic(X, y)
    assert np.allclose(beta, [-.5, 1.5], atol=.08)


def test_sign_flip_and_holm():
    assert np.isclose(sign_flip_p([1, 2, 3, 4, 5]), 2/32)
    assert sign_flip_p([1, -1, 1, -1]) == 1.0
    adjusted = holm({'a': .01, 'b': .04, 'c': .5})
    assert np.isclose(adjusted['a'], .03) and np.isclose(adjusted['b'], .08) and adjusted['c'] == .5
    c = paired({1: 2., 2: 3.}, {1: 1., 2: 1., 3: 0.})
    assert c['seeds'] == [1, 2] and c['positive_seeds'] == 2


def synthetic_rows(rng, n_opps, give_prob, choose_logits, bc=2.5):
    """Block-aligned decision rows like evaluate_world writes (9 columns)."""
    rows = []
    for _ in range(n_opps):
        n = int(rng.integers(1, 9))
        r = rng.random(n)
        b = rng.random(n)
        gave = rng.random() < give_prob(n)
        chosen = np.zeros(n)
        if gave:
            u = choose_logits(r, b)
            p = np.exp(u-u.max()); p /= p.sum()
            chosen[rng.choice(n, p=p)] = 1
        for k in range(n):
            # r_obs, r_true, b, energy_j, is_mimic, chosen, gave_any, n_neighbors, bc
            rows.append((r[k], r[k], b[k], .5, 0, chosen[k], float(gave), n, bc))
    return np.array(rows, dtype=np.float32)


def test_opportunity_blocks_follow_neighbour_counts():
    from evaluation.hamilton_report import opportunity_blocks
    rows = synthetic_rows(np.random.default_rng(0), 50, lambda n: .5, lambda r, b: 0*r)
    blocks = opportunity_blocks(rows)
    assert len(blocks) == 50 and sum(map(len, blocks)) == len(rows)
    assert all(len(b) == b[0, 7] for b in blocks)


def test_stratified_gap_removes_neighbour_count_confound():
    """A random policy (give prob n/(n+5)) has a positive raw gap but ~zero stratified gap."""
    from evaluation.hamilton_report import opportunity_blocks, stratified_gap
    rows = synthetic_rows(np.random.default_rng(1), 60000, lambda n: n/(n+5), lambda r, b: 0*r)
    above, gave = [], []
    for blk in opportunity_blocks(rows):
        above.append((blk[:, 1]*blk[0, 8] > 1).any())
        gave.append(blk[0, 6])
    above, gave = np.array(above), np.array(gave)
    raw = gave[above].mean() - gave[~above].mean()
    assert raw > .05
    assert abs(stratified_gap(rows)) < .02


def test_conditional_logit_recovers_recipient_weights():
    from evaluation.hamilton_report import conditional_logit, opportunity_blocks
    rows = synthetic_rows(np.random.default_rng(2), 6000, lambda n: 1.0, lambda r, b: 2.0*r - 1.0*b)
    beta = conditional_logit(opportunity_blocks(rows), lambda blk: blk[:, [0, 2]])
    assert np.allclose(beta, [2.0, -1.0], atol=.2)
