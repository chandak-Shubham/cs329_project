"""Scientific regressions called out by the 17 November review."""
import numpy as np
import torch
import json

from environment.grid_world import GridWorld
from models.attention_policy import GatedAttentionPolicy
from experiments.revised_study import cue_interventions, evaluate
from training.train_all_models import train_model


def test_empty_resident_rollout_skips_optimizer_and_serializes_null_losses(tmp_path):
    path = tmp_path / 'empty.pt'
    train_model('attention', environment_steps=2, rollout_length=2,
                num_epochs=1, env_config={'initial_population': 0, 'max_capacity': 1},
                save_path=str(path), seed=2)
    record = json.loads(path.with_suffix('.json').read_text(),
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert record['active_agent_transitions'] == record['ppo_updates'] == 0
    assert record['learning_curve'][0]['policy_loss'] is None
    assert record['learning_curve'][0]['value_loss'] is None
    torch.manual_seed(2)
    initial = GatedAttentionPolicy(d_model=32, num_heads=2).state_dict()
    saved = torch.load(path, weights_only=True)
    assert all(torch.equal(initial[key], saved[key]) for key in initial)


def test_vectorized_neighbors_match_grid_scan_in_training_world():
    for seed in range(5):
        env = GridWorld(grid_size=18, initial_population=45, max_capacity=45, seed=seed)
        env.reset()
        for i in range(45):
            reference = []
            x, y = env.positions[i]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    j = env.grid[(x+dx) % env.L, (y+dy) % env.L]
                    if j >= 0 and j != i:
                        reference.append((max(abs(dx), abs(dy)), int(j)))
            assert env.get_neighbors(i) == [j for _, j in sorted(reference)]


def test_demographic_reward_credits_parent_and_not_reused_dead_slot():
    env = GridWorld(grid_size=5, initial_population=2, max_capacity=2,
                    c_base=0, c_density=0, reward_mode="demographic", seed=9)
    env.reset()
    env.energy[:] = [80, 0]
    _, rewards, dones, info = env.step(np.zeros(2, dtype=int))
    assert info['births_this_step'] == info['deaths_this_step'] == 1
    np.testing.assert_array_equal(rewards, [6, 0])
    assert dones[1] and not dones[0]


def test_reward_has_no_donation_or_relatedness_bonus_and_records_mimic_transfer():
    env = GridWorld(grid_size=4, initial_population=2, max_capacity=2,
                    enable_mitosis=False, c_base=0, c_density=0,
                    reward_mode="demographic", seed=2)
    env.reset()
    env.is_parasite[1] = True
    _, rewards, _, info = env.step(np.array([5, 0]))
    np.testing.assert_array_equal(rewards, [1, 1])
    assert info['donations_to_mimics'] == 1
    assert info['energy_to_mimics'] == env.B
    assert info['donation_records'][0]['recipient_uid'] == int(env.agent_unique_ids[1])


def test_mimics_copy_modal_resident_genome_and_randomize_empty_cells():
    positions = []
    for seed in range(4):
        env = GridWorld(grid_size=10, initial_population=4, max_capacity=8, seed=seed)
        env.reset()
        env.chromosomes[:4] = [[1]*8, [0]*8, [0]*8, [1]*8]
        env.is_parasite[3] = True
        assert env.inject_mimics(2) == 2
        assert not env.chromosomes[4:6].any()
        positions.extend(map(tuple, env.positions[4:6]))
        assert len(set(map(tuple, env.positions[env.mask]))) == env.active_count
    assert len(set(positions)) > 4
    assert any(x > 1 for x, y in positions)


def test_context_gate_ignores_padding_and_sees_other_neighbors():
    torch.manual_seed(2)
    policy = GatedAttentionPolicy(d_model=32, num_heads=2, contextual_gate=True)
    s, nf = torch.rand(2, 2), torch.rand(2, 8, 5)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :2] = True
    dist, _, weights = policy(s, nf, mask)
    padded = nf.clone()
    padded[:, 2:] = 999
    changed, _, same_weights = policy(s, padded, mask)
    assert torch.allclose(dist.probs, changed.probs)
    assert torch.allclose(weights[:, :2], same_weights[:, :2])
    other = nf.clone()
    other[:, 1, 3:] += 1
    _, _, context_weights = policy(s, other, mask)
    assert not torch.allclose(weights[:, 0], context_weights[:, 0])


def test_intervention_is_deterministic_and_does_not_mutate_inputs():
    torch.manual_seed(3)
    policy = GatedAttentionPolicy(d_model=32, num_heads=2, contextual_gate=True)
    env = GridWorld(grid_size=5, initial_population=8, max_capacity=8, seed=1)
    tensors = policy.extract_tensors(env.reset())
    original = tensors[1].clone()
    first = cue_interventions(policy, tensors, np.random.default_rng(6))
    second = cue_interventions(policy, tensors, np.random.default_rng(6))
    assert first == second
    assert first['genetic_tv'] >= 0
    assert torch.equal(tensors[1], original)
