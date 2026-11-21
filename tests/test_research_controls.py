import json
import numpy as np
import pytest
import torch

from environment.grid_world import GridWorld
from models.attention_policy import GatedAttentionPolicy
from evaluation.uncertainty import mean_ci, proportion_ci
from evaluation.analysis import fit_logistic_regression
from experiments.run_phases import load_checkpoint
from training.train_all_models import train_model


def test_wilson_interval_preserves_uncertainty_at_zero_survival():
    result = proportion_ci([False] * 5)
    assert result["mean"] == 0
    assert result["ci95"][1] > .4
    assert proportion_ci([None])["ci95"] is None
    assert mean_ci([None])["mean"] is None
    assert mean_ci([1, 2, 3])["ci95"][0] < 2 < mean_ci([1, 2, 3])["ci95"][1]


def test_missing_checkpoint_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_checkpoint(GatedAttentionPolicy(), "nonexistent_research_model.pt", 999)


def test_modified_checkpoint_is_rejected(tmp_path, monkeypatch):
    import experiments.run_phases as phases
    monkeypatch.setattr(phases, "RESULTS_DIR", str(tmp_path))
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    (folder / "model_seed_1.pt").write_bytes(b"modified checkpoint")
    (folder / "model_seed_1.json").write_text(json.dumps({"checkpoint_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash differs"):
        phases.load_checkpoint(GatedAttentionPolicy(), "model.pt", 1)


def test_fixed_gating_bypasses_learned_gate():
    policy = GatedAttentionPolicy(d_model=32, num_heads=2, fixed_gate=.5)
    s = torch.rand(3, 2)
    nf = torch.rand(3, 8, 5)
    nm = torch.ones(3, 8, dtype=torch.bool)
    first, _, weights = policy(s, nf, nm)
    with torch.no_grad():
        for p in policy.gating.parameters():
            p.add_(100)
    second, _, _ = policy(s, nf, nm)
    assert torch.all(weights == .5)
    assert torch.allclose(first.probs, second.probs)


def test_random_placement_can_reproduce_when_local_cells_are_full():
    env = GridWorld(grid_size=5, initial_population=5, max_capacity=6,
                    offspring_placement="random", c_base=0, c_density=0, seed=2)
    env.reset(seed=2)
    env.grid.fill(-1)
    for i, pos in enumerate([(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]):
        env.positions[i] = pos
        env.grid[pos] = i
    env.energy[:5] = [100, 10, 10, 10, 10]
    _, _, _, info = env.step(np.zeros(6, dtype=int))
    assert info["births_this_step"] == 1
    assert env.active_count == 6
    assert env.energy[0] == 50
    assert env.parent_unique_ids[5] == env.agent_unique_ids[0]
    assert len(set(map(tuple, env.positions[env.mask]))) == 6


def test_starvation_switch_is_explicit_control():
    env = GridWorld(initial_population=1, max_capacity=1, initial_energy=.1,
                    c_base=1, enable_starvation=False, seed=3)
    env.reset(seed=3)
    env.step(np.zeros(1, dtype=int))
    assert env.active_count == 1
    assert env.energy[0] < 0


def test_cluster_regression_reports_training_units_and_intervals():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = rng.binomial(1, .4, size=200)
    result = fit_logistic_regression(x, y, ["Intercept", "x"], groups=np.repeat(np.arange(5), 40))
    assert result["num_clusters"] == 5
    assert np.array(result["ci95"]).shape == (2, 2)
    assert np.isfinite(result["std_errors"]).all()


def test_large_interaction_regression_converges_and_recovers_known_effect():
    # The submission's large likelihood previously stopped with precision loss.
    # A known synthetic model checks the fitted effect as well as termination.
    from scipy.special import expit
    rng = np.random.default_rng(7001)
    r = rng.uniform(0, 1, 30000)
    ratio = rng.choice([1., 1.5, 2.5, 3., 4.5], 30000)
    x = np.column_stack([np.ones(len(r)), r, ratio, r*ratio])
    expected = np.array([-2., .3, -.2, .4])
    y = rng.binomial(1, expit(x@expected))
    fitted = fit_logistic_regression(x, y, ['intercept','r','ratio','interaction'],
                                     groups=np.repeat(np.arange(5), 6000))
    assert fitted['converged']
    assert fitted['max_abs_mean_gradient'] < 1e-7
    assert fitted['coefficients'][3] == pytest.approx(.4, abs=.10)
    assert fitted['num_clusters'] == 5


def test_training_records_actual_budget_and_checkpoint_hash(tmp_path):
    path = tmp_path / "tiny.pt"
    train_model("mlp", total_timesteps=48, rollout_length=2,
                num_epochs=1, save_path=str(path), seed=9)
    record = json.loads(path.with_suffix('.json').read_text())
    assert record["environment_steps"] == 2
    assert record["active_agent_transitions"] == 48
    assert len(record["learning_curve"]) == 1
    assert len(record["checkpoint_sha256"]) == 64
    assert "environment/grid_world.py" in {p.replace('\\', '/') for p in record["source_sha256"]}
