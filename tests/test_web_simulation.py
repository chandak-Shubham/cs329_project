"""Cross-language checks against the repository's actual Python environment/policy."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from environment.grid_world import GridWorld
from models.mlp_policy import SimpleMLPPolicy
from models.rule_based import FixedWeightPolicy, KinOnlyPolicy, ReciprocalPolicy
from visualizer.export_web_data import build_payload

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for browser-engine parity tests")


def run_js(body, fixture=None):
    driver = r"""
const fs = require('node:fs'), vm = require('node:vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
vm.runInThisContext(html.match(/<script id="simulation-engine">([\s\S]*?)<\/script>/)[1]);
const embedded = JSON.parse(html.match(/<script id="research-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = new Function('fixture', 'embedded', payload.body)(payload.fixture, embedded);
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", driver, str(ROOT / "index.html")],
        input=json.dumps({"body": body, "fixture": fixture}), text=True,
        capture_output=True, check=True, timeout=30,
    )
    return json.loads(completed.stdout)


def fixture_world():
    world = GridWorld(grid_size=10, max_capacity=8, initial_population=4,
                      c_base=0.5, c_density=0.05, seed=9)
    world.reset(seed=9)
    world.grid.fill(-1)
    positions = [[0, 0], [9, 0], [2, 0], [8, 2]]
    genomes = [0, 0, 255, 127]
    for i, (x, y) in enumerate(positions):
        world.positions[i] = [x, y]
        world.grid[x, y] = i
        world.chromosomes[i] = [int(b) for b in f"{genomes[i]:08b}"]
    world._update_cues()
    return world, {"positions": positions, "genomes": genomes}


SETUP = """
const {SimulationEngine, NeuralActor} = Kinship;
// Set all fixture ecology explicitly so changing UI defaults cannot change parity.
const world = new SimulationEngine({L:10,maxCapacity:8,initialPop:0,B:6,C:2,cBase:0.5,cDensity:0.05,seed:9,enforceHamilton:false});
fixture.positions.forEach(([x,y], i) => world.spawn(world.cell(x,y),fixture.genomes[i]));
world.updateCues();
"""


def test_web_observations_and_rule_actions_match_python():
    world, fixture = fixture_world()
    obs = world._get_observations()
    flat, _ = SimpleMLPPolicy(hidden_dim=64).extract_flat_obs(obs)
    actual = run_js(SETUP + """
const inputs = world.agents.map(a => world.observation(a,world.neighbors(a).slice(0,8)).input);
const actions = {};
for (const strategy of ['reputation','fixed','kin']) {
  world.strategy = strategy; actions[strategy] = world.agents.map(a => world.chooseAction(a));
}
return {inputs,actions};
""", fixture)
    np.testing.assert_allclose(actual["inputs"], flat[:4].numpy(), atol=1e-7)
    for name, policy in [("reputation", ReciprocalPolicy()), ("fixed", FixedWeightPolicy(threshold=0.35)), ("kin", KinOnlyPolicy(threshold=0.3))]:
        assert actual["actions"][name] == policy.act(obs)[:4].tolist()


def test_web_energy_history_and_movement_match_python_transitions():
    world, fixture = fixture_world()
    action_steps = [[5, 0, 4, 1], [0, 5, 0, 0], [1, 2, 0, 0], [0, 0, 5, 5], [0, 0, 0, 0]]
    fixture["actions"] = action_steps
    expected = []
    for actions in action_steps:
        all_actions = np.zeros(world.N_max, dtype=int)
        all_actions[:4] = actions
        _, _, _, info = world.step(all_actions)
        expected.append({
            "positions": world.positions[:4].tolist(),
            "energy": world.energy[:4].tolist(),
            "history": world.history.get_all_scores(world.mask)[:4].tolist(),
            "donations": info["donations_count"],
        })
    actual = run_js(SETUP + """
return fixture.actions.map(actions => {
  world.step(actions);
  return {positions:world.agents.map(a=>[a.x,a.y]),energy:world.agents.map(a=>a.energy),history:world.agents.map(a=>world.reputation(a)),donations:world.lastDonations};
});
""", fixture)
    for python_step, js_step in zip(expected, actual):
        assert js_step["positions"] == python_step["positions"]
        assert js_step["donations"] == python_step["donations"]
        np.testing.assert_allclose(js_step["energy"], python_step["energy"], atol=1e-5)
        np.testing.assert_allclose(js_step["history"], python_step["history"], atol=1e-7)


def test_web_neural_logits_match_checkpoint_pytorch():
    policy = SimpleMLPPolicy(hidden_dim=64)
    policy.load_state_dict(torch.load(ROOT / "results/checkpoints/simple_mlp_seed_42.pt", map_location="cpu", weights_only=True))
    policy.eval()
    inputs = np.random.default_rng(42).normal(size=(20, 43)).astype(np.float32)
    with torch.no_grad():
        expected = policy.actor_head(policy.encoder(torch.from_numpy(inputs))).numpy()
    actual = run_js("const actor=new Kinship.NeuralActor(embedded.neural); return fixture.map(x=>actor.logits(x));", inputs.tolist())
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_embedded_web_artifacts_match_verified_research_sources():
    assert run_js("return embedded;") == build_payload()
