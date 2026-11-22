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
from models.attention_policy import GatedAttentionPolicy
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
const result = new Function('fixture', 'embedded', 'require', payload.body)(payload.fixture, embedded, require);
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", driver, str(ROOT / "index.html")],
        input=json.dumps({"body": body, "fixture": fixture}), text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=600,
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


LOAD = """
const fs = require('node:fs');
const load = path => { const b = fs.readFileSync(path); return Kinship.loadCheckpoint(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), embedded.models.layout); };
"""


def test_page_reads_every_family_network_and_genome_exactly_as_pytorch():
    """The page's own .pt reader returns the tensors torch.load returns."""
    models = build_payload()['models']
    entry = models['checkpoints'][models['default']]
    saved = torch.load(ROOT/entry['path'], map_location='cpu', weights_only=True)
    policy = GatedAttentionPolicy(d_model=32, num_heads=2, contextual_gate=True)
    inputs = np.random.default_rng(42).normal(size=(20, 43)).astype(np.float32)
    counts = np.arange(20) % 9  # zero through eight neighbours, including padding
    inputs[:, 42] = counts / 8
    masks = torch.as_tensor(np.arange(8)[None, :] < counts[:, None])
    actual = run_js(LOAD + f"""
const m = load({json.dumps(str(ROOT/entry['path']))});
const probs = actor => fixture.map(x => {{const l = actor.logits(x), mx = Math.max(...l), e = l.map(v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s);}});
return {{genomes: m.genomes, n: m.policies.length, probs: [0, 7, 23].map(i => probs(m.policies[i]))}};""", inputs.tolist())
    genomes = [int(''.join(str(int(b)) for b in row), 2) for row in saved['founder_genomes'].tolist()]
    assert actual['genomes'] == genomes and actual['n'] == 24
    for k, i in enumerate([0, 7, 23]):
        policy.load_state_dict(saved['policies'][i]); policy.eval()
        with torch.no_grad():
            expected = policy(torch.from_numpy(inputs[:, :2]), torch.from_numpy(inputs[:, 2:42].reshape(-1, 8, 5)), masks)[0].probs.numpy()
        np.testing.assert_allclose(actual['probs'][k], expected, atol=1e-6, rtol=1e-5)


def test_embedded_web_artifacts_match_verified_research_sources():
    assert run_js("return embedded;") == build_payload()


def test_browser_engine_reproduces_the_recorded_study_outcome():
    """Running the saved B/C = 4 checkpoints in the page's engine (one world per seed)
    gives the study's result: inclusive populations mostly survive, energy-only
    populations never do, and per-seed donation rates track the Python record."""
    models = build_payload()['models']
    rows = run_js(LOAD + """
const out = {};
for (const condition of ['inclusive_bc400', 'energy_bc400']) out[condition] = embedded.models.seeds.map(seed => {
  const m = load(fixture.root + '/' + embedded.models.checkpoints[`${condition}_seed_${seed}`].path);
  return Kinship.runEpisode({policies: m.policies, founderGenomes: m.genomes, B: 8, C: 2, seed: seed + 10000});
});
return out;""", {'root': str(ROOT)})
    recorded = {c: [models['checkpoints'][f'{c}_seed_{s}']['recorded'][0] for s in models['seeds']] for c in rows}
    assert sum(r['survival'] for r in rows['energy_bc400']) == 0
    assert sum(r['survival'] for r in rows['inclusive_bc400']) >= 7
    browser = [r['donationsPerResidentStep'] for r in rows['inclusive_bc400']]
    python = [r['donationsPerResidentStep'] for r in recorded['inclusive_bc400']]
    assert np.corrcoef(browser, python)[0, 1] > .9


def test_mimic_reserve_lets_mimics_enter_a_full_population():
    out = run_js("""
const w = new Kinship.SimulationEngine({strategy:'selfish', maxCapacity: 51, mimicReserve: 6, initialPop: 45, seed: 3});
const full = w.metrics().residents, injected = w.injectMimics(6);
for (const a of w.agents) a.energy = 100;
const before = w.metrics().residents; w.step(Array(51).fill(0));
return {full, injected, residentsAfterBirthAttempts: w.metrics().residents, before};""")
    assert out['full'] == 45 and out['injected'] == 6
    assert out['residentsAfterBirthAttempts'] <= 45
