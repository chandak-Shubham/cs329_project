"""Build the submission report from the complete study, with an integrity audit.

python evaluation/final_report.py --study-dir results/submission_study
No historical outcomes or manually entered performance numbers are substituted.
"""
import argparse
from itertools import product
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from environment.grid_world import GridWorld
from evaluation.uncertainty import mean_ci


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_checkpoints(root):
    checks = []
    for path in sorted((root/'checkpoints').glob('*.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        archive_bad = [name for name, value in meta['source_sha256'].items()
                       if not (root/'source'/name).exists() or digest(root/'source'/name) != value]
        current_bad = [name for name, value in meta['source_sha256'].items()
                       if not (ROOT/name).exists() or digest(ROOT/name) != value]
        checks.append(dict(checkpoint=path.stem, weight_hash_matches=digest(path.with_suffix('.pt')) == meta['checkpoint_sha256'],
                           archived_source_mismatches=archive_bad, current_source_mismatches=current_bad,
                           environment_steps=meta['environment_steps']))
    return dict(checkpoints=checks, passed=bool(checks) and all(
        row['weight_hash_matches'] and not row['archived_source_mismatches'] and not row['current_source_mismatches']
        for row in checks))


def immediate_reward_probe():
    """An actual matched environment intervention, without training or inference."""
    outcomes = {}
    for give in (False, True):
        env = GridWorld(grid_size=5, initial_population=2, max_capacity=2,
                        initial_energy=30, donation_cost=2, donation_benefit=5,
                        c_base=.6, c_density=.08, reward_mode='energy', seed=42)
        env.reset(seed=42)
        # Force the same adjacent state and stationary recipient in both worlds.
        env.positions[:2] = [[1, 1], [1, 2]]
        # GridWorld resolves occupancy from its explicit map; initialize it too.
        env.grid.fill(-1)
        env.grid[1, 1], env.grid[1, 2] = 0, 1
        env._update_cues()
        actions = np.zeros(env.N_max, dtype=int)
        actions[0] = 5 if give else 0
        _, rewards, _, _ = env.step(actions)
        outcomes['give' if give else 'stay'] = rewards[:2].tolist()
    outcomes['donor_reward_difference'] = outcomes['give'][0]-outcomes['stay'][0]
    outcomes['recipient_reward_difference'] = outcomes['give'][1]-outcomes['stay'][1]
    outcomes['interpretation'] = 'Holding the state and other action fixed, giving immediately costs the donor C and benefits the recipient B. This tests reward plumbing, not the long-term optimal action.'
    return outcomes


def paired_optimizer_tests(runs):
    """Exact sign-flip comparisons of independent seed means; Holm correction."""
    rows = []
    for condition in ('low_lr', 'high_entropy', 'long_training'):
        differences = []
        for seed in sorted({r['training_seed'] for r in runs}):
            values = {}
            for c in ('clean', condition):
                sample = [r['donations_per_resident_step'] for r in runs
                          if r['condition'] == c and r['training_seed'] == seed
                          and r['control'] == 'trained' and not r['attack']]
                if sample:
                    values[c] = float(np.mean(sample))
            if len(values) == 2:
                differences.append(values[condition]-values['clean'])
        if not differences:
            continue
        observed = abs(float(np.mean(differences)))
        null = [abs(float(np.mean(np.array(signs)*differences)))
                for signs in product((-1, 1), repeat=len(differences))]
        p = sum(v >= observed-1e-12 for v in null)/len(null)
        rows.append(dict(condition=condition, differences=differences, summary=mean_ci(differences),
                         p_sign_flip=p))
    floor = 0.
    for rank, row in enumerate(sorted(rows, key=lambda r:r['p_sign_flip'])):
        floor = max(floor, min(1., (len(rows)-rank)*row['p_sign_flip']))
        row['p_holm'] = floor
    return rows


def interval(row):
    if row['mean'] is None:
        return 'undefined'
    ci = row['ci95']
    return f"{row['mean']:.4f}" + (f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ' [CI unavailable]')


def validate_evaluation_coverage(analysis):
    expected = {(seed, seed+10000+replicate*1000) for seed in analysis['training_seeds']
                for replicate in range(analysis['evaluation_worlds_per_seed'])}
    for condition in analysis['budgets_by_condition']:
        rows = [r for r in analysis['runs'] if r['condition'] == condition
                and r['control'] == 'trained' and not r['attack']]
        keys = {(r['training_seed'],r['world_seed']) for r in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(f'Incomplete or duplicate trained evaluation worlds: {condition}')


def main(root):
    analysis = json.loads((root/'analysis.json').read_text(encoding='utf-8'))
    diagnostics = json.loads((root/'diagnostics.json').read_text(encoding='utf-8'))
    if diagnostics['analysis_sha256'] != digest(root/'analysis.json'):
        raise ValueError('Regenerate inheritance diagnostics before the final report')
    if analysis['not_run']:
        raise ValueError('The final report requires all declared conditions')
    from experiments.inheritance_study import CONDITIONS, checkpoint_matches
    if set(analysis['budgets_by_condition']) != set(CONDITIONS):
        raise ValueError('The complete study must include every declared condition')
    validate_evaluation_coverage(analysis)
    for condition in CONDITIONS:
        for seed in analysis['training_seeds']:
            meta = json.loads((root/'checkpoints'/f'{condition}_seed_{seed}.json').read_text(encoding='utf-8'))
            if not checkpoint_matches(meta, condition, seed, analysis['environment_steps_per_run']):
                raise ValueError(f'Checkpoint does not match the declared protocol: {condition}/{seed}')
    audit = audit_checkpoints(root)
    if not audit['passed']:
        raise ValueError('Checkpoint or source audit failed')
    objective = immediate_reward_probe()
    tests = paired_optimizer_tests(analysis['runs'])
    supplement = dict(analysis_sha256=digest(root/'analysis.json'), diagnostics_sha256=digest(root/'diagnostics.json'),
                      integrity=audit, reward_probe=objective, optimizer_tests=tests,
                      inference='Exact two-sided sign flips assume exchangeable paired effects under the null. Holm adjustment covers the three declared donation-rate comparisons. Five independent pairs limit power.')
    (root/'submission_diagnostics.json').write_text(json.dumps(supplement, indent=2, allow_nan=False), encoding='utf-8')
    runtime = dict(python=platform.python_version(), platform=platform.platform(),
                   packages={name:version(name) for name in ('numpy','torch','scipy','matplotlib','pillow','pytest')})
    (root/'run_environment.json').write_text(json.dumps(runtime,indent=2), encoding='utf-8')
    build = runtime['packages']['torch'].partition('+')[2]
    index = [f'--extra-index-url https://download.pytorch.org/whl/{build}'] if build == 'cpu' or build.startswith('cu') else []
    (root/'requirements-lock.txt').write_text('\n'.join(index+[f'{name}=={value}' for name,value in runtime['packages'].items()])+'\n', encoding='utf-8')
    summary = analysis['summary']
    clean = summary['clean/trained/clean']
    long = summary['long_training/trained/clean']
    base_budget = analysis['environment_steps_per_run']
    seeds = analysis['training_seeds']
    worlds = analysis['evaluation_worlds_per_seed']
    protocol_link = 'final_study_protocol.md'
    result_path = '../'+root.relative_to(ROOT).as_posix()
    lines = ['# Cooperation and cue reliability under inherited energy optimizing policies', '',
             'CS329 Multi Agent Systems | Group 21 | IIT Gandhinagar', '',
             'Vansh Barfiwala (24110380), Pratham Choksi (24110275), Shubham Chandak (24110335)', '',
             '## Abstract', '',
             'We investigate whether agents rewarded only for their own energy change learn sustained cooperation and recipient discrimination in a spatial donation game. Each founder owns an independent PPO network; offspring inherit its policy and a mutated chromosome. A survivor-based outer selection loop provides explicit competition between policy families. Gated attention combines a genetic marker with observed donation history.', '',
             f'The complete exploratory study contains {len(analysis["budgets_by_condition"])} conditions and {len(audit["checkpoints"])} runs grouped into five independent seed blocks. Each checkpoint is evaluated in {worlds} fresh worlds. Clean trained populations finish with {clean["final_residents"]["mean"]:.2f} residents on average and {clean["donations_per_resident_step"]["mean"]:.4f} donations per resident transition; the fourfold-budget condition finishes with {long["final_residents"]["mean"]:.2f} residents. These finite-budget observations do not establish emergent kin selection. The study tests model, optimizer and environmental alternatives and reports negative outcomes alongside positive controls.', '',
             '## Decision problem and research questions', '',
             'At each transition an agent chooses whether to stay, move to one of four cardinal cells, or donate to an observed neighbour. Giving costs C to the donor and adds B to the recipient. Agents observe local energy, relative positions, a corrected chromosome-similarity cue and recent donation history. They do not observe whether a neighbour is a mimic or the true corruption probability.', '',
             'The questions are whether individual energy optimization sustains cooperation; whether corruption changes full-policy use of genetic and behavioural cues; whether local reproduction increases spatial marker similarity relative to randomized placement; and whether recipient decisions are associated with the variables in Hamilton’s inequality. A diagnostic extension asks whether learning rate, exploration pressure or training duration explains the observed collapse.', '',
             '## Model and objective', '',
             'The partially observed Markov game uses an 18 × 18 torus, 24 founders, capacity 45, initial energy 30 and reproduction threshold 60. Default B=5, C=2, basal drain=0.6 and crowding cost=0.08 per living neighbour within Chebyshev radius 3. The horizon is 300 transitions. The energy increment before reproduction is', '',
             '`ΔE_i = −0.6 − 0.08 |Ω_i| − C 1[give_i] + B × incoming_donations_i`.', '',
             'Starvation terminates an individual at nonpositive energy. Reproduction divides parental energy equally, assigns a fresh persistent identity, mutates each chromosome bit with probability 0.05, and starts behavioural history at a neutral prior. Donation creates net energy B−C; reproduction conserves parent-plus-child energy. Crowding counts the full radius-3 neighbourhood; the actor observes and can target only its eight nearest neighbours, ordered by distance then storage slot. Donation recipients are fixed at decision time; crowding is computed after simultaneous movement. Movement conflicts and recyclable storage slots do not change individual identity.', '',
             'The genetic feature is normalized Hamming similarity of eight-bit tags, corrected against the mean over distinct living pairs: `r_tilde = max(0, (h_ij − h_bar) / max(1 − h_bar, epsilon))`. It is a marker-based proxy, not pedigree relatedness. The behavioural feature counts donations **given** during the last ten actions, not donations received.', '',
             'Each lineage maximizes its members’ individual discounted energy returns with gamma=0.99. An individual’s return terminates at its death; it does not continue through a child occupying the same slot. Policy sharing pools family samples but does not directly add recipients’ energy or future descendant fitness to a donor’s return. At completed horizons, surviving individuals are sampled to seed the next population, copying their genomes, networks and optimizer states. Total extinction supplies no selection winner. This explicit selection extension is distinct from the originally proposed within-episode mitosis.', '',
             '## Algorithm and implementation', '',
             'A contextual gate produces genetic and behavioural weights. Each neighbour encoder receives position, energy, both raw cues and their gated affinity. Two-head attention with 32-dimensional embeddings supplies context to movement and targeted-donation heads and a local critic. A fixed-gate ablation sets both weights to 0.5 while retaining the rest of the architecture; an MLP removes attention. Because raw cues also enter the encoder, gate weights alone cannot identify reliance.', '',
             'PPO uses 64-transition rollouts, three epochs, minibatches of 64, learning rate 0.0003, GAE lambda=0.95, clipping=0.2, value coefficient=0.5, entropy coefficient=0.01 and gradient norm limit=0.5. Samples are assigned by persistent family membership. Unrelated founder networks have independent parameter and optimizer storage. Terminal death masks block credit across slot reuse. Illegal actions and mimic donations are masked.', '',
             '## Experimental design', '',
             f'Training seeds are {seeds}. The common budget is {base_budget:,} environment transitions per run; long_training uses {4*base_budget:,}. The longer run is a budget comparison, not a convergence guarantee. Each seed receives {worlds} new evaluation worlds, using training_seed + 10000 + 1000 × replicate. World means are averaged within each training seed before inference. The sample size is five independent training runs, not fifteen independent learned policies.', '',
             'The design contains clean PPO; no outer selection; selection without PPO; fixed gate; MLP; cue bit-flip probabilities 0.25 and 0.5; noise plus stationary non-donating genetic mimics; fixed gate with the same attack; B/C of 1.5 and 4.5; mutation 0, 0.1 and 0.2; random offspring placement; the static-demography fallback; lower learning rate 0.00003; higher entropy coefficient 0.1; and fourfold training. Other than the named factor, settings are matched. This is a one-factor design, not a full factorial study.', '',
             'All conditions run regardless of the clean feasibility screen. The screen requires every clean evaluation world to survive and donate at least 0.1 times per resident transition during its final 100 transitions, with donations in at least half those transitions. Failing this screen does not erase later measurements; it prevents claims of successful sustained cooperative adaptation. All comparisons are exploratory.', '',
             'Matched clean rule controls include selfish movement, tag trust, a fixed threshold, public reputation and genuine dyadic Tit-for-tat. Unknown dyads start cooperatively; the latter remembers directed encounters by persistent identity. Untrained ancestor controls preserve the trained founder genomes and ancestry composition but restore initial neural weights. Selection-only runs separately test evolution without PPO. An untrained network has several donation actions among its available choices; its survival is not evidence that it has learned cooperation.', '',
             '## Results', '',
             '| Condition | Final residents | Donations per resident transition | Fraction of seeds with all evaluation worlds surviving |',
             '| --- | --- | --- | --- |']
    for name, row in summary.items():
        if '/trained/' in name or '/rule_' in name or name == 'clean/untrained_ancestors/clean':
            lines.append('| '+name+' | '+interval(row['final_residents'])+' | '+interval(row['donations_per_resident_step'])+' | '+interval(row['all_worlds_survive'])+' |')
    lines += ['', 'Brackets show exploratory 95% intervals. Final population and donation intervals use Student t across seed means; the all-worlds-survive event uses Wilson intervals across seeds. Small-sample t intervals can extend beyond physical bounds. Fallback survival is forced by disabling starvation and has no ecological meaning.', '',
              '### Optimizer and training budget controls', '',
              '| Contrast against clean | Paired donation-rate difference and 95% interval | Exact sign-flip p | Holm-adjusted p |',
              '| --- | --- | --- | --- |']
    for row in tests:
        lines.append(f'| {row["condition"]} | {interval(row["summary"])} | {row["p_sign_flip"]:.4f} | {row["p_holm"]:.4f} |')
    lines += ['', 'The three optimizer comparisons use paired seed means. With five pairs, exact two-sided sign-flip tests have very limited resolution; lack of significance does not establish equivalence. Higher entropy directly encourages diverse actions and can preserve giving without demonstrating learned recipient discrimination. Lower learning rate can delay collapse without resolving it. Longer training tests the duration explanation only over the stated budgets.', '',
              '### Cue reliability and spatial assortment', '',
              '| Condition | Genetic cue sensitivity | Behavioural cue sensitivity | Genetic gate weight |',
              '| --- | --- | --- | --- |']
    for name in ('clean','noise025','noise050','noise_mimics','fixed_noise_mimics'):
        row = diagnostics['common_input_summary'][name]
        lines.append('| '+name+' | '+' | '.join(interval(row[k]) for k in ('genetic_tv','behavior_tv','genetic_weight'))+' |')
    lines += ['', 'Sensitivity is total variation in the full action distribution after permuting one cue on identical synthetic inputs. This covers raw-cue paths, but the intervention can produce unrealistic observations. Neither it nor a change in gate weights alone proves adaptive trust.', '',
              '| Paired cue contrast | Mean difference and exploratory 95% interval |', '| --- | --- |']
    for name in ('noise025_minus_clean','noise050_minus_clean','noise_mimics_minus_noise050'):
        for metric in ('genetic_tv','behavior_tv'):
            row = diagnostics['cue_paired_contrasts'][f'{name}/{metric}']
            lines.append(f'| {name}: {metric} | {interval(row)} |')
    lines += ['', '| Attacked policy | Post-invasion donation share to mimics | Donation-bearing training seeds | Total observed post-invasion donations |',
              '| --- | --- | --- | --- |']
    for condition in ('noise_mimics','fixed_noise_mimics'):
        row = summary[f'{condition}/trained/attack']['mimic_donation_share']
        total = sum(r['post_invasion_donations'] for r in analysis['runs'] if r['condition'] == condition
                    and r['control'] == 'trained' and r['attack'])
        lines.append(f'| {condition} | {interval(row)} | {row["n"]} | {total} |')
    lines += ['', 'Shares are averaged within donation-bearing worlds and then training seeds; the total exposure column is descriptive, not an independent sample count. Undefined shares never count as successful defense. A policy that stops donating to everybody has not demonstrated recipient discrimination.', '',
              '| Spatial contrast | Paired difference and 95% interval |', '| --- | --- |']
    for metric in ('adjacent_hamming_similarity','opportunity_adjusted_tag_preference'):
        row = diagnostics['ecological_paired_contrasts'][f'clean_minus_random_placement/{metric}']
        lines.append(f'| Local minus random offspring placement: {metric} | {interval(row)} |')
    lines += ['', 'Adjacent similarity is averaged over the observed trajectory, including transients. Extinct populations have no equilibrium assortment. Opportunity-adjusted preference compares actual recipients with available neighbours, conditional on giving. Neither statistic identifies pedigree kin selection.', '',
              '### Hamilton association analysis', '',
              f'Regression status: **{diagnostics["hamilton_regression"]["status"]}**. The logistic model uses corrected tag similarity, donation history and B/C from independently trained conditions. It describes recipient decisions during observed trajectories, including trajectories ending in extinction. B/C is a training condition, not an input that the actor reads. The regression is an association test, not a causal validation of Hamilton’s rule.', '']
    reg = diagnostics['hamilton_regression']
    if 'coefficients' in reg:
        lines += ['| Predictor | Coefficient | Cluster 95% interval |', '| --- | --- | --- |']
        for name, coefficient, ci in zip(reg['features'],reg['coefficients'],reg['ci95']):
            lines.append(f'| {name} | {coefficient:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |')
        lines += ['', f'McFadden pseudo-R²={reg["pseudo_r2"]:.4f}; {reg["num_samples"]:,} sampled recipient decisions, clustered into {reg["num_clusters"]} training-seed units. Small-cluster inference is exploratory. Optimizer status: {reg["optimizer_message"]}.', '']
    else:
        lines += [reg.get('reason','No estimable fit.'), '']
    lines += ['## Failure analysis and interpretation', '',
              f'A matched one-transition environment intervention changes the donor reward by {objective["donor_reward_difference"]:.4f} and the recipient reward by {objective["recipient_reward_difference"]:.4f} when stay is replaced with give. This confirms the immediate private cost and social benefit. It does not determine which action has the larger long-term return.', '',
              'There is also an objective-level distinction. For a trajectory without reproduction, summing energy increments gives `sum_t gamma^t ΔE_t = −E_0 + (1−gamma) sum_{t=1}^{T−1} gamma^(t−1) E_t + gamma^(T−1) E_T`. At gamma=1 this reduces to terminal minus initial energy. At gamma=0.99 it additionally values earlier stored energy. This expression contains no descendant-fitness term. Reproduction adds bookkeeping terms because the implemented reward is computed before energy splitting, so the simple identity is deliberately restricted to trajectories without reproduction. Reciprocity can still raise future individual energy; the identity does not prove that donation is suboptimal.', '',
              'In a world without mimics, total energy change before demographic bookkeeping is `(B−C) D − 0.6 N − 0.08 sum_i |Ω_i|`, where D is the number of donations and N is the acting population. At the default B−C=3, fewer than 0.2 donations per resident transition cannot even replace basal energy loss; crowding raises the required rate. This explains the ecological consequence of low giving, but does not explain why the optimizer learns it.', '',
              'There are two different processes: within-lifetime PPO rewards immediate energy gains and delayed individual benefits, while outer selection favours policies with surviving carriers at an episode boundary. Donation’s direct cost appears in the donor’s gradient, whereas another individual’s benefit is not directly assigned to that action. Recipient reciprocity could still make giving individually beneficial, but it has to be discovered. Terminating individual returns at death further separates the objective from continuing lineage fitness. These are plausible mechanisms; the experiments do not isolate a unique causal explanation.', '',
              'The control comparisons distinguish ecological feasibility, the effect of learning, the effect of selection and sensitivity to two optimizer settings and duration. Surviving hand-written rules demonstrate that the tested ecology can sustain productive donations. They do not establish that the PPO objective must find those strategies. Selection-only survival likewise cannot demonstrate genetic discrimination. Earlier shared-policy demographic and descendant-reward studies use different objectives and remain separate historical evidence.', '',
              '## Limitations and scope', '',
              'The study has five training seeds, finite budgets, a single base ecology and one-factor sweeps. The exploratory comparisons are not a pre-registered confirmatory test. Corruption and mimicry experiments cannot establish successful cooperative adaptation when giving or survival collapses. The marker is an eight-bit proxy that may be copied or become homogeneous. Periodic restocking adds energy at episode boundaries and is an explicit model extension. Neural family policies are frozen during evaluation. Static-demography fallback results cannot establish ecological survival. The browser simulator executes an earlier shared-policy actor and is an illustration rather than a replay of the multi-policy experiment.', '',
              'The project therefore answers what occurred under the declared model and finite computational budgets. It does not prove that energy-only learning can never support cooperation, or that Hamilton’s rule is false. A positive kin-selection claim would require sustained cooperation, justified pedigree or fitness measurements, controlled recipient discrimination and robust independent replication.', '',
              '## Reproducibility and submission evidence', '',
              f'Every one of the {len(audit["checkpoints"])} new checkpoints has a verified weight hash, exact configuration, training curve, versions and source hashes. The actual training source is archived under `{result_path}/source`. Current and archived sources match the new checkpoints. Older artifacts were preserved with their original metadata; their hashes were not rewritten to claim retraining.', '',
              'Run from the repository root:', '', '```powershell',
              f'python experiments/inheritance_study.py --complete --steps {base_budget} --workers 3 --worlds {worlds} --out {root.relative_to(ROOT).as_posix()}',
              f'python evaluation/inheritance_report.py --study-dir {root.relative_to(ROOT).as_posix()}',
              f'python evaluation/final_report.py --study-dir {root.relative_to(ROOT).as_posix()}',
              'python -m pytest -q', 'node --test tests/web_simulation.test.cjs',
              'python visualizer/export_web_data.py --check', '```', '',
              'Matching checkpoints are reused only after protocol and source verification. Changed settings or source require a separate output directory. Exact library versions appear in each checkpoint sidecar. The artifact manifest covers analysis, source, checkpoints, diagnostics, report-generation sources and figures.', '',
              f'[Full results and uncertainty]({result_path}/results.md) · [Raw evaluations]({result_path}/analysis.json) · [Diagnostic data]({result_path}/submission_diagnostics.json) · [Experimental protocol]({protocol_link})', '',
              f'![Training curves]({result_path}/figures/learning_curves.png)', '',
              '## Alignment with the project guidelines', '',
              '| Guideline | Submission evidence |', '| --- | --- |',
              '| Clear computational question | Decision problem, private energy objective and stated research questions |',
              '| Model and algorithm | Markov game, energy equation, inheritance, selection and gated-attention PPO |',
              '| Implementation | Custom environment, dynamic identity handling, neural and scripted strategies |',
              '| Baselines | MLP, fixed gate, dyadic Tit-for-tat, reputation, selfish and untrained controls |',
              '| Meaningful experimental variation | Noise, attacks, B/C, mutation, placement, selection, learning rate, entropy and budget |',
              '| Quantitative evaluation | Independent seed summaries, intervals, paired comparisons, cue probes and association analysis |',
              '| Results and limitations | Negative and positive controls, objective analysis and explicit limits on interpretation |', '',
              'The guidelines do not require a positive result. This report presents the implemented study and the evidence it supports; the instructor determines marks.', '',
              '## Contributions and references', '',
              'The ideation assigned environment and demographic modeling to Vansh Barfiwala; gating, cues and attention to Pratham Choksi; PPO and evaluation to Shubham Chandak. These are the team’s declared responsibilities, not an independent authorship audit.', '',
              'Hamilton (1964), The genetical evolution of social behaviour, parts I and II; Leibo et al. (2017), Multi-agent reinforcement learning in sequential social dilemmas; Hughes et al. (2018), Inequity aversion improves cooperation in intertemporal social dilemmas; Jaques et al. (2019), Social influence as intrinsic motivation for multi-agent deep reinforcement learning; Lupu and Precup (2020), Gifting in multi-agent reinforcement learning. Bibliographic background follows the supplied ideation; no novelty or literature-completeness claim is made.', '']
    # The hand-written submission is docs/final_report.md. This generated summary
    # of the energy-only study stays beside its results and never overwrites it.
    report = root/'final_report.md'
    text = '\n'.join(lines)
    # Freeze report and analysis sources separately from training provenance.
    for path in (Path(__file__), ROOT/'evaluation/inheritance_report.py', ROOT/'evaluation/analysis.py'):
        target = root/'report_source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    report.write_text(text.replace(result_path, '.')
                      .replace('](final_study_protocol.md)', '](../../docs/final_report.md)'), encoding='utf-8')
    manifest = {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*')
                if p.is_file() and p.name != 'manifest.json' and p.suffix not in ('.log','.pyc')}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    print(f'Wrote {report}; verified {len(audit["checkpoints"])} checkpoints.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=Path('results/submission_study'))
    main(parser.parse_args().study_dir.resolve())
