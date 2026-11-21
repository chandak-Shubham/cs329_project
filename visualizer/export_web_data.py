"""Refresh the standalone demo's embedded research data; no runtime dependencies.

Run after regenerating research artifacts: python visualizer/export_web_data.py
Use --check to verify that the committed HTML matches the checkpoint and results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from html import escape

import torch

ROOT = Path(__file__).resolve().parents[1]


def interval(row, percent=False):
    if row['mean'] is None:
        return 'Not estimable (n=0)'
    fmt = '.1%' if percent else '.4f'
    ci = row['ci95']
    estimate = format(row['mean'], fmt)
    limits = 'CI unavailable' if ci is None else ', '.join(format(v, fmt) for v in ci)
    return f'{estimate} [{limits}]; n={row["n"]}'


def results_html(data):
    def table(headers, rows, caption):
        return ('<div class="research-table"><table><caption>'+escape(caption)+'</caption><thead><tr>'+''.join('<th scope="col">'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    def result_rows(summary):
        rows = []
        for name, row in summary.items():
            share = row['mimic_donation_share']
            text = interval(share, True)
            if 0 < share['n'] < 3:
                text = f'Insufficient donation-bearing runs (n={share["n"]}); no finding'
            rows.append([name.replace('_', ' '), interval(row['resident_survival'], True),
                         interval(row['final_residents']), interval(row['donations_per_resident_step']), text])
        return rows
    headers = ['Condition / evaluation', 'Resident survival [95% CI]', 'Final residents [95% CI]', 'Donations / resident-step [95% CI]', 'Post-invasion donations to mimics']
    chunks = ['<section id="research-results" aria-labelledby="research-heading"><h2 id="research-heading">Research results</h2>']
    h = data.get('hamilton')
    if h:
        count = lambda r: f"{round(r['mean']*r['n'])}/{r['n']}"
        chunks += [
            '<h3>Final study: does learning need kinship?</h3>',
            f'<p><strong>Main finding.</strong> With the purely individual energy reward, PPO <em>unlearns</em> cooperation and every population dies out. '
            f'Adding Hamilton\'s inclusive-fitness term <em>rB</em> to the donor\'s reward preserves cooperation, and learned giving follows Hamilton\'s threshold <em>rB &gt; C</em>. '
            f'{h["conditions"]} conditions, {h["seeds"]} independent training seeds each, {h["steps"]:,} transitions per run, {h["worlds"]} fresh evaluation worlds per seed. '
            f'Read the <a href="docs/final_report.md">final report</a>.</p>',
            table(['B/C', 'Inclusive PPO: donations / step [95% CI]', 'Energy-only PPO: donations / step [95% CI]',
                   'Untrained networks', 'Seeds surviving (inclusive)', 'Seeds surviving (energy-only)'],
                  [[row['bc'], interval(row['inclusive']), interval(row['energy']), f"{row['untrained']['mean']:.3f}",
                    count(row['inclusiveSurvival']), count(row['energySurvival'])] for row in h['sweep']],
                  'Hamilton study: trained populations at five benefit/cost ratios'),
            '<p>Inclusive learners gave more than energy-only learners in '
            f'{h["stacked"]["donations_per_resident_step"]["positive"]}/{h["stacked"]["donations_per_resident_step"]["pairs"]} independent paired runs '
            f'(exact sign test p = {h["stacked"]["donations_per_resident_step"]["sign_test_p"]:.1e}). Survival at B/C &ge; 4: '
            f'{h["fisher"]["B/C >= 4"]["inclusive"]} versus {h["fisher"]["B/C >= 4"]["energy"]} (Fisher p = {h["fisher"]["B/C >= 4"]["fisher_p"]:.1e}). '
            'Below the threshold (B/C &le; 1.5) inclusive learners abandon giving too; at B/C = 6 they give more than untrained networks in every seed.</p>',
            '<p class="research-caveat">Limits: learners prefer related recipients but are far less selective than the scripted Hamilton rule; when the genetic cue is corrupted its weight falls, '
            'but they do not switch to behavioural history, and they keep feeding genetic mimics (a public-reputation rule gives them nothing). Spatial assortment was not detected because selection makes populations near-clonal.</p>',
            '<div class="research-figures"><figure><img src="results/hamilton_study/figures/bc_sweep.png" width="1800" height="1200" loading="lazy" alt="Survival, donations, lifetime and Hamilton gap across five benefit-to-cost ratios for inclusive-fitness PPO, energy-only PPO and untrained networks"><figcaption>Cooperation is learnable only with inclusive credit, and only above Hamilton\'s threshold.</figcaption></figure>'
            '<figure><img src="results/hamilton_study/figures/learning_curves.png" width="1800" height="600" loading="lazy" alt="Population size during training for five seeds under energy-only and inclusive-fitness rewards"><figcaption>Energy-only populations die out during training; inclusive populations persist.</figcaption></figure>'
            '<figure><img src="results/hamilton_study/figures/cue_reliance.png" width="900" height="600" loading="lazy" alt="Weight of genetic and behavioural cues in recipient choice at three cue-noise levels"><figcaption>Reliance on the genetic cue falls as it becomes noisier.</figcaption></figure></div>',
            '<p><a href="docs/final_report.md">Final report</a> · <a href="results/hamilton_study/results.md">All conditions and statistics</a> · <a href="results/hamilton_study/analysis.json">Raw evaluations</a></p>',
            '<h3>Earlier studies</h3>']
    chunks += ['<p>The studies below preceded the final study and use the individual energy reward or other objectives. They are kept as supporting evidence. The live simulation above runs one of these earlier shared-policy checkpoints.</p>',
              '<p>The earlier revised and lineage-fitness studies use five training seeds and one evaluation world per seed. Their survival intervals are Wilson intervals; other intervals use Student t across seed-level runs.</p>',
              '<h3>Revised study: 8,192 environment steps per checkpoint</h3>',
              table(headers, result_rows(data['revised']['summary']), 'Revised study: trained policies and untrained network controls'),
              '<p class="research-caveat">Fallback survival is forced by disabling starvation; it is not ecological success. Mimic share is undefined without post-invasion donations. Estimates based on one or two donation-bearing runs, including the historical interval spanning negative percentages to above 100%, are not findings. Full raw estimates remain available below.</p>',
              '<h3>Cue sensitivity on identical synthetic inputs</h3>',
              '<p class="research-caveat">These policies barely donate. Sensitivity of their action distributions cannot establish a shift from genetic to behavioural cooperation. Raw cues bypass the gate; gate weights alone are not reliance. Permutations are exploratory and may create unrealistic observations.</p>',
              table(['Training condition','Genetic cue TV [95% CI]','Behaviour cue TV [95% CI]'],
                    [[name.replace('_',' '),interval(row['genetic_tv']),interval(row['behavior_tv'])] for name,row in data['revised']['cueSensitivity'].items()], 'Total variation after permuting one cue'),
              '<div class="research-figures"><figure><img src="results/revised_study/figures/learning_curves.png" width="1600" height="1120" loading="lazy" alt="Training donation rates and population over environment transitions, with seed means and pointwise 95% intervals"><figcaption>Training curves: cooperation collapses as training proceeds.</figcaption></figure><figure><img src="results/revised_study/figures/cue_sensitivity.png" width="1600" height="800" loading="lazy" alt="Genetic and behavioural cue permutation sensitivity across training conditions"><figcaption>Cue sensitivity measures action changes, not successful cooperative adaptation.</figcaption></figure></div>']
    if data.get('submission'):
        study = data['submission']
        selected = {name: row for name,row in study['summary'].items()
                    if name in ('clean/trained/clean','clean/untrained_ancestors/clean',
                                'evolution_only/trained/clean','low_lr/trained/clean',
                                'high_entropy/trained/clean','long_training/trained/clean')}
        at = next(i for i, c in enumerate(chunks) if c.startswith('<p>The studies below preceded')) + 1
        chunks[at:at] = [
            '<h3>Complete submission study</h3>',
            f'<p>{study["conditions"]} conditions and {study["checkpoints"]} experiment runs; five training seeds and three fresh evaluation worlds per checkpoint. Matched conditions use {study["baseBudget"]:,} transitions; the longer-training control uses {4*study["baseBudget"]:,}. All declared comparisons were executed regardless of clean feasibility.</p>',
            table(['Condition','Final residents [95% CI]','Donations / resident-step [95% CI]',
                   'Seeds with all worlds surviving [Wilson 95% CI]'],
                  [[name.replace('_',' '),interval(row['final_residents']),
                    interval(row['donations_per_resident_step']),interval(row['all_worlds_survive'],True)]
                   for name,row in selected.items()],
                  'Complete inherited-policy study: selected controls; all conditions are linked below'),
            '<p>These are exploratory finite-budget results. Full-policy cue sensitivity, spatial comparisons and a transient Hamilton association analysis are reported separately from claims of sustained cooperation. The live actor above remains the earlier shared-policy demonstration.</p>',
            '<p><a href="docs/final_report.md">Final submission report</a> · <a href="results/submission_study/results.md">All experimental conditions</a> · <a href="results/submission_study/analysis.json">Raw evaluations</a> · <a href="docs/final_report.md">Protocol and reproduction commands</a> · <a href="results/submission_study/submission_diagnostics.json">Source integrity and optimizer comparisons</a></p>']
    if data['lineage']:
        lineage = data['lineage']
        passed = lineage['feasibility']['passed']
        chunks += ['<h3>New lineage-fitness follow-up</h3>',
                   '<p>Reward: one unit for own survival plus one per living child or grandchild. No chromosome similarity, Hamilton threshold, or donation bonus enters the reward. This explicitly changes the fitness objective; neural weights remain shared.</p>',
                   '<p>'+('The clean feasibility screen passed; conditional follow-ups were evaluated. This does not establish kin selection.' if passed else 'The clean feasibility screen failed. Noise and mimic follow-ups were deliberately not run; the failed clean result is retained.')+'</p>',
                   table(headers,result_rows(lineage['summary']),'Lineage-fitness study: saved evaluation outcomes'),
                   '<div class="research-figures"><figure><img src="results/lineage_study/figures/learning_curves.png" width="1600" height="1120" loading="lazy" alt="Lineage-fitness training donation and population curves across five seeds"><figcaption>Lineage-fitness training, with seed uncertainty.</figcaption></figure><figure><img src="results/lineage_study/figures/cue_sensitivity.png" width="1600" height="800" loading="lazy" alt="Lineage-fitness genetic and behavioural cue sensitivity on identical inputs"><figcaption>Exploratory lineage cue sensitivity; not evidence of kin-directed cooperation.</figcaption></figure></div>',
                   '<p><a href="results/lineage_study/results.md">Lineage results and uncertainty</a> · <a href="results/lineage_study/analysis.json">Raw lineage analysis</a></p>']
    if data.get('inheritance'):
        study = data['inheritance']
        headers_new = ['Condition / control / evaluation', 'Final residents [95% CI]',
                       'Donations / resident-step [95% CI]', 'Mean fraction of worlds surviving [95% CI]',
                       'Post-invasion donations to mimics']
        rows = [[name.replace('_',' '),interval(row['final_residents']),
                 interval(row['donations_per_resident_step']),interval(row['resident_survival'],True),
                 interval(row['mimic_donation_share'],True)] for name,row in study['summary'].items()]
        chunks += ['<h3>Inherited neural policies with the original energy reward</h3>',
                   '<p>Each founder has a separate network and optimizer. Children inherit their family policy; periodic restocking samples surviving residents. PPO still receives only individual energy change. This is a separate multi-policy experiment, not the single-policy simulator above.</p>',
                   f'<p>{study["trainingSeeds"]} training seeds; {study["worldsPerSeed"]} evaluation worlds per seed; {study["steps"]:,} environment transitions per training run. Worlds are averaged within seeds before exploratory t intervals. Hand-written rules have no training budget.</p>',
                   '<p>'+('Clean cooperation feasibility passed; this alone does not establish kin selection.' if study['feasibility']['passed'] else 'Clean cooperation feasibility failed. The new inheritance mechanism does not establish the proposed kin-selection result.')+'</p>',
                   table(headers_new,rows,'Inherited-policy experiment and matched controls'),
                   '<p class="research-caveat">Untrained ancestors use the same evolved founder genomes and ancestry composition with initial network weights. Evolution-only is the separate control for selection without PPO. Fallback survival is forced; undefined mimic shares are not successful defense. Small-sample t intervals may exceed physical bounds.</p>',
                   '<p>Conditions not run: '+escape(', '.join(study['notRun']))+'.</p>',
                   '<p><a href="results/inheritance_study/results.md">Inherited-policy results</a> · <a href="results/inheritance_study/analysis.json">Raw evaluations</a> · <a href="docs/final_report.md">Inheritance and selection protocol</a></p>']
        if study['hasDiagnostics']:
            chunks += ['<div class="research-figures"><figure><img src="results/inheritance_study/figures/learning_curves.png" width="1600" height="1120" loading="lazy" alt="Training trajectories for independent inherited policies and mechanism controls"><figcaption>Inherited-policy training and controls.</figcaption></figure><figure><img src="results/inheritance_study/figures/cue_sensitivity.png" width="1600" height="800" loading="lazy" alt="Full-policy genetic and behavioural cue sensitivity for inherited networks"><figcaption>Descriptive input sensitivity, not successful cooperative adaptation.</figcaption></figure></div>',
                       '<p><a href="results/inheritance_study/diagnostics.md">Cue diagnostics and paired seed comparisons</a></p>']
    regression = data['regression']
    chunks += ['<h3>Historical benchmark: 1,024 training steps</h3>',
               '<p>These use the earlier stress ecology and are not matched comparisons with the revised study. Rule-based thresholds are hand-written baselines. Mimic extinction with resident survival is not evidence of discrimination.</p>',
               table(['Model / baseline','Resident survival','Final population','Episode duration','Performed attacks'],
                     [[row['model'],f"{row['survival']:.1f}%",f"{row['population']:.1f}",f"{row['duration']:.1f}",row['attacks']] for row in data['benchmarks']], 'Historical benchmark point estimates; full intervals are linked in the source artifacts'),
               '<h3>Historical regression: no evidence of learned kin preference</h3>',
               f'<p>McFadden pseudo-R²: {regression["pseudo_r2"]:.4f}. {regression["num_clusters"]} training-seed clusters; {regression["num_samples"]:,} dependent recipient observations. Unstandardized coefficients; small-cluster, exploratory inference. This analysis does not describe the new lineage checkpoints.</p>',
               table(['Predictor','Coefficient','95% CI','Exploratory p-value'],
                     [[name,f'{b:.4f}',f'[{ci[0]:.4f}, {ci[1]:.4f}]',f'{p:.4f}'] for name,b,ci,p in zip(regression['features'],regression['coefficients'],regression['ci95'],regression['p_values'])], 'Training-seed cluster regression'),
               '<h3>Interpretation and reproducibility</h3><p>Donation adds net energy B−C; mitosis conserves parent-plus-child energy. Behavioural history counts donations given. Mimics cannot donate and are excluded from PPO. The simulator and earlier studies share one resident policy. The separately labelled inherited-policy study uses family networks and explicit selection. Rule baselines can all reach capacity: compare donation frequency and exploitation, not the population cap alone.</p>',
               '<p><a href="results/revised_study/results.md">Revised results table</a> · <a href="results/revised_study/analysis.json">Raw revised data</a> · <a href="results/phase4_benchmarks.json">Historical benchmarks</a> · <a href="results/statistical_analysis.json">Regression data</a> · <a href="README.md">Reproduce the experiments</a></p></section>']
    return '\n'.join(chunks)


def build_payload() -> dict:
    checkpoint = ROOT / "results/revised_study/checkpoints/demographic_clean_seed_42.pt"
    latest = ROOT / "results/lineage_study/checkpoints/lineage_clean_seed_42.pt"
    if latest.exists() and (ROOT / "results/lineage_study/analysis.json").exists():
        checkpoint = latest
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from its research sidecar")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    benchmarks = json.loads((ROOT / "results/phase4_benchmarks.json").read_text(encoding="utf-8"))
    revised = json.loads((ROOT / 'results/revised_study/analysis.json').read_text())
    regression = json.loads((ROOT / 'results/statistical_analysis.json').read_text())['hamiltonian_regression']
    lineage_path = ROOT / 'results/lineage_study/analysis.json'
    lineage = json.loads(lineage_path.read_text()) if lineage_path.exists() else None
    inheritance_path = ROOT/'results/inheritance_study/analysis.json'
    inheritance = json.loads(inheritance_path.read_text(encoding='utf-8')) if inheritance_path.exists() else None
    diagnostics_path = ROOT/'results/inheritance_study/diagnostics.json'
    diagnostics_current = (inheritance is not None and diagnostics_path.exists()
                           and json.loads(diagnostics_path.read_text(encoding='utf-8')).get('analysis_sha256')
                           == hashlib.sha256(inheritance_path.read_bytes()).hexdigest())
    submission_root = ROOT/'results/submission_study'
    submission = None
    if (submission_root/'submission_diagnostics.json').exists():
        audit = json.loads((submission_root/'submission_diagnostics.json').read_text(encoding='utf-8'))
        for filename, field in (('analysis.json','analysis_sha256'),('diagnostics.json','diagnostics_sha256')):
            if hashlib.sha256((submission_root/filename).read_bytes()).hexdigest() != audit[field]:
                raise ValueError('Submission report is stale; regenerate evaluation/final_report.py')
        final = json.loads((submission_root/'analysis.json').read_text(encoding='utf-8'))
        submission = dict(summary=final['summary'], conditions=len(final['budgets_by_condition']),
                          checkpoints=len(audit['integrity']['checkpoints']),baseBudget=final['environment_steps_per_run'])
    hamilton = None
    hamilton_path = ROOT/'results/hamilton_study/analysis.json'
    if hamilton_path.exists():
        h = json.loads(hamilton_path.read_text(encoding='utf-8'))
        levels = ('bc100', 'bc150', 'bc250', 'bc400', 'bc600')
        hamilton = dict(
            steps=h['protocol']['environment_steps_per_run'], seeds=len(h['protocol']['training_seeds']),
            worlds=h['protocol']['evaluation_worlds_per_seed'], conditions=len(h['protocol']['conditions']),
            sweep=[dict(bc=int(k[2:])/100,
                        inclusive=h['summary'][f'inclusive_{k}/trained/clean']['donations_per_resident_step'],
                        energy=h['summary'][f'energy_{k}/trained/clean']['donations_per_resident_step'],
                        untrained=h['summary'][f'inclusive_{k}/untrained_ancestors/clean']['donations_per_resident_step'],
                        inclusiveSurvival=h['survival_by_bc'][f'inclusive_{k}/trained'],
                        energySurvival=h['survival_by_bc'][f'energy_{k}/trained']) for k in levels],
            stacked=h['stacked_sign_test'], fisher=h['survival_fisher'],
            choice=h['hamilton_choice_regression']['inclusive'].get('coefficients'),
            cue={c: h['cue_reliance'][c]['genetic'] for c in ('inclusive_bc250', 'inclusive_noise025', 'inclusive_noise050')},
            mimics={k.split('/')[1]: h['summary'][k]['mimic_donation_share'] for k in h['summary']
                    if k.startswith('inclusive_noise050_mimics/') and k.endswith('/attack')})
    return {
        'hamilton': hamilton,
        'submission': submission,
        "neural": {
            "architecture": "contextual_attention", "dModel": 32, "numHeads": 2,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "condition": checkpoint.stem.rsplit('_seed_', 1)[0],
            "rewardMode": metadata['environment']['reward_mode'],
            "seed": metadata["seed"],
            "environmentSteps": metadata["environment_steps"],
            "sha256": digest,
            "state": {key: tensor.tolist() for key, tensor in state.items() if not key.startswith(('critic_head.', 'gating.'))},
        },
        "revised": {"summary": revised['summary'], "cueSensitivity": revised['common_input_summary']},
        "lineage": None if lineage is None else {"summary": lineage['summary'], "cueSensitivity": lineage['common_input_summary'],
                                                   "feasibility": lineage['clean_feasibility'], "notRun": lineage['not_run']},
        "inheritance": None if inheritance is None else {
            "summary": inheritance['summary'], "feasibility": inheritance['clean_feasibility'],
            "notRun": inheritance['not_run'], "trainingSeeds": len(inheritance['training_seeds']),
            "worldsPerSeed": inheritance['evaluation_worlds_per_seed'], "steps": inheritance['environment_steps_per_run'],
            "hasDiagnostics": diagnostics_current and all((ROOT/'results/inheritance_study'/p).exists() for p in
                                  ('diagnostics.md','figures/learning_curves.png','figures/cue_sensitivity.png'))},
        "regression": regression,
        "benchmarks": [
            {
                "model": name.replace('Fixed Weights (w^g=0.5)', 'Fixed cue threshold (rule-based)'),
                "survival": row["survival_rate"],
                "population": row['carrying_capacity_mean'], "duration": row['lifespan_mean'],
                "defense": row["successful_defense_rate"],
                "attacks": row["faced_invasion_count"],
            }
            for name, row in benchmarks["benchmarks"].items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "index.html"
    html = target.read_text(encoding="utf-8")
    data = build_payload()
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    pattern = r'(<script id="research-data" type="application/json">).*?(</script>)'
    updated, count = re.subn(pattern, lambda match: match[1] + payload + match[2], html, flags=re.S)
    if count != 1:
        raise ValueError("Expected exactly one research-data script in index.html")
    updated, count = re.subn(r'(?<=<!-- RESEARCH_RESULTS_START -->).*?(?=<!-- RESEARCH_RESULTS_END -->)',
                            lambda _: '\n'+results_html(data)+'\n', updated, flags=re.S)
    if count != 1:
        raise ValueError('Expected exactly one research results section')
    if args.check:
        if updated != html:
            raise SystemExit("Embedded web data is stale. Run python visualizer/export_web_data.py")
        print("Embedded attention checkpoint, revised results and historical benchmarks match research artifacts.")
    else:
        target.write_text(updated, encoding="utf-8", newline="\n")
        print(f"Embedded checkpoint and benchmarks in {target.name} ({len(updated.encode('utf-8')):,} bytes).")


if __name__ == "__main__":
    main()
