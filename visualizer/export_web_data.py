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
    chunks = ['<section id="research-results" aria-labelledby="research-heading"><h2 id="research-heading">Research results</h2>',
              '<p><strong>The kin-selection hypothesis is not yet supported.</strong> These are saved, independently seeded research evaluations, separate from the live simulation above.</p>',
              '<p>Five training seeds per condition. Survival intervals are Wilson intervals; other intervals use Student t across seed-level runs. One evaluation world per trained seed limits precision.</p>',
              '<h3>Revised study: 8,192 environment steps per checkpoint</h3>',
              table(headers, result_rows(data['revised']['summary']), 'Revised study: trained policies and untrained network controls'),
              '<p class="research-caveat">Fallback survival is forced by disabling starvation; it is not ecological success. Mimic share is undefined without post-invasion donations. Estimates based on one or two donation-bearing runs, including the historical interval spanning negative percentages to above 100%, are not findings. Full raw estimates remain available below.</p>',
              '<h3>Cue sensitivity on identical synthetic inputs</h3>',
              '<p class="research-caveat">These policies barely donate. Sensitivity of their action distributions cannot establish a shift from genetic to behavioural cooperation. Raw cues bypass the gate; gate weights alone are not reliance. Permutations are exploratory and may create unrealistic observations.</p>',
              table(['Training condition','Genetic cue TV [95% CI]','Behaviour cue TV [95% CI]'],
                    [[name.replace('_',' '),interval(row['genetic_tv']),interval(row['behavior_tv'])] for name,row in data['revised']['cueSensitivity'].items()], 'Total variation after permuting one cue'),
              '<div class="research-figures"><figure><img src="results/revised_study/figures/learning_curves.png" width="1600" height="1120" loading="lazy" alt="Training donation rates and population over environment transitions, with seed means and pointwise 95% intervals"><figcaption>Training curves: cooperation collapses as training proceeds.</figcaption></figure><figure><img src="results/revised_study/figures/cue_sensitivity.png" width="1600" height="800" loading="lazy" alt="Genetic and behavioural cue permutation sensitivity across training conditions"><figcaption>Cue sensitivity measures action changes, not successful cooperative adaptation.</figcaption></figure></div>']
    if data['lineage']:
        lineage = data['lineage']
        passed = lineage['feasibility']['passed']
        chunks += ['<h3>New lineage-fitness follow-up</h3>',
                   '<p>Reward: one unit for own survival plus one per living child or grandchild. No chromosome similarity, Hamilton threshold, or donation bonus enters the reward. This explicitly changes the fitness objective; neural weights remain shared.</p>',
                   '<p>'+('The clean feasibility screen passed; conditional follow-ups were evaluated. This does not establish kin selection.' if passed else 'The clean feasibility screen failed. Noise and mimic follow-ups were deliberately not run; the failed clean result is retained.')+'</p>',
                   table(headers,result_rows(lineage['summary']),'Lineage-fitness study: saved evaluation outcomes'),
                   '<div class="research-figures"><figure><img src="results/lineage_study/figures/learning_curves.png" width="1600" height="1120" loading="lazy" alt="Lineage-fitness training donation and population curves across five seeds"><figcaption>Lineage-fitness training, with seed uncertainty.</figcaption></figure><figure><img src="results/lineage_study/figures/cue_sensitivity.png" width="1600" height="800" loading="lazy" alt="Lineage-fitness genetic and behavioural cue sensitivity on identical inputs"><figcaption>Exploratory lineage cue sensitivity; not evidence of kin-directed cooperation.</figcaption></figure></div>',
                   '<p><a href="results/lineage_study/results.md">Lineage results and uncertainty</a> · <a href="results/lineage_study/analysis.json">Raw lineage analysis</a></p>']
    regression = data['regression']
    chunks += ['<h3>Historical benchmark: 1,024 training steps</h3>',
               '<p>These use the earlier stress ecology and are not matched comparisons with the revised study. Rule-based thresholds are hand-written baselines. Mimic extinction with resident survival is not evidence of discrimination.</p>',
               table(['Model / baseline','Resident survival','Final population','Episode duration','Performed attacks'],
                     [[row['model'],f"{row['survival']:.1f}%",f"{row['population']:.1f}",f"{row['duration']:.1f}",row['attacks']] for row in data['benchmarks']], 'Historical benchmark point estimates; full intervals are linked in the source artifacts'),
               '<h3>Historical regression: no evidence of learned kin preference</h3>',
               f'<p>McFadden pseudo-R²: {regression["pseudo_r2"]:.4f}. {regression["num_clusters"]} training-seed clusters; {regression["num_samples"]:,} dependent recipient observations. Unstandardized coefficients; small-cluster, exploratory inference. This analysis does not describe the new lineage checkpoints.</p>',
               table(['Predictor','Coefficient','95% CI','Exploratory p-value'],
                     [[name,f'{b:.4f}',f'[{ci[0]:.4f}, {ci[1]:.4f}]',f'{p:.4f}'] for name,b,ci,p in zip(regression['features'],regression['coefficients'],regression['ci95'],regression['p_values'])], 'Training-seed cluster regression'),
               '<h3>Interpretation and reproducibility</h3><p>Donation adds net energy B−C; mitosis conserves parent-plus-child energy. Behavioural history counts donations given. Mimics are scripted cheaters, excluded from PPO. All neural residents use one shared policy; no evolving lineage-specific networks are claimed. Rule baselines can all reach capacity: compare donation frequency and exploitation, not the population cap alone.</p>',
               '<p><a href="results/revised_study/results.md">Revised results table</a> · <a href="results/revised_study/analysis.json">Raw revised data</a> · <a href="results/phase4_benchmarks.json">Historical benchmarks</a> · <a href="results/statistical_analysis.json">Regression data</a> · <a href="docs/review_response_2026-11-17.md">First review response</a> · <a href="docs/final_review_response_2026-11-18.md">Final review response</a> · <a href="README.md">Reproduce the experiments</a></p></section>']
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
    return {
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
