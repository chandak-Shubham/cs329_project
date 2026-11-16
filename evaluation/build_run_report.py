"""Validate artifact lineage and write the executed-study report from saved results."""
import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(name):
    return json.loads(Path("results", name).read_text())


def interval_text(summary):
    if summary["mean"] is None:
        return "not estimable"
    ci = summary["ci95"]
    return f"{summary['mean']:.4f} (95% CI {ci[0]:.4f} to {ci[1]:.4f}; n={summary['n']})" if ci else f"{summary['mean']:.4f} (CI unavailable)"


def build_report():
    test_bytes = Path("results/test_run.log").read_bytes()
    test_log = test_bytes.decode('utf-16' if test_bytes.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8', errors='replace')
    checkpoints = sorted(Path("results/checkpoints").glob("*_seed_*.pt"))
    if len(checkpoints) != 35:
        raise ValueError(f"Expected 35 replicated checkpoints, found {len(checkpoints)}")
    metadata = []
    for checkpoint in checkpoints:
        record = json.loads(checkpoint.with_suffix('.json').read_text())
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual != record['checkpoint_sha256']:
            raise ValueError(f"Checkpoint provenance mismatch: {checkpoint}")
        if record['environment_steps'] != len(record['learning_curve']) * record['rollout_length']:
            raise ValueError(f"Budget/learning curve mismatch: {checkpoint}")
        if not all(np.isfinite(point[key]) for point in record['learning_curve']
                   for key in ('mean_population', 'donations_per_step', 'policy_loss', 'value_loss')):
            raise ValueError(f"Non-finite training metric: {checkpoint}")
        metadata.append(record)
    controls = read("research_controls.json")
    if any(r['births'] == 0 for key, records in controls['runs'].items()
           if key.startswith('placement_') for r in records):
        raise ValueError("Placement control was not exercised in every run")
    for evaluation in (controls, read("evaluation_provenance.json")):
        for path, digest in evaluation["checkpoints_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Evaluation used a different checkpoint: {path}")
    phases = {p: read(f"phase{p}_{name}.json") for p, name in
              [(1, "selfish"), (2, "altruist"), (3, "cue_reliability"), (4, "benchmarks")]}
    analysis = read("statistical_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for color, kind in zip(('tab:blue', 'tab:orange', 'tab:green', 'tab:red'), ("mlp", "gating", "attention", "fixed_attention")):
        runs = [r for r in metadata if r["model_type"] == kind and r["bc_ratio"] == 2.5]
        for index, run in enumerate(runs):
            for ax, metric in zip(axes, ("mean_population", "donations_per_step")):
                ax.plot([p["environment_steps"] for p in run["learning_curve"]],
                        [p[metric] for p in run["learning_curve"]], alpha=.5, color=color,
                        label=kind if index == 0 else None)
    for ax, title in zip(axes, ("Population during training", "Donations per training transition")):
        ax.set_title(title)
        ax.set_xlabel("Environment transitions")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/fig7_learning_curves.png", dpi=180)
    plt.close(fig)

    text = ["# Completion evidence - 16 November 2026", "",
            "This report supersedes the outstanding-work status in the earlier dated reviews. Those reviews remain historical records.", "",
            "Test evidence: " + test_log.strip().splitlines()[-1], "",
            f"Validated {len(checkpoints)} newly trained checkpoint files against their SHA256 sidecars. Training budgets: {sorted(set(r['environment_steps'] for r in metadata))} environment transitions per run, five independent training seeds per condition. Active-agent transitions and full per-update curves are saved per run. These are short-budget experiments, not a convergence demonstration.", "",
            "Implemented: replicated corrected-environment training; independent evaluation/action seeds; missing-checkpoint errors; 95% intervals; matched invasion/no-invasion controls; trained neural fixed-gating ablation; random offspring-placement control; strict adjacency and seed-paired assortment inference; training-seed cluster regression; before/after mimic trust comparison; separate incentive-conditioned training; regenerated static figures, tables and replay; README/dependencies; bytecode untracking.", "",
            "## Regenerated observations", "",
            f"Phase 2 final populations: {[v['population'][-1] for v in phases[2]['trajectories'].values()]}", "",
            "| Phase 4 model | Resident survival (%) | Tested invasions | Successful defense (%) |",
            "| --- | ---: | ---: | ---: |"]
    for name, result in phases[4]["benchmarks"].items():
        defense = result['successful_defense_rate']
        text.append(f"| {name} | {result['survival_rate']:.1f} | {result['faced_invasion_count']} | {defense if defense is not None else 'not tested'} |")
    text += ["", "## Paired controls", "",
             "Intervals use independent training-seed differences. Tests are exploratory and unadjusted; zero-width mean intervals for identical outcomes are not population-level certainty.", ""]
    control_table = ["# Table 4: Paired research controls", "", "| Contrast | Metric | Mean difference and 95% CI |", "| --- | --- | --- |"]
    for name, metrics in controls["contrasts"].items():
        if name == "trust_shift_difference_in_differences":
            text.append(f"Genetic-weight pre/post difference in differences: {interval_text(metrics)}.")
            control_table.append(f"| {name} | genetic weight | {interval_text(metrics)} |")
        else:
            for metric, estimate in metrics.items():
                control_table.append(f"| {name} | {metric} | {interval_text(estimate)} |")
    text += ["", *control_table[2:], "", "## Incentive response", ""]
    incentive = controls["incentive_response"]
    text += [f"High-minus-low training incentive effect on fixed-input donation probabilities: {interval_text(incentive['incentive_response'])}.", "",
             f"Above-minus-below the theoretical rB=C threshold: {interval_text(incentive['supra_minus_sub_hamilton_threshold'])}.", "",
             incentive["note"], "", "## Spatial controls", "",
             "| Mutation | Adjacent-minus-random pairs | Paired p | Local-minus-random offspring placement |",
             "| --- | --- | --- | --- |"]
    for mu, result in controls["spatial_assortment"].items():
        text.append(f"| {mu} | {interval_text(result['gap_uncertainty'])} | {result['p_value']} | {interval_text(result['placement_control'])} |")
    text += ["", controls['design']['placement_design_note'], "",
             "The initial low-reserve data are retained in `results/placement_low_reserve_exploratory.json`.", "",
             "Placement birth counts (local / global placement, by training seed):", ""]
    for mu in controls["spatial_assortment"]:
        text.append(f"- Mutation {mu}: {[r['births'] for r in controls['runs'][f'placement_adjacent_{mu}']]} / {[r['births'] for r in controls['runs'][f'placement_random_{mu}']]}")
    reg = analysis['hamiltonian_regression']
    text += ["", "## Limits on conclusions", "",
             "Neural residents share a single policy per run. No competition among separately evolving neural lineage policies is implemented; that proposed claim has been removed from the current scope. Baseline-corrected chromosome similarity is not a validated pedigree relatedness estimate. B/C is unobserved, so incentive-conditioned training effects are not instantaneous incentive perception. Frozen-policy population weight changes can reflect changing recipients and population composition.", "",
             f"Regression uses {reg['num_clusters']} independent training-seed clusters, with {reg['num_samples']} dependent recipient observations; coefficient estimates (relatedness, behavior, B/C) are {reg['coefficients'][1:]} and p-values are {reg['p_values'][1:]}. These are rollout associations, not direct incentive perception. Small-cluster inference is exploratory. Assortment averages living states, not equilibrium populations. One evaluation environment per trained policy does not separate training variation from evaluation-world variation.", "",
             "The results above determine whether effects are positive, negative or uncertain. Donation counts or a narrow synthetic-probe association do not by themselves establish learned cooperation or Hamilton's rule. Longer training, more independent seeds and a prespecified confirmatory design remain follow-up research, not completed positive findings.", "",
             "## Reproduction and artifacts", "",
             "See README.md for exact execution order and budgets. `results/artifact_manifest.json` records checksums and installed package versions. Checkpoint sidecars retain source hashes as captured at each run's start; non-training tooling may subsequently change without invalidating trained transitions."]
    Path("docs/completion_2026-11-16.md").write_text('\n'.join(text)+'\n', encoding='utf-8')
    Path("results/tables/table4_controls.md").write_text('\n'.join(control_table)+'\n')
    paths = [p for p in Path('results').rglob('*') if p.is_file() and p.suffix in ('.json', '.pt', '.png', '.gif', '.md') and p.name != 'artifact_manifest.json']
    manifest = {"artifacts_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "packages": {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'scipy', 'matplotlib', 'pillow', 'pytest')},
                "python": sys.version, "independent_training_runs": len(checkpoints)}
    Path('results/artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"Validated {len(checkpoints)} checkpoints; wrote completion report, control table, learning curves and artifact manifest.")


if __name__ == '__main__':
    build_report()
