"""Figures and seed-level diagnostics for the inherited-policy experiment.

Never substitutes historical shared-policy statistics for a new experiment.
python evaluation/inheritance_report.py --study-dir results/inheritance_study
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.uncertainty import mean_ci
from experiments.inherited_policies import load, forward
from experiments.inheritance_study import evaluate


def common_probe(population):
    rng = np.random.default_rng(7301)
    state = torch.tensor(np.column_stack((rng.uniform(.1,1,64),rng.uniform(0,.3,64))),dtype=torch.float32)
    features = torch.tensor(rng.uniform(0,1,(64,8,5)),dtype=torch.float32)
    features[...,:2] = features[...,:2]*2-1
    mask = torch.ones(64,8,dtype=torch.bool)
    action_mask = torch.ones(64,13,dtype=torch.bool)
    # Identical interventions for every network and every condition.
    orders = {name:torch.as_tensor(rng.permutation(512)) for name in ('genetic_tv','behavior_tv')}
    rows = []
    with torch.no_grad():
        for policy in population.policies:
            dist, _, weights = forward(policy,(state,features,mask,action_mask))
            row = {'genetic_weight':float(weights[...,0].mean()) if weights is not None else None}
            for name, column in (('genetic_tv',3),('behavior_tv',4)):
                changed = features.clone()
                changed[...,column] = features[...,column].flatten()[orders[name]].reshape(64,8)
                alternative = forward(policy,(state,changed,mask,action_mask))[0].probs
                row[name] = float((.5*(alternative-dist.probs).abs().sum(-1)).mean())
            rows.append(row)
    return {key:float(np.mean([r[key] for r in rows])) if rows[0][key] is not None else None for key in rows[0]}


def paired_contrasts(runs):
    """Pair seed means, not individual worlds or recipients."""
    pairs = [('clean','no_selection'),('clean','evolution_only'),('clean','fixed_gate'),
             ('clean','mlp'),('clean','random_placement'),('noise050','clean'),
             ('noise_mimics','fixed_noise_mimics'),('low_lr','clean'),
             ('high_entropy','clean'),('long_training','clean')]
    result = {}
    for a,b in pairs:
        for metric in ('donations_per_resident_step','final_residents','adjacent_hamming_similarity',
                       'opportunity_adjusted_tag_preference'):
            differences = []
            for seed in sorted({r['training_seed'] for r in runs}):
                values = {}
                for condition in (a,b):
                    selected = [r[metric] for r in runs if r['condition'] == condition
                                and r['control'] == 'trained' and not r['attack']
                                and r['training_seed'] == seed and r[metric] is not None]
                    if selected:
                        values[condition] = float(np.mean(selected))
                if len(values) == 2:
                    differences.append(values[a]-values[b])
            result[f'{a}_minus_{b}/{metric}'] = mean_ci(differences)
    return result


def hamilton_regression(root, analysis):
    required = {'clean','bc150','bc450'}
    present = {r['condition'] for r in analysis['runs']}
    if not required <= present:
        return dict(status='not fitted',reason='Independently trained matched B/C conditions are required. No historical regression is substituted.')
    from evaluation.analysis import fit_logistic_regression
    rows = []
    class Recorder:
        def __init__(self, population, seed):
            self.population, self.seed = population, seed

        def __getattr__(self, name):
            return getattr(self.population,name)

        def act(self, env, obs):
            output = self.population.act(env,obs)
            _,owners,actions,_,_ = output
            for slot in np.flatnonzero(owners >= 0):
                for k in np.flatnonzero(obs['neighbor_mask'][slot]):
                    if obs['action_mask'][slot,5+k]:
                        feature = obs['neighbor_features'][slot,k]
                        rows.append((1.,float(feature[3]),float(feature[4]),env.B/env.C,
                                     int(actions[slot] == 5+k),self.seed))
            return output
    for condition in sorted(required):
        for seed in analysis['training_seeds']:
            population,genomes,metadata = load(root/'checkpoints'/f'{condition}_seed_{seed}.pt')
            recorder = Recorder(population,seed)
            for replicate in range(analysis['evaluation_worlds_per_seed']):
                evaluate(recorder,genomes,metadata['environment'],seed+10000+replicate*1000)
    data = np.asarray(rows,dtype=float)
    if not len(data) or len(np.unique(data[:,4])) < 2:
        return dict(status='not identifiable',reason='Recipient decision outcomes have no variation.',num_samples=len(data))
    # A fixed uniform cap within each seed prevents recipient counts from
    # dominating compute. This is NOT case-control or outcome-based sampling.
    rng = np.random.default_rng(9301)
    selected = []
    for seed in analysis['training_seeds']:
        indices = np.flatnonzero(data[:,5] == seed)
        selected.extend(rng.choice(indices,size=min(12000,len(indices)),replace=False))
    sampled = data[selected]
    x,y,groups = sampled[:,:4],sampled[:,4],sampled[:,5]
    np.savez_compressed(root/'hamilton_decisions.npz',X=x,y=y,training_seed=groups)
    if np.linalg.matrix_rank(x) < x.shape[1] or len(np.unique(groups)) < 3:
        return dict(status='not identifiable',reason='Rank-deficient predictors or fewer than three independent seeds.',num_samples=len(x))
    result = fit_logistic_regression(x,y,['intercept','corrected_tag_cue','behavior_history','B_over_C'],groups=groups)
    result['status'] = 'exploratory fit' if result['converged'] else 'optimizer did not converge'
    result['eligible_records_before_sampling'] = len(data)
    result['clean_feasibility_passed'] = analysis['clean_feasibility']['passed']
    result['scope'] = 'Transient recipient decisions, including trajectories ending in extinction; not an equilibrium cooperation test.'
    result['interpretation'] = 'Recipient decisions are dependent within seed; cluster-t inference with five seeds is exploratory. B/C indexes independently trained conditions, not an observed policy input. This association is not a causal validation of Hamilton rule.'
    return result


def main(root):
    torch.set_num_threads(1)
    analysis = json.loads((root/'analysis.json').read_text(encoding='utf-8'))
    conditions = sorted({r['condition'] for r in analysis['runs']})
    records, probes = {}, []
    for condition in conditions:
        records[condition] = []
        for seed in analysis['training_seeds']:
            population, _, metadata = load(root/'checkpoints'/f'{condition}_seed_{seed}.pt')
            records[condition].append(metadata)
            probes.append(dict(condition=condition,training_seed=seed,**common_probe(population)))
    contrasts = {}
    for a,b in (('noise025','clean'),('noise050','clean'),('noise_mimics','noise050')):
        for metric in ('genetic_tv','behavior_tv','genetic_weight'):
            differences = []
            for seed in analysis['training_seeds']:
                values = {r['condition']:r[metric] for r in probes if r['training_seed'] == seed and r[metric] is not None}
                if a in values and b in values:
                    differences.append(values[a]-values[b])
            contrasts[f'{a}_minus_{b}/{metric}'] = mean_ci(differences)
    summary = {c:{key:mean_ci([r[key] for r in probes if r['condition'] == c])
                  for key in ('genetic_tv','behavior_tv','genetic_weight')} for c in conditions}
    diagnostics = dict(analysis_sha256=hashlib.sha256((root/'analysis.json').read_bytes()).hexdigest(),
                       common_input_probes=probes,common_input_summary=summary,
                       cue_paired_contrasts=contrasts,ecological_paired_contrasts=paired_contrasts(analysis['runs']),
                       interpretation='Full-policy cue permutation is an exploratory off-manifold diagnostic, not causal reliability. Networks within a run are not independent replicates.',
                       hamilton_regression=hamilton_regression(root,analysis))
    (root/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False),encoding='utf-8')
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    # Keep the overview readable; every condition remains in raw diagnostics.
    overview = {'clean','no_selection','evolution_only','fixed_gate','mlp',
                'low_lr','high_entropy','long_training'}
    for condition, runs in records.items():
        if len(records) > 8 and condition not in overview:
            continue
        # Episode termination makes rollout endpoints irregular. Bin actual
        # transitions rather than pretending each update is exactly 64 steps.
        budget = max(r['environment_steps'] for r in runs)
        edges = np.linspace(0,budget,17)
        for axis,metric,label in zip(axes,('donations_per_step','final_residents'),('Donations / transition','Residents at rollout endpoint')):
            columns = []
            for lo,hi in zip(edges[:-1],edges[1:]):
                seed_values = []
                for run in runs:
                    weighted, duration = 0.,0.
                    for row in run['learning_curve']:
                        end = row['environment_steps']
                        start = end-row['duration']
                        overlap = max(0.,min(end,hi)-max(start,lo))
                        weighted += row[metric]*overlap
                        duration += overlap
                    if duration:
                        seed_values.append(weighted/duration)
                columns.append(mean_ci(seed_values))
            x = (edges[:-1]+edges[1:])/2
            axis.plot(x,[r['mean'] for r in columns],label=condition)
            if all(r['ci95'] is not None for r in columns):
                axis.fill_between(x,[r['ci95'][0] for r in columns],[r['ci95'][1] for r in columns],alpha=.12)
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title('Inherited policies: seed means and pointwise 95% t intervals')
    axes[-1].set_xlabel('Environment transitions per training run')
    fig.savefig(figures/'learning_curves.png',dpi=160)
    plt.close(fig)
    cue_conditions = [c for c in conditions if len(conditions) <= 8 or c in
                      ('clean','noise025','noise050','noise_mimics','fixed_noise_mimics')]
    fig, ax = plt.subplots(figsize=(10,5),layout='constrained')
    for offset,key,label in ((-.18,'genetic_tv','Genetic cue'),(.18,'behavior_tv','Behaviour cue')):
        rows = [summary[c][key] for c in cue_conditions]
        errors = np.array([[r['mean']-r['ci95'][0],r['ci95'][1]-r['mean']] if r['ci95'] else [0,0] for r in rows]).T
        ax.bar(np.arange(len(cue_conditions))+offset,[r['mean'] for r in rows],.36,yerr=errors,capsize=3,label=label)
    ax.set_xticks(np.arange(len(cue_conditions)),[c.replace('_','\n') for c in cue_conditions])
    ax.set_ylabel('Mean total variation in action probabilities')
    ax.set_title('Common-input cue sensitivity; descriptive, not successful adaptation')
    ax.legend()
    fig.savefig(figures/'cue_sensitivity.png',dpi=160)
    plt.close(fig)
    text = ['# Inherited policy diagnostics','',diagnostics['interpretation'],'',
            '| Condition | Genetic cue sensitivity | Behaviour cue sensitivity | Genetic gate weight |','| --- | --- | --- | --- |']
    for c,row in summary.items():
        def format_row(r):
            return 'not applicable' if r['mean'] is None else f"{r['mean']:.5f}; n={r['n']}; CI={r['ci95']}"
        text.append('| '+c+' | '+' | '.join(format_row(row[k]) for k in ('genetic_tv','behavior_tv','genetic_weight'))+' |')
    text += ['', 'No genetic-to-behavioural adaptation claim is supported unless clean cooperation and the independently trained cue comparisons succeed.', '',
             'Paired seed-level contrasts and the explicit regression status: [diagnostics.json](diagnostics.json).', '',
             '![Training curves](figures/learning_curves.png)', '', '![Cue sensitivity](figures/cue_sensitivity.png)']
    (root/'diagnostics.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    manifest = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.suffix in ('.json','.pt','.md','.png','.npz') and p.name != 'manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Generated figures and seed-level diagnostics in {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir',type=Path,default=Path('results/inheritance_study'))
    main(parser.parse_args().study_dir)
