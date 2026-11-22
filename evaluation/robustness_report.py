"""Analysis of the robustness and reward-control study (Study C, B/C = 4).

python evaluation/robustness_report.py

Baselines at B/C = 4 come from the main study (inclusive_bc400, energy_bc400),
which uses the same code path, seeds and evaluation worlds. The inference unit
is the training seed; worlds are averaged within seed first.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np

from evaluation.uncertainty import mean_ci, proportion_ci
import evaluation.hamilton_report as hr
import training.paths as paths

BASE = 'inclusive_bc400'
METRICS = ('donations_per_resident_step', 'duration', 'final_residents', 'hamilton_consistent_fraction',
           'kin_preference', 'adjacent_similarity', 'mimic_donation_share', 'genetic_gate_weight')
LABELS = {
    'inclusive_bc400': 'Inclusive (kin-weighted) reward', 'energy_bc400': 'Energy-only reward',
    'b4_prosocial': 'Prosocial reward (+B for any gift)', 'b4_kinblind': 'Kin-blind reward (mean neighbour r)',
    'b4_inclusive_mimics': 'Inclusive, trained with mimics', 'b4_prosocial_mimics': 'Prosocial, trained with mimics',
    'b4_kinblind_mimics': 'Kin-blind, trained with mimics', 'b4_noise025': 'Inclusive, cue noise 0.25',
    'b4_noise050': 'Inclusive, cue noise 0.50', 'b4_noise050_mimics': 'Inclusive, noise 0.50 + mimics (learned gate)',
    'b4_fixedgate_noise050_mimics': 'Inclusive, noise 0.50 + mimics (fixed gate)', 'b4_fixed_gate': 'Inclusive, fixed gate',
    'b4_mlp': 'Inclusive, MLP (no attention)', 'b4_random_placement': 'Inclusive, random offspring placement',
    'b4_mutation000': 'Inclusive, mutation 0', 'b4_mutation020': 'Inclusive, mutation 0.20',
    'b4_inclusive_gamma099': 'Inclusive, gamma 0.99', 'b4_energy_gamma099': 'Energy-only, gamma 0.99',
    'b4_evolution_only': 'Evolution only (no learning)', 'b25_inclusive_long': 'Inclusive, B/C 2.5, 4x training',
    'inclusive_bc250': 'Inclusive, B/C 2.5, 1x training',
}


def seed_values(runs, condition, metric, control='trained', attack=False):
    out = {}
    for r in runs:
        if (r['condition'], r['control'], r['attack']) == (condition, control, attack) and r[metric] is not None:
            out.setdefault(r['training_seed'], []).append(float(r[metric]))
    return {s: float(np.mean(v)) for s, v in out.items()}


def survival(runs, condition, control='trained', attack=False):
    seeds = {}
    for r in runs:
        if (r['condition'], r['control'], r['attack']) == (condition, control, attack):
            seeds.setdefault(r['training_seed'], []).append(r['resident_survival'])
    return [all(v) for _, v in sorted(seeds.items())]


def contrast(runs, a, b, metric, attack=False, a_ctrl='trained', b_ctrl='trained'):
    return hr.paired(seed_values(runs, a, metric, a_ctrl, attack), seed_values(runs, b, metric, b_ctrl, attack))


def paired_survival(runs, a, b, attack=False):
    outcomes = []
    for condition in (a, b):
        worlds = {}
        for r in runs:
            if (r['condition'], r['control'], r['attack']) == (condition, 'trained', attack):
                block = worlds.setdefault(r['training_seed'], {})
                if r['world_seed'] in block:
                    raise ValueError(f'Duplicate world for {condition}')
                block[r['world_seed']] = r['resident_survival']
        outcomes.append(worlds)
    sa, sb = outcomes
    if not sa or sa.keys() != sb.keys():
        raise ValueError(f'Unmatched seeds for {a} and {b}')
    seeds = sorted(sa)
    for seed in seeds:
        if sa[seed].keys() != sb[seed].keys() or len(sa[seed]) != 3:
            raise ValueError(f'Unmatched evaluation worlds for seed {seed}')
    first = [int(all(sa[s].values())) for s in seeds]
    second = [int(all(sb[s].values())) for s in seeds]
    differences = [x-y for x, y in zip(first, second)]
    return dict(a=f'{sum(first)}/{len(first)}', b=f'{sum(second)}/{len(second)}',
                seeds=seeds, count_differences=differences, sign_flip_p=hr.sign_flip_p(differences))


def row(runs, condition, control='trained', attack=False):
    out = {m: mean_ci(list(seed_values(runs, condition, m, control, attack).values())) for m in METRICS}
    out['survival'] = proportion_ci(survival(runs, condition, control, attack))
    return out


def survival_comparisons(runs):
    return {
        'inclusive vs energy-only (gamma 0.9)': paired_survival(runs, BASE, 'energy_bc400'),
        'inclusive vs energy-only (gamma 0.99)': paired_survival(runs, 'b4_inclusive_gamma099', 'b4_energy_gamma099'),
        'inclusive vs prosocial': paired_survival(runs, BASE, 'b4_prosocial'),
        'inclusive vs kin-blind': paired_survival(runs, BASE, 'b4_kinblind'),
        'inclusive vs prosocial (mimics)': paired_survival(runs, 'b4_inclusive_mimics', 'b4_prosocial_mimics', attack=True),
        'inclusive vs kin-blind (mimics)': paired_survival(runs, 'b4_inclusive_mimics', 'b4_kinblind_mimics', attack=True),
        '4x vs 1x training (B/C 2.5)': paired_survival(runs, 'b25_inclusive_long', 'inclusive_bc250'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-study', default=paths.STUDY_B, help='Study B folder name (baselines)')
    p.add_argument('--study', default=paths.STUDY_C, help='Study C folder name')
    args = p.parse_args()
    M, R = paths.study(args.main_study), paths.study(args.study)
    main_runs, main_rows = hr.load(M)
    rob_runs, rob_rows = hr.load(R)
    runs = [r for r in main_runs if r['condition'] in (BASE, 'energy_bc400', 'inclusive_bc250')] + rob_runs
    rows = {k: v for k, v in main_rows.items() if k[0] in (BASE, 'energy_bc400', 'inclusive_bc250')}
    rows.update(rob_rows)
    protocol = json.loads((R.results/'protocol.json').read_text(encoding='utf-8'))
    conditions = [BASE, 'energy_bc400', *protocol['conditions'], 'inclusive_bc250']

    table = {}
    for c in conditions:
        table[f'{c}/trained/clean'] = row(runs, c)
        table[f'{c}/untrained_ancestors/clean'] = row(runs, c, 'untrained_ancestors')
        if any(r['condition'] == c and r['attack'] for r in runs):
            table[f'{c}/trained/attack'] = row(runs, c, attack=True)
            table[f'{c}/untrained_ancestors/attack'] = row(runs, c, 'untrained_ancestors', attack=True)
            for rule in hr_rules(runs, c):
                table[f'{c}/{rule}/attack'] = row(runs, c, rule, attack=True)

    contrasts = {}
    def add(name, a, b, metric, **kw):
        contrasts[f'{name}|{metric}'] = contrast(runs, a, b, metric, **kw)
    for metric in ('hamilton_consistent_fraction', 'kin_preference', 'donations_per_resident_step', 'duration'):
        add('inclusive - prosocial', BASE, 'b4_prosocial', metric)
        add('inclusive - kin-blind', BASE, 'b4_kinblind', metric)
        add('inclusive - untrained', BASE, BASE, metric, b_ctrl='untrained_ancestors')
        add('prosocial - untrained', 'b4_prosocial', 'b4_prosocial', metric, b_ctrl='untrained_ancestors')
        add('kin-blind - untrained', 'b4_kinblind', 'b4_kinblind', metric, b_ctrl='untrained_ancestors')
    for metric in ('mimic_donation_share', 'duration', 'final_residents'):
        add('inclusive - prosocial (mimics)', 'b4_inclusive_mimics', 'b4_prosocial_mimics', metric, attack=True)
        add('inclusive - kin-blind (mimics)', 'b4_inclusive_mimics', 'b4_kinblind_mimics', metric, attack=True)
        add('learned - fixed gate (noise 0.5, mimics)', 'b4_noise050_mimics', 'b4_fixedgate_noise050_mimics', metric, attack=True)
    for metric in ('donations_per_resident_step', 'duration', 'hamilton_consistent_fraction'):
        add('learned - fixed gate', BASE, 'b4_fixed_gate', metric)
        add('attention - MLP', BASE, 'b4_mlp', metric)
        add('mutation 0.05 - 0', BASE, 'b4_mutation000', metric)
        add('mutation 0.05 - 0.20', BASE, 'b4_mutation020', metric)
        add('noise 0 - 0.50', BASE, 'b4_noise050', metric)
        add('gamma 0.9 - 0.99 (inclusive)', BASE, 'b4_inclusive_gamma099', metric)
        add('inclusive PPO - evolution only', BASE, 'b4_evolution_only', metric)
        add('4x - 1x training (B/C 2.5)', 'b25_inclusive_long', 'inclusive_bc250', metric)
        add('4x training - untrained (B/C 2.5)', 'b25_inclusive_long', 'b25_inclusive_long', metric, b_ctrl='untrained_ancestors')
    add('local - random placement', BASE, 'b4_random_placement', 'adjacent_similarity')
    add('local - random placement (untrained)', BASE, 'b4_random_placement', 'adjacent_similarity',
        a_ctrl='untrained_ancestors', b_ctrl='untrained_ancestors')

    survival_tests = survival_comparisons(runs)
    cue = {c: hr.cue_reliance(rows, c) for c in (BASE, 'b4_noise025', 'b4_noise050', 'b4_prosocial', 'b4_kinblind')}
    for a, b in ((BASE, 'b4_noise050'), (BASE, 'b4_noise025'), (BASE, 'b4_prosocial'), (BASE, 'b4_kinblind')):
        for key in ('genetic', 'genetic_share'):
            contrasts[f'{a} - {b}|choice_{key}'] = hr.paired({s: v[key] for s, v in cue[a]['per_seed'].items()},
                                                            {s: v[key] for s, v in cue[b]['per_seed'].items()})

    sweep = json.loads((R.results/'strategy_sweep.json').read_text(encoding='utf-8'))
    strategies = {}
    for ecology, rule in sorted({(r['ecology'], r['rule']) for r in sweep}):
        sel = [r for r in sweep if (r['ecology'], r['rule']) == (ecology, rule)]
        strategies[f'{ecology}/{rule}'] = dict(
            b_over_c=sel[0]['b_over_c'], attack=sel[0]['attack'], worlds=len(sel),
            survival=proportion_ci([r['resident_survival'] for r in sel]),
            **{m: mean_ci([r[m] for r in sel]) for m in ('donations_per_resident_step', 'duration',
                                                          'hamilton_consistent_fraction', 'mimic_donation_share')})

    analysis = dict(protocol=protocol, baseline_source=M.results.relative_to(ROOT).as_posix(),
                    summary=table, contrasts=contrasts, survival_paired=survival_tests, cue_reliance=cue,
                    strategy_sweep=strategies, runs=[{k: v for k, v in r.items() if k != 'curve'} for r in rob_runs])
    (R.results/'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding='utf-8')
    hr.write_csv(R, rob_runs, protocol)
    with (R.results/'strategy_sweep.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        cols = ['duration', 'final_residents', 'resident_survival', 'donations_per_resident_step',
                'hamilton_consistent_fraction', 'mimic_donation_share']
        w.writerow(['ecology', 'b_over_c', 'strategy', 'world_seed', 'mimic_attack', *cols])
        for r in sweep:
            w.writerow([r['ecology'], r['b_over_c'], r['rule'], r['world_seed'], r['attack'], *('' if r[c] is None else r[c] for c in cols)])
    figures(R, analysis, main_runs)
    markdown(R, analysis)
    print(f'Wrote {R.results/"results.md"}')


def hr_rules(runs, condition):
    return sorted({r['control'] for r in runs if r['condition'] == condition and r['control'].startswith('rule_')})


def figures(out_dir, a, main_runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = out_dir.plots
    out.mkdir(parents=True, exist_ok=True)
    colors = dict(inclusive='#2a6fdb', energy='#e07b39', untrained='#8a8f98', hamilton_rule='#2e9e5b',
                  reputation='#9b59b6', tag_trust='#c0392b', dyadic_tit_for_tat='#16a085')

    # Threshold: learned vs scripted strategies across B/C.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    levels = ('bc100', 'bc150', 'bc250', 'bc400', 'bc600')
    xs = [int(k[2:])/100 for k in levels]
    for reward, label in (('inclusive', 'Inclusive-fitness PPO'), ('energy', 'Energy-only PPO')):
        ys = [proportion_ci(survival(main_runs, f'{reward}_{k}')) for k in levels]
        ax.plot(xs, [y['mean'] for y in ys], 'o-', color=colors[reward], label=label, lw=2)
        ax.fill_between(xs, [y['ci95'][0] for y in ys], [y['ci95'][1] for y in ys], color=colors[reward], alpha=.15)
    s = a['strategy_sweep']
    for rule, label in (('hamilton_rule', 'Scripted Hamilton rule'), ('reputation', 'Scripted reputation'),
                        ('tag_trust', 'Scripted tag trust'), ('dyadic_tit_for_tat', 'Scripted Tit-for-tat')):
        pts = sorted((v['b_over_c'], v['survival']['mean']) for k, v in s.items()
                     if k.endswith('/'+rule) and not v['attack'] and k.split('/')[0].startswith('bc') and '_' not in k.split('/')[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 's--', color=colors[rule], label=label, alpha=.85)
    ax.set_xlabel('B / C'); ax.set_ylabel('Fraction surviving 300 transitions')
    ax.set_title('Learned vs scripted strategies across benefit/cost ratios', fontsize=10)
    ax.set_ylim(-.05, 1.05); ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout(); fig.savefig(out/'strategies_vs_bc.png', dpi=150); plt.close(fig)

    # Reward controls: clean and under mimic attack.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    clean = (('inclusive_bc400', 'Inclusive'), ('b4_kinblind', 'Kin-blind'), ('b4_prosocial', 'Prosocial'), ('energy_bc400', 'Energy-only'))
    attack = (('b4_inclusive_mimics', 'Inclusive'), ('b4_kinblind_mimics', 'Kin-blind'), ('b4_prosocial_mimics', 'Prosocial'))
    t = a['summary']
    def bars(ax, items, key, metric, title, survival_metric=False):
        vals = [t[f'{c}/{key}'][metric] for c, _ in items]
        m = [v['mean'] if v['mean'] is not None else np.nan for v in vals]
        lo = [max(0.0, v['mean']-v['ci95'][0]) if v.get('ci95') and v['mean'] is not None else 0 for v in vals]
        hi = [max(0.0, v['ci95'][1]-v['mean']) if v.get('ci95') and v['mean'] is not None else 0 for v in vals]
        ax.bar(range(len(items)), m, yerr=[lo, hi], capsize=4, color=['#2a6fdb', '#7fa7e8', '#b9c9e8', '#e07b39'][:len(items)])
        ax.set_xticks(range(len(items)), [l for _, l in items]); ax.set_title(title, fontsize=10); ax.grid(axis='y', alpha=.3)
        if survival_metric:
            ax.set_ylim(0, 1.05)
    bars(axes[0], clean, 'trained/clean', 'hamilton_consistent_fraction', 'Clean: share of gifts with rB > C (95% CI)')
    bars(axes[1], attack, 'trained/attack', 'mimic_donation_share', 'Mimic attack: share of gifts to mimics (95% CI)')
    bars(axes[2], attack, 'trained/attack', 'survival', 'Mimic attack: seeds surviving (Wilson 95%)', True)
    fig.tight_layout(); fig.savefig(out/'reward_controls.png', dpi=150); plt.close(fig)


def fmt(r, d=3):
    return hr.fmt(r, d)


def count(r):
    return f"{round(r['mean']*r['n'])}/{r['n']}" if r['n'] else 'n/a'


def markdown(out_dir, a):
    t, c = a['summary'], a['contrasts']
    n = len(a['protocol']['training_seeds'])
    L = ['# Robustness and reward-control study (B/C = 4): results', '',
         f'Generated by `evaluation/robustness_report.py`. {n} training seeds per condition, '
         f"{a['protocol']['evaluation_worlds_per_seed']} evaluation worlds per seed (averaged within seed), "
         f"{a['protocol']['environment_steps_per_run']:,} transitions per run (4x for the long-training condition). "
         'Baselines `inclusive_bc400` and `energy_bc400` come from the main study (same code, seeds and worlds). '
         'Brackets: 95% Student-t interval over seeds; survival: Wilson interval.', '',
         '**Mimic slots.** Mimics can only take free slots, so a population already at its cap of 45 would '
         'receive no attack. Every attacked ecology here therefore reserves 6 slots that only mimics can use '
         '(capacity 51, births still stop at 45 residents), and every attack delivers its 6 mimics; '
         'the `mimics_injected` column of `evaluation_runs.csv` records this for each episode. '
         'The five mimic conditions were trained and evaluated with this reserve (source archived in `source/`). '
         'The other conditions have no mimics, so the reserve never applies to them; they were trained with the '
         'previous version of the Study C training script (archived in `source_v1/`, identical apart from the reserve), '
         'and `evaluation/verify_hamilton.py --study robustness` re-simulates them exactly with the current code.', '',
         '## 1. All conditions (trained policies)', '',
         '| Condition | Evaluation | Seeds surviving | Donations / step | Lifetime | Share of gifts with rB > C | Kin preference | Mimic share | Gate w^g |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for key, r in t.items():
        cond, ctrl, ev = key.split('/')
        if ctrl != 'trained':
            continue
        L.append(f"| {LABELS.get(cond, cond)} | {ev} | {count(r['survival'])} | {fmt(r['donations_per_resident_step'])} | "
                 f"{fmt(r['duration'], 1)} | {fmt(r['hamilton_consistent_fraction'])} | {fmt(r['kin_preference'])} | "
                 f"{fmt(r['mimic_donation_share'])} | {fmt(r['genetic_gate_weight'])} |")
    L += ['', '## 2. Untrained-network and scripted controls', '',
          '| Condition | Actor | Evaluation | Seeds surviving | Donations / step | Lifetime | Mimic share |', '| --- | --- | --- | --- | --- | --- | --- |']
    for key, r in t.items():
        cond, ctrl, ev = key.split('/')
        if ctrl == 'trained':
            continue
        L.append(f"| {LABELS.get(cond, cond)} | {ctrl} | {ev} | {count(r['survival'])} | {fmt(r['donations_per_resident_step'])} | "
                 f"{fmt(r['duration'], 1)} | {fmt(r['mimic_donation_share'])} |")
    L += ['', '## 3. Survival comparisons (two-sided paired sign-flip test over seeds)', '', '| Comparison | First | Second | p |', '| --- | --- | --- | --- |']
    for name, f in a['survival_paired'].items():
        L.append(f"| {name} | {f['a']} | {f['b']} | {f['sign_flip_p']:.2e} |")
    L += ['', f'## 4. Paired contrasts by seed (exact sign-flip; minimum attainable p with {n} pairs = {2/2**n:.4f})', '',
          '| Contrast | Metric | Difference | Seeds positive | p |', '| --- | --- | --- | --- | --- |']
    for key, v in c.items():
        name, metric = key.split('|')
        p = '' if v['sign_flip_p'] is None else f"{v['sign_flip_p']:.4f}"
        L.append(f"| {name} | {metric} | {fmt(v['difference'])} | {v['positive_seeds']}/{len(v['seeds'])} | {p} |")
    L += ['', '## 5. Recipient-choice reliance (per-seed standardised conditional logit)', '',
          '| Condition | Genetic cue | Behavioural cue | Genetic share | Seeds with >= 20 gifts |', '| --- | --- | --- | --- | --- |']
    for cond, v in a['cue_reliance'].items():
        L.append(f"| {LABELS.get(cond, cond)} | {fmt(v['genetic'])} | {fmt(v['behavioural'])} | {fmt(v['genetic_share'])} | {len(v['seeds'])} |")
    L += ['', '## 6. Scripted strategies in every ecology (10 fresh worlds each, random founders)', '',
          '| Ecology | B/C | Strategy | Worlds surviving | Donations / step | Lifetime | Share of gifts with rB > C | Mimic share |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for key, v in a['strategy_sweep'].items():
        eco, rule = key.split('/')
        L.append(f"| {eco} | {v['b_over_c']} | {rule} | {count(v['survival'])} | {fmt(v['donations_per_resident_step'])} | "
                 f"{fmt(v['duration'], 1)} | {fmt(v['hamilton_consistent_fraction'])} | {fmt(v['mimic_donation_share'])} |")
    L += ['', '## Spreadsheets', '', '[seed_summary.csv](seed_summary.csv), [evaluation_runs.csv](evaluation_runs.csv), '
          '[strategy_sweep.csv](strategy_sweep.csv).', '', '## Figures', '',
          f'![Reward controls](../../plots/{out_dir.name}/reward_controls.png)', '', f'![Strategies across B/C](../../plots/{out_dir.name}/strategies_vs_bc.png)', '']
    (out_dir.results/'results.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    main()
