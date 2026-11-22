"""Figure for the report that is not produced by the study reports.

python evaluation/report_figures.py   ->  plots/report/cue_and_mimics.png

(a) How much recipient choice relies on the genetic cue as it is corrupted, at
    B/C = 4 (Study C), with the behavioural cue for comparison.
(b) Share of post-invasion gifts that go to mimics, for learners and scripted rules.
All values are read from results/study_c_robustness/analysis.json.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import training.paths as paths


def err(row):
    lo, hi = row['ci95']
    return [[max(0.0, row['mean'] - lo)], [max(0.0, hi - row['mean'])]]


def main():
    a = json.loads((paths.study(paths.STUDY_C).results/'analysis.json').read_text(encoding='utf-8'))
    cue, s = a['cue_reliance'], a['summary']
    blue, orange, grey, green = '#2a6fdb', '#e07b39', '#8a8f98', '#2e9e6b'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1, 1.35]))

    levels = [('inclusive_bc400', 0.0), ('b4_noise025', 0.25), ('b4_noise050', 0.5)]
    for key, color, label, dx in (('genetic', blue, 'Genetic cue $\\tilde r$', -0.012), ('behavioural', orange, 'Behavioural cue $b$', 0.012)):
        xs = [x + dx for _, x in levels]
        rows = [cue[c][key] for c, _ in levels]
        ax1.errorbar(xs, [r['mean'] for r in rows], yerr=[[r['mean'] - r['ci95'][0] for r in rows], [r['ci95'][1] - r['mean'] for r in rows]],
                     fmt='o-', color=color, capsize=4, label=label)
    ax1.axhline(0, color='black', lw=.8)
    ax1.set_xticks([0, .25, .5]); ax1.set_xlabel('Genetic-cue bit-flip probability')
    ax1.set_ylabel('Standardised recipient-choice coefficient')
    ax1.set_title('(a) Reliance on each cue, B/C = 4 (95% CI, 10 seeds)', fontsize=10)
    ax1.legend(fontsize=8, frameon=False)

    bars = [('Inclusive PPO', 'b4_inclusive_mimics/trained/attack', blue),
            ('Kin-blind PPO', 'b4_kinblind_mimics/trained/attack', blue),
            ('Prosocial PPO', 'b4_prosocial_mimics/trained/attack', blue),
            ('Untrained networks', 'b4_inclusive_mimics/untrained_ancestors/attack', grey),
            ('Rule: reputation', 'b4_inclusive_mimics/rule_reputation/attack', green),
            ('Rule: Hamilton', 'b4_inclusive_mimics/rule_hamilton_rule/attack', green),
            ('Rule: Tit-for-tat', 'b4_inclusive_mimics/rule_dyadic_tit_for_tat/attack', green)]
    for i, (label, key, color) in enumerate(bars):
        row = s[key]['mimic_donation_share']
        ax2.bar(i, row['mean'], color=color, alpha=.85)
        ax2.errorbar(i, row['mean'], yerr=err(row), color='black', capsize=3)
        ax2.text(i, max(row['ci95'][1], row['mean']) + .02, f"{row['mean']:.2f}", ha='center', fontsize=8)
    ax2.axhline(6/51, color='black', ls='--', lw=.8)
    ax2.text(len(bars) - .5, 6/51 + .01, 'mimics\' share of agents', ha='right', fontsize=7)
    ax2.set_xticks(range(len(bars))); ax2.set_xticklabels([b[0] for b in bars], rotation=25, ha='right', fontsize=8)
    ax2.set_ylabel('Share of post-invasion gifts to mimics')
    ax2.set_title('(b) Mimic exploitation, B/C = 4 (6 mimics at step 10)', fontsize=10)
    fig.tight_layout()
    out = ROOT/'plots/report'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'cue_and_mimics.png', dpi=200)
    print(f'Wrote {out/"cue_and_mimics.png"}')


if __name__ == '__main__':
    main()
