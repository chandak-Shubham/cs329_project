"""Analysis of the Hamilton's-rule study (experiments/hamilton_study.py).

python evaluation/hamilton_report.py --study-dir results/hamilton_study

Inference unit: the training seed. Evaluation worlds are averaged within each
seed first. Decision rows are never treated as independent samples: regression
uncertainty comes from a cluster bootstrap that resamples whole training seeds.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from evaluation.uncertainty import mean_ci, proportion_ci

BC = ('bc100', 'bc150', 'bc250', 'bc400', 'bc600')
BC_VALUE = dict(bc100=1.0, bc150=1.5, bc250=2.5, bc400=4.0, bc600=6.0)
METRICS = ('donations_per_resident_step', 'late_donation_rate', 'final_residents', 'duration',
           'resident_survival', 'give_rate', 'hamilton_gap', 'hamilton_consistent_fraction',
           'recipient_true_relatedness', 'kin_preference', 'adjacent_similarity',
           'genetic_gate_weight', 'mimic_donation_share')
ROW = dict(r_obs=0, r_true=1, b=2, energy_j=3, is_mimic=4, chosen=5, gave_any=6, n_neighbors=7, bc=8)


def load(study):
    runs, rows = [], {}
    for path in sorted((study/'evaluations').glob('*_seed_*.json')):
        runs.extend(json.loads(path.read_text(encoding='utf-8')))
        condition, seed = path.stem.rsplit('_seed_', 1)
        with np.load(path.with_name(path.stem+'_rows.npz')) as data:
            for key in data.files:
                control, attack, _ = key.split('|')
                rows.setdefault((condition, control, attack, int(seed)), []).append(data[key])
    rows = {k: np.concatenate(v) if v else np.zeros((0, 9)) for k, v in rows.items()}
    return runs, rows


def seed_means(runs, condition, control='trained', attack=False, metric='donations_per_resident_step'):
    """{seed: mean over worlds}, skipping undefined world values."""
    out = {}
    for r in runs:
        if (r['condition'], r['control'], r['attack']) == (condition, control, attack):
            out.setdefault(r['training_seed'], [])
            if r[metric] is not None:
                out[r['training_seed']].append(float(r[metric]))
    return {s: float(np.mean(v)) for s, v in out.items() if v}


def summarize(runs):
    table = {}
    for condition, control, attack in sorted({(r['condition'], r['control'], r['attack']) for r in runs}):
        row = {m: mean_ci(list(seed_means(runs, condition, control, attack, m).values())) for m in METRICS}
        seeds = sorted({r['training_seed'] for r in runs if (r['condition'], r['control'], r['attack']) == (condition, control, attack)})
        row['all_worlds_survive'] = proportion_ci([all(
            r['resident_survival'] for r in runs if (r['condition'], r['control'], r['attack'], r['training_seed'])
            == (condition, control, attack, s)) for s in seeds])
        table[f'{condition}/{control}/{"attack" if attack else "clean"}'] = row
    return table


def sign_flip_p(diffs):
    """Exact two-sided paired sign-flip p-value (min attainable 2/2^n)."""
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return None
    observed = abs(d.mean())
    signs = np.array(list(product((-1, 1), repeat=len(d))))
    return float(np.mean(np.abs((signs*d).mean(1)) >= observed - 1e-12))


def paired(a, b):
    """Paired contrast a-b over seeds present in both."""
    seeds = sorted(set(a) & set(b))
    diffs = [a[s]-b[s] for s in seeds]
    return dict(seeds=seeds, difference=mean_ci(diffs), sign_flip_p=sign_flip_p(diffs),
                positive_seeds=int(sum(d > 0 for d in diffs)))


def holm(pvalues):
    keys = [k for k, p in sorted(pvalues.items(), key=lambda kv: kv[1])]
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m-i)*pvalues[k]))
        out[k] = running
    return out


def logistic(X, y, iters=60, ridge=1e-4):
    """Newton/IRLS logistic regression with a tiny ridge for separation safety."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-np.clip(X@beta, -30, 30)))
        grad = X.T@(y-p) - ridge*beta
        hess = (X*(p*(1-p))[:, None]).T@X + ridge*np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def design(rows, interaction=True):
    r, b, bc = rows[:, ROW['r_obs']], rows[:, ROW['b']], rows[:, ROW['bc']]
    cols = [np.ones(len(rows)), r, b, bc] + ([r*bc] if interaction else [])
    return np.column_stack(cols), rows[:, ROW['chosen']]


def hamilton_regression(rows, reward, cap=3000, reps=500, seed=0):
    """logit P(choose j) = b0 + b1 r~ + b2 b + b3 B/C + b4 r~*(B/C), pooled over the B/C sweep.

    Hamilton's rule predicts b4 > 0: relatedness matters more as B/C grows.
    Rows: every observed neighbour of a sampled donation opportunity.
    Cluster bootstrap resamples training seeds within each B/C condition.
    """
    rng = np.random.default_rng(seed)
    clusters = {}
    for (condition, control, attack, s), data in rows.items():
        if control == 'trained' and attack == 'clean' and condition in [f'{reward}_{k}' for k in BC] and len(data):
            take = data if len(data) <= cap else data[rng.choice(len(data), cap, replace=False)]
            clusters.setdefault(condition, {})[s] = take
    pooled = np.concatenate([d for c in clusters.values() for d in c.values()]) if clusters else np.zeros((0, 9))
    if len(pooled) < 50 or len(np.unique(pooled[:, ROW['chosen']])) < 2:
        return dict(reward=reward, status='not identifiable: too few decisions', rows=int(len(pooled)))
    X, y = design(pooled)
    beta = logistic(X, y)
    boot = []
    for _ in range(reps):
        sample = []
        for c in clusters.values():
            seeds = list(c)
            sample += [c[seeds[i]] for i in rng.integers(len(seeds), size=len(seeds))]
        data = np.concatenate(sample)
        if len(np.unique(data[:, ROW['chosen']])) == 2:
            boot.append(logistic(*design(data)))
    boot = np.array(boot)
    names = ['intercept', 'r_obs', 'b', 'bc', 'r_obs_x_bc']
    return dict(reward=reward, status='fitted', rows=int(len(pooled)), chosen=int(y.sum()),
                clusters=int(sum(len(c) for c in clusters.values())), bootstrap_replicates=int(len(boot)),
                coefficients={n: dict(estimate=float(beta[i]),
                                      ci95=[float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))],
                                      bootstrap_fraction_positive=float(np.mean(boot[:, i] > 0)))
                              for i, n in enumerate(names)})


def opportunity_blocks(data):
    """Split decision rows into opportunities.

    evaluate_world appends one row per visible neighbour, contiguously, for each
    sampled opportunity; column n_neighbors is the block length.
    """
    blocks, i = [], 0
    while i < len(data):
        n = int(data[i, ROW['n_neighbors']])
        if n < 1 or i+n > len(data):
            raise ValueError('Decision rows are not block-aligned')
        blocks.append(data[i:i+n])
        i += n
    return blocks


def stratified_gap(data):
    """Hamilton gap holding neighbourhood size fixed.

    The raw gap compares opportunities with and without an rB > C neighbour, but
    such a neighbour is likelier when there are more neighbours, and giving
    probability itself rises with the number of give actions available. Within
    each neighbour count n we take the gap and combine strata with
    Cochran-Mantel-Haenszel risk-difference weights a*b/(a+b).
    """
    strata = {}
    for block in opportunity_blocks(data):
        n = len(block)
        above = bool((block[:, ROW['r_true']]*block[0, ROW['bc']] > 1).any())
        gave = bool(block[0, ROW['gave_any']])
        s = strata.setdefault(n, [0, 0, 0, 0])  # above opps, above gives, below opps, below gives
        s[0 if above else 2] += 1
        s[1 if above else 3] += gave
    num = den = 0.0
    for a, ga, b, gb in strata.values():
        if a and b:
            w = a*b/(a+b)
            num += w*(ga/a - gb/b)
            den += w
    return num/den if den else None


def choice_arrays(blocks, features):
    """Pad gift opportunities into (gifts, 8, features) arrays with a mask and the chosen index."""
    sets = [b for b in blocks if b[0, ROW['gave_any']] and len(b) > 1 and b[:, ROW['chosen']].sum() == 1]
    if not sets:
        return None
    F = features(sets[0]).shape[1]
    X = np.zeros((len(sets), 8, F))
    M = np.zeros((len(sets), 8), dtype=bool)
    y = np.zeros(len(sets), dtype=int)
    for g, b in enumerate(sets):
        X[g, :len(b)] = features(b)
        M[g, :len(b)] = True
        y[g] = int(np.flatnonzero(b[:, ROW['chosen']])[0])
    return X, M, y


def fit_choice(X, M, y, iters=60, ridge=1e-4):
    """Newton-Raphson for McFadden's conditional logit on padded arrays."""
    if X is None or len(y) < 20:
        return None
    beta = np.zeros(X.shape[2])
    chosen = X[np.arange(len(y)), y].sum(0)
    for _ in range(iters):
        u = np.where(M, X@beta, -np.inf)
        p = np.exp(u - u.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        mean = np.einsum('gk,gkf->gf', p, X)
        grad = chosen - mean.sum(0) - ridge*beta
        hess = np.einsum('gk,gkf,gkh->fh', p, X, X) - mean.T@mean + ridge*np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def conditional_logit(blocks, features, iters=60, ridge=1e-4):
    """McFadden conditional logit over the neighbours of each gift.

    Only opportunities that ended in a gift enter; the outcome is WHICH
    neighbour received it. Constant-within-opportunity terms (B/C, own state)
    cancel, so this isolates recipient choice from the decision to give.
    """
    arrays = choice_arrays(blocks, features)
    return None if arrays is None else fit_choice(*arrays, iters=iters, ridge=ridge)


HAMILTON_CHOICE = ('r_obs', 'b', 'energy_j', 'r_obs_x_bc')


def hamilton_choice_features(b):
    bc = b[:, ROW['bc']]
    return np.column_stack([b[:, ROW['r_obs']], b[:, ROW['b']], b[:, ROW['energy_j']], b[:, ROW['r_obs']]*bc])


def hamilton_choice_regression(rows, reward, reps=500, seed=0):
    """Conditional logit of recipient choice, pooled over the B/C sweep.

    Hamilton's rule predicts that weight on relatedness rises with B/C
    (r_obs_x_bc > 0). Cluster bootstrap resamples training seeds within B/C.
    """
    rng = np.random.default_rng(seed)
    clusters = {}
    for (condition, control, attack, s), data in rows.items():
        if control == 'trained' and attack == 'clean' and condition in [f'{reward}_{k}' for k in BC] and len(data):
            arrays = choice_arrays(opportunity_blocks(data), hamilton_choice_features)
            if arrays is not None:
                clusters.setdefault(condition, {})[s] = arrays
    join = lambda parts: tuple(np.concatenate([a[i] for a in parts]) for i in range(3))
    allparts = [a for c in clusters.values() for a in c.values()]
    beta = fit_choice(*join(allparts)) if allparts else None
    if beta is None:
        return dict(reward=reward, status='not identifiable: fewer than 20 gifts')
    boot = []
    for _ in range(reps):
        sample = []
        for c in clusters.values():
            seeds = list(c)
            sample += [c[seeds[i]] for i in rng.integers(len(seeds), size=len(seeds))]
        est = fit_choice(*join(sample), iters=25)
        if est is not None:
            boot.append(est)
    boot = np.array(boot)
    gifts = int(sum(len(a[2]) for a in allparts))
    return dict(reward=reward, status='fitted', gifts_with_choice=gifts, clusters=len(allparts), bootstrap_replicates=len(boot),
                coefficients={n: dict(estimate=float(beta[i]),
                                      ci95=[float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))],
                                      bootstrap_fraction_positive=float(np.mean(boot[:, i] > 0)))
                              for i, n in enumerate(HAMILTON_CHOICE)})


def stratified_gap_by_seed(rows, condition, control='trained', attack='clean'):
    out = {}
    for (c, ctrl, a, s), data in rows.items():
        if (c, ctrl, a) == (condition, control, attack) and len(data):
            gap = stratified_gap(data)
            if gap is not None:
                out[s] = gap
    return out


def cue_reliance(rows, condition, attack='clean'):
    """Per-seed standardized conditional-logit weights of recipient choice.

    Only gifts enter (which neighbour, not whether to give). Cues are
    standardized within seed so genetic and behavioural weights are comparable.
    """
    per_seed = {}
    for (c, control, a, s), data in rows.items():
        if (c, control, a) != (condition, 'trained', attack) or len(data) < 100:
            continue
        z = lambda v: (v-v.mean())/(v.std()+1e-8)
        zr, zb = z(data[:, ROW['r_obs']]), z(data[:, ROW['b']])
        scaled = data.copy()
        scaled[:, ROW['r_obs']], scaled[:, ROW['b']] = zr, zb
        beta = conditional_logit(opportunity_blocks(scaled),
                                 lambda b: np.column_stack([b[:, ROW['r_obs']], b[:, ROW['b']]]))
        if beta is None:
            continue
        per_seed[s] = dict(genetic=float(beta[0]), behavioural=float(beta[1]),
                           genetic_share=float(abs(beta[0])/(abs(beta[0])+abs(beta[1])+1e-12)))
    return dict(seeds=sorted(per_seed), per_seed=per_seed,
                genetic=mean_ci([v['genetic'] for v in per_seed.values()]),
                behavioural=mean_ci([v['behavioural'] for v in per_seed.values()]),
                genetic_share=mean_ci([v['genetic_share'] for v in per_seed.values()]))


def seed_survival(runs, condition, control='trained'):
    """Per training seed: True if every clean evaluation world survived all 300 transitions."""
    seeds = {}
    for r in runs:
        if (r['condition'], r['control'], r['attack']) == (condition, control, False):
            seeds.setdefault(r['training_seed'], []).append(r['resident_survival'])
    return [all(v) for _, v in sorted(seeds.items())]


def learning_curves(study, condition):
    curves = {}
    for path in sorted((study/'checkpoints').glob(f'{condition}_seed_*.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        curves[meta['seed']] = [(h['environment_steps'], h['donations_per_step'], h['final_residents'])
                                for h in meta['learning_curve']]
    return curves


def figures(study, runs, analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = study/'figures'
    out.mkdir(exist_ok=True)
    blue, orange, grey = '#2a6fdb', '#e07b39', '#8a8f98'

    def band(ax, xs, rows, color, label):
        m = [r['mean'] if r['mean'] is not None else np.nan for r in rows]
        lo = [r['ci95'][0] if r['ci95'] else np.nan for r in rows]
        hi = [r['ci95'][1] if r['ci95'] else np.nan for r in rows]
        ax.plot(xs, m, 'o-', color=color, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=.18)

    xs = [BC_VALUE[k] for k in BC]
    s = analysis['summary']
    g = analysis['stratified_hamilton_gap']
    sv = analysis['survival_by_bc']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (('survival', 'Seeds surviving all evaluation worlds (Wilson 95%)'),
              ('donations_per_resident_step', 'Donations per resident step'),
              ('duration', 'Population lifetime (transitions, max 300)'),
              ('stratified_hamilton_gap', 'Hamilton gap within neighbourhood size\nP(give | rB>C neighbour) - P(give | none)'))
    series = (('inclusive', 'trained', blue, 'Inclusive-fitness PPO'), ('energy', 'trained', orange, 'Energy-only PPO'),
              ('inclusive', 'untrained_ancestors', grey, 'Untrained networks'))
    for ax, (metric, title) in zip(axes.flat, panels):
        for reward, control, color, label in series:
            if metric == 'survival':
                rows = [sv[f'{reward}_{k}/{control}'] for k in BC]
            elif metric == 'stratified_hamilton_gap':
                rows = [g[f'{reward}_{k}/{control}'] for k in BC]
            else:
                rows = [s[f'{reward}_{k}/{control}/clean'][metric] for k in BC]
            band(ax, xs, rows, color, label)
        if metric == 'survival':
            ax.set_ylim(-.05, 1.05)
        if metric == 'stratified_hamilton_gap':
            ax.axhline(0, color='k', lw=.6)
        ax.set_xlabel('B / C'); ax.set_xticks(xs); ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8, loc='center right')
    fig.tight_layout(); fig.savefig(out/'bc_sweep.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, condition, title in ((axes[0], 'energy_bc250', 'Energy-only PPO (B/C = 2.5)'),
                                 (axes[1], 'inclusive_bc250', 'Inclusive-fitness PPO (B/C = 2.5)')):
        for seed, curve in sorted(learning_curves(study, condition).items()):
            c = np.array(curve)[:-1]   # last rollout is a partial one after a reset
            smooth = np.convolve(c[:, 2], np.ones(8)/8, mode='valid')
            line, = ax.plot(c[7:, 0], smooth, label=f'seed {seed}', lw=1.6)
            ax.plot(c[:, 0], c[:, 2], color=line.get_color(), alpha=.15, lw=.8)
        ax.set_title(title); ax.set_xlabel('Environment transitions during training'); ax.grid(alpha=.3)
    axes[0].set_ylabel('Residents at end of each 64-step rollout\n(faint: raw, solid: 8-rollout mean)')
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'learning_curves.png', dpi=150); plt.close(fig)

    noise = analysis['cue_reliance']
    labels = ['0.00', '0.25', '0.50']
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, color, label in (('genetic', blue, 'Genetic cue r~ (std. logit)'), ('behavioural', orange, 'Behavioural history b (std. logit)')):
        band(ax, [0, .25, .5], [noise[c][key] for c in ('inclusive_bc250', 'inclusive_noise025', 'inclusive_noise050')], color, label)
    ax.axhline(0, color='k', lw=.6); ax.set_xticks([0, .25, .5], labels)
    ax.set_xlabel('Genetic-cue bit-flip probability'); ax.set_title('Recipient-choice reliance on each cue'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'cue_reliance.png', dpi=150); plt.close(fig)


CSV_METRICS = ('duration', 'final_residents', 'resident_survival', 'donations_per_resident_step', 'late_donation_rate',
               'give_rate', 'hamilton_gap', 'hamilton_consistent_fraction', 'recipient_true_relatedness', 'kin_preference',
               'adjacent_similarity', 'genetic_gate_weight', 'resident_births', 'mimics_injected',
               'post_invasion_donations', 'mimic_donation_share')


def write_csv(study, runs, protocol):
    """Spreadsheet copies of the raw evaluations: one row per episode, and one per seed (the inference unit)."""
    import csv
    def reward(c):
        spec = protocol['conditions'][c]
        if spec.get('learning', True) is False:
            return 'none (no learning)'
        if not spec['config'].get('inclusive_fitness'):
            return 'energy'
        return spec['config'].get('reward_variant', 'inclusive')
    bc = lambda c: protocol['conditions'][c]['config']['donation_benefit']/protocol['conditions'][c]['config'].get('donation_cost', 2.0)
    ordered = sorted(runs, key=lambda r: (r['condition'], r['control'], r['training_seed'], r['world_seed'], r['attack']))
    with (study/'evaluation_runs.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'reward', 'b_over_c', 'actor', 'training_seed', 'world_seed', 'mimic_attack', *CSV_METRICS])
        for r in ordered:
            w.writerow([r['condition'], reward(r['condition']), bc(r['condition']), r['control'], r['training_seed'],
                        r['world_seed'], r['attack'], *('' if r[m] is None else r[m] for m in CSV_METRICS)])
    groups = {}
    for r in ordered:
        groups.setdefault((r['condition'], r['control'], r['attack'], r['training_seed']), []).append(r)
    with (study/'seed_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'reward', 'b_over_c', 'actor', 'mimic_attack', 'training_seed', 'worlds',
                    'all_worlds_survived', *(f'mean_{m}' for m in CSV_METRICS if m != 'resident_survival')])
        for (condition, control, attack, seed), rs in groups.items():
            means = []
            for m in CSV_METRICS:
                if m == 'resident_survival':
                    continue
                vals = [x[m] for x in rs if x[m] is not None]
                means.append(float(np.mean(vals)) if vals else '')
            w.writerow([condition, reward(condition), bc(condition), control, attack, seed, len(rs),
                        all(x['resident_survival'] for x in rs), *means])


def fmt(row, digits=3):
    if row is None or row.get('mean') is None:
        return 'undefined'
    ci = '' if not row.get('ci95') else f" [{row['ci95'][0]:.{digits}f}, {row['ci95'][1]:.{digits}f}]"
    return f"{row['mean']:.{digits}f}{ci}"


def write_markdown(study, a):
    s = a['summary']
    L = ['# Hamilton\'s-rule study: results', '',
         f"Generated by `evaluation/hamilton_report.py`. {a['protocol']['environment_steps_per_run']} environment transitions per run, "
         f"{len(a['protocol']['training_seeds'])} training seeds, {a['protocol']['evaluation_worlds_per_seed']} fresh evaluation worlds per seed "
         f"(averaged within seed). Brackets: 95% Student-t interval over training seeds (n = {len(a['protocol']['training_seeds'])}). "
         'Bounded metrics can have intervals outside their range.', '',
         '## 1. B/C sweep (trained policies, clean worlds)', '',
         '| B/C | Reward | Donations / resident step | Lifetime (of 300) | Final residents | Hamilton gap | Hamilton-consistent gifts |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for k in BC:
        for reward in ('energy', 'inclusive'):
            r = s[f'{reward}_{k}/trained/clean']
            L.append(f"| {BC_VALUE[k]} | {reward} | {fmt(r['donations_per_resident_step'])} | {fmt(r['duration'],1)} | "
                     f"{fmt(r['final_residents'],1)} | {fmt(r['hamilton_gap'])} | {fmt(r['hamilton_consistent_fraction'])} |")
    L += ['', '### 1b. Seeds whose trained population survived every evaluation world (Wilson 95%)', '',
          '| B/C | Inclusive-fitness PPO | Energy-only PPO | Untrained networks (inclusive runs) |', '| --- | --- | --- | --- |']
    sv = a['survival_by_bc']
    count = lambda r: f"{round(r['mean']*r['n'])}/{r['n']} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
    for k in BC:
        L.append(f"| {BC_VALUE[k]} | {count(sv[f'inclusive_{k}/trained'])} | {count(sv[f'energy_{k}/trained'])} | "
                 f"{count(sv[f'inclusive_{k}/untrained_ancestors'])} |")
    L += ['', '## 2. Primary contrast: inclusive minus energy-only (paired by seed)', '',
          '| B/C | Metric | Difference | Seeds positive | Exact sign-flip p | Holm p |', '| --- | --- | --- | --- | --- | --- |']
    for key, c in a['primary_contrasts'].items():
        k, metric = key.split('|')
        L.append(f"| {BC_VALUE[k]} | {metric} | {fmt(c['difference'])} | {c['positive_seeds']}/{len(c['seeds'])} | "
                 f"{c['sign_flip_p']:.4f} | {a['primary_holm'][key]:.4f} |")
    n_pairs = len(a['protocol']['training_seeds'])
    floor = 2/2**n_pairs
    L += ['', f'With {n_pairs} pairs the smallest attainable two-sided sign-flip p is {floor:.4g}'
          + ('; no single contrast can reach p < 0.05.' if floor >= .05 else '.')
          + ' The pooled contrast below averages each seed over all five B/C levels.', '']
    pc = a['pooled_contrast']
    L += [f"Pooled over B/C (per-seed mean of the five levels): donation difference {fmt(pc['difference'])}, "
          f"{pc['positive_seeds']}/{len(pc['seeds'])} seeds positive, sign-flip p = {pc['sign_flip_p']:.4f}.", '',
          '### 2b. Stacked tests over all independently trained runs', '',
          f"Each (B/C, seed) training run is independent, so the {a['stacked_sign_test']['donations_per_resident_step']['pairs']} inclusive-minus-energy pairs can be stacked.", '',
          '| Metric | Pairs with inclusive > energy | Exact binomial sign test p |', '| --- | --- | --- |']
    for metric, t in a['stacked_sign_test'].items():
        L.append(f"| {metric} | {t['positive']}/{t['pairs']} | {t['sign_test_p']:.2e} |")
    L += ['', '| B/C levels | Inclusive runs surviving | Energy-only runs surviving | Fisher exact p |', '| --- | --- | --- | --- |']
    for label, t in a['survival_fisher'].items():
        L.append(f"| {label} | {t['inclusive']} | {t['energy']} | {t['fisher_p']:.2e} |")
    L += ['',
          '## 3. Hamilton regression (decision level, cluster bootstrap over seeds)', '',
          'logit P(choose j) = b0 + b1 r~ + b2 b + b3 (B/C) + b4 r~ x (B/C). Hamilton\'s rule predicts b4 > 0.', '',
          '| Reward | Coefficient | Estimate | 95% bootstrap CI | Bootstrap share > 0 |', '| --- | --- | --- | --- | --- |']
    for reward, reg in a['hamilton_regression'].items():
        if reg['status'] != 'fitted':
            L.append(f"| {reward} | - | {reg['status']} | | |")
            continue
        for name, c in reg['coefficients'].items():
            L.append(f"| {reward} | {name} | {c['estimate']:.3f} | [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] | {c['bootstrap_fraction_positive']:.2f} |")
    L += ['', 'This pooled model mixes the decision to give with the choice of recipient, and the raw Hamilton gap is '
          'confounded by neighbourhood size. The two analyses below remove each confound.', '',
          '### 3a. Recipient choice given a gift (conditional logit, cluster bootstrap over seeds)', '',
          'P(j | gift) proportional to exp(c1 r~ + c2 b + c3 E_j + c4 r~ x (B/C)). Hamilton\'s rule predicts c4 > 0.', '',
          '| Reward | Coefficient | Estimate | 95% bootstrap CI | Bootstrap share > 0 |', '| --- | --- | --- | --- | --- |']
    for reward, reg in a['hamilton_choice_regression'].items():
        if reg['status'] != 'fitted':
            L.append(f"| {reward} | - | {reg['status']} | | |")
            continue
        for name, c in reg['coefficients'].items():
            L.append(f"| {reward} | {name} | {c['estimate']:.3f} | [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] | {c['bootstrap_fraction_positive']:.2f} |")
    L += ['', '### 3b. Hamilton gap within neighbourhood size (Mantel-Haenszel, sampled opportunities)', '',
          'Untrained networks give more often when more neighbours are visible, so the raw gap is positive even for them. '
          'The stratified gap compares opportunities with the same number of neighbours.', '',
          '| B/C | Reward | Trained | Untrained networks | Trained - untrained (paired) | Seeds positive |', '| --- | --- | --- | --- | --- | --- |']
    for k in BC:
        for reward in ('energy', 'inclusive'):
            c = a['secondary_contrasts'][f'{reward} B/C {BC_VALUE[k]}: trained - untrained|stratified_hamilton_gap']
            g = a['stratified_hamilton_gap']
            L.append(f"| {BC_VALUE[k]} | {reward} | {fmt(g[f'{reward}_{k}/trained'])} | {fmt(g[f'{reward}_{k}/untrained_ancestors'])} | "
                     f"{fmt(c['difference'])} | {c['positive_seeds']}/{len(c['seeds'])} |")
    L += ['', '## 4. Cue reliability (inclusive reward, B/C = 2.5)', '',
          '| Cue noise | Genetic coefficient | Behavioural coefficient | Genetic share | Gate weight w^g | Donations / step | Lifetime |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for c, label in (('inclusive_bc250', '0.00'), ('inclusive_noise025', '0.25'), ('inclusive_noise050', '0.50')):
        cr, r = a['cue_reliance'][c], s[f'{c}/trained/clean']
        L.append(f"| {label} | {fmt(cr['genetic'])} | {fmt(cr['behavioural'])} | {fmt(cr['genetic_share'])} | "
                 f"{fmt(r['genetic_gate_weight'])} | {fmt(r['donations_per_resident_step'])} | {fmt(r['duration'],1)} |")
    L += ['', 'Coefficients are per-seed standardized conditional-logit weights of WHICH neighbour receives a gift; '
          'seeds with fewer than 20 sampled gifts are excluded (n shown in analysis.json).', '',
          '## 5. Genetic mimicry (six non-donating mimics with the modal resident genome, injected at t = 10)', '',
          '| Condition / actor | Mimic share of post-invasion gifts | Donations / step | Lifetime | Final residents |', '| --- | --- | --- | --- | --- |']
    for key in sorted(k for k in s if k.endswith('/attack')):
        r = s[key]
        L.append(f"| {key[:-7]} | {fmt(r['mimic_donation_share'])} | {fmt(r['donations_per_resident_step'])} | {fmt(r['duration'],1)} | {fmt(r['final_residents'],1)} |")
    L += ['', 'A mimic share is undefined when no post-invasion donations occurred; low shares without survival are not successful defence.', '',
          '## 6. Ablations, controls and assortment (clean worlds)', '',
          '| Condition / actor | Donations / step | Lifetime | Final residents | Hamilton gap | Adjacent chromosome similarity |', '| --- | --- | --- | --- | --- | --- |']
    for key in sorted(k for k in s if k.endswith('/clean') and not any(k.startswith(f'{r}_{b}/trained') for r in ('energy', 'inclusive') for b in BC)):
        r = s[key]
        L.append(f"| {key[:-6]} | {fmt(r['donations_per_resident_step'])} | {fmt(r['duration'],1)} | {fmt(r['final_residents'],1)} | "
                 f"{fmt(r['hamilton_gap'])} | {fmt(r['adjacent_similarity'])} |")
    L += ['', '## 7. Secondary paired contrasts', '', '| Contrast | Metric | Difference | Seeds positive | Sign-flip p |', '| --- | --- | --- | --- | --- |']
    for key, c in a['secondary_contrasts'].items():
        name, metric = key.split('|')
        L.append(f"| {name} | {metric} | {fmt(c['difference'])} | {c['positive_seeds']}/{len(c['seeds'])} | "
                 f"{c['sign_flip_p']:.4f} |" if c['sign_flip_p'] is not None else f"| {name} | {metric} | undefined | | |")
    L += ['', '## Spreadsheets', '', '[seed_summary.csv](seed_summary.csv): one row per trained model (or control) and seed, worlds averaged. '
          '[evaluation_runs.csv](evaluation_runs.csv): one row per evaluation episode.']
    L += ['', '## Figures', '', '![B/C sweep](figures/bc_sweep.png)', '', '![Learning curves](figures/learning_curves.png)', '',
          '![Cue reliance](figures/cue_reliance.png)', '']
    (study/'results.md').write_text('\n'.join(L), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', type=Path, default=ROOT/'results/hamilton_study')
    args = p.parse_args()
    study = args.study_dir
    protocol = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    runs, rows = load(study)
    summary = summarize(runs)
    primary, pvals = {}, {}
    for k in BC:
        for metric in ('donations_per_resident_step', 'duration'):
            c = paired(seed_means(runs, f'inclusive_{k}', metric=metric), seed_means(runs, f'energy_{k}', metric=metric))
            primary[f'{k}|{metric}'] = c
            pvals[f'{k}|{metric}'] = c['sign_flip_p']
    pooled = {}
    for reward in ('inclusive', 'energy'):
        per = [seed_means(runs, f'{reward}_{k}') for k in BC]
        pooled[reward] = {sd: float(np.mean([m[sd] for m in per])) for sd in set.intersection(*(set(m) for m in per))}
    secondary = {}
    def add(name, a_cond, b_cond, metric, a_ctrl='trained', b_ctrl='trained', attack=False):
        secondary[f'{name}|{metric}'] = paired(seed_means(runs, a_cond, a_ctrl, attack, metric), seed_means(runs, b_cond, b_ctrl, attack, metric))
    for metric in ('donations_per_resident_step', 'duration'):
        add('inclusive (gamma 0.9) - inclusive (gamma 0.99)', 'inclusive_bc250', 'inclusive_gamma099', metric)
        add('learned gate - fixed gate', 'inclusive_bc250', 'inclusive_fixed_gate', metric)
        add('attention - MLP', 'inclusive_bc250', 'inclusive_mlp', metric)
        add('inclusive PPO - evolution only', 'inclusive_bc250', 'evolution_only', metric)
        add('inclusive PPO - untrained networks', 'inclusive_bc250', 'inclusive_bc250', metric, b_ctrl='untrained_ancestors')
        add('energy PPO - untrained networks', 'energy_bc250', 'energy_bc250', metric, b_ctrl='untrained_ancestors')
        add('noise 0.5 - noise 0', 'inclusive_noise050', 'inclusive_bc250', metric)
    add('inclusive PPO - untrained networks', 'inclusive_bc250', 'inclusive_bc250', 'hamilton_gap', b_ctrl='untrained_ancestors')
    add('inclusive PPO - untrained networks', 'inclusive_bc250', 'inclusive_bc250', 'hamilton_consistent_fraction', b_ctrl='untrained_ancestors')
    add('local - random offspring placement', 'inclusive_bc250', 'inclusive_random_placement', 'adjacent_similarity')
    add('local - random placement (untrained)', 'inclusive_bc250', 'inclusive_random_placement', 'adjacent_similarity', 'untrained_ancestors', 'untrained_ancestors')
    for k in BC:
        for reward in ('inclusive', 'energy'):
            secondary[f'{reward} B/C {BC_VALUE[k]}: trained - untrained|stratified_hamilton_gap'] = paired(
                stratified_gap_by_seed(rows, f'{reward}_{k}'), stratified_gap_by_seed(rows, f'{reward}_{k}', 'untrained_ancestors'))
    add('learned gate - fixed gate (noise 0.5, mimics)', 'inclusive_noise050_mimics', 'inclusive_fixedgate_noise050_mimics', 'mimic_donation_share', attack=True)
    add('learned gate - fixed gate (noise 0.5, mimics)', 'inclusive_noise050_mimics', 'inclusive_fixedgate_noise050_mimics', 'duration', attack=True)
    # Every (B/C, seed) training run is independent, so the 25 inclusive-energy
    # pairs can be stacked: an exact binomial sign test, plus Fisher's exact test
    # on the number of runs whose population survived every evaluation world.
    from scipy.stats import binomtest, fisher_exact
    stacked = {}
    for metric in ('donations_per_resident_step', 'duration'):
        diffs = [d for k in BC for d in (lambda a, b: [a[s]-b[s] for s in sorted(set(a) & set(b))])(
            seed_means(runs, f'inclusive_{k}', metric=metric), seed_means(runs, f'energy_{k}', metric=metric))]
        nonzero = [d for d in diffs if d != 0]
        stacked[metric] = dict(pairs=len(diffs), positive=int(sum(d > 0 for d in diffs)),
                               sign_test_p=float(binomtest(sum(d > 0 for d in nonzero), len(nonzero), .5).pvalue))
    survival_tests = {}
    for label, levels in (('all B/C', BC), ('B/C >= 4', ('bc400', 'bc600'))):
        si = [x for k in levels for x in seed_survival(runs, f'inclusive_{k}')]
        se = [x for k in levels for x in seed_survival(runs, f'energy_{k}')]
        inc, ene = sum(si), sum(se)
        survival_tests[label] = dict(inclusive=f'{inc}/{len(si)}', energy=f'{ene}/{len(se)}',
                                     fisher_p=float(fisher_exact([[inc, len(si)-inc], [ene, len(se)-ene]])[1]))
    cue = {c: cue_reliance(rows, c) for c in ('inclusive_bc250', 'inclusive_noise025', 'inclusive_noise050')}
    g0, g5 = cue['inclusive_bc250']['per_seed'], cue['inclusive_noise050']['per_seed']
    secondary['noise 0.5 - noise 0|genetic_choice_coefficient'] = paired(
        {s: v['genetic'] for s, v in g5.items()}, {s: v['genetic'] for s, v in g0.items()})
    secondary['noise 0.5 - noise 0|genetic_share'] = paired(
        {s: v['genetic_share'] for s, v in g5.items()}, {s: v['genetic_share'] for s, v in g0.items()})
    analysis = dict(protocol=protocol, summary=summary, primary_contrasts=primary, primary_holm=holm(pvals),
                    stacked_sign_test=stacked, survival_fisher=survival_tests,
                    pooled_contrast=paired(pooled['inclusive'], pooled['energy']), secondary_contrasts=secondary,
                    hamilton_regression={r: hamilton_regression(rows, r) for r in ('inclusive', 'energy')},
                    hamilton_choice_regression={r: hamilton_choice_regression(rows, r) for r in ('inclusive', 'energy')},
                    stratified_hamilton_gap={f'{r}_{k}/{ctrl}': mean_ci(list(stratified_gap_by_seed(rows, f'{r}_{k}', ctrl).values()))
                                             for r in ('inclusive', 'energy') for k in BC for ctrl in ('trained', 'untrained_ancestors')},
                    survival_by_bc={f'{r}_{k}/{ctrl}': proportion_ci(seed_survival(runs, f'{r}_{k}', ctrl))
                                    for r in ('inclusive', 'energy') for k in BC for ctrl in ('trained', 'untrained_ancestors')},
                    cue_reliance=cue,
                    learning_curves={c: learning_curves(study, c) for c in ('energy_bc250', 'inclusive_bc250')},
                    runs=[{k: v for k, v in r.items() if k != 'curve'} for r in runs])
    (study/'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding='utf-8')
    figures(study, runs, analysis)
    write_csv(study, runs, protocol)
    write_markdown(study, analysis)
    print(f'Wrote {study/"results.md"}')


if __name__ == '__main__':
    main()
