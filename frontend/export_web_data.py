"""Refresh the data that index.html embeds; the page itself never hard-codes results.

python frontend/export_web_data.py           # rewrite index.html
python frontend/export_web_data.py --check   # fail if index.html is stale

The page loads trained checkpoints (.pt files) directly from models/ and runs
them in the browser. This script embeds only what the page needs to do that:
the list of offered checkpoints with their SHA-256 (verified here against each
sidecar), the tensor layout inside a .pt file, the results the Python
evaluation recorded for each checkpoint (so the browser can compare), and a
research summary generated from the analysis files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from html import escape
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from frontend.checkpoint_layout import layout  # noqa: E402

MAIN = 'study_b_hamilton'      # folder name under models/, results/ and plots/
ROBUST = 'study_c_robustness'
UP = '../'  # the page is frontend/index.html; links and checkpoint URLs are relative to it
SEEDS = (42, 101, 202, 303, 404, 505, 606, 707, 808, 909)
MIMICS = dict(invasionStep=10, invasionCount=6)   # experiments/inherited_policies.py, hamilton_study.evaluate_world
LEVELS = (('bc100', 1.0), ('bc150', 1.5), ('bc250', 2.5), ('bc400', 4.0), ('bc600', 6.0))

# (group, study, condition, label). Attention networks in ecologies the browser
# engine implements exactly (no cue noise, standard mutation and placement).
OFFERED = [('Inclusive-fitness reward (Study B)', MAIN, f'inclusive_{k}', f'Inclusive reward · B/C = {bc:g}') for k, bc in LEVELS] + \
          [('Energy-only reward (Study B)', MAIN, f'energy_{k}', f'Energy-only reward · B/C = {bc:g}') for k, bc in LEVELS] + [
    ('Mimic attack (Study B)', MAIN, 'inclusive_mimics', 'Inclusive reward, trained with mimics · B/C = 2.5'),
    ('Reward controls at B/C = 4 (Study C)', ROBUST, 'b4_prosocial', 'Prosocial reward (+B for any gift) · B/C = 4'),
    ('Reward controls at B/C = 4 (Study C)', ROBUST, 'b4_kinblind', 'Kin-blind reward (mean neighbour r) · B/C = 4'),
    ('Mimic attack at B/C = 4 (Study C)', ROBUST, 'b4_inclusive_mimics', 'Inclusive reward, trained with mimics · B/C = 4'),
    ('Mimic attack at B/C = 4 (Study C)', ROBUST, 'b4_prosocial_mimics', 'Prosocial reward, trained with mimics · B/C = 4'),
    ('Mimic attack at B/C = 4 (Study C)', ROBUST, 'b4_kinblind_mimics', 'Kin-blind reward, trained with mimics · B/C = 4'),
]
DEFAULT = 'inclusive_bc400'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def recorded(study: str, condition: str, seed: int):
    """What the Python evaluation recorded for this checkpoint's trained population."""
    runs = load_json(ROOT/'results'/study/'evaluations'/f'{condition}_seed_{seed}.json')
    return [dict(world=r['world_seed'], attack=r['attack'], survival=r['resident_survival'], duration=r['duration'],
                 finalResidents=r['final_residents'], donationsPerResidentStep=round(r['donations_per_resident_step'], 6),
                 injected=r['mimics_injected'], mimicShare=None if r['mimic_donation_share'] is None else round(r['mimic_donation_share'], 6))
            for r in runs if r['control'] == 'trained']


def models():
    shared, entries = None, []
    for group, study, condition, label in OFFERED:
        env = None
        for seed in SEEDS:
            path = ROOT/'models'/study/f'{condition}_seed_{seed}.pt'
            meta = load_json(path.with_suffix('.json'))
            digest = sha256(path)
            if digest != meta['checkpoint_sha256']:
                raise ValueError(f'{path} differs from the SHA-256 in its sidecar')
            if meta['architecture'] != 'attention':
                raise ValueError(f'{path}: the browser implements the attention network only')
            this = layout(path)
            shared = shared or this
            if this != shared:
                raise ValueError(f'{path} stores its tensors differently from the other checkpoints')
            env = meta['environment']
            entries.append(dict(id=f'{condition}_seed_{seed}', condition=condition, seed=seed, sha256=digest,
                                path=path.relative_to(ROOT).as_posix(), recorded=recorded(study, condition, seed)))
        # The browser engine's defaults (index.html SimulationEngine) must be the training ecology.
        browser = dict(grid_size=18, initial_population=24, initial_energy=30.0, e_mitosis=60.0, mutation_rate=.05,
                       c_base=.6, c_density=.08, max_steps=300)
        if (env.get('cue_noise', 0) or env.get('offspring_placement', 'adjacent') != 'adjacent'
                or any(env[k] != v for k, v in browser.items())):
            raise ValueError(f'{condition}: ecology not implemented by the browser engine')
        reserve = int(env.get('mimic_reserve', 0))
        entries_cond = dict(condition=condition, group=group, label=label, study=f'results/{study}',
                            B=env['donation_benefit'], C=env['donation_cost'], reward=env.get('reward_variant', 'inclusive') if env['inclusive_fitness'] else 'energy',
                            attack=bool(meta['invasion']), maxCapacity=env['max_capacity'], mimicReserve=reserve,
                            trainingSteps=meta['environment_steps'], gamma=meta['ppo']['gamma'])
        entries.append(dict(conditionInfo=entries_cond))
    conditions = [e['conditionInfo'] for e in entries if 'conditionInfo' in e]
    checkpoints = {e['id']: e for e in entries if 'id' in e}
    return dict(layout={**shared, 'dModel': 32, 'numHeads': 2}, conditions=conditions, checkpoints=checkpoints,
                seeds=list(SEEDS), default=f'{DEFAULT}_seed_42', mimics=MIMICS)


# ----------------------------------------------------------------------------
# Research summary, generated from results/*/analysis.json
# ----------------------------------------------------------------------------

def ci(row, digits=3, pct=False):
    if not row or row.get('mean') is None:
        return 'n/a'
    f = (lambda v: f'{100*v:.0f}%') if pct else (lambda v: f'{v:.{digits}f}')
    lo, hi = row['ci95'] if row.get('ci95') else (None, None)
    return f'{f(row["mean"])} [{f(lo)}, {f(hi)}]' if lo is not None else f(row['mean'])


def count(row):
    return f"{round(row['mean']*row['n'])}/{row['n']}"


def pfmt(p):
    return f'{p:.2g}' if p >= .001 else f'{p:.1e}'


def table(caption, headers, rows):
    head = ''.join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows)
    return f'<div class="research-table"><table><caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def research(main: dict, robust: dict | None) -> str:
    p = main['protocol']
    s, surv = main['summary'], main['survival_by_bc']
    fisher = main['survival_fisher']
    rows = []
    for k, bc in LEVELS:
        rows.append([f'{bc:g}', count(surv[f'inclusive_{k}/trained']), count(surv[f'energy_{k}/trained']),
                     count(surv[f'inclusive_{k}/untrained_ancestors']),
                     ci(s[f'inclusive_{k}/trained/clean']['donations_per_resident_step']),
                     ci(s[f'energy_{k}/trained/clean']['donations_per_resident_step']),
                     ci(s[f'inclusive_{k}/untrained_ancestors/clean']['donations_per_resident_step'])])
    out = ['<section id="research-results" aria-labelledby="research-heading"><h2 id="research-heading">Research results</h2>',
           '<p class="fineprint">Every number in this section is generated by <code>frontend/export_web_data.py</code> from '
           f'<a href="{UP}results/{MAIN}/analysis.json">results/{MAIN}/analysis.json</a> and <a href="{UP}results/{ROBUST}/analysis.json">results/{ROBUST}/analysis.json</a>. '
           'To check them against the models yourself, use <a href="#reproduce">Reproduce a result</a> above, which reruns the saved checkpoints in your browser.</p>',
           '<h3>Study B: does learning need kinship?</h3>',
           f'<p>{len(p["conditions"])} conditions × {len(p["training_seeds"])} independent training seeds, {p["environment_steps_per_run"]:,} transitions per run, '
           f'{p["evaluation_worlds_per_seed"]} fresh evaluation worlds per seed. A seed counts as surviving only if residents are alive at step 300 in all of its worlds.</p>',
           table('Seeds whose population survives, and donations per resident-step [95% CI over seeds]',
                 ['B/C', 'Survive: inclusive', 'Survive: energy-only', 'Survive: untrained', 'Donations: inclusive', 'Donations: energy-only', 'Donations: untrained'], rows),
           f'<p>Survival over all B/C levels: inclusive {fisher["all B/C"]["inclusive"]} versus energy-only {fisher["all B/C"]["energy"]} '
           f'(Fisher exact p = {pfmt(fisher["all B/C"]["fisher_p"])}); at B/C ≥ 4: {fisher["B/C >= 4"]["inclusive"]} versus {fisher["B/C >= 4"]["energy"]} '
           f'(p = {pfmt(fisher["B/C >= 4"]["fisher_p"])}). At B/C ≥ 2.5 untrained networks with the same founders also survive, so learning with the energy-only reward '
           '<em>destroys</em> cooperation that was already present; the inclusive reward preserves it above the threshold and loses it below.</p>']
    if robust:
        rs, rf = robust['summary'], robust['survival_fisher']
        keys = [('inclusive_bc400', 'Inclusive (kin-weighted)'), ('b4_prosocial', 'Prosocial (+B for any gift)'),
                ('b4_kinblind', 'Kin-blind (mean neighbour r)'), ('energy_bc400', 'Energy-only')]
        rrows = []
        for c, label in keys:
            row = rs[f'{c}/trained/clean']
            rrows.append([label, count(row['survival']), ci(row['donations_per_resident_step']),
                          ci(row.get('hamilton_consistent_fraction')), ci(row.get('kin_preference'))])
        out += ['<h3>Study C: is it kin information, or any credit for giving?</h3>',
                table('Reward controls at B/C = 4 (10 seeds each)', ['Reward', 'Seeds surviving', 'Donations / resident-step',
                                                                    'Gifts with rB > C', 'Kin preference of gifts'], rrows)]
        c = robust['contrasts']
        def line(key, what):
            r = c[key]
            return f'{what} {r["difference"]["mean"]:+.3f} ({r["positive_seeds"]}/{len(r["seeds"])} seeds, sign-flip p = {pfmt(r["sign_flip_p"])})'
        out.append('<p>Crediting every gift keeps populations alive at least as well as kin-weighting does, so survival alone does not need kin information. '
                   'What kin information changes is <em>whom</em> agents help: ' +
                   line('inclusive - kin-blind|hamilton_consistent_fraction', 'inclusive minus kin-blind, share of gifts with rB > C:') + '; ' +
                   line('inclusive - kin-blind|kin_preference', 'kin preference:') + '.</p>')
    out += ['<div class="research-figures">',
            f'<figure><img src="{UP}plots/{MAIN}/bc_sweep.png" loading="lazy" alt="Survival, donations, lifetime and Hamilton gap across five benefit-to-cost ratios"><figcaption>Study B across five B/C levels.</figcaption></figure>',
            f'<figure><img src="{UP}plots/{MAIN}/learning_curves.png" loading="lazy" alt="Population size during training under both rewards"><figcaption>Energy-only populations die out during training.</figcaption></figure>']
    if robust:
        out.append(f'<figure><img src="{UP}plots/{ROBUST}/reward_controls.png" loading="lazy" alt="Reward controls at B/C = 4"><figcaption>Study C reward controls.</figcaption></figure>')
    out += ['</div>',
            f'<p><a href="{UP}report/final_report.pdf">Final report (PDF)</a> · <a href="{UP}results/{MAIN}/results.md">Study B tables</a> · '
            f'<a href="{UP}results/{ROBUST}/results.md">Study C tables</a> · <a href="{UP}results/{MAIN}/seed_summary.csv">One row per trained model (CSV)</a> · <a href="{UP}README.md">Reproduce</a></p></section>']
    return '\n'.join(out)


def build_payload() -> dict:
    return dict(models=models())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target = ROOT/'frontend/index.html'
    html = target.read_text(encoding='utf-8')
    payload = json.dumps(build_payload(), separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    updated, n = re.subn(r'(<script id="research-data" type="application/json">).*?(</script>)',
                         lambda m: m[1] + payload + m[2], html, flags=re.S)
    if n != 1:
        raise ValueError('Expected exactly one research-data script in index.html')
    robust_path = ROOT/'results'/ROBUST/'analysis.json'
    section = research(load_json(ROOT/'results'/MAIN/'analysis.json'), load_json(robust_path) if robust_path.exists() else None)
    updated, n = re.subn(r'(?<=<!-- RESEARCH_RESULTS_START -->).*?(?=<!-- RESEARCH_RESULTS_END -->)',
                         lambda _: '\n' + section + '\n', updated, flags=re.S)
    if n != 1:
        raise ValueError('Expected exactly one research results section')
    if args.check:
        if updated != html:
            raise SystemExit('frontend/index.html is stale. Run python frontend/export_web_data.py')
        print('index.html matches the checkpoints and analysis files.')
    else:
        target.write_text(updated, encoding='utf-8', newline='\n')
        print(f'Wrote {target.name} ({len(updated.encode("utf-8")):,} bytes).')


if __name__ == '__main__':
    main()
