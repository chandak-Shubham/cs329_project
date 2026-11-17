"""Generate revised-study figures and a hash manifest from executed artifacts."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.revised_study import OUT, CONDITIONS
from evaluation.uncertainty import mean_ci


def main():
    result = json.loads((OUT / 'analysis.json').read_text())
    figure_dir = OUT / 'figures'
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained', sharex=True)
    for condition in ('energy_clean', 'demographic_clean', 'demographic_noise_mimics'):
        records = [json.loads(p.read_text()) for p in sorted((OUT / 'checkpoints').glob(f'{condition}_seed_*.json'))]
        if not records:
            continue
        steps = [r['environment_steps'] for r in records[0]['learning_curve']]
        for ax, key, ylabel in zip(axes, ('donations_per_step', 'mean_population'), ('Donations / transition', 'Mean living population')):
            curves = np.array([[row[key] for row in r['learning_curve']] for r in records])
            means = [mean_ci(values) for values in curves.T]
            ax.plot(steps, [m['mean'] for m in means], label=condition.replace('_', ' '))
            if len(records) > 1:
                ax.fill_between(steps, [m['ci95'][0] for m in means], [m['ci95'][1] for m in means], alpha=.14)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
    axes[0].set_title('Revised training: seed means and pointwise 95% t intervals')
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Environment transitions per training run')
    fig.savefig(figure_dir / 'learning_curves.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    names = list(result['common_input_summary'])
    x = np.arange(len(names))
    for offset, metric, label in ((-.18, 'genetic_tv', 'Genetic cue'), (.18, 'behavior_tv', 'Behaviour cue')):
        rows = [result['common_input_summary'][name][metric] for name in names]
        heights = np.array([r['mean'] for r in rows])
        errors = np.array([[r['mean']-r['ci95'][0], r['ci95'][1]-r['mean']] if r['ci95'] else [0, 0] for r in rows]).T
        ax.bar(x+offset, heights, width=.36, yerr=errors, capsize=3, label=label)
    ax.set_xticks(x, [name.replace('_', '\n') for name in names], fontsize=8)
    ax.set_ylabel('Mean total variation of action probabilities')
    ax.set_title('Cue permutation sensitivity on identical synthetic inputs\nExploratory attribution; seed means and 95% t intervals')
    ax.legend()
    fig.savefig(figure_dir / 'cue_sensitivity.png', dpi=160)
    plt.close(fig)

    manifest = {str(p.relative_to(OUT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ('.pt', '.json', '.md', '.png') and p.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Generated two figures; recorded hashes for {len(manifest)} revised artifacts.')


if __name__ == '__main__':
    main()
