"""Generate a single evidence-backed report, figures, and an integrity manifest.

python evaluation/submission_report.py
Requires both complete_study evaluations. No manually entered outcome numbers.
"""
from itertools import product
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.complete_study import digest, verify_checkpoint
from evaluation.uncertainty import mean_ci
from evaluation.final_report import interval, immediate_reward_probe
from evaluation.inheritance_report import common_probe
from experiments.inherited_policies import load
from experiments.mechanism_study import CONDITIONS, RATIOS

BASE = ROOT/'results/submission_study'
MECH = ROOT/'results/mechanism_study'


def seed_means(runs, condition, metric, control='trained'):
    seeds = sorted({r['training_seed'] for r in runs})
    values = {}
    for seed in seeds:
        sample = [r.get(metric) for r in runs if r['condition']==condition and r['training_seed']==seed
                  and r['control']==control and not r['attack'] and r.get(metric) is not None]
        if sample:
            values[seed] = float(np.mean(sample))
    return values


def contrast(runs, a, b, metric):
    left, right = seed_means(runs,a,metric), seed_means(runs,b,metric)
    keys = sorted(set(left)&set(right))
    differences = [left[k]-right[k] for k in keys]
    return dict(seeds=keys,differences=differences,summary=mean_ci(differences))


def sign_flip(differences):
    if not differences:
        return None
    threshold = abs(np.mean(differences))-1e-12
    return sum(abs(np.mean(np.array(s)*differences)) >= threshold
               for s in product((-1,1),repeat=len(differences)))/2**len(differences)


def holm(rows):
    floor = 0.
    for rank,row in enumerate(sorted(rows,key=lambda r:r['p_sign_flip'])):
        floor = max(floor,min(1.,(len(rows)-rank)*row['p_sign_flip']))
        row['p_holm'] = float(floor)
    return rows


def validate_analysis(analysis):
    """Reject duplicated worlds and missing independent training blocks."""
    seeds=analysis['training_seeds']
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('distinct nonempty training seeds required')
    identities=[(r['condition'],r['control'],r['attack'],r['training_seed'],r['world_seed'])
                for r in analysis['runs']]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate evaluation identity')
    expected={(s,s+10000+k*1000) for s in seeds
              for k in range(analysis['evaluation_worlds_per_seed'])}
    for c in analysis['budgets_by_condition']:
        actual={(r['training_seed'],r['world_seed']) for r in analysis['runs']
                if r['condition']==c and r['control']=='trained' and not r['attack']}
        if actual != expected:
            raise ValueError(f'Incomplete evaluation: {c}')


def save_decisions():
    source = MECH/'decisions.json'
    target = MECH/'hamilton_decisions.npz'
    if source.exists():
        payload = json.loads(source.read_text(encoding='utf-8'))
        names = sorted({r[0] for r in payload['rows']})
        array = np.asarray([[names.index(r[0]),*r[1:]] for r in payload['rows']],dtype=float)
        np.savez_compressed(target,data=array,conditions=np.array(names),columns=np.array(payload['columns']))
    with np.load(target,allow_pickle=False) as saved:
        return saved['data'],saved['conditions'].tolist()


def regressions(data,names):
    from evaluation.analysis import fit_logistic_regression
    results = {}
    for objective in ('energy','inclusive'):
        selected = np.isin(data[:,0],[i for i,n in enumerate(names) if n.startswith(objective+'_bc')])
        rows = data[selected]
        # Sample uniformly within seed over the retained, outcome-blind world
        # reservoirs. No recipient or world is treated as an independent run.
        rng = np.random.default_rng(9301)
        indices = []
        for seed in np.unique(rows[:,-2]):
            eligible = np.flatnonzero(rows[:,-2]==seed)
            indices.extend(rng.choice(eligible,min(12000,len(eligible)),replace=False))
        rows = rows[indices]
        x,y,groups = rows[:,1:8], rows[:,8], rows[:,9]
        if len(np.unique(y)) < 2 or np.linalg.matrix_rank(x) < x.shape[1]:
            results[objective] = dict(status='not identifiable',num_samples=len(rows))
            continue
        fit = fit_logistic_regression(x,y,['intercept','tag_cue','donation_history','B_over_C',
                                          'tag_x_B_over_C','available_neighbors','own_energy'],groups=groups)
        fit['status'] = 'exploratory fit' if fit['converged'] else 'optimizer did not converge'
        fit['ratios'] = sorted(np.unique(x[:,3]).tolist())
        results[objective] = fit
    return results


def plot_learning(root,conditions,target,title):
    fig,ax = plt.subplots(figsize=(9,4.6),layout='constrained')
    colors = plt.cm.tab10.colors
    for color,condition in zip(colors,conditions):
        records = [json.loads(p.read_text(encoding='utf-8'))
                   for p in sorted((root/'checkpoints').glob(f'{condition}_seed_*.json'))]
        budget = min(r['environment_steps'] for r in records)
        edges = np.linspace(0,budget,17)
        columns = []
        for lo,hi in zip(edges[:-1],edges[1:]):
            means = []
            for record in records:
                numerator=denominator=0.
                previous_active=0
                for row in record['learning_curve']:
                    active = row['active_agent_transitions']-previous_active
                    previous_active = row['active_agent_transitions']
                    overlap=max(0.,min(hi,row['environment_steps'])-
                                max(lo,row['environment_steps']-row['duration']))
                    fraction=overlap/row['duration']
                    numerator += row['donations_per_step']*row['duration']*fraction
                    denominator += active*fraction
                if denominator:
                    means.append(numerator/denominator)
            columns.append(mean_ci(means))
        x=(edges[:-1]+edges[1:])/2
        ax.plot(x,[r['mean'] for r in columns],label=condition,color=color)
        if all(r['ci95'] for r in columns):
            ax.fill_between(x,[r['ci95'][0] for r in columns],[r['ci95'][1] for r in columns],alpha=.14,color=color)
    ax.set(xlabel='Environment transitions per run',ylabel='Donations / resident transition',title=title)
    ax.axhline(.2,color='gray',ls=':',label='Basal-only replacement rate at B/C=2.5')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8,ncol=2)
    fig.savefig(target,dpi=170)
    plt.close(fig)


def plot_factorial(analysis,target):
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for objective in ('energy','inclusive'):
        rows=[analysis['summary'][f'{objective}_bc{int(r*100):03d}/trained/clean'] for r in RATIOS]
        for ax,metric,label in zip(axes,('donations_per_resident_step','final_residents'),
                                  ('Donations / resident transition','Final residents')):
            means=[r[metric]['mean'] for r in rows]
            errors=np.array([[v-r[metric]['ci95'][0],r[metric]['ci95'][1]-v] for v,r in zip(means,rows)]).T
            ax.errorbar(RATIOS,means,yerr=errors,marker='o',capsize=3,label=objective)
            ax.set(xlabel='Benefit/cost ratio B/C',ylabel=label)
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Matched reward x incentive study: five seed means and 95% t intervals')
    fig.savefig(target,dpi=170)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    base=json.loads((BASE/'analysis.json').read_text(encoding='utf-8'))
    mech=json.loads((MECH/'analysis.json').read_text(encoding='utf-8'))
    validate_analysis(base)
    validate_analysis(mech)
    if set(mech['budgets_by_condition']) != set(CONDITIONS):
        raise ValueError('Mechanism study incomplete')
    if len(base['budgets_by_condition']) != 19:
        raise ValueError('Original complete evaluation requires 19 conditions')
    audits={}
    for root,analysis in ((BASE,base),(MECH,mech)):
        expected={(c,s) for c in analysis['budgets_by_condition'] for s in analysis['training_seeds']}
        actual=set()
        for path in sorted((root/'checkpoints').glob('*.pt')):
            meta=verify_checkpoint(path)
            c=path.stem.rsplit('_seed_',1)[0]
            actual.add((c,meta['seed']))
            if meta['environment_steps'] != analysis['budgets_by_condition'][c]:
                raise ValueError(f'Wrong budget: {path}')
        if actual != expected:
            raise ValueError(f'Missing checkpoint coverage: {root}')
        audits[root.name]=dict(verified_checkpoints=len(actual),analysis_sha256=digest(root/'analysis.json'))
    figures=MECH/'figures'
    figures.mkdir(exist_ok=True)
    plot_learning(BASE,['clean','evolution_only','high_entropy'],figures/'original_learning.png',
                  'Does PPO preserve giving sustained without learning?')
    plot_learning(MECH,['energy_bc250','inclusive_bc250','evolution_local'],figures/'reward_learning.png',
                  'Matched reward control at B/C=2.5')
    plot_factorial(mech,figures/'reward_incentive.png')
    data,names=save_decisions()
    regression=regressions(data,names)
    reward_tests=[]
    for ratio in RATIOS:
        a,b=(f'{o}_bc{int(ratio*100):03d}' for o in ('inclusive','energy'))
        result=contrast(mech['runs'],a,b,'donations_per_resident_step')
        result.update(ratio=ratio,p_sign_flip=sign_flip(result['differences']))
        reward_tests.append(result)
    holm(reward_tests)
    spatial={}
    for a,b in (('evolution_local','evolution_random'),('energy_bc250','energy_random')):
        for metric in ('late_adjacent_similarity','late_global_similarity','late_excess_similarity',
                       'donations_per_resident_step','final_residents'):
            spatial[f'{a}_minus_{b}/{metric}']=contrast(mech['runs'],a,b,metric)
    evo=spatial['evolution_local_minus_evolution_random/donations_per_resident_step']
    ppo=spatial['energy_bc250_minus_energy_random/donations_per_resident_step']
    differences=[p-e for p,e in zip(ppo['differences'],evo['differences'])]
    interaction=mean_ci(differences)
    probes=[]
    for condition in ('clean','noise025','noise050','noise_mimics','fixed_noise_mimics'):
        for seed in base['training_seeds']:
            pop,_,_=load(BASE/'checkpoints'/f'{condition}_seed_{seed}.pt')
            probes.append(dict(condition=condition,training_seed=seed,**common_probe(pop)))
    cue_summary={c:{m:mean_ci([r[m] for r in probes if r['condition']==c])
                    for m in ('genetic_tv','behavior_tv','genetic_weight')}
                 for c in sorted({r['condition'] for r in probes})}
    objective_probe=immediate_reward_probe()
    supplemental=dict(integrity=audits,reward_contrasts=reward_tests,spatial_contrasts=spatial,
                      learning_placement_interaction=interaction,hamilton_regressions=regression,
                      common_input_probes=probes,common_input_summary=cue_summary,
                      immediate_reward_probe=objective_probe,
                      packages={n:version(n) for n in ('numpy','torch','scipy','matplotlib','pytest')})
    (MECH/'diagnostics.json').write_text(json.dumps(supplemental,indent=2,allow_nan=False),encoding='utf-8')
    clean=base['summary']['clean/trained/clean']
    evolution=base['summary']['evolution_only/trained/clean']
    energy=mech['summary']['energy_bc250/trained/clean']
    inclusive=mech['summary']['inclusive_bc250/trained/clean']
    budget=mech['environment_steps_per_run']
    worlds=mech['evaluation_worlds_per_seed']
    seeds=mech['training_seeds']
    lines=[
        '# Does private-reward learning preserve cooperation?', '',
        '## Inherited policies, cue reliability, and spatial assortment in a donation game', '',
        'CS329 Multi-Agent Systems | Group 21 | IIT Gandhinagar', '',
        'Vansh Barfiwala (24110380), Pratham Choksi (24110275), Shubham Chandak (24110335)', '',
        '## Abstract', '',
        'We investigate whether individual energy-reward PPO preserves cooperation supported by inherited policies and survivor selection, and whether incentive alignment, cue reliability, and spatial placement alter the outcome. '
        'We complete evaluation of 95 existing checkpoints (19 conditions, five seeds) and add 65 new training runs: a two-objective by five-incentive factorial and a learning by placement control. '
        f'The original clean PPO policies yield {clean["donations_per_resident_step"]["mean"]:.4f} donations per resident transition and {clean["final_residents"]["mean"]:.2f} final residents; '
        f'selection without PPO yields {evolution["donations_per_resident_step"]["mean"]:.4f} and {evolution["final_residents"]["mean"]:.2f}, respectively. '
        f'In the new matched B/C=2.5 experiment, private and marker-weighted rewards yield donation rates of {energy["donations_per_resident_step"]["mean"]:.4f} and {inclusive["donations_per_resident_step"]["mean"]:.4f}. '
        'We report uncertainty at the training-seed level. The experiments distinguish ecological cooperation, marker preference, reward shaping, and evolutionary kin selection; none is assumed equivalent.', '',
        '## 1. Computational question and success criteria', '',
        '**Primary question:** Does private energy optimization preserve, create, or suppress giving that an inherited-policy population sustains without PPO? '
        '**Mechanism questions:** Does adding marker-weighted social energy to the reward change this answer across five B/C ratios? Does local reproduction cause spatial assortment, and does that depend on learning? '
        '**Robustness question:** Do independently trained noise and mimic conditions change the full policy\'s dependence on genetic and behavioural information?', '',
        'Sustained cooperation requires survival through 300 evaluation transitions and a late donation rate at least 0.1 per resident transition, with giving on at least half of the final 100 transitions, in every expected seed/world. '
        'Cue adaptation additionally requires maintained cooperation and evidence of changed full-policy sensitivity; a gate weight alone is insufficient. '
        'Spatial assortment requires a matched local-minus-random difference among surviving populations and a comparison with contemporaneous global similarity. '
        'Hamilton-style discrimination is explored with an interaction regression and a direct reward intervention, not inferred from survival alone. These are declared operational criteria for this analysis, not a preregistered protocol.', '',
        '### Known and unknown information', '',
        '| Quantity | Information available to the actor |',
        '| --- | --- |',
        '| Own state | Energy/E_mitosis and age/1000 |',
        '| Neighbours | Up to eight nearest: relative position/radius, normalized energy, corrected genetic cue, donation history |',
        '| Incentives and rules | Fixed by each training environment; B/C is not a numerical policy input |',
        '| Hidden | True corruption probability, mimic identity, other networks and internal states |',
        '| Trainer only | Family membership for assigning transitions; full energy vector for the explicitly shaped reward control |', '',
        '## 2. One consistent model', '',
        'A partially observed Markov game runs on a toroidal grid. Each living resident chooses stay (action 0), one of four cardinal moves (1-4), or a donation to an observed neighbour (5-12). '
        'Stay means no move and no donation; there is no separate request/refusal game. A donation is legal only when the donor has strictly more than C energy. '
        'Movements and donations are mutually exclusive. Recipients are resolved from the decision-time neighbourhood; movement conflicts are resolved before metabolic crowding is charged.', '',
        '`Delta E_i = -c_base - c_density * |N_i(after movement)| - C * 1[give_i] + B * incoming_i`', '',
        'The energy reward is recorded before mitosis. Death at energy <= 0 terminates that individual. A parent with energy >= 60 can place a child in a free adjacent cardinal cell if capacity and a free tensor slot exist. '
        'The parent and child each receive half the pre-split energy. If no legal cell exists, reproduction is skipped. The child receives a fresh persistent identity, a mutated eight-bit chromosome, the family policy, and history prior 0.5. '
        'Donation creates B-C net energy; only the reproduction split conserves parent-plus-child energy.', '',
        '`h_ij = 1 - HammingDistance(c_i,c_j)/8`', '',
        '`r_tilde_ij = max(0, (h_ij - h_bar) / max(1-h_bar, 1e-6))`', '',
        'Here h is **similarity**, and h_bar is the distinct-active-pair population mean. This baseline correction uses population information computed by the simulator; the actor receives its result locally. '
        'The true chromosome determines assortment measurements; corrupted visible chromosomes determine the observed cue. Recent behaviour is the fraction of the neighbour\'s last ten actions that were donations **given**. '
        'The chromosome is a tag proxy and may be copied by mimics; it is not pedigree relatedness.', '',
        '### Policy ownership and evolutionary selection', '',
        'There are 24 independent founder networks and optimizers. Descendants share their founder family\'s network. PPO pools individual-return transitions within each family; unrelated founders never share optimizer updates. '
        'Individual returns terminate at death even if a newborn reuses the storage slot. At episode end, survivor restocking samples 24 living individuals with replacement and copies their genomes, networks, and optimizer states into independent next-generation founders. '
        'Thus selection weights families by surviving abundance. If all residents die, there is no winner; the last founder genomes and learned networks are retained for a new episode. Restocking reintroduces initial energy and is an explicit ecological intervention.', '',
        'This resolves the proposal\'s ambiguous use of "independent PPO" and "shared parameters": the submitted formulation is **independent across founders and shared within descendant families**. '
        'The outer selection loop extends the proposal. Shared weights alone do not put a recipient\'s benefit in the donor\'s PPO return; selection and learning are distinct processes.', '',
        '### Architecture and algorithm', '',
        'A contextual gate uses each neighbour\'s cues plus masked neighbourhood means and standard deviations. The gated affinity and raw cues enter a 32-dimensional encoder and two-head attention. '
        'Five movement/stay logits and eight targeted-donation logits are concatenated and normalized in **one masked 13-action categorical distribution**. A local critic predicts individual return. '
        'A decentralized critic matches the individual objective; variable population size does not itself preclude a centralized attention critic.', '',
        '```text',
        'Initialize 24 independent family networks and genomes.',
        'For each rollout:',
        '  Observe active identities; sample legal actions from their family policies.',
        '  Snapshot decision-time cues and identities; step movement, energy and demographics.',
        '  Assign terminal death credit to old identities; bind each child to its parent family.',
        '  Build individual-return GAE targets; update each family on its own samples.',
        '  At episode end, restock from living residents if any; record extinction otherwise.',
        'Freeze all policies for evaluation in fresh seeded worlds.',
        '```', '',
        '## 3. Experimental design', '',
        '| Parameter | Value |', '| --- | --- |',
        '| Grid / founders / capacity | 18 x 18 torus / 24 / 45 |',
        '| Initial energy / mitosis threshold | 30 / 60 |',
        '| Basal / per-neighbour drain | 0.6 / 0.08 within Chebyshev radius 3 |',
        '| Donation C / default B | 2 / 5 |',
        '| Neighbour limit / history / initial prior | 8 / 10 / 0.5 |',
        '| Chromosome / mutation | 8 bits / independent bit flips with probability 0.05 |',
        '| Episode / evaluation horizon | 300 transitions |',
        '| PPO rollout / epochs / minibatch | 64 transitions / 3 / 64 |',
        '| Learning rate / gamma / GAE lambda | 0.0003 / 0.99 / 0.95 |',
        '| Clip / value coefficient / entropy / gradient limit | 0.2 / 0.5 / 0.01 / 0.5 |',
        f'| New matched budget / seeds / worlds per seed | {budget} environment transitions / {seeds} / {worlds} |', '',
        'The original 19-condition study includes architecture, noise, attack, selection, mutation, placement, optimizer and budget controls. Most use 2,048 transitions; the longer run uses 8,192. '
        'We complete its frozen evaluation without pretending it was retrained. Original training used PyTorch 2.5.1+cu121 and NumPy 2.3.2; evaluation and all new training use the separately recorded CPU runtime. '
        'All claims about reward changes use newly trained matched conditions on the same runtime. Original results and new results are displayed separately.', '',
        'The new factorial crosses two objectives with B/C in {1, 1.5, 2.5, 3, 4.5}, holding C=2 and setting B=C*(B/C). '
        'Three extra conditions complete learning x offspring-placement comparisons at B/C=2.5: selection without PPO under local and random placement, and private-reward PPO with random placement. '
        'Each uses the same five seed blocks and three fresh worlds per checkpoint. Paired seeds reduce initialization variation; differing trajectories need not consume identical RNG draws.', '',
        'The **marker-weighted inclusive reward control** is', '',
        '`R_i = Delta E_i + sum_{j != i, acting at decision time} r_tilde_ij * Delta E_j`.', '',
        'This intentionally changes the objective and supplies privileged social reward information to training. It uses decision-time masks and cues, excluding diagonal self-credit and inactive slots; death or birth cannot redirect old-identity rewards. '
        'It is an incentive diagnostic, not biological inclusive fitness. The original ecology and pre-mitosis energy deltas remain unchanged. '
        'For an isolated donor-recipient intervention with all other actions fixed, the immediate shaped reward difference is `-C + r_tilde_ij*B`; its zero lies at `r_tilde=1/(B/C)`. '
        'Thresholds across the five ratios are 1, 0.6667, 0.4, 0.3333 and 0.2222. At B/C=1, strict rB>C is impossible for r<=1; that is an informative boundary condition.', '',
        '### Metrics and uncertainty', '',
        'We report donations divided by acting-resident transitions, final resident count, full-horizon survival, late-window giving, donation share to mimics, recipient preference relative to available neighbours, and spatial similarity. '
        'World means are averaged **within training seed** before Student t intervals or paired contrasts. The effective sample size is five seed blocks, not fifteen worlds, 24 family networks, or thousands of recipient decisions. '
        'Survival events use Wilson intervals across seeds. Small-sample mean intervals can exceed physical bounds; these are exploratory intervals, not guaranteed coverage under all distributions.', '',
        'Reward contrasts use exact paired sign-flip tests with Holm adjustment across five ratios. Five pairs give a minimum two-sided p of 0.0625, so these runs cannot establish conventional 5% significance with that test. '
        'Effect sizes and intervals are reported without equating a nonsignificant p with no effect. All-condition evaluation proceeds even after the original cooperation screen fails.', '',
        '## 4. Results: completed original study', '',
        f'Clean sustained-cooperation screen passed: **{base["clean_feasibility"]["passed"]}**.', '',
        '| Condition / control | Final residents (95% interval) | Donations / resident transition (95% interval) | All worlds survive (seed proportion, Wilson interval) |',
        '| --- | --- | --- | --- |',
    ]
    for name,row in base['summary'].items():
        if '/trained/clean' in name or name.startswith('clean/'):
            lines.append(f'| {name} | {interval(row["final_residents"])} | {interval(row["donations_per_resident_step"])} | {interval(row["all_worlds_survive"])} |')
    lines += ['', 'The rule baselines demonstrate feasibility under the matched ecology; they do not establish that the neural objective finds the same strategies. '
              'Dyadic Tit-for-tat remembers directed encounters, while public reputation uses an individual\'s public donation history. '
              'The untrained-ancestor comparator restores initial neural weights but retains the trained founder chromosomes and ancestry composition; the evolution-only condition is the separate selection-without-PPO control. '
              'The static fallback disables birth/death and forces persistence; its survival count is not ecological success.', '',
              '![Original donation learning curves](../results/mechanism_study/figures/original_learning.png)', '',
              'Curves are binned by actual environment transitions and weighted by resident exposure within each run; shaded regions are pointwise seed-level 95% t intervals. '
              'The dotted 0.2 line replaces basal drain only at B/C=2.5; crowding increases the required giving rate. It is not a learned decision threshold.', '',
              '## 5. Results: matched reward x benefit/cost experiment', '',
              '| Objective / B:C | Final residents (95% interval) | Donations / resident transition (95% interval) | Seed fraction surviving every world |',
              '| --- | --- | --- | --- |']
    for objective in ('energy','inclusive'):
        for ratio in RATIOS:
            name=f'{objective}_bc{int(ratio*100):03d}'
            row=mech['summary'][name+'/trained/clean']
            lines.append(f'| {objective} / {ratio:g} | {interval(row["final_residents"])} | {interval(row["donations_per_resident_step"])} | {interval(row["all_worlds_survive"])} |')
    lines += ['', '![Matched reward and incentive results](../results/mechanism_study/figures/reward_incentive.png)', '',
              '| B/C | Shaped minus private donation rate (paired 95% interval) | Exact paired p | Holm p |',
              '| --- | --- | --- | --- |']
    for row in reward_tests:
        lines.append(f'| {row["ratio"]:g} | {interval(row["summary"])} | {row["p_sign_flip"]:.4f} | {row["p_holm"]:.4f} |')
    default_effect=next(r for r in reward_tests if r['ratio']==2.5)
    direction='increases' if default_effect['summary']['mean'] > 0 else 'decreases'
    lines += ['', f'At the default B/C=2.5, the shaped objective {direction} mean giving relative to matched private PPO: '
              f'{interval(default_effect["summary"])}. The proportion of seed blocks surviving every fresh world is '
              f'{inclusive["all_worlds_survive"]["mean"]:.2f} for shaped reward and {energy["all_worlds_survive"]["mean"]:.2f} for private reward. '
              'This is a measured incentive-control result at the declared budget. It does not validate the proposal\'s claim that private-reward policy sharing inherently values kin. '
              'Giving more and sustaining a population are separate outcomes; neither establishes biological kin selection. '
              'The finite budget and coarse marker reward remain possible limits on what the optimizer can discover.', '',
              '![Matched reward learning curves](../results/mechanism_study/figures/reward_learning.png)', '',
              '## 6. Results: spatial structure where survival is possible', '',
              '| Condition | Final residents | Late adjacent similarity | Late global similarity | Late adjacent minus global similarity |',
              '| --- | --- | --- | --- | --- |']
    for c in ('evolution_local','evolution_random','energy_bc250','energy_random'):
        lines.append('| '+c+' | '+' | '.join(interval(mean_ci(list(seed_means(mech['runs'],c,m).values())))
                     for m in ('final_residents','late_adjacent_similarity','late_global_similarity','late_excess_similarity'))+' |')
    lines += ['', '| Local minus random placement | Paired late excess similarity | Paired donation rate |',
              '| --- | --- | --- |']
    for a,b in (('evolution_local','evolution_random'),('energy_bc250','energy_random')):
        lines.append(f'| {a} vs {b} | {interval(spatial[f"{a}_minus_{b}/late_excess_similarity"]["summary"])} | {interval(spatial[f"{a}_minus_{b}/donations_per_resident_step"]["summary"])} |')
    lines += ['', f'The learning-by-placement interaction in donation rate, `(PPO local - PPO random) - (evolution local - evolution random)`, is {interval(interaction)}.', '',
              'Late statistics use the final 100 transitions and are undefined after extinction. Their n therefore reports the available surviving seed means; a paired spatial comparison requires defined measurements in both placements. '
              'The excess over global similarity controls for population-wide tag homogenization. This is spatial marker assortment, not a pedigree coefficient or a demonstrated evolutionary fitness advantage. '
              'The no-PPO comparison prevents an extinct learned population from making the spatial question empty.', '',
              '## 7. Cue reliability and mimicry', '',
              '| Original training condition | Genetic cue TV sensitivity | Behaviour cue TV sensitivity | Genetic gate weight |',
              '| --- | --- | --- | --- |']
    for c,row in cue_summary.items():
        lines.append('| '+c+' | '+' | '.join(interval(row[m]) for m in ('genetic_tv','behavior_tv','genetic_weight'))+' |')
    lines += ['', 'TV is total variation between the complete action distributions on identical synthetic inputs before and after permuting one cue. '
              'Raw cues also enter the encoder, so gate weights cannot identify reliance. These probes cover both paths but can leave the realistic observation manifold; they are descriptive sensitivity tests.', '',
              '| Attacked condition | Donation share to mimics | Post-invasion donation count |', '| --- | --- | --- |']
    for c in ('noise_mimics','fixed_noise_mimics'):
        row=base['summary'][c+'/trained/attack']['mimic_donation_share']
        total=sum(r['post_invasion_donations'] for r in base['runs'] if r['condition']==c and r['control']=='trained' and r['attack'])
        lines.append(f'| {c} | {interval(row)} | {total} |')
    lines += ['', 'Mimics enter at transition 10, copy the most common resident tag, remain stationary, and never donate. '
              'Mimic donation shares condition on actual post-invasion giving; no giving yields an undefined share, never successful defense. '
              'Because clean cooperation is a prerequisite, changes under noise or mimicry cannot be called successful adaptive trust when the learned ecology collapses.', '',
              '## 8. Hamilton association and objective intervention', '',
              '`logit P(give to j) = beta0 + beta_r*r + beta_b*b + beta_q*(B/C) + beta_rq*r*(B/C) + beta_n*available_neighbors + beta_E*own_energy`', '',
              'The corrected model includes the missing tag-by-incentive interaction and five independently trained incentive levels. '
              'It records only legal recipient opportunities, includes competition among available targets, and fits each objective separately. '
              'Uniform outcome-blind reservoirs retain at most 4,000 opportunities per world, followed by a cap of 12,000 per seed for fitting. '
              'Recipient decisions are dependent; cluster sandwich intervals use the five training-seed blocks. This small-cluster regression is exploratory and does not estimate causal pedigree fitness. '
              'B/C is a training condition, not an actor input, so coefficients cannot be read as an instantaneous actor response to an observed B/C value.', '',
              '| Objective | Fit status | Tag x B/C coefficient and cluster interval | Pseudo-R2 | Seed clusters / opportunities |',
              '| --- | --- | --- | --- | --- |']
    for o,fit in regression.items():
        if 'coefficients' in fit:
            i=fit['features'].index('tag_x_B_over_C')
            lines.append(f'| {o} | {fit["status"]} | {fit["coefficients"][i]:.4f} [{fit["ci95"][i][0]:.4f}, {fit["ci95"][i][1]:.4f}] | {fit["pseudo_r2"]:.4f} | {fit["num_clusters"]} / {fit["num_samples"]} |')
        else:
            lines.append(f'| {o} | {fit["status"]} | undefined | undefined | {fit["num_samples"]} opportunities |')
    lines += ['', f'The matched one-transition environment intervention gives donor reward difference {objective_probe["donor_reward_difference"]:.4f} and recipient difference {objective_probe["recipient_reward_difference"]:.4f}. '
              'Thus private reward immediately prices giving at -C; the shaped two-agent control prices it at -C+rB. '
              'This directly tests reward plumbing. It is separate from whether learned long-horizon actions follow that threshold, and neither test validates biological Hamilton\'s rule.', '',
              '## 9. Interpretation and limitations', '',
              'Individual PPO and evolutionary selection act on different signals. Giving\'s immediate private cost is assigned to the donor, while another resident\'s benefit is not directly in its return. '
              'Delayed reciprocity can still make giving privately beneficial, but it must be discovered under partial observations, moving neighbours, terminal death, and nonstationary family policies. '
              'The negative learning comparison is scientifically informative: ecological feasibility does not imply that a private-return optimizer learns a cooperative rule. '
              'Untrained policies have many possible donation actions, so their survival is not learned altruism. Selection without PPO is likewise not proof of kin discrimination.', '',
              'Summing pre-demographic energy changes yields `(B-C)*D - c_base*N - c_density*sum_i |N_i|`, where D is the donation count. '
              'At the default B-C=3, fewer than 0.2 donations per resident transition cannot replace even basal energy loss. Crowding raises that threshold. '
              'The plotted collapse in giving therefore has a concrete ecological consequence, but this energy balance alone cannot identify why PPO chose those actions.', '',
              'The main limits are five seeds, short finite training budgets, one base ecology, eight-bit marker proxies, restocking energy, and a privileged shaped reward. '
              'The original one-factor controls and new reward/incentive factorial do not span all interactions. '
              'Noise/mimic effects are measured under trajectories that may go extinct; late assortment is reported only where defined. '
              'Pointwise learning intervals are not simultaneous bands. Regression samples are dependent and small-cluster intervals can be unreliable. '
              'Evaluation on a newer CPU runtime may alter sampled trajectories even with the same seed; the original training provenance and replay runtime are both retained. '
              'No finite negative experiment proves that cooperation is impossible, and a successful explicitly shaped reward would not establish spontaneous kin selection.', '',
              'The browser simulation still runs the earlier single shared-policy checkpoint; it is an illustration, not an execution or replay of the new inherited-family experiment. '
              'The report, raw world results, and saved policies are the evidence for the submission. Historical demographic and descendant-reward variants are retained separately and are not pooled into these results.', '',
              '## 10. Reproducibility', '',
              f'We verify {audits["submission_study"]["verified_checkpoints"]} original and {audits["mechanism_study"]["verified_checkpoints"]} new checkpoint files against their recorded SHA256 values and original archived training sources. '
              'Actor and environment source must match the trained architecture/ecology byte for byte; evaluation versions and source hashes are recorded separately. '
              'Archive line endings are preserved in Git so Windows checkout cannot silently corrupt source hashes. '
              'Original checkpoint metadata and claimed training versions are never rewritten to pass a replay check.', '',
              '```powershell',
              'python -m pip install -r requirements.txt',
              'python experiments/mechanism_study.py --steps 2048 --workers 3',
              'python evaluation/complete_study.py --study-dir results/submission_study --workers 3',
              'python evaluation/complete_study.py --study-dir results/mechanism_study --workers 3',
              'python evaluation/submission_report.py',
              'python -m pytest -q',
              'node --test tests/web_simulation.test.cjs',
              'python -m pip install -r requirements-report.txt',
              'python evaluation/export_submission.py',
              '```', '',
              'New training refuses to overwrite incompatible files; matching completed runs are checked before reuse. '
              'Training and evaluation run in separate worker processes to isolate PyTorch RNG state. '
              'The full provenance manifest includes checkpoints, source archives, summaries, diagnostics, and figures. Exact package versions for this execution are recorded alongside each analysis.', '',
              '- [Original study results](../results/submission_study/results.md) and [raw worlds](../results/submission_study/analysis.json).',
              '- [New study results](../results/mechanism_study/results.md), [raw worlds](../results/mechanism_study/analysis.json), and [diagnostics](../results/mechanism_study/diagnostics.json).',
              '- [Complete model and execution protocol](final_study_protocol.md).', '',
              '## 11. Proposal corrections and scope', '',
              'The final study replaces the unproven title claim of emergent kin selection with a testable cooperation question. '
              'It resolves independent versus shared PPO, specifies all parameters, fixes the free-slot rule, defines affordability and stay, normalizes all actions jointly, and separates donation energy creation from mitosis conservation. '
              'It measures full-policy cue sensitivity, uses a tag-by-B/C interaction, tests five incentive levels, and measures late assortment in surviving selection-only populations. '
              'The privileged reward control is explicitly outside the original private-energy objective. These changes make the mechanism and the scope of the conclusions reviewable.', '',
              'The project matches the course sequence: decision problem, model, algorithm, implementation, quantitative evaluation, and limitations. '
              'The rubric does not require a positive hypothesis result; the contribution is a reproducible investigation of when private learning and survivor selection diverge.', '',
              '## Team responsibilities and background', '',
              'The proposal assigns environment and demographics to Vansh Barfiwala, cues and attention to Pratham Choksi, and PPO and evaluation to Shubham Chandak. '
              'These are the team\'s stated areas of responsibility.', '',
              'Biological background: Hamilton (1964), The genetical evolution of social behaviour, Parts I and II; Dawkins (1976), The Selfish Gene. '
              'MARL background from the proposal: Leibo et al. (2017), Multi-agent reinforcement learning in sequential social dilemmas; '
              'Hughes et al. (2018), Inequity aversion improves cooperation in intertemporal social dilemmas; '
              'Jaques et al. (2019), Social influence as intrinsic motivation for multi-agent deep reinforcement learning; '
              'Lupu and Precup (2020), Gifting in multi-agent reinforcement learning. '
              'Dawkins supplies biological context and is not cited as MARL research.', '']
    report=ROOT/'docs/final_report.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    for root in (BASE,MECH):
        archive=root/'report_source/evaluation'
        archive.mkdir(parents=True,exist_ok=True)
        for name in ('complete_study.py','submission_report.py','analysis.py','uncertainty.py','inheritance_report.py','final_report.py'):
            (archive/name).write_bytes((ROOT/'evaluation'/name).read_bytes())
        manifest={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*')
                  if p.is_file() and p.suffix not in ('.log','.pyc') and p.name not in ('manifest.json','decisions.json')
                  and 'evaluations' not in p.relative_to(root).parts}
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Wrote {report} using {sum(v["verified_checkpoints"] for v in audits.values())} verified checkpoints')


if __name__ == '__main__':
    main()
