"""Compare frozen evaluations from this branch and the pinned ten-seed study."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://github.com/chandak-Shubham/cs329_project/tree/7b591fa5e7b88127b899ef85c2c8bbf1b1a7d545'
TITLE = 'What changed with longer training?'
HEADERS = ['Study and reward', 'B/C', 'Training steps', 'Donations per resident step',
           'Mean final residents', 'Seeds surviving all 3 worlds']
CONTEXT = ('Each row uses frozen trained policies in three fresh worlds per training seed. '
           'We average worlds within seeds. Survival requires residents at step 300 in every world. '
           'The five-seed study uses a discount of 0.99 and weights changes in neighbours\' energy. '
           'The ten-seed study uses 0.90 and credits the donor for the recipient of its own gift. '
           'Only the two ten-seed shaped runs at B/C 2.5 isolate the training budget.')
INTERPRETATION = ('At B/C 2.5, quadrupling the ten-seed training budget lowered average giving '
                  'while more seeds survived all three worlds. The clearest survival gain came at '
                  'higher benefits under donor-specific reward shaping. These experiments support '
                  'a response to an explicit incentive. They leave spontaneous kin selection unproven.')

def comparison():
    mechanism = json.loads((ROOT/'results/mechanism_study/analysis.json').read_text())
    rows = []

    def add(label, ratio, steps, runs, seeds):
        blocks = defaultdict(dict)
        for r in runs:
            seed, world = int(r['training_seed']), int(r['world_seed'])
            if world in blocks[seed]:
                raise ValueError(f'Duplicate world for {label}')
            blocks[seed][world] = r
        if set(blocks) != set(seeds) or any(set(worlds) != {s+10000,s+11000,s+12000}
                                             for s, worlds in blocks.items()):
            raise ValueError(f'Incomplete seed or world coverage for {label}')
        groups = [worlds.values() for worlds in blocks.values()]
        def avg(key):
            return mean(mean(float(r[key]) for r in group) for group in groups)
        survivors = sum(all(int(r['final_residents']) > 0 for r in group) for group in groups)
        rows.append([label, str(ratio), f'{steps:,}', f'{avg("donations_per_resident_step"):.4f}',
                     f'{avg("final_residents"):.2f}', f'{survivors}/{len(groups)}'])

    seeds = [42,101,202,303,404]
    for condition, reward in [('energy_bc250','private'), ('inclusive_bc250','shaped')]:
        add('Five-seed / '+reward, 2.5, mechanism['budgets_by_condition'][condition],
            [r for r in mechanism['runs'] if r['condition']==condition and r['control']=='trained' and not r['attack']], seeds)
    sources = [
        ('hamilton_study','0b577d9aebb7c5d5411638ed56fee0dde7473679fbb904b0f8073f3b36db74ee',
         [('energy_bc250','private',2.5),('inclusive_bc250','shaped',2.5),
          ('energy_bc400','private',4),('inclusive_bc400','shaped',4),
          ('energy_bc600','private',6),('inclusive_bc600','shaped',6)]),
        ('hamilton_robustness','b935928fefa654c3fa45b182b718682caeb24a49fb5c140b4d5898533581af4c',
         [('b25_inclusive_long','shaped',2.5),('b4_prosocial','equal gift credit',4)])]
    for study, digest, conditions in sources:
        path = ROOT/'results/main_comparison'/f'{study}.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'World data differ from pinned source for {study}')
        with path.open(newline='') as stream:
            runs = list(csv.DictReader(stream))
        for condition, reward, ratio in conditions:
            add('Ten-seed / '+reward, ratio, 32768 if condition=='b25_inclusive_long' else 8192,
                [r for r in runs if r['condition']==condition and r['actor']=='trained' and r['mimic_attack']=='False'],
                seeds+[505,606,707,808,909])
    return sorted(rows, key=lambda row: (float(row[1]), int(row[2].replace(',',''))))

def markdown():
    lines = ['### '+TITLE, '', CONTEXT, '']
    for cells in [HEADERS, ['---']*len(HEADERS), *comparison()]:
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join([*lines,'',INTERPRETATION,'',f'[Ten-seed study, checkpoints and protocols]({SOURCE}).'])
