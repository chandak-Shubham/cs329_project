from copy import deepcopy
import numpy as np
import pytest
import torch

from environment.grid_world import GridWorld
from evaluation.complete_study import MeasurementPopulation
from evaluation.submission_report import contrast, holm, sign_flip, validate_analysis
from experiments.inherited_policies import LineagePopulation
from experiments.inheritance_study import evaluate


def test_world_replication_does_not_inflate_paired_n_or_flip_resolution():
    rows=[]
    for seed in range(5):
        for condition,value in [('a',.4),('b',.2)]:
            for world in range(3):
                rows.append(dict(condition=condition,training_seed=seed,world_seed=world,
                                 control='trained',attack=False,metric=value))
    result=contrast(rows,'a','b','metric')
    assert result['summary']['n'] == 5
    assert result['summary']['mean'] == pytest.approx(.2)
    assert sign_flip(result['differences']) == .0625
    assert sign_flip([0]*5) == 1.
    adjusted=holm([{'p_sign_flip':p} for p in (.01,.03,.04)])
    assert [r['p_holm'] for r in adjusted] == pytest.approx([.03,.06,.06])


def test_measurement_observer_does_not_change_actions_or_ecological_trajectory():
    torch.set_num_threads(1)
    config=dict(grid_size=6,initial_population=3,max_capacity=4,max_steps=8)
    env=GridWorld(**config,seed=71)
    env.reset(seed=71)
    torch.manual_seed(71)
    population=LineagePopulation(3)
    genomes=env.chromosomes[:3].copy()
    reference=evaluate(deepcopy(population),genomes,config,10071)
    measured=MeasurementPopulation(deepcopy(population),10071,True)
    observed=evaluate(measured,genomes,config,10071)
    assert reference == observed
    assert measured.decisions
    assert all(len(r)==8 for r in measured.decisions)
    assert np.isfinite(np.array(measured.decisions)).all()


def test_report_rejects_duplicated_worlds_and_missing_seed_blocks():
    rows=[dict(condition='clean',control='trained',attack=False,training_seed=s,
               world_seed=s+10000+k*1000) for s in (42,101) for k in range(3)]
    analysis=dict(training_seeds=[42,101],evaluation_worlds_per_seed=3,
                  budgets_by_condition={'clean':2048},runs=rows)
    validate_analysis(analysis)
    rows[-1]=rows[0]
    with pytest.raises(ValueError,match='Duplicate'):
        validate_analysis(analysis)
    rows.pop()
    with pytest.raises(ValueError,match='Incomplete'):
        validate_analysis(analysis)
