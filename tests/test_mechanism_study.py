import numpy as np
import pytest
from experiments.mechanism_study import shaped_reward, train, CONDITIONS, RATIOS
from evaluation.complete_study import verify_checkpoint


def test_two_agent_intervention_has_hamilton_threshold_without_rewarding_newborn_slots():
    # Stay vs give: only the donor and recipient energy increments change.
    r = np.array([[0,.5,1],[.5,0,1],[1,1,0]],dtype=np.float32)
    acting = np.array([True,True,False])
    stay = np.array([-.6,-.6,0])
    give = stay+np.array([-2,5,1000])  # inactive garbage cannot affect credit
    delta = shaped_reward(give,r,acting,'inclusive')-shaped_reward(stay,r,acting,'inclusive')
    assert delta[0] == pytest.approx(-2+.5*5)
    assert delta[2] == 0
    assert shaped_reward(give,r,acting,'energy')[0] == pytest.approx(give[0])


def test_reward_incentive_factorial_has_matched_five_levels():
    assert len(RATIOS) == 5
    for ratio in RATIOS:
        for objective in ('energy','inclusive'):
            spec = CONDITIONS[f'{objective}_bc{int(ratio*100):03d}']
            assert spec == dict(objective=objective,ratio=ratio)


def test_new_objective_checkpoint_is_labelled_and_integrity_checked(tmp_path):
    path = tmp_path/'checkpoints'/'inclusive_bc250_seed_42.pt'
    meta = train(path,'inclusive_bc250',42,2)
    assert meta['objective'] == 'inclusive'
    assert meta['environment']['reward_mode'] == 'energy'  # raw ecology unchanged
    assert meta['environment_steps'] == 2
    assert verify_checkpoint(path) == meta
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_checkpoint(path)
