import numpy as np
from training.dyadic_baseline import DyadicTitForTat


def observation():
    return dict(active_mask=np.ones(3,dtype=bool), neighbor_mask=np.ones((3,2),dtype=bool),
                neighbor_indices=np.array([[1,2],[0,2],[0,1]]),
                action_mask=np.ones((3,7),dtype=bool))


def test_dyadic_reciprocity_tracks_recipient_not_public_donation_score():
    policy = DyadicTitForTat(3)
    obs, uids = observation(), np.array([10,20,30])
    # Agent 20 gives to 30, not 10. Agent 30 gives to 10.
    policy.observe(obs,uids,np.array([0,6,5]))
    assert policy.memory[(10,20)] is False
    assert policy.memory[(30,20)] is True
    assert policy.act(obs,uids)[0] == 6  # 10 reciprocates with 30 only


def test_newborn_uid_does_not_inherit_recycled_slots_social_debt():
    policy = DyadicTitForTat(4)
    obs, uids = observation(), np.array([10,20,30])
    policy.observe(obs,uids,np.zeros(3,dtype=int))
    assert policy.act(obs,uids)[0] == 0
    uids[1] = 99  # unrelated newborn in former defector's slot
    assert policy.act(obs,uids)[0] == 5


def test_unaffordable_attempt_is_not_remembered_as_cooperation():
    policy = DyadicTitForTat(4)
    obs, uids = observation(), np.array([10,20,30])
    obs['action_mask'][1,5] = False
    policy.observe(obs,uids,np.array([0,5,0]))
    assert policy.memory[(10,20)] is False
