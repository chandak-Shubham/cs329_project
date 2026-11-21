from experiments.inheritance_study import summarize, feasibility
from evaluation.inheritance_report import paired_contrasts


def row(seed,world,value,condition='clean'):
    return dict(condition=condition,control='trained',attack=False,training_seed=seed,
                world_seed=world,donations_per_resident_step=value,final_residents=10,
                resident_survival=True,mimic_donation_share=None,
                opportunity_adjusted_tag_preference=None,adjacent_hamming_similarity=.5,
                late_donation_rate=value,late_donation_frequency=1.)


def test_evaluation_worlds_do_not_inflate_independent_sample_count():
    rows = [row(42,10042,.1),row(42,11042,.9),row(101,10101,.2),row(101,11101,.2)]
    summary = summarize(rows)['clean/trained/clean']
    assert summary['donations_per_resident_step']['n'] == 2
    assert summary['donations_per_resident_step']['mean'] == .35
    assert summary['mimic_donation_share']['mean'] is None
    assert summary['all_worlds_survive']['n'] == 2


def test_feasibility_requires_each_expected_distinct_seed_and_world():
    rows = [row(42,10042,.5),row(42,11042,.5),row(101,10101,.5),row(101,11101,.5)]
    assert feasibility(rows,[42,101],2)['passed']
    assert not feasibility(rows[:-1]+rows[:1],[42,101],2)['passed']
    rows[0]['resident_survival'] = False
    assert not feasibility(rows,[42,101],2)['passed']


def test_paired_contrasts_average_worlds_before_pairing_seeds():
    rows = [row(42,1,.8),row(42,2,.4),row(101,3,.5),row(101,4,.5),
            row(42,1,.2,'no_selection'),row(42,2,.2,'no_selection'),
            row(101,3,.3,'no_selection'),row(101,4,.3,'no_selection')]
    result = paired_contrasts(rows)['clean_minus_no_selection/donations_per_resident_step']
    assert result['n'] == 2
    assert abs(result['mean']-.3) < 1e-10
