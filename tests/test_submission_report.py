import json

import pytest

from evaluation.final_report import immediate_reward_probe, paired_optimizer_tests, audit_checkpoints, digest, validate_evaluation_coverage


def test_reward_probe_is_an_actual_private_cost_and_recipient_benefit():
    result = immediate_reward_probe()
    assert result['donor_reward_difference'] == pytest.approx(-2.)
    assert result['recipient_reward_difference'] == pytest.approx(5.)


def test_optimizer_inference_uses_seed_means_and_corrects_three_comparisons():
    rows = []
    for seed in range(5):
        for condition, value in (('clean', .1), ('low_lr', .2), ('high_entropy', .2), ('long_training', .1)):
            for world in range(3):
                rows.append(dict(condition=condition, training_seed=seed, world_seed=world,
                                 control='trained', attack=False, donations_per_resident_step=value))
    result = {r['condition']:r for r in paired_optimizer_tests(rows)}
    assert result['low_lr']['summary']['n'] == 5
    assert result['low_lr']['p_sign_flip'] == .0625
    assert result['low_lr']['p_holm'] == .1875
    assert result['long_training']['p_sign_flip'] == 1.


def test_report_audit_rejects_missing_archive_and_corrupted_weights(tmp_path):
    checkpoints = tmp_path/'checkpoints'
    checkpoints.mkdir()
    weight = checkpoints/'test.pt'
    weight.write_bytes(b'weights')
    # No source entry claims that an unknown current file exists.
    meta = dict(checkpoint_sha256=digest(weight), environment_steps=2,
                source_sha256={'nonexistent.py':'0'*64})
    (checkpoints/'test.json').write_text(json.dumps(meta))
    report = audit_checkpoints(tmp_path)
    assert not report['passed']
    assert report['checkpoints'][0]['archived_source_mismatches'] == ['nonexistent.py']
    weight.write_bytes(b'changed')
    assert not audit_checkpoints(tmp_path)['checkpoints'][0]['weight_hash_matches']


def test_final_report_rejects_duplicate_worlds_disguised_as_complete_evaluation():
    rows = [dict(condition='clean',training_seed=42,world_seed=world,control='trained',attack=False)
            for world in (10042,11042,12042)]
    analysis = dict(training_seeds=[42],evaluation_worlds_per_seed=3,
                    budgets_by_condition={'clean':2048},runs=rows)
    validate_evaluation_coverage(analysis)
    rows[-1] = rows[0]
    with pytest.raises(ValueError,match='duplicate'):
        validate_evaluation_coverage(analysis)
