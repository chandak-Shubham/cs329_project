"""A valid weight file must not silently stand in for a different experiment."""
import json
from pathlib import Path
import shutil

import pytest

from experiments import revised_study as study
from experiments.lineage_study import CONDITIONS


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = root / 'results/lineage_study/checkpoints/lineage_clean_seed_42.pt'
    target = tmp_path / 'checkpoints' / source.name
    target.parent.mkdir()
    for suffix in ('.pt', '.json'):
        shutil.copyfile(source.with_suffix(suffix), target.with_suffix(suffix))
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(study, 'CONDITIONS', CONDITIONS)
    return target


def test_matching_checkpoint_is_reusable_and_portable(checkpoint):
    assert study.reusable('lineage_clean', 42, 8192)
    sidecar = checkpoint.with_suffix('.json')
    record = json.loads(sidecar.read_text())
    record['source_sha256'] = {key.replace('\\', '/'): value
                               for key, value in record['source_sha256'].items()}
    sidecar.write_text(json.dumps(record))
    assert study.reusable('lineage_clean', 42, 8192)


@pytest.mark.parametrize('field,value', [
    ('seed', 101), ('model_type', 'mlp'), ('lr', .01),
    ('invasion_count', 99), ('torch_version', 'different'),
    ('numpy_version', 'different'), ('source_sha256', {}),
])
def test_reuse_rejects_mismatched_provenance(checkpoint, field, value):
    sidecar = checkpoint.with_suffix('.json')
    record = json.loads(sidecar.read_text())
    record[field] = value
    sidecar.write_text(json.dumps(record))
    assert not study.reusable('lineage_clean', 42, 8192)


def test_reuse_rejects_changed_source_and_corrupted_weights(checkpoint):
    sidecar = checkpoint.with_suffix('.json')
    original = sidecar.read_text()
    record = json.loads(original)
    key = next(iter(record['source_sha256']))
    record['source_sha256'][key] = '0' * 64
    sidecar.write_text(json.dumps(record))
    assert not study.reusable('lineage_clean', 42, 8192)
    sidecar.write_text(original)
    checkpoint.write_bytes(b'not the trained checkpoint')
    with pytest.raises(ValueError, match='Checkpoint hash mismatch'):
        study.reusable('lineage_clean', 42, 8192)
