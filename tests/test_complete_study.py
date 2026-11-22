"""Protocol guards for the complete, exploratory inherited-policy study."""
import hashlib

import pytest

from training.inherited_policies import train, archive_sources, source_hashes
from training.inheritance_study import checkpoint_matches, training_spec


def test_long_budget_and_optimizer_controls_change_only_declared_factors():
    assert training_spec('long_training', 2048) == (8192, {})
    assert training_spec('low_lr', 2048) == (2048, {'ppo': {'lr': 3e-5}})
    assert training_spec('high_entropy', 2048) == (2048, {'ppo': {'ent_coef': .1}})


def test_sources_are_archived_and_protocol_mismatch_is_rejected(tmp_path):
    path = tmp_path/'checkpoints'/'clean_seed_42.pt'
    metadata = train(path, steps=2, seed=42)
    assert checkpoint_matches(metadata, 'clean', 42, 2)
    assert not checkpoint_matches(metadata, 'long_training', 42, 2)
    assert not checkpoint_matches(metadata, 'low_lr', 42, 2)
    for name, digest in metadata['source_sha256'].items():
        assert hashlib.sha256((tmp_path/'source'/name).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match='overwrite'):
        train(path, steps=2, seed=42)
    metadata['source_sha256'] = {}
    assert not checkpoint_matches(metadata, 'clean', 42, 2)


def test_concurrent_source_archives_are_complete_and_immutable(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    sources = source_hashes()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: archive_sources(tmp_path, sources), range(6)))
    for name, value in sources.items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() == value
    target = tmp_path/next(iter(sources))
    target.write_bytes(b'different version')
    with pytest.raises(ValueError, match='different source'):
        archive_sources(tmp_path, sources)
    assert target.read_bytes() == b'different version'
