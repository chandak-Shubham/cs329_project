"""Locate each actor tensor inside a PyTorch checkpoint without loading torch.

A .pt file written by torch.save is an uncompressed zip: data.pkl describes the
object and every tensor's bytes are stored raw in <prefix>/data/<key>. The web
page reads the committed .pt file directly; this module tells it which storage
key, element offset and shape belong to each tensor of each family network.
"""
from __future__ import annotations

import collections
import pickle
import zipfile
from pathlib import Path

ACTOR_EXCLUDE = ('critic_head.', 'gating.')  # value head and unused legacy gate


class _Ref:
    def __init__(self, storage, offset, size):
        self.key, self.offset, self.shape = storage['key'], int(offset), [int(v) for v in size]

    def entry(self):
        return [self.key, self.offset, self.shape]


class _Unpickler(pickle.Unpickler):
    """Resolves torch tensors to storage references; refuses anything else."""

    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return lambda storage, offset, size, stride, *rest: _Ref(storage, offset, size)
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'torch' and name.endswith('Storage'):
            return name
        raise pickle.UnpicklingError(f'unexpected object in checkpoint: {module}.{name}')

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        return dict(type=pid[1], key=str(pid[2]), numel=int(pid[4]))


def layout(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        if any(info.compress_type != zipfile.ZIP_STORED for info in archive.infolist()):
            raise ValueError(f'{path} has compressed entries')
        prefix = archive.namelist()[0].split('/')[0]
        obj = _Unpickler(archive.open(f'{prefix}/data.pkl')).load()
    policies = [{name: ref.entry() for name, ref in state.items() if not name.startswith(ACTOR_EXCLUDE)}
                for state in obj['policies']]
    return dict(policies=policies, founderGenomes=obj['founder_genomes'].entry(),
                founders=len(obj['policies']))
