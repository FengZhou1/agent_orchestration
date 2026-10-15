"""Small, fingerprinted greedy reward references, not historical simulator states."""
from functools import lru_cache
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import pickle
import sys
import tempfile


@lru_cache(maxsize=1)
def _implementation_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    # Conservative invalidation across the whole package also covers changes
    # to Simulator accounting, route validation and slot materialization.
    for path in sorted(root.rglob('*.py')):
        relative = str(path.relative_to(root))
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reference_signature(scenario, trajectory_digest, spec, references, samples, routing):
    # Pickle is only used to HASH trusted in-memory configuration, never loaded.
    environment = (sys.version, sys.executable, platform.node(), platform.machine(),
                   tuple((name, version(name)) for name in ('numpy', 'scipy', 'networkx')))
    payload = (_implementation_digest(), environment, scenario, trajectory_digest, spec,
               references, samples, routing, 'deterministic-greedy-seed-0')
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


class FixedReferenceCache:
    VERSION = 'fixed-greedy-reference-v1'

    def __init__(self):
        self.path = None
        self.entries = {}
        self.hits = self.misses = 0
        self._loaded_path = None

    def _load(self):
        if self.path is None or self._loaded_path == Path(self.path):
            return
        path = Path(self.path)
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            if data['version'] != self.VERSION:
                raise ValueError('Unknown fixed-reference cache version')
            encoded = json.dumps(data['entries'], sort_keys=True, allow_nan=False)
            if hashlib.sha256(encoded.encode()).hexdigest() != data['sha256']:
                raise ValueError('Corrupt fixed-reference cache')
            self.entries.update(data['entries'])
        self._loaded_path = path

    def get_or_compute(self, signature, compute):
        self._load()
        if signature in self.entries:
            self.hits += 1
            values = self.entries[signature]
        else:
            self.misses += 1
            values = compute()
            self.entries[signature] = {str(k): dict(v) for k, v in values.items()}
            while len(self.entries) > 16:
                self.entries.pop(next(iter(self.entries)))
            if self.path is not None:
                path = Path(self.path)
                encoded = json.dumps(self.entries, sort_keys=True, allow_nan=False)
                data = {'version': self.VERSION, 'entries': self.entries,
                        'sha256': hashlib.sha256(encoded.encode()).hexdigest()}
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                        dir=path.parent, prefix=path.stem + '.', suffix='.tmp.json', delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(data, handle, indent=2, allow_nan=False)
                try:
                    temporary.replace(path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
        # Reward bookkeeping must not mutate the shared cache's dictionaries.
        return {int(k): dict(v) for k, v in values.items()}
