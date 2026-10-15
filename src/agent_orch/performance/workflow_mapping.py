"""Exact representation-only replacement for the reference mapping sampler."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import time

import numpy as np


@dataclass
class MappingBatch:
    nodes: tuple[str, ...]
    servers: np.ndarray
    candidates: np.ndarray
    probabilities: np.ndarray


class UniformCache:
    """Only immutable, seeded uniforms; never probabilities or evaluated results."""
    def __init__(self, limit_bytes=64 * 1024 * 1024):
        self.limit_bytes = limit_bytes
        self.bytes = 0
        self.hits = self.misses = 0
        self.entries = OrderedDict()

    def get(self, identity, nodes, samples):
        key = (*identity, samples, nodes)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        # Keep the reference's big-endian conversion and draw order.
        seed = int.from_bytes(hashlib.sha256('|'.join(identity).encode('utf-8')).digest()[:8], 'big')
        value = np.random.default_rng(seed).random((samples, len(nodes)))
        value.flags.writeable = False
        if value.nbytes <= self.limit_bytes:
            while self.entries and self.bytes + value.nbytes > self.limit_bytes:
                _, old = self.entries.popitem(last=False)
                self.bytes -= old.nbytes
            self.entries[key] = value
            self.bytes += value.nbytes
        return value


def sample_mappings(identity, nodes, choices, samples, uniforms):
    """choices: (server IDs, candidate IDs, once-normalized probabilities).

    Python sums and multiplication order are retained by the assembler/caller.
    The sampled branch duplicates Generator.choice's CDF and uniform lookup.
    Singleton/missing supports still consume one draw per node per sample.
    """
    started = time.perf_counter()
    cardinality = 1
    for servers, _, _ in choices:
        cardinality *= len(servers)
    count = cardinality if cardinality <= samples else samples
    servers = np.empty((count, len(nodes)), dtype=np.int32)
    candidates = np.empty_like(servers)
    weights = np.empty(count, dtype=np.float64)
    if cardinality <= samples:
        for row, combination in enumerate(itertools.product(*(range(len(c[0])) for c in choices))):
            weight = 1.0
            for col, index in enumerate(combination):
                s, c, p = choices[col]
                servers[row, col], candidates[row, col] = s[index], c[index]
                weight *= float(p[index])
            weights[row] = weight
    else:
        draws = uniforms.get(identity, nodes, samples)
        for col, (s, c, p) in enumerate(choices):
            p = p.copy()
            p /= p.sum()  # The reference performs this second normalization.
            cumulative = p.cumsum()
            cumulative /= cumulative[-1]  # Generator.choice implementation.
            index = cumulative.searchsorted(draws[:, col], side='right')
            servers[:, col], candidates[:, col] = s[index], c[index]
        weights.fill(1.0 / samples)
    return MappingBatch(nodes, servers, candidates, weights), time.perf_counter() - started


def summarize(batch, probabilities):
    """Do not replace sequential float64 additions with a vector reduction."""
    e2e = ttft = tbt = slo = completed = 0.0
    stages = {}
    for row, probability in enumerate(probabilities):
        probability = float(probability)
        if not batch.valid[row]:
            continue
        completed += probability
        e2e += probability * float(batch.e2e[row])
        ttft += probability * float(batch.ttft[row])
        tbt += probability * float(batch.tbt[row])
        if batch.slo[row]:
            slo += probability
        for name, column in batch.stages:
            stages[name] = stages.get(name, 0.0) + probability * float(batch.completions[row, column])
    value = ((e2e / completed, ttft / completed, tbt / completed, slo,
              {n: v / completed for n, v in stages.items()}) if completed else
             (None, None, None, 0.0, {}))
    return value, completed
