"""Run-owned performance options and bounded, persistent application workers."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import multiprocessing
import os
import pickle
import time

from .workflow_mapping import UniformCache, sample_mappings, summarize


@dataclass(frozen=True)
class WorkflowOptions:
    engine: str = 'reference'
    workers: int = 1

    def __post_init__(self):
        if self.engine not in ('reference', 'array') or self.workers not in (1, 2, 4):
            raise ValueError('Invalid workflow engine or worker count')
        if self.engine != 'array' and self.workers != 1:
            raise ValueError('Workflow parallelism requires --workflow-engine array')

    @classmethod
    def resolve(cls, manifest=None, engine=None, workers=None):
        saved = (manifest or {}).get('workflow_runtime', {})
        return cls(engine if engine is not None else saved.get('engine', 'reference'),
                   workers if workers is not None else saved.get('workers', 1))

    def to_dict(self):
        return asdict(self)


_worker_uniforms = None


def _evaluate_application(task, uniforms=None):
    global _worker_uniforms
    if uniforms is None:
        if _worker_uniforms is None:
            _worker_uniforms = UniformCache()
        uniforms = _worker_uniforms
    app, context, requests, samples = task
    result = {}
    timers = {'sampling': 0.0, 'numeric': 0.0, 'aggregation': 0.0}
    for identity, nodes, choices, flow in requests:
        mappings, elapsed = sample_mappings(identity, nodes, choices, samples, uniforms)
        timers['sampling'] += elapsed
        permutation = [nodes.index(node) for node in flow.nodes]
        started = time.perf_counter()
        batch = context.evaluate_arrays(app, identity[1], identity[2], flow,
            mappings.servers[:, permutation], mappings.candidates[:, permutation])
        timers['numeric'] += time.perf_counter() - started
        started = time.perf_counter()
        result[identity] = summarize(batch, mappings.probabilities)
        timers['aggregation'] += time.perf_counter() - started
    return result, timers, {'pid': os.getpid(), 'bytes': uniforms.bytes, 'hits': uniforms.hits, 'misses': uniforms.misses}


def _evaluate_serialized(payload):
    started = time.perf_counter()
    task = pickle.loads(payload)
    elapsed = time.perf_counter() - started
    result, timers, cache = _evaluate_application(task)
    timers['deserialization'] = elapsed
    return result, timers, cache


class WorkflowRuntime:
    """Own once per run, pass into every per-slot Simulator, close in finally.

    At most `workers` application payloads are outstanding. Ordered delivery
    allows the parent to retain its original per-flow accumulation order.
    """
    def __init__(self, options=None):
        self.options = options or WorkflowOptions()
        self.uniforms = UniformCache()
        self.pool = None
        self.last_timings = {}
        self.last_cache = {}
        self.worker_cache = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def applications(self, applications, prepare):
        self.last_timings = {}
        self.last_cache = {}
        def make(app):
            started = time.perf_counter()
            task = prepare(app)
            self.last_timings['assembly'] = self.last_timings.get('assembly', 0.0) + time.perf_counter() - started
            return task
        def unpack(output):
            result, timers, cache = output
            for key, seconds in timers.items():
                self.last_timings[key] = self.last_timings.get(key, 0.0) + seconds
            self.worker_cache[cache['pid']] = cache
            self.last_cache = {'bytes': sum(v['bytes'] for v in self.worker_cache.values()),
                               'per_process': dict(self.worker_cache)}
            return result
        if self.options.workers == 1:
            for app in applications:
                yield unpack(_evaluate_application(make(app), self.uniforms))
            return
        if self.pool is None:
            self.pool = ProcessPoolExecutor(self.options.workers,
                mp_context=multiprocessing.get_context('spawn'))
        iterator = iter(applications)
        pending = deque()
        def submit(app):
            task = make(app)
            started = time.perf_counter()
            payload = pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL)
            self.last_timings['serialization'] = self.last_timings.get('serialization', 0.0) + time.perf_counter() - started
            return self.pool.submit(_evaluate_serialized, payload)
        try:
            for _ in range(self.options.workers):
                app = next(iterator, None)
                if app is not None:
                    pending.append(submit(app))
            while pending:
                started = time.perf_counter()
                output = pending.popleft().result()  # submission order, not completion order
                self.last_timings['process_wait'] = self.last_timings.get('process_wait', 0.0) + time.perf_counter() - started
                yield unpack(output)
                app = next(iterator, None)
                if app is not None:
                    pending.append(submit(app))
        except BaseException:
            self.close()
            raise
