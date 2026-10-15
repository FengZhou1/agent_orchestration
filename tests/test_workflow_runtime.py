from dataclasses import asdict, replace
import hashlib
import itertools
import math

import numpy as np
import pytest

from agent_orch.performance.analytical import AnalyticalBackend
from agent_orch.performance.workflow import WorkflowEvaluator
from agent_orch.performance.workflow_mapping import UniformCache, sample_mappings
from agent_orch.performance.workflow_runtime import WorkflowOptions, WorkflowRuntime
from agent_orch.schema.models import NodeType, RoutingDecision, SLOType, SLOSpec
from test_indexed_workflow import parallel_case


@pytest.mark.parametrize('size', [1, 127, 128, 129])
@pytest.mark.parametrize('extreme', [False, True])
def test_sampler_exact_order_weights_and_random_consumption(size, extreme):
    identity = ('app', 'g', 'm', 'flow')
    nodes = ('a', 'b')
    raw = ([1e-200] * (size - 1) + [1.0]) if extreme else list(range(1, size + 1))
    p = np.asarray([v / sum(raw) for v in raw])
    choices = [(np.arange(size, dtype=np.int32), np.arange(size, dtype=np.int32), p),
               (np.array([0], dtype=np.int32), np.array([0], dtype=np.int32), np.array([1.]))]
    cache = UniformCache()
    batch, _ = sample_mappings(identity, nodes, choices, 128, cache)
    if size <= 128:
        assert np.array_equal(batch.servers[:, 0], np.arange(size))
        assert np.array_equal(batch.probabilities, p)
        assert cache.misses == 0
    else:
        seed = int.from_bytes(hashlib.sha256('|'.join(identity).encode()).digest()[:8], 'big')
        reference = np.random.default_rng(seed)
        prepared = [v[2] / v[2].sum() for v in choices]
        expected = np.array([[reference.choice(len(v), p=v) for v in prepared] for _ in range(128)])
        assert np.array_equal(batch.servers, expected)
        vector = np.random.default_rng(seed)
        draws = vector.random((128, 2))
        assert np.array_equal(draws, cache.get(identity, nodes, 128))
        assert vector.random() == reference.random()
        assert cache.hits == 1


def test_uniform_cache_lru_byte_limit_and_identity():
    cache = UniformCache(128)
    first = cache.get(('a', 'g', 'm', 'f'), ('a',), 8)
    cache.get(('b', 'g', 'm', 'f'), ('a',), 8)
    assert cache.bytes == 128
    cache.get(('c', 'g', 'm', 'f'), ('a',), 8)
    assert len(cache.entries) == 2 and cache.bytes == 128
    assert np.array_equal(first, cache.get(('a', 'g', 'm', 'f'), ('a',), 8))
    cache.get(('a', 'g', 'm', 'f'), ('a', 'b', 'c'), 8)
    assert cache.bytes <= 128


def test_runtime_option_legacy_resume_and_explicit_override():
    assert WorkflowOptions.resolve({}) == WorkflowOptions('reference', 1)
    saved = {'workflow_runtime': {'engine': 'array', 'workers': 2}}
    assert WorkflowOptions.resolve(saved) == WorkflowOptions('array', 2)
    assert WorkflowOptions.resolve(saved, workers=4) == WorkflowOptions('array', 4)
    with pytest.raises(ValueError):
        WorkflowOptions('reference', 2)


@pytest.mark.parametrize('workers', [1, 2, 4])
@pytest.mark.parametrize('kind', list(SLOType))
def test_complete_workflow_matches_reference(parallel_case, workers, kind):
    scene, app, model, ingress, flow, result, _, _ = parallel_case
    app = replace(app, slo=SLOSpec(kind, 10., 1., 10.))
    scene = replace(scene, applications={app.id: app})
    llm = {}
    for name, node in app.nodes.items():
        if node.type is NodeType.LLM:
            candidates = [c for c in scene.candidates.values() if c.model == model]
            for c in candidates:
                llm[app.id, ingress, name, c.id] = 1. / len(candidates)
    distribution = {(app.id, ingress, model, flow.id, n): {s: 1. / len(scene.servers) for s in scene.servers}
                    for n in flow.nodes if app.nodes[n].type is NodeType.TOOL}
    result = replace(result, node_server_distribution=distribution)
    routing = RoutingDecision({(app.id, ingress, model): 1.}, llm, {})
    reference = WorkflowEvaluator(scene, AnalyticalBackend(scene), max_mapping_samples=7)
    expected = reference.evaluate(None, routing, result)
    with WorkflowRuntime(WorkflowOptions('array', workers)) as runtime:
        evaluator = WorkflowEvaluator(scene, AnalyticalBackend(scene), max_mapping_samples=7, runtime=runtime)
        assert asdict(evaluator.evaluate(None, routing, result)) == asdict(expected)
        pool = runtime.pool
        if workers > 1:
            assert pool is not None
        # Same placement, CHANGED global queue state: cannot reuse evaluation.
        bad = replace(result, llm_utilization={c: 1. for c in result.llm_utilization})
        assert evaluator.evaluate(None, routing, bad).completed_rps == 0
        assert evaluator.evaluate(None, routing, result) == expected
        assert runtime.pool is pool  # Never respawn once per slot/evaluation.


@pytest.mark.parametrize('samples', [1, 4096])
@pytest.mark.parametrize('missing', [False, True])
def test_array_location_rows_match_real_reference(parallel_case, samples, missing):
    scene, app, model, ingress, flow, result, routing, _ = parallel_case
    if not missing:
        candidates = [c for c in scene.candidates.values() if c.model == model]
        routing = RoutingDecision({(app.id, ingress, model): 1.}, {
            (app.id, ingress, n, c.id): (i + 1) / sum(range(1, len(candidates) + 1))
            for n in flow.nodes if app.nodes[n].type is NodeType.LLM
            for i, c in enumerate(candidates)}, {})
        result = replace(result, node_server_distribution={
            (app.id, ingress, model, flow.id, n): {s: (i + 1) / sum(range(1, len(scene.servers) + 1))
                for i, s in enumerate(scene.servers)} for n in flow.nodes if app.nodes[n].type is NodeType.TOOL})
    # Missing supports still produce sentinel positions and consume draws.
    evaluator = WorkflowEvaluator(scene, AnalyticalBackend(scene), max_mapping_samples=samples)
    old = evaluator._physical_mapping_outcomes(app.id, ingress, model, flow.id, routing, result)
    _, kernel, requests, _ = evaluator._prepare_application(app, routing, result)
    identity, nodes, choices, _ = requests[0]
    batch, _ = sample_mappings(identity, nodes, choices, samples, UniformCache())
    server_ids = ('', *scene.servers)
    candidate_ids = (None, *scene.candidates)
    for row, (mapping, weight) in enumerate(old):
        assert float(batch.probabilities[row]) == weight
        assert {n: (server_ids[batch.servers[row, col]], candidate_ids[batch.candidates[row, col]])
                for col, n in enumerate(nodes)} == mapping


def test_three_phase_learning_and_resume_engine_switch(scenario):
    import copy
    import torch
    from agent_orch.agents import train_ppo
    from agent_orch.agents.config import PPOConfig
    from agent_orch.envs import SlotSequentialJointEnv
    from agent_orch.workload import SlotTrajectory
    trace = SlotTrajectory.sample(scenario, slots=2, seed=71)
    schedule = (('composition', 1), ('deployment', 1), ('joint', 1))
    config = PPOConfig(hidden_size=16, hidden_layers=1, update_epochs=1,
                       minibatch_size=8, exploration_mode='none')
    def train(engine, *, resume=None, stop=None):
        saved = []
        with WorkflowRuntime(WorkflowOptions(engine, 1)) as runtime:
            env = SlotSequentialJointEnv(scenario, trace, mapping_samples=7, workflow_runtime=runtime)
            def checkpoint(state):
                saved.append(copy.deepcopy(state))
                if state['next_update'] == stop:
                    raise InterruptedError('intentional test interruption')
            try:
                policy, history = train_ppo(env, updates=3, rollout_steps=1, rollout_periods=2,
                    seed=93, config=config, phase_schedule=schedule, resume_state=resume,
                    on_checkpoint=checkpoint)
            except InterruptedError:
                return saved[-1]
            return saved[-1]
    reference = train('reference')
    array = train('array')
    assert [r['training_phase'] for r in reference['history']] == ['composition', 'deployment', 'joint']
    interrupted = train('reference', stop=1)
    resumed = train('array', resume=interrupted)
    for actual in (array, resumed):
        assert reference['environment_reward_state'] == actual['environment_reward_state']
        np.testing.assert_array_equal(reference['lagrange_multipliers'], actual['lagrange_multipliers'])
        assert reference['optimizer_training_phase'] == actual['optimizer_training_phase']
        for name, weight in reference['policy_state_dict'].items():
            torch.testing.assert_close(weight, actual['policy_state_dict'][name], atol=1e-7, rtol=1e-6)
        assert reference['optimizer_state_dict']['param_groups'] == actual['optimizer_state_dict']['param_groups']
        for parameter, state in reference['optimizer_state_dict']['state'].items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor, actual['optimizer_state_dict']['state'][parameter][name], atol=1e-7, rtol=1e-6)
        for a, b in zip(reference['history'], actual['history']):
            for key in a:
                if key.startswith('timing_') or key in ('collection_time_s', 'optimization_time_s'):
                    continue
                if isinstance(a[key], float):
                    assert np.isclose(a[key], b[key], atol=1e-7, rtol=1e-6, equal_nan=True), key
                else:
                    assert a[key] == b[key], key


def test_worker_exception_is_not_silently_ignored_and_pool_closes():
    choice = (np.array([0], dtype=np.int32), np.array([0], dtype=np.int32), np.array([1.]))
    task = (None, None, [(('a', 'g', 'm', 'f'), ('n',), [choice], None)], 7)
    with WorkflowRuntime(WorkflowOptions('array', 2)) as runtime:
        with pytest.raises(AttributeError):
            list(runtime.applications([object()], lambda app: task))
        assert runtime.pool is None


def test_cli_interruption_resume_preserves_science_and_records_runtime_switch(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_slot_joint
    real_train = run_slot_joint.train_ppo
    arguments = ['run_slot_joint.py', '--scenario', 'configs/toy.yaml', '--output', str(tmp_path),
        '--updates', '3', '--route-updates', '1', '--deploy-updates', '1', '--slots', '1',
        '--mapping-samples', '7', '--hidden-size', '16', '--hidden-layers', '1',
        '--update-epochs', '1', '--minibatch-size', '8', '--smoke']
    def interrupted(env, **kwargs):
        save = kwargs['on_checkpoint']
        def stop(state):
            save(state)
            raise KeyboardInterrupt('intentional CLI resume test')
        return real_train(env, **{**kwargs, 'on_checkpoint': stop})
    monkeypatch.setattr(run_slot_joint, 'train_ppo', interrupted)
    monkeypatch.setattr(sys, 'argv', [*arguments, '--workflow-engine', 'reference'])
    with pytest.raises(KeyboardInterrupt):
        run_slot_joint.main()
    names = ('train_trajectory.json', 'warmup_deployments.json')
    hashes = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in names}
    original = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    # Also exercise the old-manifest missing-runtime interpretation.
    original.pop('workflow_runtime')
    (tmp_path / 'manifest.json').write_text(json.dumps(original), encoding='utf-8')
    monkeypatch.setattr(run_slot_joint, 'train_ppo', real_train)
    monkeypatch.setattr(sys, 'argv', [*arguments, '--resume', '--workflow-engine', 'array'])
    run_slot_joint.main()
    resumed = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    assert resumed['training_settings'] == original['training_settings']
    assert resumed['workflow_runtime'] == {'engine': 'array', 'workers': 1}
    switch = resumed['workflow_runtime_switches'][-1]
    assert switch['from'] == {'engine': 'reference', 'workers': 1}
    assert switch['to'] == {'engine': 'array', 'workers': 1}
    assert switch['checkpoint_next_update'] == 1
    assert {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in names} == hashes
