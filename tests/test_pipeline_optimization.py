"""Differential oracles are the immutable, pre-task Git implementation."""
from dataclasses import asdict
from functools import lru_cache
import subprocess
import sys
import types

import numpy as np
import pytest

from agent_orch.performance import llm_queueing as current


@lru_cache(None)
def reference_module(path, name, package):
    source = subprocess.check_output(['git', 'show', f'de4a017:{path}']).decode('utf-8')
    module = types.ModuleType(name)
    module.__package__ = package
    sys.modules[name] = module
    exec(compile(source, f'de4a017/{path}', 'exec'), module.__dict__)
    return module


def reference_queue():
    return reference_module('src/agent_orch/performance/llm_queueing.py',
        'agent_orch.performance._reference_queue', 'agent_orch.performance')


def reference_analytical():
    old = reference_module('src/agent_orch/performance/analytical.py',
        'agent_orch.performance._reference_analytical', 'agent_orch.performance')
    queue = reference_queue()
    for name, value in list(vars(old).items()):
        if getattr(value, '__module__', '') == 'agent_orch.performance.llm_queueing' and hasattr(queue, name):
            setattr(old, name, getattr(queue, name))
    return old


@pytest.mark.parametrize('limit', [1, 7, 32])
@pytest.mark.parametrize('budget', [128, 2048, 8192])
def test_prepared_solver_matches_original_float64(scenario, limit, budget):
    old = reference_queue()
    model = next(iter(scenario.models.values()))
    classes = [(256., 1.), (1024.5, 200.), (8192.2, 1200.), (64., 16.)]
    weights = [0.13, 0., 0.51, 0.36]
    a = current.TwoModeCalibration(current.IterationCalibration(5e12, 5e11), current.IterationCalibration(5e12, 5e11))
    b = old.TwoModeCalibration(old.IterationCalibration(5e12, 5e11), old.IterationCalibration(5e12, 5e11))
    expected = old.two_mode_capacity(model, b, classes, weights, 512, budget, limit)
    actual = current.two_mode_capacity(model, a, classes, weights, 512, budget, limit)
    assert actual == expected
    for load in (.05, .7, 1., 1.2):
        rate = expected[0] * load
        x = asdict(current.two_mode_operating_point(model, a, classes, weights, rate, 512, budget, limit))
        y = asdict(old.two_mode_operating_point(model, b, classes, weights, rate, 512, budget, limit))
        assert x == y
    prepared = current._prepare_workload(model, classes, weights, 512)
    assert current._packed_prefill_chunks(classes, weights, 512, budget, _prepared=prepared) == old._packed_prefill_chunks(classes, weights, 512, budget)


@pytest.mark.parametrize('phase', ['composition', 'deployment', 'joint'])
def test_all_observations_masks_and_feedback_match_original(scenario, phase):
    from agent_orch.envs import SlotSequentialJointEnv
    from agent_orch.workload import SlotTrajectory
    old = reference_module('src/agent_orch/envs/slot_joint_env.py',
        'agent_orch.envs._reference_slot_env', 'agent_orch.envs')
    trace = SlotTrajectory.sample(scenario, slots=2, seed=31)
    first = SlotSequentialJointEnv(scenario, trace, mapping_samples=7)
    second = old.SlotSequentialJointEnv(scenario, trace, mapping_samples=7)
    first.training_phase = second.training_phase = phase
    a, _ = first.reset(seed=3)
    b, _ = second.reset(seed=3)
    saved, copies = [], []
    while True:
        assert list(a) == list(b)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        saved.append(a)
        copies.append({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in a.items()})
        valid = np.flatnonzero(a['deploy_mask'])
        action = {'deploy': int(valid[-1]) if len(valid) else 0,
                  'model': np.ones(first.layout.model_action_size, dtype=np.float32)}
        a, r1, done, _, info1 = first.step(action)
        b, r2, done2, _, info2 = second.step(action)
        assert r1 == r2 and done == done2
        assert info1.get('discount') == info2.get('discount')
        if info1.get('period_complete'):
            assert asdict(info1['metrics']) == asdict(info2['metrics'])
            assert info1['constraint_vector'] == info2['constraint_vector']
        if done:
            break
    for observation, snapshot in zip(saved, copies):
        for key in observation:
            np.testing.assert_array_equal(observation[key], snapshot[key])


def test_sparse_analytical_and_route_match_original(scenario):
    from agent_orch.capacity import CapacityPlanner
    from agent_orch.routing import CapacityProportionalRouter
    from agent_orch.performance.analytical import AnalyticalBackend
    old = reference_analytical()
    router_module = reference_module('src/agent_orch/routing/capacity.py',
        'agent_orch.routing._reference_capacity', 'agent_orch.routing')
    router_module.reference_llm_capacity = old.reference_llm_capacity
    placement = CapacityPlanner(scenario).initial_deployment()
    current_router = CapacityProportionalRouter(scenario)
    reference_router = router_module.CapacityProportionalRouter(scenario)
    for offset in (0., .07, .2):
        shares = {(app.id, g, m): 1. / len(scenario.models) + offset
                  for app in scenario.applications.values() for g in app.ingress_rates for m in scenario.models}
        first = current_router.route(placement, shares)
        second = reference_router.route(placement, shares)
        assert first == second
        a = AnalyticalBackend(scenario)
        b = old.AnalyticalBackend(scenario)
        assert a.llm_arrivals(first) == b.llm_arrivals(second)
        assert list(a.llm_arrivals(first)) == list(b.llm_arrivals(second))
        assert asdict(a.evaluate(placement, first)) == asdict(b.evaluate(placement, second))


@pytest.mark.parametrize('count', [0, 1, 127, 128])
def test_native_prefix_aggregation_preserves_every_float_bit(count):
    from agent_orch.performance.workflow_mapping import summarize
    from agent_orch.performance.workflow_batch import BatchValues
    old = reference_module('src/agent_orch/performance/workflow_mapping.py',
        'agent_orch.performance._reference_mapping', 'agent_orch.performance')
    rng = np.random.default_rng(28)
    for _ in range(12):
        weights = rng.random(count)
        values = rng.normal(size=(count, 5)) * 10. ** rng.integers(-12, 12, size=(count, 5))
        valid = rng.random(count) > .2
        batch = BatchValues(valid, rng.random(count) > .3, values[:, 0], values[:, 1], values[:, 2],
                            values[:, 3:], (('a', 0), ('b', 1)))
        actual, expected = summarize(batch, weights), old.summarize(batch, weights)
        def bits(value):
            if isinstance(value, dict):
                return [(k, bits(v)) for k, v in value.items()]
            if isinstance(value, tuple):
                return tuple(bits(v) for v in value)
            return None if value is None else np.float64(value).tobytes()
        assert bits(actual) == bits(expected)


def test_fixed_reference_cache_roundtrip_and_invalidation(scenario, tmp_path):
    from agent_orch.envs import SlotSequentialJointEnv
    from agent_orch.performance.workflow_runtime import WorkflowRuntime
    from agent_orch.performance.fixed_reference import FixedReferenceCache
    from agent_orch.workload import SlotTrajectory
    trace = SlotTrajectory.sample(scenario, slots=2, seed=81)
    with WorkflowRuntime() as runtime:
        runtime.fixed_reference.path = tmp_path / 'cache.json'
        first = SlotSequentialJointEnv(scenario, trace, mapping_samples=7, workflow_runtime=runtime)
        expected = first.baseline_report()
        second = SlotSequentialJointEnv(scenario, trace, mapping_samples=7, workflow_runtime=runtime)
        second._fixed_baseline = lambda: pytest.fail('Fixed reference recomputed')
        assert second.baseline_report() == expected
        assert runtime.fixed_reference.hits == 1
        second._baseline[0]['utility'] = 1000.
        third = SlotSequentialJointEnv(scenario, trace, mapping_samples=7, workflow_runtime=runtime)
        assert third.baseline_report() == expected
        changed = SlotSequentialJointEnv(scenario, trace, mapping_samples=8, workflow_runtime=runtime)
        changed.baseline_report()
        assert runtime.fixed_reference.misses == 2
    with WorkflowRuntime() as restored:
        restored.fixed_reference.path = tmp_path / 'cache.json'
        fourth = SlotSequentialJointEnv(scenario, trace, mapping_samples=7, workflow_runtime=restored)
        fourth._fixed_baseline = lambda: pytest.fail('Saved reference recomputed')
        assert fourth.baseline_report() == expected
    import json
    path = tmp_path / 'cache.json'
    corrupted = json.loads(path.read_text(encoding='utf-8'))
    corrupted['sha256'] = 'corrupt'
    path.write_text(json.dumps(corrupted), encoding='utf-8')
    cache = FixedReferenceCache()
    cache.path = path
    with pytest.raises(ValueError, match='Corrupt'):
        cache.get_or_compute('any', lambda: {})


@pytest.mark.parametrize('phase', ['composition', 'deployment', 'joint'])
def test_frozen_unused_encoder_keeps_outputs_and_gradients(scenario, phase):
    import torch
    from agent_orch.agents import StructuredActorCritic
    from agent_orch.agents.config import PPOConfig
    from agent_orch.agents.rollout import _stack_observations, _stack_actions
    from agent_orch.envs import SlotSequentialJointEnv
    from agent_orch.workload import SlotTrajectory
    old = reference_module('src/agent_orch/agents/networks.py',
        'agent_orch.agents._reference_networks', 'agent_orch.agents')
    env = SlotSequentialJointEnv(scenario, SlotTrajectory.sample(scenario, 1, 13), mapping_samples=7)
    env.training_phase = phase
    observation, _ = env.reset(seed=7)
    config = PPOConfig(hidden_size=16, hidden_layers=1)
    current_policy = StructuredActorCritic(env, config)
    old_policy = old.StructuredActorCritic(env, config)
    old_policy.load_state_dict(current_policy.state_dict())
    current_policy.set_training_phase(phase)
    old_policy.set_training_phase(phase)
    action = old_policy.act(observation, deterministic=True)[0]
    states = _stack_observations([observation] * 4, 'cpu')
    actions = _stack_actions([action] * 4, 'cpu')
    actual = current_policy.evaluate_actions(states, actions)
    expected = old_policy.evaluate_actions(states, actions)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    sum(t.sum() for t in actual).backward()
    sum(t.sum() for t in expected).backward()
    for a, b in zip(current_policy.parameters(), old_policy.parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
