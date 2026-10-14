from dataclasses import replace
import json

import numpy as np
import pytest
import networkx as nx

from agent_orch.agents import TrainingProgressReporter
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.envs.base import BaseOrchestrationEnv
from agent_orch.performance.analytical import (
    _evaluate_llm_instance_uncached, _llm_instance_cached, evaluate_llm_instance,
)
from agent_orch.performance.workflow import _workflow_topology
from agent_orch.workload import SlotTrajectory


def test_topology_cache_preserves_orders_and_uses_full_structure(monkeypatch):
    _workflow_topology.cache_clear()
    nodes = ('z', 'b', 'a', 'final')
    edges = (('z', 'b'), ('z', 'a'), ('b', 'final'), ('a', 'final'))
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    expected = (tuple((n, tuple(graph.predecessors(n))) for n in nx.lexicographical_topological_sort(graph)),
                tuple(n for n in graph if graph.in_degree(n) == 0))
    assert _workflow_topology(nodes, edges) == expected
    def no_build(*args, **kwargs):
        raise AssertionError('an identical graph was rebuilt')
    with monkeypatch.context() as patch:
        patch.setattr(nx, 'DiGraph', no_build)
        assert _workflow_topology(nodes, edges) == expected
    assert _workflow_topology(nodes, edges[:-1]) != expected


@pytest.mark.parametrize('mode', ['macro', 'two_mode', 'resident', 'arrival', 'occupancy_ps'])
def test_analytical_cache_is_exact_and_cannot_be_poisoned(scenario, mode):
    config = next(iter(scenario.llm_configs.values()))
    model = scenario.models[config.model]
    args = (model, config, [(256., 8.), (512., 16.)], [.6, .4], .01, 512, mode)
    _llm_instance_cached.cache_clear()
    expected = _evaluate_llm_instance_uncached(*args)
    actual = evaluate_llm_instance(*args)
    assert actual == expected
    actual[1][0].response_s = -999.
    actual[1].clear()
    assert evaluate_llm_instance(*args) == expected
    assert _llm_instance_cached.cache_info().hits == 1
    changed = list(args)
    changed[4] = np.nextafter(args[4], np.inf)
    evaluate_llm_instance(*changed)
    assert _llm_instance_cached.cache_info().misses == 2
    changed = list(args)
    changed[1] = replace(config, effective_flops=np.nextafter(config.effective_flops, np.inf))
    evaluate_llm_instance(*changed)
    assert _llm_instance_cached.cache_info().misses == 3
    changed = list(args)
    changed[2] = [(np.nextafter(256., np.inf), 8.), (512., 16.)]
    evaluate_llm_instance(*changed)
    assert _llm_instance_cached.cache_info().misses == 4
    changed = list(args)
    changed[3] = [np.nextafter(.6, np.inf), .4]
    evaluate_llm_instance(*changed)
    assert _llm_instance_cached.cache_info().misses == 5
    changed = list(args)
    changed[5] += 1
    evaluate_llm_instance(*changed)
    assert _llm_instance_cached.cache_info().misses == 6


def test_path_feasibility_is_not_reused_after_link_load_changes(scenario):
    from agent_orch.baselines import GreedyPolicy
    from agent_orch.performance.analytical import AnalyticalBackend
    from agent_orch.performance.workflow import WorkflowEvaluator
    policy = GreedyPolicy(scenario)
    deployment = policy.deployment()
    routing = policy.routing(deployment)
    backend = AnalyticalBackend(scenario)
    result = backend.evaluate(deployment, routing)
    evaluator = WorkflowEvaluator(scenario, backend, max_mapping_samples=2)
    edge = scenario.links[0]
    good = replace(result, link_load_mbps={})
    assert evaluator._path_feasible(edge.source, edge.target, good)
    bad = replace(result, link_load_mbps={key: link.capacity_mbps for key, link in backend.network.links.items()})
    evaluator.evaluate(deployment, routing, bad)
    assert not evaluator._path_feasible(edge.source, edge.target, bad)
    evaluator.evaluate(deployment, routing, good)
    assert evaluator._path_feasible(edge.source, edge.target, good)


def test_one_observation_reuses_features_without_changing_any_array(scenario, monkeypatch):
    env = SlotSequentialJointEnv(scenario, SlotTrajectory.sample(scenario, 1, 13), mapping_samples=2)
    expected = BaseOrchestrationEnv._observation(env)
    expected['model_group'] = env._model_group
    expected['deployment_features'] = env._deployment_features()
    expected['routing_features'] = env._routing_features()
    calls = []
    original = env._feature_vector
    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(env, '_feature_vector', counted)
    actual = env._observation()
    assert len(calls) == 1
    assert list(actual) == list(expected)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_progress_prints_timings_and_status_details(tmp_path, capsys):
    with TrainingProgressReporter(run_id='probe', output_dir=tmp_path, updates=1,
        rollout_steps=1, update_epochs=1, minibatch_size=1, device='cpu',
        show_progress=False, print_updates=True, details=lambda: {'model_groups': 304}) as progress:
        progress.on_phase(0, 'workflow')
        progress.on_update({'update': 0., 'training_phase': 'composition',
            'collection_time_s': 2., 'optimization_time_s': .1, 'timing_workflow_s': 1.8})
        progress.on_phase(1, 'loading_completed_checkpoint')
    output = capsys.readouterr().out
    assert 'update 1/1' in output and 'workflow=1.80s' in output
    assert 'update 2/1' not in output
    status = json.loads((tmp_path/'training_status.json').read_text())
    assert status['decision_progress']['model_groups'] == 304
    assert status['latest_metrics']['timing_workflow_s'] == 1.8


def test_unchanged_slot_count_does_not_write_status_per_model_group(tmp_path, monkeypatch):
    with TrainingProgressReporter(run_id='probe', output_dir=tmp_path, updates=1,
        rollout_steps=1, update_epochs=1, minibatch_size=1, device='cpu',
        show_progress=False, heartbeat_s=3600) as progress:
        calls = []
        monkeypatch.setattr(progress, '_write_status', lambda *args, **kwargs: calls.append(1))
        for _ in range(304):
            progress.on_rollout_step(0, 0)
        assert not calls


def test_resume_does_not_count_prior_updates_as_new_process_speed(tmp_path):
    with TrainingProgressReporter(run_id='resume', output_dir=tmp_path, updates=2000,
        rollout_steps=1, update_epochs=3, minibatch_size=32, device='cpu',
        show_progress=False, initial_update=2, append_history=True):
        status = json.loads((tmp_path/'training_status.json').read_text())
        assert status['completed_updates'] == 2
        assert status['eta_seconds'] is None
