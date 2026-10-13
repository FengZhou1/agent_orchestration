from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from agent_orch.agents.config import PPOConfig
from agent_orch.agents.ppo import _constrained_utility, build_update_record
from agent_orch.agents.rollout import RolloutBatch
from agent_orch.envs import SlotSequentialJointEnv
from agent_orch.objective import ObjectiveEvaluator, ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import SlotTrajectory


def _complete_slot(env, observation):
    steps = 0
    while True:
        phase = observation["action_type"]
        action = {
            "deploy": 1 if phase < env.COMPOSITION and observation["deploy_mask"][1] else 0,
            "model": np.ones(env.layout.model_action_size, dtype=np.float32),
        }
        observation, reward, terminated, _, info = env.step(action)
        steps += 1
        if info.get("period_complete"):
            return observation, reward, terminated, info, steps
        assert env.simulator.slot == env._period_index


def test_trajectory_replays_and_varies_between_slots():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    first = SlotTrajectory.sample(scenario, slots=2, seed=9)
    second = SlotTrajectory.sample(scenario, slots=2, seed=9)
    assert first.digest() == second.digest()
    assert SlotTrajectory.from_dict(first.to_dict()).digest() == first.digest()
    assert first.slots[0].arrivals != first.slots[1].arrivals
    assert first.slots[0].link_capacity_mbps != first.slots[1].link_capacity_mbps
    assert first.slots[0].prompt_tokens == first.slots[1].prompt_tokens
    assert first.slots[0].output_tokens == first.slots[1].output_tokens
    assert first.slots[0].flow_probabilities == first.slots[1].flow_probabilities
    assert first.slots[0].server_resources != first.slots[1].server_resources
    # SLOs are application contracts, and the physical topology is stable.
    assert first.scenario_at(0, scenario).applications[next(iter(scenario.applications))].slo == scenario.applications[next(iter(scenario.applications))].slo


def test_fixed_workload_matches_base_exactly_and_rejects_tampered_slots(scenario):
    trace = SlotTrajectory.sample(scenario, slots=3, seed=61)
    for snapshot in trace.slots:
        for app in scenario.applications.values():
            assert snapshot.flow_probabilities[app.id] == {
                flow.id: flow.probability for flow in app.pattern_flows
            }
            for node in app.nodes.values():
                if node.type.value == "llm":
                    assert snapshot.prompt_tokens[app.id][node.id] == node.prompt_tokens
                    assert snapshot.output_tokens[app.id][node.id] == node.output_tokens
    import copy
    payload = copy.deepcopy(trace.to_dict())
    app = next(iter(scenario.applications.values()))
    node = next(node for node in app.nodes.values() if node.type.value == "llm")
    model = next(iter(scenario.models))
    payload["slots"][1]["prompt_tokens"][app.id][node.id][model] += 1
    with pytest.raises(ValueError, match="stay fixed"):
        SlotTrajectory.from_dict(payload)


def test_evaluation_loads_only_saved_training_trajectory(scenario, tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_slot_joint import load_training_trajectory, PROTOCOL, PHYSICAL_ROUTING, EVALUATION_PROTOCOL
    trace = SlotTrajectory.sample(scenario, slots=2, seed=71)
    (tmp_path / "train_trajectory.json").write_text(json.dumps(trace.to_dict()), encoding="utf-8")
    manifest = {"protocol": PROTOCOL, "physical_routing": PHYSICAL_ROUTING,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "trajectory_digest": trace.digest(), "slots": 2, "trajectory_seed": 71}
    def no_sampling(*args, **kwargs):
        raise AssertionError("Evaluation must not sample another trajectory")
    monkeypatch.setattr(SlotTrajectory, "sample", no_sampling)
    assert load_training_trajectory(tmp_path, manifest).digest() == trace.digest()
    with pytest.raises(ValueError, match="differs"):
        load_training_trajectory(tmp_path, {**manifest, "trajectory_digest": "tampered"})


def test_main_evaluates_training_trace_without_another_sampling_call(tmp_path, monkeypatch):
    """Exercise the CLI plumbing with a mock trainer, not a training experiment."""
    import json
    import sys
    from pathlib import Path
    from agent_orch.agents import StructuredActorCritic
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_slot_joint
    original = SlotTrajectory.sample
    calls = []
    def sample_once(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    def fake_train(env, **kwargs):
        return StructuredActorCritic(env, kwargs["config"]), []
    monkeypatch.setattr(SlotTrajectory, "sample", sample_once)
    monkeypatch.setattr(run_slot_joint, "train_ppo", fake_train)
    monkeypatch.setattr(sys, "argv", ["run_slot_joint.py", "--scenario", "configs/toy.yaml",
        "--output", str(tmp_path), "--slots", "1", "--updates", "1",
        "--route-updates", "1", "--deploy-updates", "0", "--mapping-samples", "2"])
    run_slot_joint.main()
    assert len(calls) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    baselines = json.loads((tmp_path / "baselines.json").read_text(encoding="utf-8"))
    assert manifest["trajectory_digest"] == evaluation["trajectory_digest"] == baselines["trajectory_digest"]
    assert evaluation["evaluation_protocol"] == run_slot_joint.EVALUATION_PROTOCOL
    assert not (tmp_path / "evaluation_trajectory.json").exists()


def test_joint_substeps_and_previous_episode_slot_reward():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    trace = SlotTrajectory.sample(scenario, slots=2, seed=7)
    env = SlotSequentialJointEnv(scenario, trace, mapping_samples=16)
    observation, _ = env.reset(seed=1)
    expected_steps = len(scenario.candidates) + (
        len(scenario.tools) * len(scenario.servers)
    ) + len(env.layout.model_groups)
    first = []
    for slot in range(2):
        observation, reward, terminated, info, steps = _complete_slot(env, observation)
        assert steps == expected_steps
        assert info["physical_slot"] == slot
        assert info["reward_baseline"] == "fixed_greedy"
        assert np.isclose(reward, info["reward_components"]["delta_utility"])
        first.append(info["reward_components"])
    assert terminated
    observation, _ = env.reset(seed=999)
    assert env.simulator.current_arrival_rates() == {
        (app.id, ingress): rate
        for app in trace.scenario_at(0, scenario).applications.values()
        for ingress, rate in app.ingress_rates.items()
    }
    for slot in range(2):
        observation, reward, terminated, info, _ = _complete_slot(env, observation)
        assert info["physical_slot"] == slot
        assert info["reward_baseline"] == "previous_round"
        assert np.isclose(reward, 0.0, atol=1e-8)
        for component in ("cost", "latency", "goodput", "quality"):
            assert np.isclose(info["reward_components"][f"delta_{component}_normalized"], 0.0, atol=1e-8)


def test_reward_state_rejects_different_trajectory():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    env = SlotSequentialJointEnv(scenario, SlotTrajectory.sample(scenario, 1, 2), mapping_samples=8)
    other = SlotSequentialJointEnv(scenario, SlotTrajectory.sample(scenario, 1, 3), mapping_samples=8)
    try:
        other.restore_reward_state(env.reward_state())
    except ValueError:
        pass
    else:
        raise AssertionError("A reward baseline from another trajectory was accepted")


def test_absolute_constraint_is_charged_even_when_slot_delta_is_zero():
    # The previous episode's constraint was also 0.4. Its zero *delta* does
    # not make a persistent violation feasible in the current episode.
    result = _constrained_utility(
        0.0, np.asarray([0.4]), np.asarray([2.0]), np.asarray([0.0]), True
    )
    assert result == -0.8


def test_slot_profile_separates_link_and_service_constraints():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    spec = replace(ObjectiveSpec.slo_constrained(0.9), network_utilization_target=0.9)
    evaluator = ObjectiveEvaluator(scenario, spec)
    value = evaluator.evaluate_arrays(
        cost=0.0, mean_latency=0.0, attainment=1.0, quality=0.0,
        app_latency={}, arrival_rates={},
        tool_utilization={"tool@server": 0.95},
        link_utilization={"a->b": 1.1},
        violation_labels=("link_overload:a->b",),
    )
    assert spec.constraint_names == ("llm", "service", "network", "attainment")
    assert np.isclose(value.constraints[1], 0.05)
    assert value.constraints[2] == 1.0


def test_phase_observations_include_previous_placement_and_local_group():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    env = SlotSequentialJointEnv(
        scenario, SlotTrajectory.sample(scenario, 2, 31), mapping_samples=8
    )
    obs, _ = env.reset(seed=2)
    assert obs["deployment_features"].shape == env.observation_space["deployment_features"].shape
    assert obs["routing_features"].shape == env.observation_space["routing_features"].shape
    assert np.all(obs["deployment_features"][-(
        len(env.layout.candidates) + len(env.layout.tools) * len(env.layout.servers)
    ):] == 0)
    obs, _, _, _, _ = _complete_slot(env, obs)
    previous = env.simulator.previous_deployment
    assert previous is not None
    beginning = -len(env.layout.candidates) - len(env.layout.tools) * len(env.layout.servers)
    for index, candidate in enumerate(env.layout.candidates):
        assert obs["deployment_features"][beginning + index] == previous.llm_active.get(candidate, 0)


def test_routing_warmup_placement_is_fixed_for_each_slot():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    env = SlotSequentialJointEnv(
        scenario, SlotTrajectory.sample(scenario, 2, 37), mapping_samples=8
    )
    env.training_phase = "composition"
    first, _ = env.reset(seed=1)
    target = env._stage_deployment.copy()
    while first["action_type"] != env.COMPOSITION:
        first, _, _, _, _ = env.step({"deploy": 0, "model": np.zeros(env.layout.model_action_size)})
    assert env.current_deployment == target
    env.reset(seed=999)
    assert env._stage_deployment == target


def test_deployment_uses_one_masked_count_action_per_pool():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    env = SlotSequentialJointEnv(
        scenario, SlotTrajectory.sample(scenario, 1, 41), mapping_samples=8
    )
    observation, _ = env.reset(seed=1)
    width = max(2, scenario.simulation.max_tool_replicas_per_server + 1)
    assert env.action_space["deploy"].n == width
    assert observation["deploy_mask"].shape == (width,)
    for _ in scenario.candidates:
        assert env._current_demand[0] == "llm"
        assert np.all(observation["deploy_mask"][2:] == 0)
        observation, _, _, _, _ = env.step({"deploy": 0})
    assert env._current_demand[0] == "tool"
    tool, server = env._current_demand[1:]
    for count in range(width):
        proposed = env.current_deployment.copy()
        proposed.tool_replicas[(tool, server)] = count
        assert bool(observation["deploy_mask"][count]) == env.planner.deployment_feasible(proposed)
    feasible_count = int(np.flatnonzero(observation["deploy_mask"])[-1])
    observation, _, _, _, _ = env.step({"deploy": feasible_count})
    assert env.current_deployment.tool_replicas[(tool, server)] == feasible_count
    assert env._deployment_target_index == len(scenario.candidates) + 1


def test_deployment_rejects_masked_count_without_advancing():
    scenario = ScenarioLoader.load("configs/toy.yaml")
    env = SlotSequentialJointEnv(
        scenario, SlotTrajectory.sample(scenario, 1, 43), mapping_samples=8
    )
    observation, _ = env.reset(seed=1)
    assert observation["deploy_mask"][2] == 0
    try:
        env.step({"deploy": 2})
    except ValueError:
        pass
    else:
        raise AssertionError("A masked LLM deployment action was accepted")
    assert env._deployment_target_index == 0


def test_slot_constraint_dual_update_ignores_intermediate_zero_substeps():
    intermediate = {
        "phase": 2, "period_complete": False,
        "constraint_vector": np.asarray([0.0]), "utility": 0.0,
        "learning_utility": 0.0, "reward": 0.0, "external_reward": 0.0,
    }
    terminal = {
        "phase": 2, "period_complete": True,
        "constraint_vector": np.asarray([0.4]), "utility": 1.0,
        "learning_utility": 0.2, "reward": 0.2, "external_reward": 0.2,
        "episode_utility_sum": 1.0, "episode_cost_sum": 0.0,
        "episode_latency_sum": 0.0, "episode_slot": 1, "trace_offset": 0,
    }
    batch = RolloutBatch(
        records=[intermediate, terminal], composition_indices=[0, 1],
        deployment_indices=[], completed_periods=1, episode_seeds=[],
        final_observation={}, episode_counter=0, exploration_weight=0.0,
        collection_time_s=0.0, intrinsic_raw=np.zeros(2),
        intrinsic_normalized=np.zeros(2), icm_raw=np.zeros(2),
    )
    config = replace(
        PPOConfig().for_constraint_count(1), lagrangian_learning_rates=(0.5,)
    )
    env = type("ConstraintEnv", (), {"constraint_names": ("attainment",)})()
    record, multipliers = build_update_record(
        update=0, batch=batch, config=config, env=env, losses=[],
        lagrange_multipliers=np.zeros(1),
    )
    assert np.isclose(record["mean_constraint_attainment"], 0.4)
    assert np.isclose(record["mean_utility"], 1.0)
    assert np.isclose(record["mean_period_return"], 0.2)
    assert np.isclose(multipliers[0], 0.2)
