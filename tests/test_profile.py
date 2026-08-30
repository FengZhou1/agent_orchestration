import pandas as pd

from agent_orch.backends import ProfileBackend
from agent_orch.baselines import GreedyPolicy
from agent_orch.simulator import Simulator


def test_profile_backend_returns_finite_estimate():
    rows = []
    for prompt in (128.0, 512.0):
        for output in (16.0, 64.0):
            for arrival in (1.0, 4.0):
                for fraction in (0.0, 1.0):
                    rows.append(
                        {
                            "model": "small",
                            "config": "edge",
                            "prompt_tokens": prompt,
                            "output_tokens": output,
                            "arrival_rate_rps": arrival,
                            "long_request_fraction": fraction,
                            "ttft_s": 0.1 + prompt / 10000 + arrival / 100,
                            "tbt_s": 0.02 + output / 10000,
                            "response_s": 0.2 + output / 1000,
                            "stable_capacity_rps": 8.0,
                            "kv_tokens": prompt + output,
                        }
                    )
    backend = ProfileBackend(pd.DataFrame(rows))
    estimate = backend.estimate("small", "edge", 256, 32, 2, 0.5)
    assert estimate.ttft_s > 0
    assert estimate.response_s > estimate.ttft_s
    assert estimate.interpolation == "linear"


def test_profile_backend_can_replace_analytical_llm_component(scenario):
    rows = []
    for config in scenario.llm_configs.values():
        for prompt in (128.0, 1024.0):
            for output in (16.0, 128.0):
                for arrival in (0.1, 8.0):
                    for fraction in (0.0, 1.0):
                        rows.append(
                            {
                                "model": config.model,
                                "config": config.id,
                                "prompt_tokens": prompt,
                                "output_tokens": output,
                                "arrival_rate_rps": arrival,
                                "long_request_fraction": fraction,
                                "ttft_s": 0.1 + prompt / 5000 + arrival / 100,
                                "tbt_s": 0.02 + output / 10000,
                                "response_s": 0.3 + prompt / 5000 + output / 1000,
                                "stable_capacity_rps": 10.0,
                                "kv_tokens": prompt + output,
                            }
                        )
    profile = ProfileBackend(pd.DataFrame(rows))
    policy = GreedyPolicy(scenario, seed=4)
    deployment, routing = policy.decide()
    simulator = Simulator(scenario, llm_profile_backend=profile)
    metrics = simulator.step(deployment, routing).metrics
    assert metrics.mean_latency_s > 0.0
    assert metrics.total_arrival_rps > 0.0
