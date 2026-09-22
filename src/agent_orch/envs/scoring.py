"""One scoring path for a composition, shared by the solver and the gate.

The inner solver used to score candidates on a raw :class:`Simulator` rollout with
a constant arrival trace, while the gate scored them through
:class:`CompositionLibraryEnv` on the real trace.  For the same deployment,
composition and protocol the two produced different utilities from *identical*
metrics (0.189 vs 0.246 on ``agent-abilene-20`` index 2), so on 10 of 30 test
deployments the solver's answer lost to a plain ``quality_greedy`` composition
under the metric the gate reports.  A reference that is not the argmax of the
metric it is compared against cannot support a rank or capture claim.

Everything that scores a composition now goes through :func:`build_composition_env`
and :func:`score_composition`, so the two can no longer drift apart.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..deployment import DeploymentLibrary
from ..objective import ObjectiveEvaluator, ObjectiveSpec
from ..schema.models import Scenario
from ..workload import ArrivalTrace
from .composition_env import CompositionLibraryEnv


def build_composition_env(
    scenario: Scenario,
    objective: ObjectiveSpec,
    arrival_trace: ArrivalTrace,
    deployment_library: DeploymentLibrary,
    *,
    position: int,
    periods: int,
    mapping_samples: int = 128,
    seed: int = 0,
    use_uniform_baseline: bool = False,
) -> CompositionLibraryEnv:
    """An environment pinned to one library deployment, for scoring compositions.

    ``use_uniform_baseline`` is off by default: the control variate shifts the
    observation and the training reward but not ``info["utility"]``, and computing
    it costs a full scratch rollout per reset.
    """

    return CompositionLibraryEnv(
        scenario,
        max_slots=max(1, int(periods)),
        seed=seed,
        arrival_trace=arrival_trace,
        mapping_samples=mapping_samples,
        objective=objective,
        deployment_library=deployment_library,
        fixed_deployment_index=position,
        use_uniform_baseline=use_uniform_baseline,
    )


def dense_composition_action(env: CompositionLibraryEnv, share: Mapping) -> np.ndarray:
    """``share`` laid out in the environment's flat model action."""

    dense = np.zeros(
        (len(env.layout.model_groups), len(env.layout.models)), dtype=np.float32
    )
    for group_index, (app_id, ingress) in enumerate(env.layout.model_groups):
        for model_index, model in enumerate(env.layout.models):
            dense[group_index, model_index] = float(share.get((app_id, ingress, model), 0.0))
    return dense.reshape(-1)


def score_composition(
    env: CompositionLibraryEnv,
    share: Mapping,
    *,
    periods: int,
    warmup: int,
) -> float:
    """Mean post-warmup ``info["utility"]`` of holding ``share`` for ``periods``.

    The composition is held fixed while the environment's own router and
    utilization feedback run, so the score is the objective a composition policy
    trained in this environment actually optimises.
    """

    action = dense_composition_action(env, share)
    observation, _ = env.reset(seed=env._seed)  # noqa: SLF001 - env owns its seed
    utilities: list[float] = []
    for period in range(max(1, int(periods))):
        observation, _, terminated, truncated, info = env.step(
            {"deploy": 0, "model": action.copy()}
        )
        if period >= warmup and info.get("period_complete"):
            utilities.append(float(info["utility"]))
        if terminated or truncated:
            break
    if not utilities:
        return float("nan")
    return float(np.mean(utilities))


class CompositionEnvScorer:
    """Scores compositions for one fixed deployment through the environment."""

    def __init__(
        self,
        scenario: Scenario,
        evaluator: ObjectiveEvaluator,
        objective: ObjectiveSpec,
        arrival_trace: ArrivalTrace,
        deployment_library: DeploymentLibrary,
        *,
        position: int,
        periods: int,
        warmup: int,
        mapping_samples: int = 128,
        seed: int = 0,
    ) -> None:
        self.periods = max(1, int(periods))
        self.warmup = max(0, int(warmup))
        self.deployment = deployment_library.entries[position].to_deployment()
        self.env = build_composition_env(
            scenario,
            objective,
            arrival_trace,
            deployment_library,
            position=position,
            periods=self.periods,
            mapping_samples=mapping_samples,
            seed=seed,
        )
        # The evaluator stays reachable for callers that want the decomposed value
        # of the last scored period.
        self.evaluator = evaluator

    def __call__(self, deployment, share: Mapping) -> float:
        """Score ``share`` for ``deployment``.

        The deployment argument is checked rather than ignored: the environment is
        pinned at construction, so scoring a different deployment's composition
        here would silently return a number for the wrong problem.
        """

        if deployment != self.deployment:
            raise ValueError(
                "this scorer is pinned to one deployment; rebuild it per deployment"
            )
        return score_composition(
            self.env, share, periods=self.periods, warmup=self.warmup
        )
