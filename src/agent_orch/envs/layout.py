"""Action-space layout shared by every orchestration environment."""

from __future__ import annotations

from dataclasses import dataclass

from agent_orch.schema.models import Scenario


@dataclass(frozen=True)
class StructuredActionLayout:
    """Index-to-identity mapping for the hybrid action space."""

    models: tuple[str, ...]
    candidates: tuple[str, ...]
    servers: tuple[str, ...]
    tools: tuple[str, ...]
    model_groups: tuple[tuple[str, str], ...]
    deployment_targets: tuple[tuple[str, str], ...]

    @staticmethod
    def build(scenario: Scenario) -> "StructuredActionLayout":
        return StructuredActionLayout(
            models=tuple(scenario.models),
            candidates=tuple(scenario.candidates),
            servers=tuple(scenario.servers),
            tools=tuple(scenario.tools),
            model_groups=tuple(
                (app.id, ingress)
                for app in scenario.applications.values()
                for ingress in app.ingress_rates
            ),
            deployment_targets=(("keep", "0"), ("add", "1"), ("remove", "2")),
        )

    @property
    def deployment_action_size(self) -> int:
        return len(self.deployment_targets)

    @property
    def model_action_size(self) -> int:
        return len(self.model_groups) * len(self.models)
