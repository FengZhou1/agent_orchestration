from pathlib import Path

import pytest

from agent_orch.schema.loader import ScenarioLoader


@pytest.fixture(scope="session")
def scenario():
    path = Path(__file__).parents[1] / "configs" / "toy.yaml"
    return ScenarioLoader.load(path)

