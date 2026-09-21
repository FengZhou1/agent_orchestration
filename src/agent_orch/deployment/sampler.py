from __future__ import annotations

import random
from typing import Mapping

from agent_orch.schema.models import DeploymentDecision

from .library import DeploymentEntry, DeploymentLibrary


class StratifiedSampler:
    """Sample deployments from a library, either stratified or uniformly.

    ``cycle`` walks the strata in a fixed order and draws one entry per stratum per
    epoch (``weights`` raise a stratum's per-epoch quota), so over full epochs every
    stratum is drawn the same number of times and the composition policy sees the
    whole deployment shape space. ``uniform`` ignores the strata unless ``weights``
    is given, in which case the stratum is drawn proportionally to its weight and the
    entry uniformly inside it. ``fixed`` always returns one deployment.
    """

    MODES = ("cycle", "uniform", "fixed")

    def __init__(
        self,
        library: DeploymentLibrary,
        seed: int = 0,
        mode: str = "cycle",
        stratum: str | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown sampling mode {mode!r}; expected {self.MODES}")
        if not library.entries:
            raise ValueError("Cannot sample from an empty deployment library")
        self.library = library
        self.seed = int(seed)
        self.mode = mode
        self.stratum = stratum
        self.weights = dict(weights or {})
        self._groups = library.by_stratum()
        if stratum is not None and stratum not in self._groups:
            raise ValueError(f"Unknown stratum {stratum!r}")
        if mode == "fixed":
            self._fixed_index = (
                self._groups[stratum][0].index if stratum is not None else 0
            )
        self._plan: list[str] = []
        for name in self._groups:
            quota = 1
            if self.weights:
                quota = max(1, int(round(self.weights.get(name, 1.0))))
            self._plan.extend([name] * quota)
        self.reset()

    def __len__(self) -> int:
        return len(self.library.entries)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._counts: dict[str, int] = {name: 0 for name in self._groups}
        self._plan_position = 0
        self._rng = random.Random(self.seed)
        self._queues: dict[str, list[DeploymentEntry]] = {}
        self._rngs: dict[str, random.Random] = {}
        for name, entries in self._groups.items():
            self._rngs[name] = random.Random(f"{self.seed}:{name}")
            self._queues[name] = self._shuffled(name, entries)

    def next_index(self) -> int:
        return self.next_entry().index

    def next_entry(self) -> DeploymentEntry:
        if self.mode == "fixed":
            return self.library.entry(self._fixed_index)
        if self.mode == "uniform":
            return self._uniform_entry()
        name = self._plan[self._plan_position]
        self._plan_position = (self._plan_position + 1) % len(self._plan)
        return self._draw(name)

    def next_deployment(self) -> DeploymentDecision:
        return self.next_entry().to_deployment()

    def strata_coverage_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def _uniform_entry(self) -> DeploymentEntry:
        if not self.weights:
            return self._rng.choice(self.library.entries)
        names = list(self._groups)
        weights = [max(0.0, float(self.weights.get(name, 1.0))) for name in names]
        if sum(weights) <= 0.0:
            weights = [1.0] * len(names)
        name = self._rng.choices(names, weights=weights, k=1)[0]
        return self._draw(name)

    def _draw(self, name: str) -> DeploymentEntry:
        queue = self._queues[name]
        if not queue:
            queue = self._shuffled(name, self._groups[name])
            self._queues[name] = queue
        entry = queue.pop()
        self._counts[name] += 1
        return entry

    def _shuffled(
        self, name: str, entries: tuple[DeploymentEntry, ...]
    ) -> list[DeploymentEntry]:
        values = list(entries)
        self._rngs[name].shuffle(values)
        return values


class FixedDeploymentSampler:
    """Always return one library deployment (diagnostic single-deployment runs).

    ``index`` is positional within the library's ``entries``.  Sub-libraries keep
    their source indices, so resolving by index value would raise on a split
    whose entries are numbered 2, 19, 22, ... rather than 0, 1, 2, ...
    """

    def __init__(self, library: DeploymentLibrary, index: int = 0) -> None:
        if not library.entries:
            raise ValueError("Cannot sample from an empty deployment library")
        self.library = library
        self.index = int(index) % len(library.entries)
        self._entry = library.entries[self.index]

    def __len__(self) -> int:
        return len(self.library.entries)

    def reset(self, seed: int | None = None) -> None:
        return None

    def next_index(self) -> int:
        return self.index

    def next_entry(self) -> DeploymentEntry:
        return self._entry

    def next_deployment(self) -> DeploymentDecision:
        return self._entry.to_deployment()

    def strata_coverage_counts(self) -> dict[str, int]:
        return {self._entry.stratum: 1}
