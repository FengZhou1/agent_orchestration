from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

from agent_orch.schema.models import DeploymentDecision, Scenario


SCHEMA_VERSION = 1
TOOL_KEY_SEPARATOR = "|"


def deployment_signature(
    llm_active: dict[str, int], tool_replicas: dict[tuple[str, str], int]
) -> str:
    """Return the stable fingerprint used to deduplicate deployments."""
    payload = json.dumps(
        [
            sorted((key, int(value)) for key, value in llm_active.items()),
            sorted(
                (f"{tool}{TOOL_KEY_SEPARATOR}{server}", int(value))
                for (tool, server), value in tool_replicas.items()
            ),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _encode_tool_key(tool_id: str, server_id: str) -> str:
    return f"{tool_id}{TOOL_KEY_SEPARATOR}{server_id}"


def _decode_tool_key(key: str) -> tuple[str, str]:
    tool_id, server_id = key.split(TOOL_KEY_SEPARATOR, 1)
    return tool_id, server_id


def _split_test_count(n_entries: int, test_fraction: float) -> int:
    """Test-set size of one split group, never emptying either side when avoidable.

    ``ceil(n * test_fraction)`` entries go to the test set, except that a strict
    fraction keeps at least one entry in train and at least one in test; a group
    with a single entry (or none) goes entirely to train.
    """

    if n_entries <= 1:
        return 0
    count = int(math.ceil(n_entries * test_fraction))
    if 0.0 < test_fraction < 1.0:
        return min(max(count, 1), n_entries - 1)
    return min(max(count, 0), n_entries)


@dataclass(frozen=True)
class DeploymentEntry:
    """One auditable feasible deployment inside a :class:`DeploymentLibrary`.

    ``llm_active`` and ``tool_replicas`` are stored complete: every candidate and
    every ``(tool, server)`` pair of the scenario is present, with a zero value when
    inactive, so ``to_deployment()`` reproduces the planner's own layout.
    """

    index: int
    stratum: str
    llm_active: dict[str, int]
    tool_replicas: dict[tuple[str, str], int]
    n_models: int
    n_llm: int
    n_tool_replicas: int
    total_gpu: int
    cost_per_period: float
    signature: str

    def to_deployment(self) -> DeploymentDecision:
        return DeploymentDecision(dict(self.llm_active), dict(self.tool_replicas))

    @property
    def n_active_models(self) -> int:
        """Alias of :attr:`n_models`, read as "how many models this deployment runs"."""

        return self.n_models

    def active_models(self, scenario: Scenario) -> tuple[str, ...]:
        models = {
            scenario.candidates[candidate_id].model
            for candidate_id, active in self.llm_active.items()
            if active and candidate_id in scenario.candidates
        }
        return tuple(sorted(models))

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "stratum": self.stratum,
            "llm_active": {
                key: int(value) for key, value in sorted(self.llm_active.items())
            },
            "tool_replicas": {
                _encode_tool_key(tool_id, server_id): int(value)
                for (tool_id, server_id), value in sorted(self.tool_replicas.items())
            },
            "n_models": self.n_models,
            "n_llm": self.n_llm,
            "n_tool_replicas": self.n_tool_replicas,
            "total_gpu": self.total_gpu,
            "cost_per_period": self.cost_per_period,
            "signature": self.signature,
        }

    @staticmethod
    def from_json(payload: dict[str, Any]) -> "DeploymentEntry":
        llm_active = {
            str(key): int(value) for key, value in payload["llm_active"].items()
        }
        tool_replicas = {
            _decode_tool_key(key): int(value)
            for key, value in payload["tool_replicas"].items()
        }
        signature = deployment_signature(llm_active, tool_replicas)
        stored = str(payload["signature"])
        if stored != signature:
            raise ValueError(
                f"Deployment entry {payload['index']} has a stale signature"
            )
        return DeploymentEntry(
            index=int(payload["index"]),
            stratum=str(payload["stratum"]),
            llm_active=llm_active,
            tool_replicas=tool_replicas,
            n_models=int(payload["n_models"]),
            n_llm=int(payload["n_llm"]),
            n_tool_replicas=int(payload["n_tool_replicas"]),
            total_gpu=int(payload["total_gpu"]),
            cost_per_period=float(payload["cost_per_period"]),
            signature=signature,
        )


@dataclass(frozen=True)
class DeploymentLibrary:
    """An offline, stratified, feasible deployment catalog for one scenario.

    ``metadata`` carries the provenance needed to interpret the entries; in
    particular ``metadata["candidate_models"]`` maps every candidate id to its model
    so that :meth:`active_model_sets` works without the scenario object.
    """

    scenario_id: str
    scenario_hash: str
    entries: tuple[DeploymentEntry, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def entry(self, index: int) -> DeploymentEntry:
        if 0 <= index < len(self.entries):
            entry = self.entries[index]
            if entry.index == index:
                return entry
        # A sub-library (:meth:`subset`, :meth:`train_test_split`) keeps the
        # source indices, so its entries are not positional; fall back to a
        # lookup by index value.  For a positional library the loop is
        # unreachable and this method behaves exactly as before.
        for candidate in self.entries:
            if candidate.index == index:
                return candidate
        raise IndexError(f"Deployment index {index} is out of range")

    def to_deployment(self, index: int) -> DeploymentDecision:
        return self.entry(index).to_deployment()

    def by_stratum(self) -> dict[str, tuple[DeploymentEntry, ...]]:
        groups: dict[str, list[DeploymentEntry]] = {}
        for entry in self.entries:
            groups.setdefault(entry.stratum, []).append(entry)
        return {name: tuple(items) for name, items in groups.items()}

    def strata_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.stratum] = counts.get(entry.stratum, 0) + 1
        return counts

    def _candidate_model_map(self, scenario: Scenario | None) -> dict[str, str]:
        if scenario is not None:
            return {
                candidate_id: candidate.model
                for candidate_id, candidate in scenario.candidates.items()
            }
        raw = self.metadata.get("candidate_models")
        if raw is None:
            raise ValueError(
                "The library has no candidate_models metadata; pass the scenario"
            )
        return {str(key): str(value) for key, value in raw.items()}

    def active_model_sets(self, scenario: Scenario | None = None) -> set[tuple[str, ...]]:
        mapping = self._candidate_model_map(scenario)
        model_sets: set[tuple[str, ...]] = set()
        for entry in self.entries:
            models = {
                mapping[candidate_id]
                for candidate_id, active in entry.llm_active.items()
                if active and candidate_id in mapping
            }
            model_sets.add(tuple(sorted(models)))
        return model_sets

    def coverage_rows(self, scenario: Scenario | None = None) -> list[dict[str, Any]]:
        mapping = self._candidate_model_map(scenario)
        rows: list[dict[str, Any]] = []
        for entry in self.entries:
            models = sorted(
                {
                    mapping[candidate_id]
                    for candidate_id, active in entry.llm_active.items()
                    if active and candidate_id in mapping
                }
            )
            rows.append(
                {
                    "index": entry.index,
                    "stratum": entry.stratum,
                    "n_models": entry.n_models,
                    "active_models": "+".join(models),
                    "n_llm": entry.n_llm,
                    "n_tool_replicas": entry.n_tool_replicas,
                    "total_gpu": entry.total_gpu,
                    "cost_per_period": entry.cost_per_period,
                }
            )
        return rows

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "metadata": self.metadata,
            "entries": [entry.to_json() for entry in self.entries],
        }

    # ------------------------------------------------------------------ subsets

    def subset(self, indices: Sequence[int]) -> "DeploymentLibrary":
        """Return a library holding only the entries with these ``index`` values.

        The entries keep their original ``index``, so the subset is a *view* of the
        source library: index ``i`` still names the same deployment, which is what
        makes a train/test split comparable across libraries.  ``metadata`` records
        the provenance (``subset_of`` fingerprint and ``subset_size``); every other
        field is inherited unchanged, including the scenario hash.
        """

        requested = [int(index) for index in indices]
        entries = tuple(self.entry(index) for index in requested)
        fingerprint = hashlib.sha256(
            ",".join(entry.signature for entry in self.entries).encode("utf-8")
        ).hexdigest()[:16]
        metadata = dict(self.metadata)
        metadata["subset_of"] = f"{self.scenario_id}:{fingerprint}"
        metadata["subset_size"] = len(entries)
        return DeploymentLibrary(
            scenario_id=self.scenario_id,
            scenario_hash=self.scenario_hash,
            entries=entries,
            metadata=metadata,
        )

    def train_test_split(
        self,
        test_fraction: float = 0.25,
        seed: int = 2026,
        stratify: bool = True,
        group_by: str = "stratum",
    ) -> tuple["DeploymentLibrary", "DeploymentLibrary"]:
        """Split the library into ``(train, test)`` for a held-out evaluation gate.

        With ``stratify=True`` the split happens inside every group, using
        ``random.Random(f"{seed}:{group}")``, so each group contributes the same
        fraction to the test set.  At least one entry per group stays in train (a
        singleton group goes entirely to train); with ``stratify=False`` the whole
        library is shuffled with ``random.Random(f"{seed}:all")`` instead.

        ``group_by`` picks what "same fraction" is enforced over:

        * ``"stratum"`` splits the deployment-shape strata, which keeps every
          stratum's mix of placements together but sends singleton strata (the
          single-model subsets, for instance) entirely to train.
        * ``"n_models"`` splits by the number of active models instead, so the
          held-out side covers the model-availability axis.  Prefer this when the
          gate must exercise the composition mask; a stratum split can leave the
          test side with only full-model deployments.

        Both sides are non-empty whenever the library has at least two entries.  The
        two subsets keep their source indices and partition the source library, so
        ``train.strata_counts()`` plus ``test.strata_counts()`` equals the source
        counts and the union of the two index sets is the source index set.
        """

        if not 0.0 <= test_fraction <= 1.0:
            raise ValueError("test_fraction must lie in [0, 1]")
        if group_by not in ("stratum", "n_models"):
            raise ValueError(f"Unknown group_by {group_by!r}; expected stratum or n_models")
        groups: list[tuple[str, list[DeploymentEntry]]] = []
        if not stratify:
            groups = [("all", list(self.entries))]
        elif group_by == "n_models":
            bucketed: dict[str, list[DeploymentEntry]] = {}
            for entry in self.entries:
                bucketed.setdefault(f"n_models={entry.n_models}", []).append(entry)
            groups = [(name, bucketed[name]) for name in sorted(bucketed)]
        else:
            by_stratum = self.by_stratum()
            groups = [(name, list(by_stratum[name])) for name in sorted(by_stratum)]

        train_indices: list[int] = []
        test_indices: list[int] = []
        for name, entries in groups:
            shuffled = list(entries)
            random.Random(f"{seed}:{name}").shuffle(shuffled)
            n_test = _split_test_count(len(shuffled), test_fraction)
            test_indices.extend(entry.index for entry in shuffled[:n_test])
            train_indices.extend(entry.index for entry in shuffled[n_test:])

        train = self.subset(sorted(train_indices))
        test = self.subset(sorted(test_indices))
        for library, role in ((train, "train"), (test, "test")):
            library.metadata.update(
                {
                    "split_role": role,
                    "split_seed": int(seed),
                    "split_test_fraction": float(test_fraction),
                    "split_stratified": bool(stratify),
                    "split_group_by": group_by,
                }
            )
        return train, test

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "DeploymentLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(payload.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported deployment-library schema_version {version}; "
                f"expected {SCHEMA_VERSION}"
            )
        return cls(
            scenario_id=str(payload["scenario_id"]),
            scenario_hash=str(payload["scenario_hash"]),
            entries=tuple(
                DeploymentEntry.from_json(item) for item in payload["entries"]
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def default_path(scenario_id: str, base: str | Path = "data/processed") -> Path:
        return Path(base) / f"deployment_library_{scenario_id}.json"

    @staticmethod
    def scenario_hash_of(scenario: Scenario) -> str:
        """Content hash of the deployment-relevant scenario catalog.

        The repository hashes scenario *files* with ``sha256(path.read_bytes())``
        (see ``scripts/calibrate_load_levels.py``); an in-memory ``Scenario`` has no
        path, so this uses a canonical JSON rendering of the same catalog instead.
        Only fields that change deployment feasibility are covered: servers, links,
        models, LLM configs, tools, candidates and the simulation spec. Editing the
        application/workload section alone therefore does not invalidate a library.
        """
        payload = {
            "id": scenario.id,
            "simulation": asdict(scenario.simulation),
            "servers": [asdict(item) for item in scenario.servers.values()],
            "links": [asdict(item) for item in scenario.links],
            "models": [asdict(item) for item in scenario.models.values()],
            "llm_configs": [asdict(item) for item in scenario.llm_configs.values()],
            "tools": [asdict(item) for item in scenario.tools.values()],
            "candidates": [asdict(item) for item in scenario.candidates.values()],
        }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
