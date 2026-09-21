"""Run-scoped logging facade that namespaces training records by metric group."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .writers import MetricWriter, _as_float, make_writer, to_jsonable

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps tqdm/torch optional
    from agent_orch.agents.progress import TrainingProgressReporter


# Ordered: the first group with a matching needle wins.  That matters for the
# real PPO record, where "mean_rnd_loss" is a loss rather than an exploration
# term, and where "mean_lagrangian_reward" (which does not contain "lagrange")
# is an objective term rather than a constraint one.
_GROUP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("loss", ("loss",)),
    ("constraint", ("lagrange", "constraint", "violation")),
    ("exploration", ("intrinsic", "exploration", "rnd", "icm")),
    ("timing", ("time", "elapsed")),
    ("objective", ("utility", "return", "reward")),
)
_DEFAULT_GROUP = "train"
_CONFIG_FILENAME = "run_config.json"


def _group_for(name: str) -> str:
    lowered = name.lower()
    for group, needles in _GROUP_RULES:
        if any(needle in lowered for needle in needles):
            return group
    return _DEFAULT_GROUP


class RunLogger:
    """Write one training run's records to a :class:`MetricWriter` and to disk.

    Grouping turns a flat PPO update record into stable dashboard namespaces
    (``loss/``, ``objective/``, ``constraint/``, ...) instead of one flat series
    list.  Values that cannot be coerced to ``float`` are dropped rather than
    raised: logging is never allowed to abort a run.
    """

    def __init__(
        self,
        writer: MetricWriter,
        output_dir: str | Path,
        run_name: str,
        config: Mapping[str, Any] | None = None,
        progress: TrainingProgressReporter | None = None,
    ) -> None:
        self.writer = writer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.progress = progress
        self._config: dict[str, Any] = {}
        self._finished = False
        if config is not None:
            self.log_config(config)

    @property
    def config(self) -> dict[str, Any]:
        """The configuration accumulated by :meth:`log_config`."""
        return dict(self._config)

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Forward the configuration to the writer and persist it as JSON.

        The writer receives the payload as given, while ``run_config.json``
        always holds everything accumulated so far, so a second call with extra
        fields cannot drop the ones recorded at start-up.
        """
        payload = dict(config)
        self._config.update(payload)
        self.writer.log_config(payload)
        path = self.output_dir / _CONFIG_FILENAME
        path.write_text(
            json.dumps(
                to_jsonable(self._config),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def log_update(
        self, update: int, record: Mapping[str, float]
    ) -> dict[str, dict[str, float]]:
        """Write one PPO update record, grouped by metric namespace."""
        grouped: dict[str, dict[str, float]] = {}
        for name, value in record.items():
            amount = _as_float(value)
            if amount is None:
                continue
            grouped.setdefault(_group_for(name), {})[name] = amount
        for group, values in grouped.items():
            self.writer.log_scalars(update, values, prefix=f"{group}/")
        if self.progress is not None:
            self.progress.on_update(dict(record))
        return grouped

    def log_validation(
        self,
        update: int,
        metrics: Mapping[str, float],
        contexts: int | None = None,
    ) -> dict[str, float]:
        """Write one validation pass under the ``validation/`` namespace."""
        flat = {
            f"validation/{name}": amount
            for name, value in metrics.items()
            if (amount := _as_float(value)) is not None
        }
        if contexts is not None:
            flat["validation/contexts"] = float(contexts)
        if flat:
            self.writer.log_scalars(update, flat)
        return flat

    def log_action_distribution(
        self,
        update: int,
        model_share: Mapping[Any, float],
        deploy_action_counts: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Write model-selection shares and deployment action counts.

        Keys of ``model_share`` are ``(app_id, ingress, model)`` triples, which
        are aggregated per ``(app_id, model)`` because the ingress dimension is
        not comparable across access points.  Any other key shape is written
        verbatim so that an unexpected producer still yields a readable series.
        """
        aggregated: dict[str, float] = {}
        for key, value in model_share.items():
            amount = _as_float(value)
            if amount is None:
                continue
            if isinstance(key, tuple) and len(key) == 3:
                app_id, _ingress, model = key
                tag = f"action/model_share/{app_id}/{model}"
            else:
                tag = f"action/model_share/{key}"
            aggregated[tag] = aggregated.get(tag, 0.0) + amount
        if deploy_action_counts is not None:
            for name, value in deploy_action_counts.items():
                amount = _as_float(value)
                if amount is None:
                    continue
                aggregated[f"action/deploy/{name}"] = amount
        if aggregated:
            self.writer.log_scalars(update, aggregated)
        return aggregated

    def log_reward_audit(
        self,
        step: int,
        per_term: Mapping[str, float],
        prefix: str = "audit/reward",
    ) -> dict[str, float]:
        """Write the sensitivity span of each reward term.

        Each term is written as ``<prefix>/<term>`` and the overall span of the
        terms as ``<prefix>/span``, which is what a reward-shaping audit reads to
        see whether any single term dominates the utility.
        """
        flat = {
            f"{prefix}/{name}": amount
            for name, value in per_term.items()
            if (amount := _as_float(value)) is not None
        }
        if not flat:
            return flat
        values = list(flat.values())
        flat[f"{prefix}/span"] = max(values) - min(values)
        self.writer.log_scalars(step, flat)
        return flat

    def log_episode(
        self, update: int, episode_index: int, info: Mapping[str, Any]
    ) -> dict[str, float]:
        """Write the scalar fields of one evaluation episode."""
        flat: dict[str, float] = {"episode/index": float(episode_index)}
        for name, value in info.items():
            amount = _as_float(value)
            if amount is not None:
                flat[f"episode/{name}"] = amount
        self.writer.log_scalars(update, flat)
        return flat

    def log_histogram(self, step: int, name: str, values: Iterable[float]) -> None:
        """Forward a histogram to the writer."""
        self.writer.log_histogram(step, name, values)

    def log_text(self, step: int, name: str, text: str) -> None:
        """Forward a text record to the writer."""
        self.writer.log_text(step, name, text)

    def finish(self) -> None:
        """End the run; safe to call more than once."""
        if self._finished:
            return
        self._finished = True
        self.writer.finish()

    def close(self) -> None:
        """Release writer resources; safe to call more than once."""
        self.finish()
        self.writer.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        self.close()
        return False


def build_run_logger(
    kind: str,
    output_dir: str | Path,
    run_name: str,
    config: Mapping[str, Any] | None = None,
    progress: TrainingProgressReporter | None = None,
    project: str = "agent-orch",
    offline: bool = True,
) -> RunLogger:
    """Build a :class:`RunLogger` for the backend named by ``kind``.

    ``kind`` is one of ``"none"``, ``"tensorboard"``, ``"swanlab"`` or ``"both"``;
    an unavailable backend degrades to a no-op instead of raising.
    """
    writer = make_writer(
        kind,
        log_dir=output_dir,
        run_name=run_name,
        project=project,
        config=config,
        offline=offline,
    )
    return RunLogger(
        writer, output_dir, run_name, config=config, progress=progress
    )
