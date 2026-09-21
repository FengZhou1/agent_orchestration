"""Metric sinks that mirror training records to TensorBoard and SwanLab.

Every backend is optional: the third-party package it needs is imported inside the
constructor, and ``make_writer`` degrades to :class:`NullWriter` when the import or
the backend initialisation fails.  Telemetry must never be able to abort a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable
import warnings

import numpy as np


@runtime_checkable
class MetricWriter(Protocol):
    """Minimal metric sink shared by every telemetry backend."""

    def log_scalars(
        self, step: int, values: Mapping[str, float], prefix: str = ""
    ) -> None:
        """Record one scalar per key, tagged ``prefix + key``, at ``step``."""

    def log_histogram(self, step: int, name: str, values: Iterable[float]) -> None:
        """Record the distribution of ``values`` under ``name`` at ``step``."""

    def log_text(self, step: int, name: str, text: str) -> None:
        """Record free-form text under ``name`` at ``step``."""

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Record the run configuration once."""

    def finish(self) -> None:
        """Flush pending records and end the run; idempotent."""

    def close(self) -> None:
        """Release backend resources; idempotent."""


class NullWriter(MetricWriter):
    """Writer that discards everything, used when no backend is available."""

    def log_scalars(
        self, step: int, values: Mapping[str, float], prefix: str = ""
    ) -> None:
        pass

    def log_histogram(self, step: int, name: str, values: Iterable[float]) -> None:
        pass

    def log_text(self, step: int, name: str, text: str) -> None:
        pass

    def log_config(self, config: Mapping[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardWriter(MetricWriter):
    """TensorBoard-backed writer.

    ``torch.utils.tensorboard`` is imported lazily so a missing ``tensorboard``
    package only fails when this writer is actually requested.  The configuration
    is written with ``add_text`` rather than ``add_hparams`` because the latter
    mixes badly with the ``global_step`` of the training scalars.
    """

    def __init__(self, log_dir: str | Path, *, flush_secs: int = 30) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoardWriter requires the optional dependency 'tensorboard' "
                "(imported as torch.utils.tensorboard). Install it with "
                "`pip install tensorboard` or `pip install -e .[telemetry]`."
            ) from error
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(
            log_dir=str(self.log_dir), flush_secs=flush_secs
        )
        self._closed = False

    def log_scalars(
        self, step: int, values: Mapping[str, float], prefix: str = ""
    ) -> None:
        if self._closed:
            return
        for name, value in values.items():
            self._writer.add_scalar(
                f"{prefix}{name}", float(value), global_step=int(step)
            )

    def log_histogram(self, step: int, name: str, values: Iterable[float]) -> None:
        if self._closed:
            return
        array = np.asarray([float(value) for value in values], dtype=np.float64)
        if array.size == 0:
            return
        self._writer.add_histogram(name, array, global_step=int(step))

    def log_text(self, step: int, name: str, text: str) -> None:
        if self._closed:
            return
        self._writer.add_text(name, text, global_step=int(step))

    def log_config(self, config: Mapping[str, Any]) -> None:
        if self._closed:
            return
        body = json.dumps(
            to_jsonable(config), indent=2, sort_keys=True, ensure_ascii=False
        )
        self._writer.add_text("config", f"```yaml\n{body}\n```", global_step=0)

    def finish(self) -> None:
        if self._closed:
            return
        self._writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.flush()
        self._writer.close()


class SwanLabWriter(MetricWriter):
    """SwanLab-backed writer.

    ``swanlab`` is imported lazily.  A missing package raises ``RuntimeError`` so
    that :func:`make_writer` can degrade, while a failed ``swanlab.init`` (no
    login, no network, another live run) only warns once and turns this writer
    into a no-op: losing telemetry is always preferable to aborting training.
    """

    def __init__(
        self,
        run_name: str,
        logdir: str | Path,
        *,
        project: str = "agent-orch",
        config: Mapping[str, Any] | None = None,
        offline: bool = True,
    ) -> None:
        try:
            import swanlab
        except ImportError as error:
            raise RuntimeError(
                "SwanLabWriter requires the optional dependency 'swanlab'. "
                "Install it with `pip install swanlab` or "
                "`pip install -e .[telemetry]`."
            ) from error
        self._swanlab = swanlab
        self._warned: set[str] = set()
        self._config: dict[str, Any] = dict(config or {})
        self._active = False
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        try:
            swanlab.init(
                project=project,
                name=run_name,
                config=self._config,
                log_dir=str(self.logdir),
                mode="offline" if offline else "online",
                # A process-level singleton run: replacing an unfinished one is
                # what a new training run in the same process wants.
                reinit=True,
            )
        except Exception as error:  # noqa: BLE001 - login/network failures
            self._warn_once(
                "init",
                f"telemetry: swanlab.init failed ({error!r}); swanlab logging "
                "is disabled for this writer.",
            )
            return
        self._active = True

    @property
    def active(self) -> bool:
        """Whether a SwanLab run is live and records are being forwarded."""
        return self._active

    def log_scalars(
        self, step: int, values: Mapping[str, float], prefix: str = ""
    ) -> None:
        if not self._active:
            return
        payload = {f"{prefix}{name}": float(value) for name, value in values.items()}
        if not payload:
            return
        self._call("log", payload, step=int(step))

    def log_histogram(self, step: int, name: str, values: Iterable[float]) -> None:
        if not self._active:
            return
        array = np.asarray([float(value) for value in values], dtype=np.float64)
        if array.size == 0:
            return
        # swanlab 0.10 has no histogram primitive: logging a raw list is rejected
        # by its parser, so the distribution is summarised with scalars.
        self._call(
            "log",
            {
                f"{name}/mean": float(array.mean()),
                f"{name}/std": float(array.std()),
                f"{name}/min": float(array.min()),
                f"{name}/max": float(array.max()),
                f"{name}/count": float(array.size),
            },
            step=int(step),
        )

    def log_text(self, step: int, name: str, text: str) -> None:
        if not self._active:
            return
        self._call("log", {name: self._swanlab.Text(text)}, step=int(step))

    def log_config(self, config: Mapping[str, Any]) -> None:
        payload = dict(config)
        self._config.update(payload)
        if not self._active:
            return
        try:
            self._swanlab.config.update(payload)
        except Exception as error:  # noqa: BLE001 - never abort a run
            self._warn_once(
                "config", f"telemetry: swanlab.config.update failed ({error!r})."
            )

    def finish(self) -> None:
        if not self._active:
            return
        self._active = False
        self._call("finish")

    def close(self) -> None:
        self.finish()

    def _call(self, name: str, *args: Any, **kwargs: Any) -> None:
        if not self._active:
            return
        try:
            getattr(self._swanlab, name)(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - never abort a run
            self._warn_once(
                name, f"telemetry: swanlab.{name} failed ({error!r}); the record "
                "was dropped."
            )

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


class MultiWriter(MetricWriter):
    """Fan out every record to several writers, tolerating individual failures.

    A child that raises is reported once and then skipped for the rest of the run,
    so a backend that dies mid-training cannot flood the log or stop the others.
    """

    def __init__(self, writers: Sequence[MetricWriter]) -> None:
        self._writers = list(writers)
        self._failed: set[str] = set()

    @property
    def writers(self) -> tuple[MetricWriter, ...]:
        """The child writers, in fan-out order."""
        return tuple(self._writers)

    @property
    def failed_writers(self) -> frozenset[str]:
        """Labels of the children disabled after raising."""
        return frozenset(self._failed)

    def log_scalars(
        self, step: int, values: Mapping[str, float], prefix: str = ""
    ) -> None:
        self._dispatch("log_scalars", step, values, prefix)

    def log_histogram(self, step: int, name: str, values: Iterable[float]) -> None:
        self._dispatch("log_histogram", step, name, values)

    def log_text(self, step: int, name: str, text: str) -> None:
        self._dispatch("log_text", step, name, text)

    def log_config(self, config: Mapping[str, Any]) -> None:
        self._dispatch("log_config", config)

    def finish(self) -> None:
        self._dispatch("finish")

    def close(self) -> None:
        self._dispatch("close")

    def _dispatch(self, method: str, *args: Any, **kwargs: Any) -> None:
        for index, writer in enumerate(self._writers):
            label = f"{index}:{type(writer).__name__}"
            if label in self._failed:
                continue
            try:
                getattr(writer, method)(*args, **kwargs)
            except Exception as error:  # noqa: BLE001 - isolate child failures
                self._failed.add(label)
                warnings.warn(
                    f"telemetry: {type(writer).__name__} failed on {method} "
                    f"({error!r}); it is disabled for the rest of the run.",
                    RuntimeWarning,
                    stacklevel=3,
                )


WRITER_KINDS: tuple[str, ...] = ("none", "tensorboard", "swanlab", "both")


def make_writer(
    kind: str,
    *,
    log_dir: str | Path,
    run_name: str,
    project: str = "agent-orch",
    config: Mapping[str, Any] | None = None,
    offline: bool = True,
    verbose: bool = True,
) -> MetricWriter:
    """Build the writer named by ``kind``, degrading instead of failing.

    Each backend owns a subdirectory of ``log_dir`` (``tensorboard/`` and
    ``swanlab/``) so that the two can be enabled together.  A backend whose
    dependency is missing is dropped with a warning; if none survives the result
    is a :class:`NullWriter`.
    """
    normalized = str(kind).strip().lower()
    if normalized not in WRITER_KINDS:
        raise ValueError(
            f"unknown telemetry writer kind {kind!r}; expected one of "
            f"{', '.join(WRITER_KINDS)}"
        )
    if normalized == "none":
        return NullWriter()

    root = Path(log_dir)
    writers: list[MetricWriter] = []
    if normalized in ("tensorboard", "both"):
        try:
            writers.append(TensorBoardWriter(root / "tensorboard"))
        except RuntimeError as error:
            _warn_unavailable("TensorBoard", error, verbose)
    if normalized in ("swanlab", "both"):
        try:
            writers.append(
                SwanLabWriter(
                    run_name,
                    root / "swanlab",
                    project=project,
                    config=config,
                    offline=offline,
                )
            )
        except RuntimeError as error:
            _warn_unavailable("SwanLab", error, verbose)

    if not writers:
        return NullWriter()
    if len(writers) == 1:
        return writers[0]
    return MultiWriter(writers)


def _warn_unavailable(backend: str, error: RuntimeError, verbose: bool) -> None:
    if not verbose:
        return
    warnings.warn(
        f"telemetry: {backend} writer unavailable ({error}); falling back to "
        "no-op for this backend.",
        RuntimeWarning,
        stacklevel=3,
    )


def to_jsonable(value: Any) -> Any:
    """Convert a configuration tree into JSON/YAML-serialisable primitives."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _as_float(value: Any) -> float | None:
    """Coerce a metric to ``float``, returning ``None`` when it is not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
