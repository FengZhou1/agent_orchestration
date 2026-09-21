from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import pytest

from agent_orch.telemetry import (
    MetricWriter,
    MultiWriter,
    NullWriter,
    RunLogger,
    SwanLabWriter,
    TensorBoardWriter,
    build_run_logger,
    make_writer,
)


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[int, dict[str, float], str]] = []
        self.histograms: list[tuple[int, str, list[float]]] = []
        self.texts: list[tuple[int, str, str]] = []
        self.configs: list[dict[str, Any]] = []
        self.finish_calls = 0
        self.close_calls = 0

    def log_scalars(
        self, step: int, values: Mapping[str, float], prefix: str = ""
    ) -> None:
        self.scalars.append((step, dict(values), prefix))

    def log_histogram(self, step: int, name: str, values: Iterable[float]) -> None:
        self.histograms.append((step, name, list(values)))

    def log_text(self, step: int, name: str, text: str) -> None:
        self.texts.append((step, name, text))

    def log_config(self, config: Mapping[str, Any]) -> None:
        self.configs.append(dict(config))

    def finish(self) -> None:
        self.finish_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _FailingWriter(_RecordingWriter):
    def log_scalars(
        self, step: int, values: Mapping[str, float], prefix: str = ""
    ) -> None:
        raise RuntimeError("backend exploded")


class _StubProgress:
    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []

    def on_update(self, record: dict[str, float]) -> None:
        self.records.append(dict(record))


def _exercise(writer: MetricWriter) -> None:
    writer.log_scalars(1, {"a": 1.0}, prefix="train/")
    writer.log_histogram(1, "hist", [1.0, 2.0, 3.0])
    writer.log_text(1, "note", "hello")
    writer.log_config({"seed": 1})
    writer.finish()
    writer.close()


def test_null_writer_methods_are_no_ops():
    writer = NullWriter()
    assert isinstance(writer, MetricWriter)
    _exercise(writer)


def test_multi_writer_fans_out_to_every_child():
    first = _RecordingWriter()
    second = _RecordingWriter()
    writer = MultiWriter([first, second])

    _exercise(writer)

    for child in (first, second):
        assert child.scalars == [(1, {"a": 1.0}, "train/")]
        assert child.histograms == [(1, "hist", [1.0, 2.0, 3.0])]
        assert child.texts == [(1, "note", "hello")]
        assert child.configs == [{"seed": 1}]
        assert child.finish_calls == 1
        assert child.close_calls == 1
    assert writer.failed_writers == frozenset()


def test_multi_writer_isolates_a_failing_child():
    broken = _FailingWriter()
    healthy = _RecordingWriter()
    writer = MultiWriter([broken, healthy])

    with pytest.warns(RuntimeWarning, match="disabled for the rest of the run"):
        _exercise(writer)

    assert healthy.scalars == [(1, {"a": 1.0}, "train/")]
    assert healthy.finish_calls == 1
    assert healthy.close_calls == 1
    assert len(writer.failed_writers) == 1
    assert "0:_FailingWriter" in writer.failed_writers
    assert writer.writers == (broken, healthy)


def test_run_logger_groups_update_record(tmp_path):
    writer = _RecordingWriter()
    logger = RunLogger(writer, tmp_path, "run")
    record = {
        "mean_loss": 0.5,
        "mean_utility": 0.3,
        "lagrange_multiplier": 1.5,
        "exploration_weight": 0.01,
        "collection_time_s": 2.0,
        "mean_reward": 0.7,
    }

    grouped = logger.log_update(7, record)

    assert set(grouped) == {
        "loss",
        "objective",
        "constraint",
        "exploration",
        "timing",
    }
    assert grouped["loss"] == {"mean_loss": 0.5}
    assert grouped["objective"] == {"mean_utility": 0.3, "mean_reward": 0.7}
    assert grouped["constraint"] == {"lagrange_multiplier": 1.5}
    assert grouped["exploration"] == {"exploration_weight": 0.01}
    assert grouped["timing"] == {"collection_time_s": 2.0}
    assert all(step == 7 for step, _, _ in writer.scalars)
    assert sorted(prefix for _, _, prefix in writer.scalars) == [
        "constraint/",
        "exploration/",
        "loss/",
        "objective/",
        "timing/",
    ]


def test_run_logger_groups_unknown_and_unconstrained_keys(tmp_path):
    writer = _RecordingWriter()
    logger = RunLogger(writer, tmp_path, "run")

    grouped = logger.log_update(
        1,
        {
            "transition_steps": 12.0,
            "mean_lagrangian_reward": 0.2,
            "mean_rnd_loss": 0.1,
            "note": "not-a-number",
        },
    )

    assert grouped["train"] == {"transition_steps": 12.0}
    # Group order is fixed, so "loss" wins over "rnd" and "reward" over
    # "lagrangian" (which does not spell the "lagrange" needle).
    assert grouped["loss"] == {"mean_rnd_loss": 0.1}
    assert grouped["objective"] == {"mean_lagrangian_reward": 0.2}
    assert "note" not in json.dumps(grouped)


def test_run_logger_forwards_updates_to_progress(tmp_path):
    progress = _StubProgress()
    logger = RunLogger(_RecordingWriter(), tmp_path, "run", progress=progress)

    logger.log_update(3, {"mean_reward": 1.0})

    assert progress.records == [{"mean_reward": 1.0}]


def test_run_logger_validation_and_episode_namespaces(tmp_path):
    writer = _RecordingWriter()
    logger = RunLogger(writer, tmp_path, "run")

    validation = logger.log_validation(5, {"mean_utility": 0.4}, contexts=3)
    episode = logger.log_episode(5, 2, {"reward": 0.6, "label": "x"})

    assert validation == {"validation/mean_utility": 0.4, "validation/contexts": 3.0}
    assert episode == {"episode/index": 2.0, "episode/reward": 0.6}
    assert writer.scalars[0][1] == validation
    assert writer.scalars[1][1] == episode


def test_run_logger_action_distribution_tolerates_odd_keys(tmp_path):
    writer = _RecordingWriter()
    logger = RunLogger(writer, tmp_path, "run")

    flat = logger.log_action_distribution(
        4,
        {
            ("app-a", "ingress-1", "qwen-8b"): 0.5,
            ("app-a", "ingress-2", "qwen-8b"): 0.25,
            ("app-b", "ingress-1", "qwen-4b"): 1.0,
            "unstructured-key": 0.1,
        },
        deploy_action_counts={"add": 2.0, "remove": 1.0},
    )

    assert flat["action/model_share/app-a/qwen-8b"] == 0.75
    assert flat["action/model_share/app-b/qwen-4b"] == 1.0
    assert flat["action/model_share/unstructured-key"] == 0.1
    assert flat["action/deploy/add"] == 2.0
    assert flat["action/deploy/remove"] == 1.0


def test_run_logger_reward_audit_reports_span(tmp_path):
    writer = _RecordingWriter()
    logger = RunLogger(writer, tmp_path, "run")

    flat = logger.log_reward_audit(9, {"cost": 1.0, "latency": 0.25})

    assert flat == {
        "audit/reward/cost": 1.0,
        "audit/reward/latency": 0.25,
        "audit/reward/span": 0.75,
    }


def test_run_logger_writes_config_and_finishes_idempotently(tmp_path):
    writer = _RecordingWriter()
    with RunLogger(writer, tmp_path, "run", config={"seed": 3}) as logger:
        logger.log_config({"tag": "中文"})

    payload = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert payload == {"seed": 3, "tag": "中文"}
    assert (tmp_path / "run_config.json").read_text(encoding="utf-8").startswith("{\n")
    assert writer.configs == [{"seed": 3}, {"tag": "中文"}]
    assert logger.config == {"seed": 3, "tag": "中文"}
    assert writer.finish_calls == 1
    assert writer.close_calls == 1

    logger.finish()
    logger.close()
    assert writer.finish_calls == 1


def test_run_logger_forwards_histogram_and_text(tmp_path):
    writer = _RecordingWriter()
    logger = RunLogger(writer, tmp_path, "run")

    logger.log_histogram(2, "action/entropy", [0.1, 0.2])
    logger.log_text(2, "notes", "body")

    assert writer.histograms == [(2, "action/entropy", [0.1, 0.2])]
    assert writer.texts == [(2, "notes", "body")]


def test_tensorboard_writer_writes_event_file(tmp_path):
    pytest.importorskip("tensorboard")
    writer = TensorBoardWriter(tmp_path)
    _exercise(writer)

    events = list(Path(tmp_path).rglob("events.out.tfevents.*"))
    assert events


def test_swanlab_writer_logs_offline(tmp_path):
    pytest.importorskip("swanlab")
    writer = SwanLabWriter(
        "telemetry-test",
        tmp_path / "swanlab",
        project="agent-orch-test",
        config={"seed": 1},
        offline=True,
    )

    _exercise(writer)

    assert writer.active is False
    assert list(Path(tmp_path).rglob("*.swanlab"))


def test_swanlab_writer_survives_a_failed_init(tmp_path, monkeypatch):
    pytest.importorskip("swanlab")
    import swanlab

    def _broken_init(**_: Any) -> None:
        raise RuntimeError("no login available")

    monkeypatch.setattr(swanlab, "init", _broken_init)
    with pytest.warns(RuntimeWarning, match="swanlab.init failed"):
        writer = SwanLabWriter("telemetry-test", tmp_path / "swanlab")

    _exercise(writer)
    assert writer.active is False


def test_make_writer_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="unknown telemetry writer kind"):
        make_writer("wandb", log_dir=tmp_path, run_name="run")


def test_make_writer_none_returns_null_writer(tmp_path):
    assert isinstance(
        make_writer("none", log_dir=tmp_path, run_name="run"), NullWriter
    )


def test_make_writer_degrades_when_tensorboard_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    with pytest.warns(RuntimeWarning, match="TensorBoard writer unavailable"):
        writer = make_writer("tensorboard", log_dir=tmp_path, run_name="run")

    assert isinstance(writer, NullWriter)
    _exercise(writer)


def test_make_writer_degrades_when_swanlab_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    with pytest.warns(RuntimeWarning, match="SwanLab writer unavailable"):
        writer = make_writer("swanlab", log_dir=tmp_path, run_name="run")

    assert isinstance(writer, NullWriter)


def test_make_writer_both_survives_one_missing_backend(tmp_path, monkeypatch):
    pytest.importorskip("swanlab")
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)

    with pytest.warns(RuntimeWarning, match="TensorBoard writer unavailable"):
        writer = make_writer("both", log_dir=tmp_path, run_name="run")

    assert isinstance(writer, SwanLabWriter)
    _exercise(writer)


def test_make_writer_both_keeps_backends_separate(tmp_path):
    pytest.importorskip("tensorboard")
    pytest.importorskip("swanlab")
    writer = make_writer("both", log_dir=tmp_path, run_name="run")

    assert isinstance(writer, MultiWriter)
    _exercise(writer)

    assert (tmp_path / "tensorboard").is_dir()
    assert (tmp_path / "swanlab").is_dir()


def test_build_run_logger_with_none_kind(tmp_path):
    with build_run_logger("none", tmp_path, "run", config={"seed": 2}) as logger:
        logger.log_update(1, {"mean_loss": 1.0})

    assert isinstance(logger.writer, NullWriter)
    assert (tmp_path / "run_config.json").exists()
