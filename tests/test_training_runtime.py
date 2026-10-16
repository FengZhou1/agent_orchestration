import json
from pathlib import Path

import pytest
import torch

from agent_orch.agents import TrainingProgressReporter, resolve_device


def test_progress_hides_evaluation_phases_but_keeps_update_summary(tmp_path, capsys):
    with TrainingProgressReporter(
        run_id="smoke", output_dir=tmp_path, updates=1, rollout_steps=1,
        rollout_unit="slot", update_epochs=1, minibatch_size=1,
        device="cpu", show_progress=True, print_updates=True,
    ) as reporter:
        reporter.on_phase(0, "collecting")
        for phase in ("physical_routing", "analytical", "workflow"):
            reporter.on_evaluation_phase(phase)
            status = json.loads((tmp_path / "training_status.json").read_text())
            assert status["phase"] == phase
        reporter.on_rollout_step(0, 1)
        reporter.on_optimization_step(0, 1, 1)
        reporter.on_update({
            "update": 0.0, "mean_utility": 0.5,
            "collection_time_s": 2.0, "optimization_time_s": 0.1,
        })
    captured = capsys.readouterr()
    assert captured.out.count("update 1/1") == 1
    assert "utility=0.500000" in captured.out
    assert "ETA " in captured.err
    assert "slot 1/1" in captured.err
    assert "100.0%" in captured.err
    for phase in ("physical_routing", "analytical", "workflow", "collecting"):
        assert phase not in captured.out + captured.err


def test_resumed_progress_estimates_eta_from_saved_update_durations(tmp_path):
    with TrainingProgressReporter(
        run_id="resume", output_dir=tmp_path, updates=1000, rollout_steps=32,
        rollout_unit="slot", update_epochs=3, minibatch_size=64,
        device="cpu", initial_update=40, append_history=True,
        show_progress=True, status_interval_steps=1, initial_update_time_s=26.0,
    ) as reporter:
        status = json.loads((tmp_path / "training_status.json").read_text())
        assert status["eta_seconds"] == 960 * 26.0
        assert status["eta_basis"].startswith("recent completed-update")
        reporter.on_phase(40, "collecting")
        reporter.on_rollout_step(40, 32)
        reporter.on_optimization_step(40, 3, 3)
        reporter.on_update({
            "update": 40.0, "collection_time_s": 24.0,
            "optimization_time_s": 6.0,
        })
        status = json.loads((tmp_path / "training_status.json").read_text())
        assert status["eta_seconds"] == pytest.approx(959 * 26.8)
        assert "41/1000 slot 32/32 ETA" in reporter._bar.postfix


def test_repeated_phase_callbacks_do_not_refresh_bar_per_phase(tmp_path, monkeypatch):
    with TrainingProgressReporter(
        run_id="no-spam", output_dir=tmp_path, updates=2, rollout_steps=32,
        update_epochs=1, minibatch_size=32, device="cpu", show_progress=True,
    ) as reporter:
        # Freeze only the reporter clock; rendering must not follow every phase.
        monkeypatch.setattr("agent_orch.agents.progress.time.perf_counter", lambda: reporter._last_bar_refresh)
        refreshes = []
        monkeypatch.setattr(reporter._bar, "refresh", lambda: refreshes.append(1))
        for _ in range(32):
            for phase in ("physical_routing", "analytical", "workflow"):
                reporter.on_evaluation_phase(phase)
        assert refreshes == []


@pytest.mark.parametrize("duration", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_initial_eta_duration_is_not_used(tmp_path, duration):
    with TrainingProgressReporter(
        run_id="resume", output_dir=tmp_path, updates=10, rollout_steps=32,
        update_epochs=1, minibatch_size=32, device="cpu", show_progress=False,
        initial_update=3, initial_update_time_s=duration,
    ) as reporter:
        status = json.loads((tmp_path / "training_status.json").read_text())
        assert status["eta_seconds"] is None
        assert reporter._format_duration(None) == "--:--:--"


def test_device_resolution_supports_cpu_and_auto():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda:0"}


def test_explicit_cuda_request_fails_cleanly_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this test host")
    with pytest.raises(ValueError, match="CUDA was requested"):
        resolve_device("cuda:0")


def test_progress_reporter_streams_status_and_history(tmp_path):
    with TrainingProgressReporter(
        run_id="smoke",
        output_dir=tmp_path,
        updates=1,
        rollout_steps=2,
        update_epochs=1,
        minibatch_size=2,
        device="cpu",
        status_interval_steps=1,
        show_progress=False,
    ) as reporter:
        reporter.on_phase(0, "collecting")
        reporter.on_rollout_step(0, 1)
        running = json.loads((tmp_path / "training_status.json").read_text())
        assert running["status"] == "running"
        assert running["completed_work_units"] == 1
        reporter.on_rollout_step(0, 2)
        reporter.on_phase(0, "optimizing")
        reporter.on_optimization_step(0, 1, 1)
        reporter.on_update(
            {
                "update": 0.0,
                "mean_reward": 0.5,
                "mean_utility": 0.6,
                "mean_constraint_cost": 0.1,
                "mean_loss": 0.2,
            }
        )

    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["status"] == "completed"
    assert status["progress_percent"] == 100.0
    lines = (tmp_path / "training_history.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["mean_reward"] == 0.5


def test_progress_reporter_retries_transient_windows_replace_lock(
    tmp_path, monkeypatch
):
    original_replace = Path.replace
    attempts = {"count": 0}

    def transient_lock(path, target):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise PermissionError(5, "destination is temporarily locked")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", transient_lock)
    with TrainingProgressReporter(
        run_id="locked-status",
        output_dir=tmp_path,
        updates=1,
        rollout_steps=1,
        update_epochs=1,
        minibatch_size=1,
        device="cpu",
        status_interval_steps=1,
        show_progress=False,
    ) as reporter:
        reporter.on_phase(0, "collecting")
        reporter.on_rollout_step(0, 1)

    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["status"] == "completed"
    assert attempts["count"] >= 3


def test_progress_reporter_appends_after_resumed_update(tmp_path):
    history = tmp_path / "training_history.jsonl"
    history.write_text('{"update": 0}\n', encoding="utf-8")
    with TrainingProgressReporter(
        run_id="resume",
        output_dir=tmp_path,
        updates=2,
        rollout_steps=2,
        update_epochs=1,
        minibatch_size=2,
        device="cpu",
        initial_update=1,
        append_history=True,
        show_progress=False,
    ) as reporter:
        reporter.on_phase(1, "collecting")
        reporter.on_rollout_step(1, 2)
        reporter.on_phase(1, "optimizing")
        reporter.on_optimization_step(1, 1, 1)
        reporter.on_update(
            {
                "update": 1.0,
                "mean_reward": 0.2,
                "mean_utility": 0.3,
                "mean_constraint_cost": 0.0,
                "mean_loss": 0.1,
            }
        )
    lines = history.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[-1])["update"] == 1.0
