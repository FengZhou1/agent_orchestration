import json

import pytest
import torch

from agent_orch.agents import TrainingProgressReporter, resolve_device


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
