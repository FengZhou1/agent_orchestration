from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import TracebackType
from typing import TextIO

from tqdm.auto import tqdm


class TrainingProgressReporter:
    """Report PPO progress to both the terminal and durable JSON files."""

    def __init__(
        self,
        *,
        run_id: str,
        output_dir: str | Path,
        updates: int,
        rollout_steps: int,
        update_epochs: int,
        minibatch_size: int,
        device: str,
        rollout_unit: str = "transition",
        status_interval_steps: int = 32,
        show_progress: bool = True,
        initial_update: int = 0,
        append_history: bool = False,
        status_filename: str = "training_status.json",
        history_filename: str = "training_history.jsonl",
    ) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.updates = updates
        self.rollout_steps = rollout_steps
        self.rollout_unit = rollout_unit
        self.optimizer_steps = update_epochs * math.ceil(
            rollout_steps / minibatch_size
        )
        self.units_per_update = rollout_steps + self.optimizer_steps
        self.total_units = updates * self.units_per_update
        self.device = device
        self.status_interval_steps = max(1, status_interval_steps)
        self.show_progress = show_progress
        self.initial_update = min(max(0, initial_update), updates)
        self.append_history = append_history
        self.status_path = self.output_dir / status_filename
        self.history_path = self.output_dir / history_filename

        self._started = 0.0
        self._completed_units = self.initial_update * self.units_per_update
        self._update = self.initial_update
        self._rollout_step = 0
        self._optimizer_step = 0
        self._optimizer_total_current = self.optimizer_steps
        self._phase = "initializing"
        self._latest_metrics: dict[str, float] = {}
        self._history_handle: TextIO | None = None
        self._bar = None

    def __enter__(self) -> TrainingProgressReporter:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started = time.perf_counter()
        self._history_handle = self.history_path.open(
            "a" if self.append_history else "w", encoding="utf-8", buffering=1
        )
        self._bar = tqdm(
            total=self.total_units,
            desc=self.run_id,
            unit="step",
            dynamic_ncols=True,
            disable=not self.show_progress,
            initial=self._completed_units,
        )
        self._write_status("running")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_value is None:
            self._completed_units = self.total_units
            self._phase = "completed"
            self._write_status("completed")
        else:
            self._phase = "failed"
            self._write_status("failed", error=repr(exc_value))
        if self._bar is not None:
            self._bar.close()
        if self._history_handle is not None:
            self._history_handle.flush()
            self._history_handle.close()
        return False

    def on_phase(self, update: int, phase: str) -> None:
        self._update = update + 1
        self._phase = phase
        if phase == "collecting":
            self._rollout_step = 0
            self._optimizer_step = 0
        elif phase == "optimizing":
            self._optimizer_step = 0
        if self._bar is not None:
            self._bar.set_description_str(
                f"{self.run_id} | {phase} {update + 1}/{self.updates}",
                refresh=True,
            )
        self._write_status("running")

    def on_rollout_step(self, update: int, rollout_step: int) -> None:
        self._update = update + 1
        self._phase = "collecting"
        self._rollout_step = rollout_step
        target = update * self.units_per_update + rollout_step
        self._advance(target)

    def on_optimization_step(
        self, update: int, optimization_step: int, total_steps: int
    ) -> None:
        self._update = update + 1
        self._phase = "optimizing"
        self._optimizer_step = optimization_step
        self._optimizer_total_current = total_steps
        mapped_step = math.ceil(
            optimization_step * self.optimizer_steps / max(1, total_steps)
        )
        target = (
            update * self.units_per_update
            + self.rollout_steps
            + mapped_step
        )
        self._advance(target)

    def on_update(self, record: dict[str, float]) -> None:
        self._latest_metrics = dict(record)
        persisted = {
            **record,
            "elapsed_time_s": self._elapsed(),
            "timestamp_utc": self._timestamp(),
        }
        if self._history_handle is None:
            raise RuntimeError("TrainingProgressReporter has not been opened")
        self._history_handle.write(
            json.dumps(persisted, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._history_handle.flush()
        self._phase = "update_completed"
        if self._bar is not None:
            self._bar.set_postfix(
                reward=f"{record.get('mean_reward', 0.0):.3f}",
                utility=f"{record.get('mean_utility', 0.0):.3f}",
                constraint=f"{record.get('mean_constraint_cost', 0.0):.3f}",
                loss=f"{record.get('mean_loss', 0.0):.3f}",
                refresh=True,
            )
        self._write_status("running")

    def _advance(self, target: int) -> None:
        target = min(max(target, self._completed_units), self.total_units)
        delta = target - self._completed_units
        self._completed_units = target
        if self._bar is not None and delta:
            self._bar.update(delta)
        if (
            self._completed_units % self.status_interval_steps == 0
            or self._completed_units == self.total_units
        ):
            self._write_status("running")

    def _write_status(self, status: str, error: str | None = None) -> None:
        elapsed = self._elapsed()
        rate = self._completed_units / elapsed if elapsed > 0.0 else 0.0
        remaining = max(0, self.total_units - self._completed_units)
        eta = remaining / rate if rate > 0.0 else None
        payload = {
            "run_id": self.run_id,
            "status": status,
            "phase": self._phase,
            "device": self.device,
            "update": self._update,
            "updates_total": self.updates,
            "rollout_step": self._rollout_step,
            "rollout_steps_per_update": self.rollout_steps,
            "rollout_unit": self.rollout_unit,
            "optimizer_step": self._optimizer_step,
            "optimizer_steps_per_update": self._optimizer_total_current,
            "completed_work_units": self._completed_units,
            "total_work_units": self.total_units,
            "progress_percent": (
                100.0 * self._completed_units / self.total_units
                if self.total_units
                else 100.0
            ),
            "elapsed_time_s": elapsed,
            "eta_seconds": eta,
            "latest_metrics": self._latest_metrics,
            "updated_at_utc": self._timestamp(),
        }
        if error is not None:
            payload["error"] = error
        serialized = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        )
        temporary = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        for attempt in range(8):
            try:
                temporary.replace(self.status_path)
                return
            except PermissionError:
                if attempt < 7:
                    time.sleep(0.02 * (attempt + 1))

        # A progress snapshot is observational state.  On Windows, a reader can
        # temporarily hold the destination open and prevent an atomic replace.
        # Fall back to an in-place update and, if that is also locked, leave the
        # previous valid snapshot in place without interrupting PPO training.
        try:
            self.status_path.write_text(serialized, encoding="utf-8")
        except PermissionError:
            pass
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except PermissionError:
                pass

    def _elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self._started)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
