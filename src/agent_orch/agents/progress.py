from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import TracebackType
from typing import Callable, TextIO

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
        print_updates: bool = False,
        details: Callable[[], dict] | None = None,
        heartbeat_s: float = 10.0,
        initial_update_time_s: float | None = None,
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
        self.print_updates = print_updates
        self.details = details
        self.heartbeat_s = heartbeat_s
        self._last_status_time = 0.0
        self._last_bar_refresh = 0.0
        self._update_time_s = (
            initial_update_time_s
            if initial_update_time_s is not None
            and math.isfinite(initial_update_time_s)
            and initial_update_time_s > 0.0
            else None
        )
        self.status_path = self.output_dir / status_filename
        self.history_path = self.output_dir / history_filename

        self._started = 0.0
        self._completed_units = self.initial_update * self.units_per_update
        self._initial_completed_units = self._completed_units
        self._completed_updates = self.initial_update
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
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            disable=not self.show_progress,
            initial=self._completed_units,
            mininterval=1.0,
            bar_format="{desc} {percentage:5.1f}%|{bar:10}| {postfix}",
        )
        self._refresh_bar(force=True)
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
            status = "interrupted" if isinstance(exc_value, KeyboardInterrupt) else "failed"
            self._phase = status
            self._write_status(status, error=repr(exc_value))
        if self._bar is not None:
            self._bar.n = self._completed_units
            self._refresh_bar(force=True)
            self._bar.close()
        if self._history_handle is not None:
            self._history_handle.flush()
            self._history_handle.close()
        return False

    def on_phase(self, update: int, phase: str) -> None:
        self._update = min(update + 1, self.updates)
        self._phase = phase
        if phase == "collecting":
            self._rollout_step = 0
            self._optimizer_step = 0
        elif phase == "optimizing":
            self._optimizer_step = 0
        self._refresh_bar()
        self._write_status("running")

    def on_rollout_step(self, update: int, rollout_step: int) -> None:
        self._update = update + 1
        self._phase = "collecting"
        self._rollout_step = rollout_step
        target = update * self.units_per_update + rollout_step
        self._advance(target)

    def on_evaluation_phase(self, phase: str) -> None:
        self.on_phase(max(0, self._update - 1), phase)

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
        self._completed_updates = int(record["update"]) + 1
        duration = record.get("collection_time_s", 0.0) + record.get("optimization_time_s", 0.0)
        if math.isfinite(duration) and duration > 0.0:
            self._update_time_s = (
                duration if self._update_time_s is None
                else 0.8 * self._update_time_s + 0.2 * duration
            )
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
        if self.print_updates:
            timings = " ".join(f"{name}={record[key]:.2f}s" for name, key in (
                ("collect", "collection_time_s"), ("optimize", "optimization_time_s"),
                ("route", "timing_physical_routing_s"), ("analytical", "timing_analytical_s"),
                ("workflow", "timing_workflow_s"), ("obs", "timing_observation_s"),
                ("reset", "timing_env_reset_s")) if key in record)
            tqdm.write(f"[{record.get('training_phase', 'PPO')}] update "
                f"{int(record['update']) + 1}/{self.updates} "
                f"utility={record.get('mean_utility', 0.):.6f} "
                f"constraint={record.get('mean_constraint_cost', 0.):.6f} {timings}")
        self._refresh_bar(force=True)
        self._write_status("running")

    def _advance(self, target: int) -> None:
        target = min(max(target, self._completed_units), self.total_units)
        delta = target - self._completed_units
        self._completed_units = target
        if self._bar is not None and delta:
            self._refresh_bar()
            self._bar.update(delta)
        if (
            (delta > 0 and self._completed_units % self.status_interval_steps == 0)
            or self._completed_units == self.total_units
            or time.perf_counter() - self._last_status_time >= self.heartbeat_s
        ):
            self._write_status("running")

    def _write_status(self, status: str, error: str | None = None) -> None:
        self._last_status_time = time.perf_counter()
        elapsed = self._elapsed()
        eta = self._eta_seconds()
        payload = {
            "run_id": self.run_id,
            "status": status,
            "phase": self._phase,
            "device": self.device,
            "update": self._update,
            "updates_total": self.updates,
            "completed_updates": self._completed_updates,
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
            "eta_basis": (
                "recent completed-update duration; approximate across training stages"
                if self._update_time_s is not None
                else "current-process work-unit rate; approximate across training stages"
            ),
            "latest_metrics": self._latest_metrics,
            "updated_at_utc": self._timestamp(),
        }
        if error is not None:
            payload["error"] = error
        if self.details is not None:
            payload["decision_progress"] = self.details()
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

    def _eta_seconds(self) -> float | None:
        remaining = max(0, self.total_units - self._completed_units)
        if remaining == 0:
            return 0.0
        if self._update_time_s is not None and self.units_per_update > 0:
            return remaining / self.units_per_update * self._update_time_s
        # Only work performed in this process contributes to its measured rate.
        elapsed = self._elapsed()
        new_units = max(0, self._completed_units - self._initial_completed_units)
        return remaining * elapsed / new_units if new_units and elapsed > 0.0 else None

    def _refresh_bar(self, *, force: bool = False) -> None:
        if self._bar is None or not self.show_progress:
            return
        now = time.perf_counter()
        if not force and now - self._last_bar_refresh < 1.0:
            return
        self._last_bar_refresh = now
        update = min(max(1, self._update), self.updates)
        self._bar.set_postfix_str(
            f"{update}/{self.updates} {self.rollout_unit} "
            f"{self._rollout_step}/{self.rollout_steps} "
            f"ETA {self._format_duration(self._eta_seconds())}",
            refresh=False,
        )
        self._bar.refresh()

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "--:--:--"
        hours, remainder = divmod(max(0, math.ceil(seconds)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self._started)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
