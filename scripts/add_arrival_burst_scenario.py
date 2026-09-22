"""Derive the arrival-burst stress scenario from the calibrated main scenario.

`build_benchmark_scenarios.py` emits *pre-calibration* scenarios; the frozen SLO
thresholds are written into those files afterwards by `calibrate_slos.py`.  So a
new stress variant cannot be produced by re-running the generator -- that erases
the calibration block.  This derives it from the file that is actually in use and
copies that file's frozen SLO provenance, which is exactly how the other three
stress variants were produced.

The burst itself is an arrival-process property, so only metadata is added: the
trace generator reads it.  No resource parameter changes, which keeps this
variant directly comparable to the main scenario.

Usage::

    python scripts/add_arrival_burst_scenario.py \\
        --source configs/benchmarks/main_abilene.yaml \\
        --output configs/benchmarks/stress_arrival_burst.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_orch.schema.loader import ScenarioLoader

BURST_METADATA = {
    "pattern": "gaussian_burst",
    "period_slots": 60,
    "sigma_slots": 8.0,
    "phase": 0.25,
    "low_fraction": 0.4,
    "jitter": 0.05,
    "note": (
        "periodic Gaussian bursts on top of the load-level baseline rate; "
        "consumed by ArrivalTrace.gaussian_burst_intensity"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="configs/benchmarks/main_abilene.yaml")
    parser.add_argument("--output", default="configs/benchmarks/stress_arrival_burst.yaml")
    args = parser.parse_args()

    source_path = Path(args.source)
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    burst = deepcopy(payload)
    burst["id"] = f"{payload['id']}-arrival-burst"
    metadata = burst.setdefault("metadata", {})
    metadata["role"] = "stress-arrival"
    metadata["arrival_burst"] = BURST_METADATA
    # Same provenance the other stress variants carry, so the frozen thresholds of
    # the main scenario are visibly inherited rather than re-derived.
    calibration = metadata.get("slo_calibration")
    if isinstance(calibration, dict):
        calibration["frozen_slo_source"] = source_path.as_posix()
        calibration["frozen_slo_source_sha256"] = source_hash
    metadata["slo_status"] = "frozen SLOs inherited from the main reference scenario"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(burst, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    scenario = ScenarioLoader.load(output_path)
    print(
        f"{output_path}: id={scenario.id} role={scenario.metadata['role']} "
        f"burst={scenario.metadata['arrival_burst']['pattern']} "
        f"({len(scenario.applications)} applications, SLO source sha256 {source_hash[:16]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
