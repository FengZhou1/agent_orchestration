"""Calibrate one frozen contract set; never overwrite workloads or run PPO."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from agent_orch.parameters import DEFAULT_SETTINGS, load_settings, file_sha256
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload.slo_calibration import (
    VERSION, CalibrationSpec, calibrate_application, sign_contracts, validate_contracts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True, help="Largest fixed-pool scenario containing every app")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--config", type=Path, default=Path("configs/slo_calibration.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="New contract directory, not an existing scene")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("SLO calibration never overwrites an existing contract or scenario")
    settings = load_settings(args.settings)
    spec = CalibrationSpec(**yaml.safe_load(args.config.read_text(encoding="utf-8")))
    spec.validate()
    scene = ScenarioLoader.load(args.scenario)
    config = next(c for c in scene.llm_configs.values() if c.model == spec.model and
                  c.gpu_type == spec.gpu_type and c.gpu_count == spec.tensor_parallel)
    contracts, samples = {}, {}
    for index, app in enumerate(scene.applications.values(), 1):
        contracts[app.id], samples[app.id] = calibrate_application(scene, app, spec)
        if index % 20 == 0 or index == len(scene.applications):
            print(f"Calibrated {index}/{len(scene.applications)} applications", flush=True)
    payload = sign_contracts({"version": VERSION, "spec": asdict(spec),
        "scope": "pattern-weighted analytical reference quantiles; not empirical request tails or real-server SLA",
        "reference_model": asdict(scene.models[spec.model]), "reference_llm_config": asdict(config),
        "source_scenario_sha256": file_sha256(args.scenario),
        "source_settings_sha256": file_sha256(settings["_path"]),
        "calibration_config_sha256": file_sha256(args.config),
        "implementation_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha256(p)
             for p in [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "src/agent_orch/workload/slo_calibration.py",
                       Path(__file__).resolve().parents[1] / "src/agent_orch/performance/workflow.py",
                       Path(__file__).resolve().parents[1] / "src/agent_orch/performance/analytical.py"]},
        "applications": contracts})
    validate_contracts(payload)
    args.output.mkdir(parents=True)
    (args.output / "contracts.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                          allow_nan=False), encoding="utf-8")
    (args.output / "reference_patterns.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False,
                                                                  allow_nan=False), encoding="utf-8")
    print(f"Frozen contracts {payload['contract_digest']}; source workloads unchanged", flush=True)


if __name__ == "__main__":
    main()
