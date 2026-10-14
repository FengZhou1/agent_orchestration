"""Read-only numeric/structural audit against the current parameter register."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from agent_orch.parameters import DEFAULT_SETTINGS, audit_scenario, load_settings
from agent_orch.schema.loader import ScenarioLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--infrastructure-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_scenario(ScenarioLoader.load(args.scenario), load_settings(args.settings),
                            infrastructure_only=args.infrastructure_only)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    for check in report["checks"]:
        print(f"{check['status'].upper()}: {','.join(check['parameters'])} {check['check']}")
    print(f"Rows inventoried: {len(report['register_rows'])}; formal_ready={report['formal_ready']}")
    raise SystemExit(0 if report["consistent"] else 2)


if __name__ == "__main__":
    main()
