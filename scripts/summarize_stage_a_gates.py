"""Report a Stage A verdict across seeds (A0.3).

G_A is defined over seeds, not over one run: a single seed's capture rate is a
draw, and earlier work established that a 49% result was a lucky seed against
28% for the same configuration.  This reads one ``gate_report.json`` per training
seed, reports each, and gives the median that the criterion is judged on.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _find_reports(roots: list[Path]) -> list[Path]:
    reports: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "gate_report.json":
            reports.append(root)
        elif root.is_dir():
            reports.extend(sorted(root.rglob("gate_report.json")))
    return sorted(set(reports))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="gate_report.json files or directories")
    parser.add_argument("--min-rho", type=float, default=0.8)
    parser.add_argument("--min-lift-capture", type=float, default=0.8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = _find_reports([Path(value) for value in args.reports])
    if not paths:
        raise SystemExit("no gate_report.json found")

    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        gates = dict(payload.get("gates", {}))
        rows.append(
            {
                "report": str(path),
                "policy": payload.get("policy", ""),
                "n_deployments": payload.get("n_deployments"),
                "rho": payload.get("rho"),
                "lift_capture": payload.get("lift_capture"),
                "passed": bool(payload.get("passed")),
                "gate_rho": bool(gates.get("rho")),
                "gate_lift": bool(gates.get("lift_capture")),
                "gate_strata": bool(gates.get("per_stratum")),
                "mean_policy": payload.get("mean_policy_cold"),
                "mean_reference": payload.get("mean_reference_cold"),
                "mean_uniform": payload.get("mean_uniform_cold"),
            }
        )

    rhos = [row["rho"] for row in rows if row["rho"] is not None]
    captures = [row["lift_capture"] for row in rows if row["lift_capture"] is not None]
    median_rho = statistics.median(rhos) if rhos else float("nan")
    median_capture = statistics.median(captures) if captures else float("nan")
    all_strata = all(row["gate_strata"] for row in rows)
    passed = (
        median_rho >= args.min_rho
        and median_capture >= args.min_lift_capture
        and all_strata
    )

    print(f"{'seed run':<52} {'rho':>8} {'capture':>9} {'strata':>7} {'pass':>5}")
    for row in rows:
        name = Path(row["report"]).parent.name or row["report"]
        print(
            f"{name[:52]:<52} {row['rho']:>8.4f} {row['lift_capture']:>9.4f} "
            f"{str(row['gate_strata']):>7} {str(row['passed']):>5}"
        )
    print()
    print(f"seeds                    : {len(rows)}")
    print(f"median rho               : {median_rho:.4f}  (min {min(rhos):.4f}, max {max(rhos):.4f})")
    print(
        f"median lift capture      : {median_capture:.4f}  "
        f"(min {min(captures):.4f}, max {max(captures):.4f})"
    )
    print(f"per-stratum pass in all  : {all_strata}")
    print()
    print(
        f"G_A (median rho >= {args.min_rho}, median capture >= {args.min_lift_capture}, "
        f"strata all pass): {'PASS' if passed else 'FAIL'}"
    )
    if len(rows) < 5:
        print(f"WARNING: {len(rows)} seeds; the criterion requires at least 5.")

    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "seeds": len(rows),
                    "median_rho": median_rho,
                    "median_lift_capture": median_capture,
                    "per_stratum_all_pass": all_strata,
                    "passed": passed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {target}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
