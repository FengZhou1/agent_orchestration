"""Run a small, reproducible LLMServingSim matrix on synthetic Roofline bundles."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "zf@192.168.234.128"
CONTAINER = "servingsim_docker"
CONTAINER_ROOT = "/app/LLMServingSim"
MODELS = [
    ("Qwen3-0.6B", 1),
    ("Qwen3-1.7B", 1),
    ("Qwen3-4B", 1),
    ("Qwen3-8B", 1),
    ("Qwen3-14B", 1),
    ("Qwen3-32B", 1),
    ("Qwen3-30B-A3B-Instruct-2507", 1),
]
GPUS = ["A10", "L20", "H20"]


def run(command: list[str]) -> str:
    p = subprocess.run(command, check=False, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{p.stdout}")
    return p.stdout


def write_workload(path: Path) -> None:
    rows = []
    cases = [(64, 32), (512, 128), (1300, 512), (2048, 512)]
    for idx, (prompt, output) in enumerate(cases):
        rows.append({
            "input_toks": prompt,
            "output_toks": output,
            "arrival_time_ns": idx * 20_000_000_000,
            "input_tok_ids": list(range(prompt)),
            "output_tok_ids": list(range(prompt, prompt + output)),
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def config(model: str, gpu: str, tp: int) -> dict:
    memory = {"A10": 24, "L20": 48, "H20": 96}[gpu]
    return {
        "num_nodes": 1,
        "link_bw": 16,
        "link_latency": 1000,
        "nodes": [{
            "num_instances": 1,
            "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
            "instances": [{
                "model_name": f"Qwen/{model}",
                "hardware": gpu,
                "npu_mem": {"mem_size": memory, "mem_bw": {"A10": 600, "L20": 864, "H20": 4000}[gpu], "mem_latency": 0, "mem_util": 0.9},
                "num_npus": tp,
                "tp_size": tp,
                "pd_type": None,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 128,
                "long_prefill_token_threshold": 512,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "block_size": 16,
            }],
        }],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "roofline_qwen_matrix")
    parser.add_argument("--models", default=",".join(x[0] for x in MODELS))
    parser.add_argument("--gpus", default=",".join(GPUS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workloads = args.output / "workloads"
    configs = args.output / "configs"
    runs = args.output / "runs"
    logs = args.output / "logs"
    for directory in (workloads, configs, runs, logs):
        directory.mkdir(parents=True, exist_ok=True)
    workload = workloads / "roofline_probe.jsonl"
    write_workload(workload)

    remote_root = f"{CONTAINER_ROOT}/outputs/roofline_qwen_matrix"
    run(["ssh", REMOTE, "mkdir", "-p", f"/home/zf/桌面/LLMServingSim/outputs/roofline_qwen_matrix/workloads", f"/home/zf/桌面/LLMServingSim/outputs/roofline_qwen_matrix/configs", f"/home/zf/桌面/LLMServingSim/outputs/roofline_qwen_matrix/runs", f"/home/zf/桌面/LLMServingSim/outputs/roofline_qwen_matrix/logs"])
    run(["scp", "-q", str(workload), f"{REMOTE}:/home/zf/桌面/LLMServingSim/outputs/roofline_qwen_matrix/workloads/roofline_probe.jsonl"])

    selected_models = [x.strip() for x in args.models.split(",") if x.strip()]
    selected_gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    results = []
    for model in selected_models:
        for gpu in selected_gpus:
            # Large dense and MoE checkpoints use TP=2 on L20; the remaining
            # probes use TP=1 when the model fits on one GPU.
            tp = 2 if model in {"Qwen3-32B", "Qwen3-30B-A3B-Instruct-2507"} and gpu == "L20" else 1
            run_id = f"roofline-{model.lower()}-{gpu.lower()}-tp{tp}"
            cfg_path = configs / f"{run_id}.json"
            csv_path = runs / f"{run_id}.csv"
            # serving/__main__.py changes cwd to astra-sim and resolves
            # relative paths from there, hence the ../ prefix.
            # The CLI's cluster-config resolver adds one ../ itself.
            remote_cfg = f"outputs/roofline_qwen_matrix/configs/{run_id}.json"
            remote_csv = f"outputs/roofline_qwen_matrix/runs/{run_id}.csv"
            cfg_path.write_text(json.dumps(config(model, gpu, tp), indent=2), encoding="utf-8")
            if csv_path.exists() and not args.force:
                results.append({"run_id": run_id, "status": "skipped"})
                continue
            run(["scp", "-q", str(cfg_path), f"{REMOTE}:/home/zf/桌面/LLMServingSim/outputs/roofline_qwen_matrix/configs/{run_id}.json"])
            command = [
                "ssh", REMOTE, "docker", "exec", "-w", CONTAINER_ROOT, CONTAINER,
                "python", "-m", "serving", "--cluster-config", remote_cfg,
                "--dtype", "bfloat16", "--block-size", "16", "--dataset", "outputs/roofline_qwen_matrix/workloads/roofline_probe.jsonl",
                "--num-reqs", "4", "--output", remote_csv, "--run-id", run_id, "--log-level", "WARNING",
                "--no-enable-prefix-caching",
            ]
            try:
                output = run(command)
                (logs / f"{run_id}.log").write_text(output, encoding="utf-8")
                run(["scp", "-q", f"{REMOTE}:/home/zf/桌面/LLMServingSim/outputs/roofline_qwen_matrix/runs/{run_id}.csv", str(csv_path)])
                results.append({"run_id": run_id, "status": "completed", "model": model, "gpu": gpu, "tp": tp})
                print(f"completed {run_id}", flush=True)
            except Exception as exc:
                (logs / f"{run_id}.log").write_text(str(exc), encoding="utf-8")
                results.append({"run_id": run_id, "status": "failed", "error": str(exc), "model": model, "gpu": gpu, "tp": tp})
                print(f"failed {run_id}: {type(exc).__name__}", flush=True)
    (args.output / "matrix_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
