from __future__ import annotations

import re

import torch


_CUDA_DEVICE = re.compile(r"cuda(?::(?P<index>\d+))?")


def resolve_device(requested: str | torch.device) -> str:
    """Resolve and validate a CPU/CUDA training device."""

    value = str(requested).strip().lower()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"

    match = _CUDA_DEVICE.fullmatch(value)
    if match is None:
        raise ValueError(
            "Unsupported device. Use auto, cpu, cuda, or cuda:<index>."
        )
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested, but the installed PyTorch build cannot access a CUDA GPU."
        )

    index = int(match.group("index") or 0)
    device_count = torch.cuda.device_count()
    if index >= device_count:
        raise ValueError(
            f"CUDA device cuda:{index} is unavailable; detected {device_count} device(s)."
        )
    return f"cuda:{index}"


def device_metadata(requested: str, resolved: str) -> dict[str, object]:
    """Return serializable device information for checkpoints and manifests."""

    metadata: dict[str, object] = {
        "requested": requested,
        "resolved": resolved,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if resolved.startswith("cuda:"):
        index = int(resolved.split(":", maxsplit=1)[1])
        metadata.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return metadata
