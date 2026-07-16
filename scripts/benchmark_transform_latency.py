#!/usr/bin/env python3
"""Benchmark reference online transforms and a representative Llama linear."""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fft_com.torch_transforms import (  # noqa: E402
    GroupedOrthogonalTransform,
    SUPPORTED_TRANSFORMS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def benchmark_cuda(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    return {
        "mean_ms": sum(samples) / len(samples),
        "median_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "std_ms": math.sqrt(
            sum((sample - sum(samples) / len(samples)) ** 2 for sample in samples)
            / len(samples)
        ),
    }


def make_permutations(
    feature_dim: int,
    group_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(group_size, generator=generator, device=device)
            for _ in range(feature_dim // group_size)
        ]
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this latency benchmark requires CUDA events")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    shapes = {
        "hidden": (
            args.batch_size,
            args.sequence_length,
            args.hidden_size,
        ),
        "intermediate": (
            args.batch_size,
            args.sequence_length,
            args.intermediate_size,
        ),
    }
    rows: list[dict[str, Any]] = []
    for shape_name, shape in shapes.items():
        values = torch.randn(shape, device=device, dtype=dtype)
        for kind in SUPPORTED_TRANSFORMS:
            transform = GroupedOrthogonalTransform(
                kind,
                args.group_size,
                boundary="latency_contiguous_groups",
            ).to(device)
            timing = benchmark_cuda(
                lambda transform=transform, values=values: transform(values),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            rows.append(
                {
                    "benchmark": "transform_forward",
                    "shape": shape_name,
                    "dimensions": list(shape),
                    "transform": kind,
                    "permutation": "none",
                    "metadata_bits": 0.0,
                    **timing,
                }
            )
            if kind != "identity":
                inverse_timing = benchmark_cuda(
                    lambda transform=transform, values=values: transform.inverse(values),
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                rows.append(
                    {
                        "benchmark": "transform_inverse",
                        "shape": shape_name,
                        "dimensions": list(shape),
                        "transform": kind,
                        "permutation": "none",
                        "metadata_bits": 0.0,
                        **inverse_timing,
                    }
                )

        permutations = make_permutations(
            shape[-1],
            args.group_size,
            generator=generator,
            device=device,
        )
        permuted_dct = GroupedOrthogonalTransform(
            "dct",
            args.group_size,
            permutation=permutations,
            boundary="latency_contiguous_groups",
        ).to(device)
        timing = benchmark_cuda(
            lambda: permuted_dct(values),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rows.append(
            {
                "benchmark": "transform_forward",
                "shape": shape_name,
                "dimensions": list(shape),
                "transform": "dct",
                "permutation": "per_group_gather",
                "metadata_bits": permuted_dct.metadata_bits(shape[-1]),
                **timing,
            }
        )
        del values

    # Representative q_proj path. Every transform is placed at the same
    # input-side location and uses the correspondingly transformed weight.
    q_input = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    base_weight = torch.randn(
        args.hidden_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    for kind in SUPPORTED_TRANSFORMS:
        transform = GroupedOrthogonalTransform(kind, args.group_size).to(device)
        with torch.no_grad():
            transformed_weight = transform(base_weight.float()).to(dtype)
        timing = benchmark_cuda(
            lambda transform=transform, weight=transformed_weight: F.linear(
                transform(q_input),
                weight,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rows.append(
            {
                "benchmark": "q_proj_online_input_rotation",
                "shape": "q_proj_4096x4096",
                "dimensions": [
                    args.batch_size,
                    args.sequence_length,
                    args.hidden_size,
                    args.hidden_size,
                ],
                "transform": kind,
                "permutation": "none",
                "metadata_bits": 0.0,
                **timing,
            }
        )
        del transformed_weight

    properties = torch.cuda.get_device_properties(device)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": args.dtype,
            "group_size": args.group_size,
            "reference_implementation": True,
            "fused_kernel": False,
            "packed_integer_kernel": False,
            "cuda_events": True,
        },
        "rows": rows,
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": properties.name,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
