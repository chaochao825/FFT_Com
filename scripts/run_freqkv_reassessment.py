#!/usr/bin/env python3
"""Reassess the historical FreqKV synthetic results with explicit numerics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fft_com.freqkv_audit import (  # noqa: E402
    frequency_threshold_rows,
    make_historical_synthetic_kv,
    make_smooth_positive_control_kv,
    measure_low_frequency_retention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", default="32,64,128")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--ratios",
        default="0.1,0.25,0.3,0.5,0.7,0.75,0.9",
    )
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "freqkv_reassessment_20260717",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Write aggregate and evidence files under docs/.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        ""
                        if row.get(field) is None
                        else f"{row[field]:.12g}"
                        if isinstance(row.get(field), float)
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    aggregated: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        output = {field: value for field, value in zip(group_fields, key)}
        output["samples"] = len(members)
        for metric in metric_fields:
            values = [
                float(member[metric])
                for member in members
                if member.get(metric) is not None
                and math.isfinite(float(member[metric]))
            ]
            output[f"{metric}_mean"] = float(np.mean(values)) if values else None
            output[f"{metric}_std"] = float(np.std(values)) if values else None
        aggregated.append(output)
    return aggregated


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_row(rows: list[dict[str, Any]], **matches: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(field) == value for field, value in matches.items()):
            return row
    raise KeyError(matches)


def main() -> int:
    args = parse_args()
    sequence_lengths = tuple(
        int(value) for value in args.sequence_lengths.split(",") if value
    )
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    ratios = tuple(float(value) for value in args.ratios.split(",") if value)
    thresholds = (0.90, 0.95, 0.99)
    started = datetime.now(timezone.utc)

    retention_raw: list[dict[str, Any]] = []
    threshold_raw: list[dict[str, Any]] = []
    for sequence_length in sequence_lengths:
        for seed in seeds:
            datasets = {
                "historical_gaussian_rope_like": make_historical_synthetic_kv(
                    sequence_length,
                    head_dim=args.head_dim,
                    heads=args.heads,
                    seed=seed,
                ),
                "smooth_dct_decay_positive_control": (
                    make_smooth_positive_control_kv(
                        sequence_length,
                        head_dim=args.head_dim,
                        heads=args.heads,
                        seed=seed,
                    )
                ),
            }
            for data_model, caches in datasets.items():
                for cache_kind, values in caches.items():
                    common = {
                        "data_model": data_model,
                        "cache_kind": cache_kind,
                        "sequence_length": sequence_length,
                        "head_dim": args.head_dim,
                        "heads": args.heads,
                        "seed": seed,
                    }
                    for transform_path in (
                        "dct_ii_ortho",
                        "rfft_parseval",
                        "rfft_drop_imag_historical",
                    ):
                        for measurement in measure_low_frequency_retention(
                            values,
                            ratios,
                            transform_path=transform_path,
                        ):
                            retention_raw.append(
                                {
                                    **common,
                                    "transform_path": transform_path,
                                    **measurement.to_dict(),
                                }
                            )
                    for energy_definition in (
                        "dct_ii_ortho",
                        "rfft_parseval",
                        "rfft_historical_unweighted",
                    ):
                        for row in frequency_threshold_rows(
                            values,
                            thresholds,
                            energy_definition=energy_definition,
                        ):
                            threshold_raw.append({**common, **row})

    retention_aggregate = aggregate_rows(
        retention_raw,
        (
            "data_model",
            "cache_kind",
            "sequence_length",
            "transform_path",
            "requested_retention_ratio",
        ),
        (
            "retained_components",
            "total_components",
            "actual_component_fraction",
            "selected_frequency_energy_retention",
            "reconstruction_energy_retention",
            "reported_energy_retention",
            "reconstruction_relative_mse",
            "reconstruction_mse",
        ),
    )
    threshold_aggregate = aggregate_rows(
        threshold_raw,
        (
            "data_model",
            "cache_kind",
            "sequence_length",
            "energy_definition",
            "energy_threshold",
        ),
        (
            "zero_based_index",
            "required_components",
            "total_components",
            "true_required_component_fraction",
            "historical_plot_label_fraction",
            "historical_label_understatement",
        ),
    )

    gaussian_dct_k_25 = find_row(
        retention_aggregate,
        data_model="historical_gaussian_rope_like",
        cache_kind="K",
        sequence_length=64,
        transform_path="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    gaussian_dct_v_25 = find_row(
        retention_aggregate,
        data_model="historical_gaussian_rope_like",
        cache_kind="V",
        sequence_length=64,
        transform_path="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    gaussian_bug_k_90 = find_row(
        retention_aggregate,
        data_model="historical_gaussian_rope_like",
        cache_kind="K",
        sequence_length=64,
        transform_path="rfft_drop_imag_historical",
        requested_retention_ratio=0.9,
    )
    smooth_dct_k_25 = find_row(
        retention_aggregate,
        data_model="smooth_dct_decay_positive_control",
        cache_kind="K",
        sequence_length=64,
        transform_path="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    gaussian_threshold_k_90 = find_row(
        threshold_aggregate,
        data_model="historical_gaussian_rope_like",
        cache_kind="K",
        sequence_length=64,
        energy_definition="dct_ii_ortho",
        energy_threshold=0.9,
    )

    finished = datetime.now(timezone.utc)
    summary = {
        "metadata": {
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "sequence_lengths": sequence_lengths,
            "seeds": seeds,
            "ratios": ratios,
            "head_dim": args.head_dim,
            "heads": args.heads,
            "scope": (
                "Synthetic falsification and numerical-path audit only; no real "
                "model KV capture, perplexity, long-context task, memory, or "
                "latency claim."
            ),
            "source_sha256": {
                "src/fft_com/freqkv_audit.py": sha256_file(
                    ROOT / "src" / "fft_com" / "freqkv_audit.py"
                ),
                "scripts/run_freqkv_reassessment.py": sha256_file(
                    ROOT / "scripts" / "run_freqkv_reassessment.py"
                ),
            },
        },
        "decision_signals": {
            "seq64_gaussian_K_dct_low25_energy_mean": gaussian_dct_k_25[
                "selected_frequency_energy_retention_mean"
            ],
            "seq64_gaussian_V_dct_low25_energy_mean": gaussian_dct_v_25[
                "selected_frequency_energy_retention_mean"
            ],
            "seq64_gaussian_K_historical_bug_reported_energy_at_90pct_mean": (
                gaussian_bug_k_90["reported_energy_retention_mean"]
            ),
            "seq64_smooth_control_K_dct_low25_energy_mean": smooth_dct_k_25[
                "selected_frequency_energy_retention_mean"
            ],
            "seq64_gaussian_K_dct_components_for_90pct_energy_mean": (
                gaussian_threshold_k_90["true_required_component_fraction_mean"]
            ),
            "interpretation": (
                "The historical Gaussian/RoPE-like generator is frequency-flat. "
                "The positive control is compressible, so the corrected probe can "
                "detect genuine low-frequency structure. Dropping the rFFT "
                "imaginary part explains the roughly half-energy historical plots."
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "retention_raw.csv", retention_raw)
    write_csv(args.output_dir / "retention_aggregate.csv", retention_aggregate)
    write_csv(args.output_dir / "frequency_thresholds_raw.csv", threshold_raw)
    write_csv(
        args.output_dir / "frequency_thresholds_aggregate.csv",
        threshold_aggregate,
    )
    write_json(args.output_dir / "summary.json", summary)

    if args.publish:
        write_csv(
            ROOT
            / "docs"
            / "evidence"
            / "freqkv_reassessment_raw_20260717.csv",
            retention_raw,
        )
        write_csv(
            ROOT / "docs" / "tables" / "freqkv_reassessment_20260717.csv",
            retention_aggregate,
        )
        write_csv(
            ROOT
            / "docs"
            / "tables"
            / "freqkv_frequency_thresholds_20260717.csv",
            threshold_aggregate,
        )
        write_json(
            ROOT
            / "docs"
            / "evidence"
            / "freqkv_reassessment_summary_20260717.json",
            summary,
        )

    print(json.dumps(json_ready(summary["decision_signals"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
