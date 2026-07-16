#!/usr/bin/env python3
"""Run the FFT_Com transform-potential study on synthetic and Llama weights.

The study is CPU-only and intentionally block-level. It measures coefficient
statistics and reconstruction error; it does not claim end-to-end model
accuracy or runtime speedups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fft_com.compression import (  # noqa: E402
    CompressionResult,
    dense_transform_quantize,
    dct_hadamard_hybrid,
    fft_dense_quantize,
    fft_hermitian_groups,
    fft_topk_energy,
    two_sided_transform,
)
from fft_com.data import (  # noqa: E402
    BlockRecord,
    collect_model_blocks,
    load_layer0_embedding_activations,
)
from fft_com.kuramoto import (  # noqa: E402
    kuramoto_rotation_probe,
    learn_circular_codebook,
    phase_only_relative_mse,
    quantize_phase_codebook,
    quantize_phase_uniform,
    weighted_phase_template,
)
from fft_com.learning import (  # noqa: E402
    evaluate_butterfly,
    learn_butterfly_rotation,
)
from fft_com.metrics import (  # noqa: E402
    excess_kurtosis,
    output_relative_mse,
    papr,
    structured_energy,
    topk_energy,
)
from fft_com.transforms import (  # noqa: E402
    dct_matrix,
    haar_orthogonal_matrix,
    hadamard_matrix,
    randomized_hadamard_matrix,
    zigzag_indices,
)


DENSITIES = {
    "topk_1_over_64": 1.0 / 64.0,
    "topk_1_over_16": 1.0 / 16.0,
    "topk_1_over_8": 1.0 / 8.0,
    "topk_1_over_4": 1.0 / 4.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/data2/wangmeiqi/Llama-2-7b-chat-hf",
        help="Local Hugging Face safetensors checkpoint.",
    )
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--blocks-per-tensor", type=int, default=8)
    parser.add_argument("--synthetic-blocks", type=int, default=12)
    parser.add_argument("--embedding-samples", type=int, default=512)
    parser.add_argument("--learned-block-size", type=int, default=32)
    parser.add_argument("--learned-block-count", type=int, default=48)
    parser.add_argument("--learned-stages", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "transform_potential_20260716",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Also write aggregate tables and summary JSON under docs/.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller smoke configuration for code validation.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_source_hashes(model_dir: str | Path) -> dict[str, Any]:
    source_paths = sorted((ROOT / "src" / "fft_com").glob("*.py")) + [
        ROOT / "scripts" / "run_transform_potential.py",
        ROOT / "scripts" / "summarize_seed_sweep.py",
        ROOT / "tests" / "test_core.py",
        ROOT / "requirements.txt",
    ]
    model_root = Path(model_dir)
    index_path = model_root / "model.safetensors.index.json"
    return {
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
        "model_index": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
            "size_bytes": index_path.stat().st_size,
        },
        "model_shards": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(model_root.glob("*.safetensors"))
        ],
    }


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
        output["blocks"] = len(members)
        for metric in metric_fields:
            values = [
                float(member[metric])
                for member in members
                if member.get(metric) is not None
                and math.isfinite(float(member[metric]))
            ]
            output[f"{metric}_count"] = len(values)
            output[f"{metric}_mean"] = float(np.mean(values)) if values else None
            output[f"{metric}_std"] = float(np.std(values)) if values else None
        aggregated.append(output)
    return aggregated


def make_synthetic_records(
    n: int,
    count: int,
    seed: int,
) -> list[BlockRecord]:
    rng = np.random.default_rng(seed)
    dct = dct_matrix(n)
    row = np.arange(n, dtype=np.float64)[:, None]
    col = np.arange(n, dtype=np.float64)[None, :]
    decay = np.exp(-0.22 * (row + col))
    records: list[BlockRecord] = []
    for index in range(count):
        smooth_coefficients = rng.standard_normal((n, n)) * decay
        smooth = dct.T @ smooth_coefficients @ dct
        records.append(
            BlockRecord(
                dataset="synthetic_smooth",
                tensor_name=f"synthetic_smooth_{index}",
                layer=-1,
                row_start=0,
                col_start=0,
                values=smooth,
            )
        )

        outlier = 0.10 * rng.standard_normal((n, n))
        locations = rng.choice(n * n, size=max(1, n // 8), replace=False)
        outlier.reshape(-1)[locations] += rng.normal(0.0, 20.0, size=locations.size)
        records.append(
            BlockRecord(
                dataset="synthetic_outlier",
                tensor_name=f"synthetic_outlier_{index}",
                layer=-1,
                row_start=0,
                col_start=0,
                values=outlier,
            )
        )

        gaussian = rng.standard_normal((n, n))
        records.append(
            BlockRecord(
                dataset="synthetic_gaussian",
                tensor_name=f"synthetic_gaussian_{index}",
                layer=-1,
                row_start=0,
                col_start=0,
                values=gaussian,
            )
        )
    return records


def transform_metric_rows(
    records: list[BlockRecord],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    n = records[0].values.shape[0]
    identity = np.eye(n, dtype=np.float64)
    dct = dct_matrix(n)
    hadamard = hadamard_matrix(n)
    random_left = randomized_hadamard_matrix(n, seed + 101)
    random_right = randomized_hadamard_matrix(n, seed + 102)
    haar_left = haar_orthogonal_matrix(n, seed + 103)
    haar_right = haar_orthogonal_matrix(n, seed + 104)
    fixed_order = zigzag_indices(n)
    transforms = {
        "identity": (identity, identity),
        "dct2": (dct, dct),
        "hadamard": (hadamard, hadamard),
        "randomized_hadamard": (random_left, random_right),
        "random_haar": (haar_left, haar_right),
    }
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        block = record.values
        common = {
            "dataset": record.dataset,
            "tensor_name": record.tensor_name,
            "layer": record.layer,
            "row_start": record.row_start,
            "col_start": record.col_start,
            "record_index": record_index,
        }
        for name, (left, right) in transforms.items():
            coefficients = two_sided_transform(block, left, right)
            result = {
                **common,
                "transform": name,
                "papr": papr(coefficients),
                "excess_kurtosis": excess_kurtosis(coefficients),
                "structured_energy_1_over_8": structured_energy(
                    coefficients, fixed_order, 1.0 / 8.0
                ),
                "q3_relative_mse": dense_transform_quantize(
                    block, left, right, 3
                ).relative_mse,
                "q4_relative_mse": dense_transform_quantize(
                    block, left, right, 4
                ).relative_mse,
            }
            result.update(
                {
                    metric: topk_energy(coefficients, density)
                    for metric, density in DENSITIES.items()
                }
            )
            rows.append(result)

        fft_coefficients = np.fft.fft2(block, norm="ortho")
        fft_result = {
            **common,
            "transform": "fft2_hermitian",
            "papr": papr(fft_coefficients),
            "excess_kurtosis": excess_kurtosis(fft_coefficients),
            "structured_energy_1_over_8": None,
            "q3_relative_mse": fft_dense_quantize(block, 3).relative_mse,
            "q4_relative_mse": fft_dense_quantize(block, 4).relative_mse,
        }
        fft_result.update(
            {
                metric: fft_topk_energy(block, density)
                for metric, density in DENSITIES.items()
            }
        )
        rows.append(fft_result)

        u, singular_values, vt = np.linalg.svd(block, full_matrices=True)
        svd_coefficients = np.zeros_like(block)
        np.fill_diagonal(svd_coefficients, singular_values)
        svd_result = {
            **common,
            "transform": "svd_oracle_unpriced",
            "papr": papr(svd_coefficients),
            "excess_kurtosis": excess_kurtosis(svd_coefficients),
            "structured_energy_1_over_8": structured_energy(
                svd_coefficients, fixed_order, 1.0 / 8.0
            ),
            "q3_relative_mse": dense_transform_quantize(
                block, u.T, vt, 3
            ).relative_mse,
            "q4_relative_mse": dense_transform_quantize(
                block, u.T, vt, 4
            ).relative_mse,
        }
        svd_result.update(
            {
                metric: topk_energy(svd_coefficients, density)
                for metric, density in DENSITIES.items()
            }
        )
        rows.append(svd_result)
    return rows


def compression_results_for_block(
    block: np.ndarray,
    *,
    seed: int,
) -> dict[str, CompressionResult]:
    n = block.shape[0]
    identity = np.eye(n, dtype=np.float64)
    dct = dct_matrix(n)
    hadamard = hadamard_matrix(n)
    random_left = randomized_hadamard_matrix(n, seed + 201)
    random_right = randomized_hadamard_matrix(n, seed + 202)
    return {
        "identity_q3": dense_transform_quantize(block, identity, identity, 3),
        "identity_q4": dense_transform_quantize(block, identity, identity, 4),
        "dct_q3": dense_transform_quantize(block, dct, dct, 3),
        "hadamard_q3": dense_transform_quantize(block, hadamard, hadamard, 3),
        "randomized_hadamard_q3": dense_transform_quantize(
            block, random_left, random_right, 3
        ),
        "fft_q3": fft_dense_quantize(block, 3),
        "hybrid_dct_zigzag12p5_q8_hadamard_q2": dct_hadamard_hybrid(
            block,
            density=0.125,
            base_bits=8,
            residual_bits=2,
            topk_base=False,
        ),
        "hybrid_dct_topk12p5_q8_hadamard_q2_oracle": dct_hadamard_hybrid(
            block,
            density=0.125,
            base_bits=8,
            residual_bits=2,
            topk_base=True,
        ),
        "hybrid_dct_zigzag6p25_q8_hadamard_q3": dct_hadamard_hybrid(
            block,
            density=0.0625,
            base_bits=8,
            residual_bits=3,
            topk_base=False,
        ),
        "hybrid_dct_zigzag12p5_q8_hadamard_q3": dct_hadamard_hybrid(
            block,
            density=0.125,
            base_bits=8,
            residual_bits=3,
            topk_base=False,
        ),
    }


def compression_metric_rows(
    records: list[BlockRecord],
    *,
    seed: int,
    layer0_activations: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        methods = compression_results_for_block(record.values, seed=seed)
        for method, result in methods.items():
            activation_error = None
            if (
                layer0_activations is not None
                and record.dataset == "llama_q_proj"
                and record.layer == 0
            ):
                start = record.col_start
                stop = start + record.values.shape[1]
                activation_error = output_relative_mse(
                    record.values,
                    result.reconstructed,
                    layer0_activations[:, start:stop],
                )
            rows.append(
                {
                    "dataset": record.dataset,
                    "tensor_name": record.tensor_name,
                    "layer": record.layer,
                    "row_start": record.row_start,
                    "col_start": record.col_start,
                    "record_index": record_index,
                    "method": method,
                    "bits_per_weight": result.bits_per_weight,
                    "relative_mse": result.relative_mse,
                    "activation_output_relative_mse": activation_error,
                }
            )
    return rows


def split_subblocks(
    records: list[BlockRecord],
    subblock_size: int,
) -> np.ndarray:
    subblocks: list[np.ndarray] = []
    for record in records:
        block = record.values
        if block.shape[0] % subblock_size or block.shape[1] % subblock_size:
            continue
        for row in range(0, block.shape[0], subblock_size):
            for col in range(0, block.shape[1], subblock_size):
                subblocks.append(
                    block[row : row + subblock_size, col : col + subblock_size]
                )
    return np.asarray(subblocks, dtype=np.float64)


def matrix_transform_batch(
    blocks: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.asarray([left @ block @ right.T for block in blocks])


def batch_matrix_transform_evaluation(
    blocks: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bits: int,
) -> dict[str, float]:
    from fft_com.metrics import batch_quantization_relative_mse

    coefficients = matrix_transform_batch(blocks, left, right)
    return {
        "relative_mse": batch_quantization_relative_mse(coefficients, bits),
        "papr_mean": float(np.mean([papr(block) for block in coefficients])),
    }


def learned_rotation_rows(
    q_records: list[BlockRecord],
    *,
    subblock_size: int,
    block_count: int,
    stages: int,
    seed: int,
    quick: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks = split_subblocks(q_records, subblock_size)
    rng = np.random.default_rng(seed + 300)
    order = rng.permutation(blocks.shape[0])
    selected = blocks[order[: min(block_count, blocks.shape[0])]]
    midpoint = selected.shape[0] // 2
    train = selected[:midpoint]
    test = selected[midpoint:]
    if train.size == 0 or test.size == 0:
        raise ValueError("not enough learned-rotation blocks")

    learned = learn_butterfly_rotation(
        train,
        bits=3,
        stages=stages,
        random_candidates=4 if quick else 12,
        coordinate_steps=(0.20,) if quick else (0.25, 0.10, 0.04),
        seed=seed + 301,
    )
    zero_angles = np.zeros((stages, subblock_size // 2), dtype=np.float64)
    transforms = {
        "identity": (zero_angles, zero_angles, 0),
        "selected_initial_butterfly": (
            learned.initial_left_angles,
            learned.initial_right_angles,
            learned.initial_left_angles.size + learned.initial_right_angles.size,
        ),
        "learned_butterfly": (
            learned.left_angles,
            learned.right_angles,
            learned.left_angles.size + learned.right_angles.size,
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, (left, right, parameters) in transforms.items():
        for split, values in (("train", train), ("held_out", test)):
            metrics = evaluate_butterfly(values, left, right, 3)
            rows.append(
                {
                    "transform": name,
                    "split": split,
                    "blocks": values.shape[0],
                    "relative_mse": metrics["relative_mse"],
                    "papr_mean": metrics["papr_mean"],
                    "angle_parameters": parameters,
                    "angle_metadata_bits_per_weight": (
                        parameters * 16.0 / (values.size) if parameters else 0.0
                    ),
                }
            )

    identity = np.eye(subblock_size, dtype=np.float64)
    hadamard = hadamard_matrix(subblock_size)
    matrix_candidates: list[
        tuple[float, int, np.ndarray, np.ndarray, dict[str, float]]
    ] = []
    deterministic_train = batch_matrix_transform_evaluation(
        train, hadamard, hadamard, 3
    )
    deterministic_test = batch_matrix_transform_evaluation(
        test, hadamard, hadamard, 3
    )
    for split, values in (
        ("train", deterministic_train),
        ("held_out", deterministic_test),
    ):
        rows.append(
            {
                "transform": "hadamard",
                "split": split,
                "blocks": train.shape[0] if split == "train" else test.shape[0],
                "relative_mse": values["relative_mse"],
                "papr_mean": values["papr_mean"],
                "angle_parameters": 0,
                "angle_metadata_bits_per_weight": 0.0,
            }
        )
    for candidate in range(4 if quick else 16):
        left = randomized_hadamard_matrix(subblock_size, seed + 400 + candidate * 2)
        right = randomized_hadamard_matrix(
            subblock_size, seed + 401 + candidate * 2
        )
        train_metrics = batch_matrix_transform_evaluation(train, left, right, 3)
        matrix_candidates.append(
            (train_metrics["relative_mse"], candidate, left, right, train_metrics)
        )
    _, selected_candidate, selected_left, selected_right, train_metrics = min(
        matrix_candidates, key=lambda item: item[0]
    )
    test_metrics = batch_matrix_transform_evaluation(
        test, selected_left, selected_right, 3
    )
    for split, values in (("train", train_metrics), ("held_out", test_metrics)):
        rows.append(
            {
                "transform": "best_of_randomized_hadamard",
                "split": split,
                "blocks": train.shape[0] if split == "train" else test.shape[0],
                "relative_mse": values["relative_mse"],
                "papr_mean": values["papr_mean"],
                "angle_parameters": 0,
                "angle_metadata_bits_per_weight": 128.0 / (
                    train.size if split == "train" else test.size
                ),
            }
        )

    metadata = {
        "subblock_size": subblock_size,
        "train_blocks": train.shape[0],
        "held_out_blocks": test.shape[0],
        "stages": stages,
        "selected_random_hadamard_candidate": selected_candidate,
        "selected_butterfly_initialization": learned.initial_source,
        "history": list(learned.history),
    }
    return rows, metadata


def kuramoto_rows(
    q_records: list[BlockRecord],
    *,
    subblock_size: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    blocks = split_subblocks(q_records, subblock_size)[:count]
    rows: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        probe = kuramoto_rotation_probe(block, bits=3, seed=seed + 500 + index)
        rows.append({"block": index, **asdict(probe)})
    return rows


def fft_phase_dataset(
    blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = blocks.shape[1]
    groups = [group for group in fft_hermitian_groups(n) if not group[4]]
    magnitudes = np.empty((blocks.shape[0], len(groups)), dtype=np.float64)
    phases = np.empty_like(magnitudes)
    total_energy = 0.0
    for block_index, block in enumerate(blocks):
        coefficients = np.fft.fft2(block, norm="ortho")
        total_energy += float(np.sum(block**2, dtype=np.float64))
        for group_index, (row, col, _, _, _) in enumerate(groups):
            value = coefficients[row, col]
            magnitudes[block_index, group_index] = abs(value)
            phases[block_index, group_index] = np.angle(value)
    return magnitudes, phases, total_energy


def phase_probe_rows(q_records: list[BlockRecord]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    blocks = np.asarray([record.values for record in q_records], dtype=np.float64)
    magnitudes, phases, total_energy = fft_phase_dataset(blocks)
    weights = magnitudes**2
    template, concentration = weighted_phase_template(phases, weights)
    n = blocks.shape[1]
    pairs = phases.shape[1]
    block_count = blocks.shape[0]

    rows: list[dict[str, Any]] = []
    for bits in (2, 3, 4):
        quantized = quantize_phase_uniform(phases, bits)
        rows.append(
            {
                "method": f"uniform_phase_{bits}bit",
                "phase_bits_per_pair": float(bits),
                "phase_bits_per_weight": block_count * pairs * bits
                / (block_count * n * n),
                "phase_only_relative_mse": phase_only_relative_mse(
                    magnitudes, phases, quantized, total_energy
                ),
            }
        )
    for bits in (2, 3):
        quantized = quantize_phase_uniform(phases, bits, offset=template)
        payload = block_count * pairs * bits + pairs * 16
        rows.append(
            {
                "method": f"frequency_template_residual_{bits}bit",
                "phase_bits_per_pair": bits + 16.0 / block_count,
                "phase_bits_per_weight": payload / (block_count * n * n),
                "phase_only_relative_mse": phase_only_relative_mse(
                    magnitudes, phases, quantized, total_energy
                ),
            }
        )
    for levels in (4, 8, 16):
        centers = learn_circular_codebook(phases, weights, levels)
        quantized = quantize_phase_codebook(phases, centers)
        assignment_bits = int(math.ceil(math.log2(levels)))
        payload = block_count * pairs * assignment_bits + levels * 16
        rows.append(
            {
                "method": f"global_circular_codebook_{levels}",
                "phase_bits_per_pair": assignment_bits
                + levels * 16.0 / (block_count * pairs),
                "phase_bits_per_weight": payload / (block_count * n * n),
                "phase_only_relative_mse": phase_only_relative_mse(
                    magnitudes, phases, quantized, total_energy
                ),
            }
        )
    concentration_summary = {
        "weighted_frequency_concentration_mean": float(np.mean(concentration)),
        "weighted_frequency_concentration_median": float(np.median(concentration)),
        "weighted_frequency_concentration_p90": float(
            np.quantile(concentration, 0.90)
        ),
        "blocks": float(block_count),
        "nonself_conjugate_pairs_per_block": float(pairs),
    }
    return rows, concentration_summary


def find_aggregate(
    rows: list[dict[str, Any]],
    **matches: str,
) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in matches.items()):
            return row
    raise KeyError(matches)


def build_decision_signals(
    transform_aggregate: list[dict[str, Any]],
    compression_aggregate: list[dict[str, Any]],
    learned_rows: list[dict[str, Any]],
    kuramoto_aggregate: dict[str, Any],
    phase_rows: list[dict[str, Any]],
) -> dict[str, float]:
    identity = find_aggregate(
        transform_aggregate, dataset="llama_q_proj", transform="identity"
    )
    dct = find_aggregate(
        transform_aggregate, dataset="llama_q_proj", transform="dct2"
    )
    hadamard = find_aggregate(
        transform_aggregate, dataset="llama_q_proj", transform="hadamard"
    )
    hybrid = find_aggregate(
        compression_aggregate,
        dataset="llama_q_proj",
        method="hybrid_dct_zigzag12p5_q8_hadamard_q2",
    )
    random_hadamard = find_aggregate(
        compression_aggregate,
        dataset="llama_q_proj",
        method="randomized_hadamard_q3",
    )
    learned_test = next(
        row
        for row in learned_rows
        if row["transform"] == "learned_butterfly" and row["split"] == "held_out"
    )
    initial_test = next(
        row
        for row in learned_rows
        if row["transform"] == "selected_initial_butterfly"
        and row["split"] == "held_out"
    )
    random_hadamard_test = next(
        row
        for row in learned_rows
        if row["transform"] == "best_of_randomized_hadamard"
        and row["split"] == "held_out"
    )
    phase_uniform = next(
        row for row in phase_rows if row["method"] == "uniform_phase_3bit"
    )
    phase_template = next(
        row
        for row in phase_rows
        if row["method"] == "frequency_template_residual_3bit"
    )
    return {
        "llama_q_dct_topk_1_over_8_gain_vs_identity": (
            dct["topk_1_over_8_mean"] / identity["topk_1_over_8_mean"] - 1.0
        ),
        "llama_q_hadamard_q3_error_ratio_vs_identity": (
            hadamard["q3_relative_mse_mean"]
            / identity["q3_relative_mse_mean"]
        ),
        "llama_q_hybrid_error_ratio_vs_randomized_hadamard_q3": (
            hybrid["relative_mse_mean"] / random_hadamard["relative_mse_mean"]
        ),
        "learned_butterfly_held_out_error_ratio_vs_selected_initial": (
            learned_test["relative_mse"] / initial_test["relative_mse"]
        ),
        "learned_butterfly_held_out_error_ratio_vs_best_randomized_hadamard": (
            learned_test["relative_mse"] / random_hadamard_test["relative_mse"]
        ),
        "kuramoto_rotation_error_ratio_vs_identity": kuramoto_aggregate[
            "error_ratio_mean"
        ],
        "phase_template_3bit_error_ratio_vs_uniform_3bit": (
            phase_template["phase_only_relative_mse"]
            / phase_uniform["phase_only_relative_mse"]
        ),
    }


def main() -> int:
    args = parse_args()
    if args.quick:
        args.layers = "0,8"
        args.blocks_per_tensor = min(args.blocks_per_tensor, 2)
        args.synthetic_blocks = min(args.synthetic_blocks, 2)
        args.embedding_samples = min(args.embedding_samples, 64)
        args.learned_block_count = min(args.learned_block_count, 12)
        args.learned_stages = min(args.learned_stages, 2)

    layers = tuple(int(value) for value in args.layers.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    synthetic_records = make_synthetic_records(
        args.block_size, args.synthetic_blocks, args.seed
    )
    model_records = collect_model_blocks(
        args.model_dir,
        layers=layers,
        tensor_suffixes={
            "llama_q_proj": "self_attn.q_proj.weight",
            "llama_down_proj": "mlp.down_proj.weight",
        },
        block_size=args.block_size,
        blocks_per_tensor=args.blocks_per_tensor,
        seed=args.seed + 1,
    )
    all_records = synthetic_records + model_records
    q_records = [record for record in model_records if record.dataset == "llama_q_proj"]

    transform_raw = transform_metric_rows(all_records, seed=args.seed)
    transform_metrics = (
        "papr",
        "excess_kurtosis",
        "structured_energy_1_over_8",
        "q3_relative_mse",
        "q4_relative_mse",
        *DENSITIES.keys(),
    )
    transform_aggregate = aggregate_rows(
        transform_raw,
        ("dataset", "transform"),
        transform_metrics,
    )

    layer0_activations = load_layer0_embedding_activations(
        args.model_dir,
        sample_count=args.embedding_samples,
        seed=args.seed + 2,
    )
    compression_raw = compression_metric_rows(
        all_records,
        seed=args.seed,
        layer0_activations=layer0_activations,
    )
    compression_aggregate = aggregate_rows(
        compression_raw,
        ("dataset", "method"),
        (
            "bits_per_weight",
            "relative_mse",
            "activation_output_relative_mse",
        ),
    )

    learned_rows, learned_metadata = learned_rotation_rows(
        q_records,
        subblock_size=args.learned_block_size,
        block_count=args.learned_block_count,
        stages=args.learned_stages,
        seed=args.seed,
        quick=args.quick,
    )
    kuramoto_raw = kuramoto_rows(
        q_records,
        subblock_size=args.learned_block_size,
        count=4 if args.quick else 16,
        seed=args.seed,
    )
    kuramoto_aggregate_rows = aggregate_rows(
        ({**row, "probe": "attractive_relative_phase"} for row in kuramoto_raw),
        ("probe",),
        (
            "initial_order_parameter",
            "final_order_parameter",
            "paired_angle_rms",
            "identity_relative_mse",
            "kuramoto_relative_mse",
            "error_ratio",
        ),
    )
    kuramoto_aggregate = kuramoto_aggregate_rows[0]
    phase_rows, phase_concentration = phase_probe_rows(q_records)
    decision_signals = build_decision_signals(
        transform_aggregate,
        compression_aggregate,
        learned_rows,
        kuramoto_aggregate,
        phase_rows,
    )

    finished = datetime.now(timezone.utc)
    metadata = {
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_dir": str(args.model_dir),
        "layers": layers,
        "block_size": args.block_size,
        "blocks_per_tensor": args.blocks_per_tensor,
        "synthetic_blocks_per_family": args.synthetic_blocks,
        "seed": args.seed,
        "study_revision": 2,
        "quick": args.quick,
        "model_block_count": len(model_records),
        "synthetic_block_count": len(synthetic_records),
        "scope": (
            "CPU block-level coefficient and reconstruction study; no end-to-end "
            "model accuracy, latency, or deployment-kernel claim."
        ),
        "provenance": experiment_source_hashes(args.model_dir),
    }
    summary = {
        "metadata": metadata,
        "decision_signals": decision_signals,
        "phase_concentration": phase_concentration,
        "learned_rotation": learned_metadata,
        "transform_aggregate": transform_aggregate,
        "compression_aggregate": compression_aggregate,
        "learned_rotation_rows": learned_rows,
        "kuramoto_aggregate": kuramoto_aggregate,
        "phase_rows": phase_rows,
    }

    write_csv(args.output_dir / "transform_raw.csv", transform_raw)
    write_csv(args.output_dir / "transform_aggregate.csv", transform_aggregate)
    write_csv(args.output_dir / "compression_raw.csv", compression_raw)
    write_csv(args.output_dir / "compression_aggregate.csv", compression_aggregate)
    write_csv(args.output_dir / "learned_rotation.csv", learned_rows)
    write_csv(args.output_dir / "kuramoto_raw.csv", kuramoto_raw)
    write_csv(args.output_dir / "phase_probe.csv", phase_rows)
    write_json(args.output_dir / "summary.json", summary)

    if args.publish:
        write_csv(
            ROOT / "docs" / "tables" / "transform_metrics_20260716.csv",
            transform_aggregate,
        )
        write_csv(
            ROOT / "docs" / "tables" / "compression_rate_distortion_20260716.csv",
            compression_aggregate,
        )
        write_csv(
            ROOT / "docs" / "tables" / "learned_rotation_20260716.csv",
            learned_rows,
        )
        write_csv(
            ROOT / "docs" / "tables" / "kuramoto_phase_probe_20260716.csv",
            [
                {"section": "kuramoto_rotation", **kuramoto_aggregate},
                *({"section": "phase_coding", **row} for row in phase_rows),
            ],
        )
        write_json(
            ROOT
            / "docs"
            / "evidence"
            / "transform_potential_summary_20260716.json",
            summary,
        )

    print(json.dumps(json_ready(decision_signals), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
