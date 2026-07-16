#!/usr/bin/env python3
"""Aggregate several transform-potential run summaries into robustness tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def find(rows: list[dict[str, Any]], **matches: str) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in matches.items()):
            return row
    raise KeyError(matches)


def extract(summary: dict[str, Any]) -> dict[str, float | int]:
    transform = summary["transform_aggregate"]
    compression = summary["compression_aggregate"]
    learned = summary["learned_rotation_rows"]
    identity_q = find(transform, dataset="llama_q_proj", transform="identity")
    dct_q = find(transform, dataset="llama_q_proj", transform="dct2")
    hadamard_q = find(transform, dataset="llama_q_proj", transform="hadamard")
    fft_q = find(transform, dataset="llama_q_proj", transform="fft2_hermitian")
    identity_down = find(
        transform, dataset="llama_down_proj", transform="identity"
    )
    hadamard_down = find(
        transform, dataset="llama_down_proj", transform="hadamard"
    )
    identity_q3 = find(
        compression, dataset="llama_q_proj", method="identity_q3"
    )
    identity_q4 = find(
        compression, dataset="llama_q_proj", method="identity_q4"
    )
    fft_q3 = find(compression, dataset="llama_q_proj", method="fft_q3")
    hadamard_q3 = find(
        compression, dataset="llama_q_proj", method="hadamard_q3"
    )
    hybrid_q2 = find(
        compression,
        dataset="llama_q_proj",
        method="hybrid_dct_zigzag12p5_q8_hadamard_q2",
    )
    hybrid_q3 = find(
        compression,
        dataset="llama_q_proj",
        method="hybrid_dct_zigzag12p5_q8_hadamard_q3",
    )
    learned_test = find(
        learned, transform="learned_butterfly", split="held_out"
    )
    random_hadamard_test = find(
        learned, transform="best_of_randomized_hadamard", split="held_out"
    )
    return {
        "seed": int(summary["metadata"]["seed"]),
        "q_dct_topk_1_over_8_ratio_vs_identity": dct_q["topk_1_over_8_mean"]
        / identity_q["topk_1_over_8_mean"],
        "q_hadamard_q3_weight_error_ratio_vs_identity": hadamard_q[
            "q3_relative_mse_mean"
        ]
        / identity_q["q3_relative_mse_mean"],
        "q_fft_q3_weight_error_ratio_vs_identity": fft_q["q3_relative_mse_mean"]
        / identity_q["q3_relative_mse_mean"],
        "down_hadamard_q3_weight_error_ratio_vs_identity": hadamard_down[
            "q3_relative_mse_mean"
        ]
        / identity_down["q3_relative_mse_mean"],
        "q_fft_q3_activation_error_ratio_vs_identity_q3": fft_q3[
            "activation_output_relative_mse_mean"
        ]
        / identity_q3["activation_output_relative_mse_mean"],
        "q_fft_q3_activation_error_ratio_vs_identity_q4": fft_q3[
            "activation_output_relative_mse_mean"
        ]
        / identity_q4["activation_output_relative_mse_mean"],
        "q_hadamard_q3_activation_error_ratio_vs_identity_q3": hadamard_q3[
            "activation_output_relative_mse_mean"
        ]
        / identity_q3["activation_output_relative_mse_mean"],
        "q_hybrid_q2_error_ratio_vs_hadamard_q3": hybrid_q2[
            "relative_mse_mean"
        ]
        / hadamard_q3["relative_mse_mean"],
        "q_hybrid_q3_error_ratio_vs_identity_q4": hybrid_q3[
            "relative_mse_mean"
        ]
        / identity_q4["relative_mse_mean"],
        "learned_error_ratio_vs_best_randomized_hadamard": learned_test[
            "relative_mse"
        ]
        / random_hadamard_test["relative_mse"],
        "kuramoto_error_ratio_vs_identity": summary["kuramoto_aggregate"][
            "error_ratio_mean"
        ],
        "phase_concentration_median": summary["phase_concentration"][
            "weighted_frequency_concentration_median"
        ],
    }


def main() -> int:
    args = parse_args()
    rows = []
    for path in args.summaries:
        with path.open("r", encoding="utf-8") as handle:
            rows.append(extract(json.load(handle)))
    rows.sort(key=lambda row: int(row["seed"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict[str, float]] = {}
    for metric in rows[0]:
        if metric == "seed":
            continue
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    payload = {
        "seeds": [int(row["seed"]) for row in rows],
        "runs": rows,
        "aggregate": aggregate,
        "all_finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key != "seed"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
