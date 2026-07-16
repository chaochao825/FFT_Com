#!/usr/bin/env python3
"""Measure real decoder-only LM KV-cache frequency structure without modification.

This probe captures projected K before RoPE, cached K after RoPE, and cached V
from selected attention layers. It measures sequence-axis DCT/rFFT energy
curves on disjoint WikiText-2 segments. It does not inject compressed caches
back into the model and therefore makes no perplexity, generation-quality,
memory, or latency claim.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/data2/wangmeiqi/Llama-2-7b-chat-hf"),
    )
    parser.add_argument(
        "--test-arrow",
        type=Path,
        default=Path(
            "/home/wangmeiqi/.cache/huggingface/datasets/wikitext/"
            "wikitext-2-raw-v1/0.0.0/"
            "b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-test.arrow"
        ),
    )
    parser.add_argument("--layers", default="0,8,16,24,31")
    parser.add_argument("--token-offsets", default="0,4096,8192,12288")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--ratios", default="0.1,0.25,0.5,0.75,0.9")
    parser.add_argument("--thresholds", default="0.9,0.95,0.99")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "freqkv_real_kv_20260717",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Write aggregate and evidence files under docs/.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run transform-metric checks without loading a model.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_if_exists(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def sha256_tokens(values: Any) -> str:
    array = values.detach().cpu().numpy().astype(np.int64, copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


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
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
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


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output_rows: list[dict[str, Any]] = []
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
        output_rows.append(output)
    return output_rows


def _rfft_weights(sequence_length: int) -> np.ndarray:
    count = sequence_length // 2 + 1
    weights = np.ones(count, dtype=np.float64)
    if sequence_length % 2 == 0:
        if count > 2:
            weights[1:-1] = 2.0
    elif count > 1:
        weights[1:] = 2.0
    return weights


def analyze_frequency_array(
    values: np.ndarray,
    *,
    transform: str,
    ratios: Iterable[float],
    thresholds: Iterable[float],
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Analyze a ``[heads, sequence, head_dim]`` real cache tensor."""

    from scipy.fft import dct

    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("values must have shape [heads, sequence, head_dim]")
    sequence_length = data.shape[1]

    if transform == "dct_ii_ortho":
        coefficients = dct(data, type=2, axis=1, norm="ortho")
        weights = np.ones(sequence_length, dtype=np.float64)
    elif transform == "rfft_parseval":
        coefficients = np.fft.rfft(data, axis=1, norm="ortho")
        weights = _rfft_weights(sequence_length)
    else:
        raise ValueError(f"unknown transform: {transform}")

    per_head_component_energy = np.sum(
        np.abs(coefficients) ** 2,
        axis=2,
        dtype=np.float64,
    )
    per_head_component_energy *= weights[None, :]
    component_energy = np.sum(
        per_head_component_energy,
        axis=0,
        dtype=np.float64,
    )
    total_energy = float(np.sum(component_energy, dtype=np.float64))
    per_head_total = np.sum(
        per_head_component_energy,
        axis=1,
        dtype=np.float64,
    )
    if total_energy == 0.0 or np.any(per_head_total == 0.0):
        raise ValueError("zero-energy cache tensor is not supported")

    component_count = component_energy.size
    retention_rows: list[dict[str, float | int | str]] = []
    for ratio in ratios:
        requested = float(ratio)
        if not 0.0 < requested <= 1.0:
            raise ValueError("ratios must be in (0, 1]")
        retained = max(1, min(component_count, int(requested * component_count)))
        actual_fraction = retained / component_count

        low_global = (
            float(np.sum(component_energy[:retained], dtype=np.float64))
            / total_energy
        )
        selected_global = np.partition(
            component_energy,
            component_count - retained,
        )[-retained:]
        top_global = (
            float(np.sum(selected_global, dtype=np.float64)) / total_energy
        )

        low_per_head = (
            np.sum(
                per_head_component_energy[:, :retained],
                axis=1,
                dtype=np.float64,
            )
            / per_head_total
        )
        selected_per_head = np.partition(
            per_head_component_energy,
            component_count - retained,
            axis=1,
        )[:, -retained:]
        top_per_head = (
            np.sum(selected_per_head, axis=1, dtype=np.float64) / per_head_total
        )
        retention_rows.append(
            {
                "transform": transform,
                "requested_retention_ratio": requested,
                "retained_components": retained,
                "total_components": component_count,
                "actual_component_fraction": actual_fraction,
                "global_low_frequency_energy": low_global,
                "global_top_frequency_bin_energy": top_global,
                "global_top_vs_low_gain": top_global - low_global,
                "global_reconstruction_relative_mse": 1.0 - low_global,
                "per_head_low_frequency_energy_mean": float(np.mean(low_per_head)),
                "per_head_low_frequency_energy_std": float(np.std(low_per_head)),
                "per_head_low_frequency_energy_min": float(np.min(low_per_head)),
                "per_head_low_frequency_energy_max": float(np.max(low_per_head)),
                "per_head_top_frequency_bin_energy_mean": float(
                    np.mean(top_per_head)
                ),
                "per_head_top_frequency_bin_energy_max": float(np.max(top_per_head)),
                "dc_energy_fraction": float(component_energy[0] / total_energy),
            }
        )

    cumulative = np.cumsum(component_energy, dtype=np.float64) / total_energy
    per_head_cumulative = np.cumsum(
        per_head_component_energy,
        axis=1,
        dtype=np.float64,
    ) / per_head_total[:, None]
    threshold_rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        target = float(threshold)
        if not 0.0 < target <= 1.0:
            raise ValueError("thresholds must be in (0, 1]")
        global_index = min(
            int(np.searchsorted(cumulative, target)),
            component_count - 1,
        )
        per_head_required = np.asarray(
            [
                min(
                    int(np.searchsorted(per_head_cumulative[head], target)) + 1,
                    component_count,
                )
                for head in range(per_head_cumulative.shape[0])
            ],
            dtype=np.float64,
        )
        threshold_rows.append(
            {
                "transform": transform,
                "energy_threshold": target,
                "global_required_components": global_index + 1,
                "total_components": component_count,
                "global_required_component_fraction": (
                    (global_index + 1) / component_count
                ),
                "per_head_required_component_fraction_mean": float(
                    np.mean(per_head_required / component_count)
                ),
                "per_head_required_component_fraction_std": float(
                    np.std(per_head_required / component_count)
                ),
                "per_head_required_component_fraction_min": float(
                    np.min(per_head_required / component_count)
                ),
                "per_head_required_component_fraction_max": float(
                    np.max(per_head_required / component_count)
                ),
            }
        )
    return retention_rows, threshold_rows


def run_self_test() -> None:
    from scipy.fft import dct, idct

    rng = np.random.default_rng(7)
    white = rng.standard_normal((8, 128, 32))
    rows, _ = analyze_frequency_array(
        white,
        transform="dct_ii_ortho",
        ratios=(0.25, 0.5),
        thresholds=(0.9,),
    )
    if abs(float(rows[0]["global_low_frequency_energy"]) - 0.25) > 0.04:
        raise AssertionError("white-noise DCT energy should track retained fraction")
    coefficients = rng.standard_normal((8, 128, 32))
    coefficients *= np.exp(-0.20 * np.arange(128))[None, :, None]
    smooth = idct(coefficients, type=2, axis=1, norm="ortho")
    smooth_rows, _ = analyze_frequency_array(
        smooth,
        transform="dct_ii_ortho",
        ratios=(0.25,),
        thresholds=(0.9,),
    )
    if float(smooth_rows[0]["global_low_frequency_energy"]) < 0.99:
        raise AssertionError("smooth positive control should be low-frequency")
    roundtrip = idct(
        dct(smooth, type=2, axis=1, norm="ortho"),
        type=2,
        axis=1,
        norm="ortho",
    )
    np.testing.assert_allclose(roundtrip, smooth, atol=1e-5)
    print("self_test_ok")


def load_token_segments(
    arrow_path: Path,
    tokenizer: Any,
    *,
    sequence_length: int,
    offsets: tuple[int, ...],
) -> list[Any]:
    import torch
    from datasets import Dataset

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if not offsets or min(offsets) < 0:
        raise ValueError("token offsets must be non-negative")
    required_tokens = max(offsets) + sequence_length
    dataset = Dataset.from_file(str(arrow_path))
    texts: list[str] = []
    next_check_chars = max(4096, required_tokens * 5)
    character_count = 0
    encoded: list[int] = []
    for row in dataset:
        text = str(row.get("text", ""))
        texts.append(text)
        character_count += len(text) + 2
        if character_count < next_check_chars:
            continue
        encoded = tokenizer.encode(
            "\n\n".join(texts),
            add_special_tokens=False,
        )
        if len(encoded) >= required_tokens:
            break
        next_check_chars *= 2
    if len(encoded) < required_tokens:
        encoded = tokenizer.encode(
            "\n\n".join(texts),
            add_special_tokens=False,
        )
    if len(encoded) < required_tokens:
        raise RuntimeError(
            f"{arrow_path} produced {len(encoded)} tokens, need {required_tokens}"
        )
    return [
        torch.tensor(
            encoded[offset : offset + sequence_length],
            dtype=torch.long,
        ).unsqueeze(0)
        for offset in offsets
    ]


def find_row(rows: list[dict[str, Any]], **matches: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(field) == value for field, value in matches.items()):
            return row
    raise KeyError(matches)


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    import torch
    import transformers
    from datasets import __version__ as datasets_version
    from scipy import __version__ as scipy_version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layers = tuple(int(value) for value in args.layers.split(",") if value)
    offsets = tuple(int(value) for value in args.token_offsets.split(",") if value)
    ratios = tuple(float(value) for value in args.ratios.split(",") if value)
    thresholds = tuple(float(value) for value in args.thresholds.split(",") if value)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    started = datetime.now(timezone.utc)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
    )
    token_segments = load_token_segments(
        args.test_arrow,
        tokenizer,
        sequence_length=args.sequence_length,
        offsets=offsets,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.eval().to(args.device)
    config = model.config
    num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(config.hidden_size // config.num_attention_heads)
    if max(layers) >= int(config.num_hidden_layers):
        raise ValueError("requested layer is outside model depth")

    projected_keys: dict[int, Any] = {}
    handles = []
    for layer in layers:
        projection = model.model.layers[layer].self_attn.k_proj

        def make_hook(layer_index: int) -> Any:
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                projected_keys[layer_index] = output.detach()

            return hook

        handles.append(projection.register_forward_hook(make_hook(layer)))

    retention_raw: list[dict[str, Any]] = []
    threshold_raw: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    token_metadata: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for segment_index, (offset, input_ids) in enumerate(
            zip(offsets, token_segments)
        ):
            projected_keys.clear()
            device_ids = input_ids.to(args.device)
            with torch.inference_mode():
                output = model.model(
                    input_ids=device_ids,
                    use_cache=True,
                    return_dict=True,
                )
            cache = output.past_key_values
            legacy_cache = (
                cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
            )
            token_hash = sha256_tokens(input_ids)
            token_metadata.append(
                {
                    "segment_index": segment_index,
                    "token_offset": offset,
                    "sequence_length": args.sequence_length,
                    "token_ids_sha256": token_hash,
                }
            )

            for layer in layers:
                key_post, value_cache = legacy_cache[layer]
                key_projected = projected_keys[layer]
                expected_width = num_kv_heads * head_dim
                if key_projected.shape[-1] != expected_width:
                    raise RuntimeError(
                        f"layer {layer} k_proj width {key_projected.shape[-1]} "
                        f"!= {expected_width}"
                    )
                key_pre = key_projected.reshape(
                    key_projected.shape[0],
                    key_projected.shape[1],
                    num_kv_heads,
                    head_dim,
                ).transpose(1, 2)
                arrays = {
                    "K_pre_rope": key_pre,
                    "K_post_rope": key_post,
                    "V_cache": value_cache,
                }
                pre_energy = float(torch.sum(key_pre.float() ** 2).item())
                post_energy = float(torch.sum(key_post.float() ** 2).item())
                norm_rows.append(
                    {
                        "segment_index": segment_index,
                        "token_offset": offset,
                        "layer": layer,
                        "k_post_vs_pre_energy_ratio": post_energy / pre_energy,
                    }
                )

                for cache_stage, tensor in arrays.items():
                    data = tensor[0].float().cpu().numpy()
                    sequence_variants = {
                        "raw": data,
                        "mean_removed": (
                            data - np.mean(data, axis=1, keepdims=True)
                        ),
                    }
                    for sequence_variant, variant_data in sequence_variants.items():
                        common = {
                            "segment_index": segment_index,
                            "token_offset": offset,
                            "token_ids_sha256": token_hash,
                            "layer": layer,
                            "cache_stage": cache_stage,
                            "sequence_variant": sequence_variant,
                            "heads": variant_data.shape[0],
                            "sequence_length": variant_data.shape[1],
                            "head_dim": variant_data.shape[2],
                        }
                        for transform in ("dct_ii_ortho", "rfft_parseval"):
                            retention, threshold_data = analyze_frequency_array(
                                variant_data,
                                transform=transform,
                                ratios=ratios,
                                thresholds=thresholds,
                            )
                            retention_raw.extend(
                                {**common, **row} for row in retention
                            )
                            threshold_raw.extend(
                                {**common, **row} for row in threshold_data
                            )

            del output, cache, legacy_cache, device_ids
    finally:
        for handle in handles:
            handle.remove()

    metric_fields = (
        "retained_components",
        "total_components",
        "actual_component_fraction",
        "global_low_frequency_energy",
        "global_top_frequency_bin_energy",
        "global_top_vs_low_gain",
        "global_reconstruction_relative_mse",
        "per_head_low_frequency_energy_mean",
        "per_head_low_frequency_energy_std",
        "per_head_low_frequency_energy_min",
        "per_head_low_frequency_energy_max",
        "per_head_top_frequency_bin_energy_mean",
        "per_head_top_frequency_bin_energy_max",
        "dc_energy_fraction",
    )
    retention_aggregate = aggregate_rows(
        retention_raw,
        (
            "layer",
            "cache_stage",
            "sequence_variant",
            "transform",
            "requested_retention_ratio",
        ),
        metric_fields,
    )
    retention_overall = aggregate_rows(
        retention_raw,
        (
            "cache_stage",
            "sequence_variant",
            "transform",
            "requested_retention_ratio",
        ),
        metric_fields,
    )
    threshold_metric_fields = (
        "global_required_components",
        "total_components",
        "global_required_component_fraction",
        "per_head_required_component_fraction_mean",
        "per_head_required_component_fraction_std",
        "per_head_required_component_fraction_min",
        "per_head_required_component_fraction_max",
    )
    threshold_aggregate = aggregate_rows(
        threshold_raw,
        (
            "layer",
            "cache_stage",
            "sequence_variant",
            "transform",
            "energy_threshold",
        ),
        threshold_metric_fields,
    )
    threshold_overall = aggregate_rows(
        threshold_raw,
        ("cache_stage", "sequence_variant", "transform", "energy_threshold"),
        threshold_metric_fields,
    )
    norm_aggregate = aggregate_rows(
        norm_rows,
        ("layer",),
        ("k_post_vs_pre_energy_ratio",),
    )

    key_pre_dct_25 = find_row(
        retention_overall,
        cache_stage="K_pre_rope",
        sequence_variant="raw",
        transform="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    key_post_dct_25 = find_row(
        retention_overall,
        cache_stage="K_post_rope",
        sequence_variant="raw",
        transform="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    value_dct_25 = find_row(
        retention_overall,
        cache_stage="V_cache",
        sequence_variant="raw",
        transform="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    key_pre_centered_dct_25 = find_row(
        retention_overall,
        cache_stage="K_pre_rope",
        sequence_variant="mean_removed",
        transform="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    key_post_centered_dct_25 = find_row(
        retention_overall,
        cache_stage="K_post_rope",
        sequence_variant="mean_removed",
        transform="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    value_centered_dct_25 = find_row(
        retention_overall,
        cache_stage="V_cache",
        sequence_variant="mean_removed",
        transform="dct_ii_ortho",
        requested_retention_ratio=0.25,
    )
    key_post_dct_threshold = find_row(
        threshold_overall,
        cache_stage="K_post_rope",
        sequence_variant="raw",
        transform="dct_ii_ortho",
        energy_threshold=0.9,
    )
    key_post_centered_dct_threshold = find_row(
        threshold_overall,
        cache_stage="K_post_rope",
        sequence_variant="mean_removed",
        transform="dct_ii_ortho",
        energy_threshold=0.9,
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
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets_version,
        "scipy": scipy_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "model_dir": str(args.model_dir),
        "model_type": config.model_type,
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "test_arrow": str(args.test_arrow),
        "layers": layers,
        "token_offsets": offsets,
        "sequence_length": args.sequence_length,
        "position_ids": "reset to 0..sequence_length-1 for every segment",
        "dataset_token_offsets_are_not_rope_position_offsets": True,
        "ratios": ratios,
        "thresholds": thresholds,
        "sequence_variants": {
            "raw": "unmodified captured tensor",
            "mean_removed": (
                "per-head, per-channel sequence mean removed before transform"
            ),
        },
        "token_segments": token_metadata,
        "scope": (
            f"Real {config.model_type} decoder-only LM KV spectral capture on "
            "WikiText-2 test segments. No compressed-cache injection, "
            "perplexity, generation quality, compact payload, memory reduction, "
            "or latency claim."
        ),
        "provenance": {
            "model_config_sha256": sha256_file(args.model_dir / "config.json"),
            "model_index_sha256": sha256_if_exists(
                args.model_dir / "model.safetensors.index.json"
            ),
            "model_weights_sha256": sha256_if_exists(
                args.model_dir / "model.safetensors"
            ),
            "tokenizer_json_sha256": sha256_if_exists(
                args.model_dir / "tokenizer.json"
            ),
            "dataset_arrow_sha256": sha256_file(args.test_arrow),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    summary = {
        "metadata": metadata,
        "decision_signals": {
            "overall_dct_low25_energy": {
                "K_pre_rope": key_pre_dct_25[
                    "global_low_frequency_energy_mean"
                ],
                "K_post_rope": key_post_dct_25[
                    "global_low_frequency_energy_mean"
                ],
                "V_cache": value_dct_25["global_low_frequency_energy_mean"],
            },
            "overall_mean_removed_dct_low25_energy": {
                "K_pre_rope": key_pre_centered_dct_25[
                    "global_low_frequency_energy_mean"
                ],
                "K_post_rope": key_post_centered_dct_25[
                    "global_low_frequency_energy_mean"
                ],
                "V_cache": value_centered_dct_25[
                    "global_low_frequency_energy_mean"
                ],
            },
            "overall_mean_removed_dct_top25_frequency_bin_energy": {
                "K_pre_rope": key_pre_centered_dct_25[
                    "global_top_frequency_bin_energy_mean"
                ],
                "K_post_rope": key_post_centered_dct_25[
                    "global_top_frequency_bin_energy_mean"
                ],
                "V_cache": value_centered_dct_25[
                    "global_top_frequency_bin_energy_mean"
                ],
            },
            "overall_raw_dct_dc_energy": {
                "K_pre_rope": key_pre_dct_25["dc_energy_fraction_mean"],
                "K_post_rope": key_post_dct_25["dc_energy_fraction_mean"],
                "V_cache": value_dct_25["dc_energy_fraction_mean"],
            },
            "overall_dct_top25_frequency_bin_energy": {
                "K_pre_rope": key_pre_dct_25[
                    "global_top_frequency_bin_energy_mean"
                ],
                "K_post_rope": key_post_dct_25[
                    "global_top_frequency_bin_energy_mean"
                ],
                "V_cache": value_dct_25[
                    "global_top_frequency_bin_energy_mean"
                ],
            },
            "rope_delta_dct_low25_energy": (
                key_post_dct_25["global_low_frequency_energy_mean"]
                - key_pre_dct_25["global_low_frequency_energy_mean"]
            ),
            "K_post_rope_dct_fraction_for_90pct_energy": (
                key_post_dct_threshold[
                    "global_required_component_fraction_mean"
                ]
            ),
            "K_post_rope_mean_removed_dct_fraction_for_90pct_energy": (
                key_post_centered_dct_threshold[
                    "global_required_component_fraction_mean"
                ]
            ),
            "K_post_rope_dct_per_head_low25_max_mean": key_post_dct_25[
                "per_head_low_frequency_energy_max_mean"
            ],
            "interpretation_boundary": (
                "These metrics test spectral concentration only. They do not "
                "establish whether truncation preserves attention outputs or "
                "language-model quality."
            ),
        },
        "retention_overall": retention_overall,
        "threshold_overall": threshold_overall,
        "k_rope_energy_norm": norm_aggregate,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "retention_raw.csv", retention_raw)
    write_csv(args.output_dir / "retention_aggregate.csv", retention_aggregate)
    write_csv(args.output_dir / "retention_overall.csv", retention_overall)
    write_csv(args.output_dir / "thresholds_raw.csv", threshold_raw)
    write_csv(args.output_dir / "thresholds_aggregate.csv", threshold_aggregate)
    write_csv(args.output_dir / "thresholds_overall.csv", threshold_overall)
    write_csv(args.output_dir / "rope_norm.csv", norm_rows)
    write_json(args.output_dir / "summary.json", summary)

    if args.publish:
        write_csv(
            ROOT / "docs" / "evidence" / "freqkv_real_kv_raw_20260717.csv",
            retention_raw,
        )
        write_csv(
            ROOT / "docs" / "tables" / "freqkv_real_kv_results_20260717.csv",
            retention_aggregate,
        )
        write_csv(
            ROOT
            / "docs"
            / "tables"
            / "freqkv_real_kv_thresholds_20260717.csv",
            threshold_aggregate,
        )
        write_json(
            ROOT / "docs" / "evidence" / "freqkv_real_kv_summary_20260717.json",
            summary,
        )

    print(json.dumps(json_ready(summary["decision_signals"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
