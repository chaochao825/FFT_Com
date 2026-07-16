#!/usr/bin/env python3
"""Evaluate legal online rotations with real calibration activations and PPL.

This is a fake-quantization quality experiment. Quantized values are
dequantized back to the model dtype and executed by ordinary floating-point
``F.linear``. The measured end-to-end time therefore includes explicit online
transform overhead but is not the latency of a packed INT3/INT4 kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fft_com.model_rotation import (  # noqa: E402
    SUPPORTED_SCOPES,
    build_online_rotated_linear,
    make_rotation_plan,
    model_topology_summary,
    replace_module,
    spectral_channel_permutation,
)
from fft_com.torch_transforms import SUPPORTED_TRANSFORMS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/wangmeiqi/zjh/meta-llama/Llama-2-7b-hf"),
    )
    parser.add_argument(
        "--calibration-arrow",
        type=Path,
        default=Path(
            "/home/wangmeiqi/.cache/huggingface/datasets/wikitext/"
            "wikitext-2-raw-v1/0.0.0/"
            "b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-train.arrow"
        ),
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
    parser.add_argument("--variant", required=True)
    parser.add_argument("--scope", choices=SUPPORTED_SCOPES, default="all_input")
    parser.add_argument(
        "--transform",
        choices=SUPPORTED_TRANSFORMS,
        default="identity",
    )
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--unquantized",
        action="store_true",
        help="Apply the requested transforms without fake quantization.",
    )
    parser.add_argument(
        "--permutation",
        choices=("none", "spectral"),
        default="none",
    )
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--calibration-tokens", type=int, default=512)
    parser.add_argument("--calibration-token-offset", type=int, default=0)
    parser.add_argument("--calibration-samples-per-module", type=int, default=32)
    parser.add_argument("--eval-tokens", type=int, default=8192)
    parser.add_argument("--eval-token-offset", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-modules", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *arguments),
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_provenance(model_dir: Path, arrows: list[Path]) -> dict[str, Any]:
    sources = [
        ROOT / "src" / "fft_com" / "torch_transforms.py",
        ROOT / "src" / "fft_com" / "model_rotation.py",
        ROOT / "scripts" / "run_model_rotation_eval.py",
        ROOT / "tests" / "test_torch_rotation.py",
    ]
    model_files = [
        model_dir / "config.json",
        model_dir / "model.safetensors.index.json",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer.model",
    ]
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sources
            if path.exists()
        },
        "model_manifest_sha256": {
            path.name: sha256_file(path) for path in model_files if path.exists()
        },
        "dataset_arrow": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in arrows
        ],
    }


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_token_prefix(
    arrow_path: Path,
    tokenizer: Any,
    token_count: int,
    *,
    token_offset: int = 0,
) -> torch.Tensor:
    from datasets import Dataset

    if token_count < 2:
        raise ValueError("token_count must be at least 2")
    if token_offset < 0:
        raise ValueError("token_offset must be non-negative")
    required_tokens = token_count + token_offset
    dataset = Dataset.from_file(str(arrow_path))
    texts: list[str] = []
    next_check_chars = max(4096, required_tokens * 5)
    character_count = 0
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
            return torch.tensor(
                encoded[token_offset:required_tokens],
                dtype=torch.long,
            ).unsqueeze(0)
        next_check_chars *= 2
    encoded = tokenizer.encode(
        "\n\n".join(texts),
        add_special_tokens=False,
    )
    if len(encoded) < required_tokens:
        raise RuntimeError(
            f"{arrow_path} produced only {len(encoded)} tokens, requested "
            f"offset {token_offset} + count {token_count}"
        )
    return torch.tensor(
        encoded[token_offset:required_tokens],
        dtype=torch.long,
    ).unsqueeze(0)


def target_linear_names(
    model: nn.Module,
    model_config: Any,
    *,
    scope: str,
) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        plan = make_rotation_plan(
            name,
            module,
            model_config,
            scope=scope,
            transform_kind="identity",
        )
        if plan.targeted:
            names.append(name)
    return names


def capture_calibration_inputs(
    model: nn.Module,
    module_names: list[str],
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    samples_per_module: int,
) -> dict[str, torch.Tensor]:
    requested = set(module_names)
    captured: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if name in captured:
                return
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            count = min(samples_per_module, values.shape[0])
            if count <= 0:
                raise RuntimeError(f"no calibration rows observed for {name}")
            indices = torch.linspace(
                0,
                values.shape[0] - 1,
                steps=count,
                device=values.device,
            ).round().long()
            captured[name] = (
                values.index_select(0, indices)
                .to(device="cpu", dtype=torch.float16)
                .contiguous()
            )

        return hook

    for name, module in model.named_modules():
        if name in requested:
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        base_model = getattr(model, "model", model)
        with torch.inference_mode():
            base_model(
                input_ids=input_ids.to(device),
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in hooks:
            handle.remove()
    missing = sorted(requested - captured.keys())
    if missing:
        raise RuntimeError(f"calibration hooks did not observe: {missing[:5]}")
    return captured


def relative_output_mse(
    original: nn.Linear,
    replacement: nn.Module,
    calibration_inputs: torch.Tensor,
) -> float:
    values = calibration_inputs.to(
        device=original.weight.device,
        dtype=original.weight.dtype,
    )
    with torch.inference_mode():
        reference = F.linear(values, original.weight, original.bias)
        estimate = replacement(values)
        numerator = torch.sum((reference.float() - estimate.float()) ** 2).double()
        denominator = torch.sum(reference.float() ** 2).double()
    if denominator.item() == 0.0:
        return 0.0 if numerator.item() == 0.0 else math.inf
    return float((numerator / denominator).item())


def apply_rotation_variant(
    model: nn.Module,
    model_config: Any,
    calibration: dict[str, torch.Tensor],
    *,
    scope: str,
    transform_kind: str,
    bits: int | None,
    quant_group_size: int,
    permutation_strategy: str,
    max_modules: int,
) -> list[dict[str, Any]]:
    linear_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    records: list[dict[str, Any]] = []
    target_index = 0
    for name in linear_names:
        linear = model.get_submodule(name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} was replaced before it could be processed")
        base_plan = make_rotation_plan(
            name,
            linear,
            model_config,
            scope=scope,
            transform_kind=transform_kind,
        )
        if not base_plan.targeted:
            continue
        if max_modules > 0 and target_index >= max_modules:
            break
        target_index += 1

        input_permutation = None
        output_permutation = None
        permutation_diagnostics: dict[str, float] = {}
        if permutation_strategy == "spectral":
            has_input = base_plan.input_transform is not None
            has_output = base_plan.output_transform is not None
            if not has_input and not has_output:
                raise ValueError("spectral permutation requires a non-identity plan")
            if has_output and (
                base_plan.output_transform.layout == "rope_pairs"
            ):
                raise ValueError("RoPE-pair layout uses a fixed algorithmic permutation")
            side_results: list[tuple[str, torch.Tensor, dict[str, float]]] = []
            if has_input:
                input_permutation, diagnostics = spectral_channel_permutation(
                    linear.weight.detach(),
                    base_plan.input_transform.group_size,
                    side="input",
                )
                side_results.append(("input", input_permutation, diagnostics))
            if has_output:
                output_permutation, diagnostics = spectral_channel_permutation(
                    linear.weight.detach(),
                    base_plan.output_transform.group_size,
                    side="output",
                )
                side_results.append(("output", output_permutation, diagnostics))
            if len(side_results) == 1:
                permutation_diagnostics.update(side_results[0][2])
            else:
                for side, _, diagnostics in side_results:
                    permutation_diagnostics.update(
                        {
                            f"{side}_{key}": value
                            for key, value in diagnostics.items()
                        }
                    )
            plan = make_rotation_plan(
                name,
                linear,
                model_config,
                scope=scope,
                transform_kind=transform_kind,
                input_permutation=input_permutation,
                output_permutation=output_permutation,
            )
        else:
            plan = base_plan

        start = time.perf_counter()
        replacement, quant_stats = build_online_rotated_linear(
            linear,
            bits=bits,
            quant_group_size=quant_group_size,
            input_transform=plan.input_transform,
            output_transform=plan.output_transform,
        )
        calibration_mse = relative_output_mse(
            linear,
            replacement,
            calibration[name],
        )
        replace_module(model, name, replacement)
        elapsed = time.perf_counter() - start
        record = {
            "module": name,
            "projection": name.rsplit(".", 1)[-1],
            "in_features": linear.in_features,
            "out_features": linear.out_features,
            "boundary": plan.boundary,
            "head_count": plan.head_count,
            "queries_per_kv_head": plan.kv_group_size,
            "input_transform": (
                None
                if plan.input_transform is None
                else {
                    "kind": plan.input_transform.kind,
                    "group_size": plan.input_transform.group_size,
                    "layout": plan.input_transform.layout,
                    "boundary": plan.input_transform.boundary,
                }
            ),
            "output_transform": (
                None
                if plan.output_transform is None
                else {
                    "kind": plan.output_transform.kind,
                    "group_size": plan.output_transform.group_size,
                    "layout": plan.output_transform.layout,
                    "boundary": plan.output_transform.boundary,
                }
            ),
            "calibration_output_relative_mse": calibration_mse,
            "build_seconds": elapsed,
            **asdict(quant_stats),
            **permutation_diagnostics,
        }
        records.append(record)
        if target_index == 1 or target_index % 8 == 0:
            print(
                json.dumps(
                    {
                        "event": "module_progress",
                        "completed": target_index,
                        "module": name,
                        "weight_relative_mse": quant_stats.weight_relative_mse,
                        "calibration_output_relative_mse": calibration_mse,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return records


def evaluate_perplexity(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    device: torch.device,
    sequence_length: int,
) -> dict[str, float | int | str]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    total_nll = 0.0
    predicted_tokens = 0
    windows = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, token_ids.shape[1] - 1, sequence_length):
            end = min(start + sequence_length, token_ids.shape[1])
            if end - start < 2:
                break
            window = token_ids[:, start:end].to(device)
            output = model(
                input_ids=window,
                labels=window,
                use_cache=False,
                return_dict=True,
            )
            count = window.shape[1] - 1
            total_nll += float(output.loss.float().item()) * count
            predicted_tokens += count
            windows += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    mean_nll = total_nll / predicted_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "predicted_tokens": predicted_tokens,
        "input_tokens": int(token_ids.shape[1]),
        "windows": windows,
        "sequence_length": sequence_length,
        "elapsed_seconds": elapsed,
        "input_tokens_per_second": token_ids.shape[1] / elapsed,
        "predicted_tokens_per_second": predicted_tokens / elapsed,
        "protocol": "non_overlapping_contiguous_windows_first_token_unscored",
    }


def aggregate_module_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "module_count": 0,
            "parameter_count": 0,
            "metadata_bits": 0.0,
        }
    parameter_count = sum(int(row["parameter_count"]) for row in records)

    def weighted(field: str) -> float:
        return sum(
            float(row[field]) * int(row["parameter_count"]) for row in records
        ) / parameter_count

    result: dict[str, Any] = {
        "module_count": len(records),
        "parameter_count": parameter_count,
        "metadata_bits": sum(float(row["metadata_bits"]) for row in records),
        "weight_relative_mse_parameter_weighted": weighted("weight_relative_mse"),
        "transformed_weight_relative_mse_parameter_weighted": weighted(
            "transformed_weight_relative_mse"
        ),
        "calibration_output_relative_mse_parameter_weighted": weighted(
            "calibration_output_relative_mse"
        ),
        "build_seconds": sum(float(row["build_seconds"]) for row in records),
    }
    for prefix in ("", "input_", "output_"):
        before_key = f"{prefix}adjacent_abs_cosine_before"
        after_key = f"{prefix}adjacent_abs_cosine_after"
        adjacency_before = [
            float(row[before_key]) for row in records if before_key in row
        ]
        adjacency_after = [
            float(row[after_key]) for row in records if after_key in row
        ]
        if adjacency_before:
            result[f"{before_key}_mean"] = sum(adjacency_before) / len(
                adjacency_before
            )
            result[f"{after_key}_mean"] = sum(adjacency_after) / len(
                adjacency_after
            )
    return result


def main() -> None:
    args = parse_args()
    if args.baseline and args.unquantized:
        raise ValueError("--baseline and --unquantized are mutually exclusive")
    if not args.baseline and not args.unquantized and args.bits < 2:
        raise ValueError("--bits must be at least 2 for fake quantization")
    if args.permutation == "spectral" and args.transform == "identity":
        raise ValueError("spectral permutation must be paired with a transform")

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype_from_name(args.dtype)
    print(
        json.dumps(
            {
                "event": "load_start",
                "variant": args.variant,
                "device": str(device),
                "model_dir": str(args.model_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    load_seconds = time.perf_counter() - load_start

    calibration_ids = load_token_prefix(
        args.calibration_arrow,
        tokenizer,
        args.calibration_tokens,
        token_offset=args.calibration_token_offset,
    )
    test_ids = load_token_prefix(
        args.test_arrow,
        tokenizer,
        args.eval_tokens,
        token_offset=args.eval_token_offset,
    )
    module_records: list[dict[str, Any]] = []
    if not args.baseline:
        names = target_linear_names(model, model.config, scope=args.scope)
        if args.max_modules > 0:
            names = names[: args.max_modules]
        print(
            json.dumps(
                {
                    "event": "calibration_start",
                    "target_modules": len(names),
                    "calibration_tokens": args.calibration_tokens,
                    "samples_per_module": args.calibration_samples_per_module,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        calibration = capture_calibration_inputs(
            model,
            names,
            calibration_ids,
            device=device,
            samples_per_module=args.calibration_samples_per_module,
        )
        module_records = apply_rotation_variant(
            model,
            model.config,
            calibration,
            scope=args.scope,
            transform_kind=args.transform,
            bits=None if args.unquantized else args.bits,
            quant_group_size=args.quant_group_size,
            permutation_strategy=args.permutation,
            max_modules=args.max_modules,
        )
        del calibration
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "event": "ppl_start",
                "variant": args.variant,
                "eval_tokens": args.eval_tokens,
                "sequence_length": args.sequence_length,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    perplexity = evaluate_perplexity(
        model,
        test_ids,
        device=device,
        sequence_length=args.sequence_length,
    )
    gpu_info: dict[str, Any] = {}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_info = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "experiment": {
            "baseline": args.baseline,
            "unquantized": args.unquantized,
            "scope": args.scope,
            "transform": args.transform,
            "bits": None if args.baseline or args.unquantized else args.bits,
            "quantizer": "symmetric_absmax_per_row_input_group",
            "quant_group_size": args.quant_group_size,
            "permutation": args.permutation,
            "calibration_tokens": args.calibration_tokens,
            "calibration_token_offset": args.calibration_token_offset,
            "calibration_samples_per_module": args.calibration_samples_per_module,
            "eval_token_offset": args.eval_token_offset,
            "max_modules": args.max_modules,
            "fake_quantization_only": True,
            "packed_integer_kernel": False,
            "online_transform_cost_included_in_ppl_elapsed_time": True,
            "weight_mse_measurement": (
                "float32_dequantized_before_final_model_dtype_storage"
            ),
        },
        "model": {
            "path": str(args.model_dir),
            "class": model.__class__.__name__,
            "dtype": args.dtype,
            "load_seconds": load_seconds,
            "topology": model_topology_summary(model.config),
        },
        "data": {
            "name": "WikiText-2 raw",
            "calibration_split": "train",
            "evaluation_split": "test",
            "calibration_arrow": str(args.calibration_arrow),
            "test_arrow": str(args.test_arrow),
            "calibration_token_offset": args.calibration_token_offset,
            "evaluation_token_offset": args.eval_token_offset,
            "calibration_and_evaluation_are_disjoint": True,
        },
        "aggregate": aggregate_module_stats(module_records),
        "modules": module_records,
        "perplexity": perplexity,
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": gpu_info,
        },
        "provenance": source_provenance(
            args.model_dir,
            [args.calibration_arrow, args.test_arrow],
        ),
        "limitations": [
            "Weights are fake-quantized then dequantized to floating point.",
            "No packed INT3/INT4 storage or integer GEMM kernel is measured.",
            "Online transforms are reference PyTorch implementations, not fused kernels.",
            "Reported weight MSE isolates transform and fake quantization in "
            "float32; calibration-output MSE and perplexity include final model-"
            "dtype storage and online-transform roundoff.",
            "The Llama-2-7B model has equal query and KV head counts, so the code "
            "enforces GQA boundaries but this checkpoint does not exercise GQA.",
        ],
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "event": "complete",
                "variant": args.variant,
                "perplexity": perplexity["perplexity"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
