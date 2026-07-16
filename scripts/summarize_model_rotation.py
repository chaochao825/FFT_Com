#!/usr/bin/env python3
"""Aggregate model-rotation JSON evidence into publishable CSV/Markdown files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / "runs" / "model_rotation_20260717",
    )
    parser.add_argument(
        "--latency-json",
        type=Path,
        default=ROOT / "runs" / "model_rotation_20260717" / "latency.json",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "model_rotation_summary_20260717",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
            writer.writerow(row)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def discover_results(runs_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(runs_dir.rglob("*.json")):
        if "smoke" in path.parts:
            continue
        try:
            payload = read_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if "variant" not in payload or "perplexity" not in payload:
            continue
        candidates.append((path, payload))
    latest: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, payload in candidates:
        variant = str(payload["variant"])
        previous = latest.get(variant)
        if previous is None or str(payload.get("created_at", "")) > str(
            previous[1].get("created_at", "")
        ):
            latest[variant] = (path, payload)
    return sorted(latest.values(), key=lambda item: str(item[1]["variant"]))


def protocol_key(payload: dict[str, Any]) -> tuple[int, int, int, str]:
    ppl = payload["perplexity"]
    experiment = payload["experiment"]
    return (
        int(ppl["input_tokens"]),
        int(ppl["sequence_length"]),
        int(experiment.get("eval_token_offset", 0)),
        str(ppl["protocol"]),
    )


def baseline_map(
    results: Iterable[tuple[Path, dict[str, Any]]],
) -> dict[tuple[int, int, int, str], float]:
    baselines: dict[tuple[int, int, int, str], float] = {}
    for _, payload in results:
        if not payload["experiment"].get("baseline"):
            continue
        baselines[protocol_key(payload)] = float(payload["perplexity"]["perplexity"])
    return baselines


def result_rows(
    results: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    baselines = baseline_map(results)
    rows: list[dict[str, Any]] = []
    for path, payload in results:
        experiment = payload["experiment"]
        aggregate = payload["aggregate"]
        ppl = payload["perplexity"]
        baseline = baselines.get(protocol_key(payload))
        current_ppl = float(ppl["perplexity"])
        topology = payload["model"]["topology"]
        is_baseline = bool(experiment.get("baseline"))
        is_unquantized = bool(experiment.get("unquantized"))
        row = {
            "variant": payload["variant"],
            "evidence_file": str(path),
            "baseline": is_baseline,
            "unquantized": is_unquantized,
            "scope": experiment.get("scope"),
            "transform": experiment.get("transform"),
            "bits": experiment.get("bits"),
            "permutation": experiment.get("permutation"),
            "module_count": aggregate.get("module_count"),
            "parameter_count": aggregate.get("parameter_count"),
            "weight_relative_mse_parameter_weighted": finite(
                aggregate.get("weight_relative_mse_parameter_weighted")
            ),
            "calibration_output_relative_mse_parameter_weighted": finite(
                aggregate.get(
                    "calibration_output_relative_mse_parameter_weighted"
                )
            ),
            "metadata_bits": aggregate.get("metadata_bits", 0.0),
            "adjacent_abs_cosine_before_mean": finite(
                aggregate.get("adjacent_abs_cosine_before_mean")
            ),
            "adjacent_abs_cosine_after_mean": finite(
                aggregate.get("adjacent_abs_cosine_after_mean")
            ),
            "input_adjacent_abs_cosine_before_mean": finite(
                aggregate.get("input_adjacent_abs_cosine_before_mean")
            ),
            "input_adjacent_abs_cosine_after_mean": finite(
                aggregate.get("input_adjacent_abs_cosine_after_mean")
            ),
            "output_adjacent_abs_cosine_before_mean": finite(
                aggregate.get("output_adjacent_abs_cosine_before_mean")
            ),
            "output_adjacent_abs_cosine_after_mean": finite(
                aggregate.get("output_adjacent_abs_cosine_after_mean")
            ),
            "perplexity": current_ppl,
            "fp16_baseline_perplexity_same_protocol": baseline,
            "perplexity_delta_vs_fp16": (
                current_ppl - baseline if baseline is not None else None
            ),
            "eval_input_tokens": ppl["input_tokens"],
            "eval_token_offset": experiment.get("eval_token_offset", 0),
            "eval_sequence_length": ppl["sequence_length"],
            "eval_seconds": ppl["elapsed_seconds"],
            "input_tokens_per_second": ppl["input_tokens_per_second"],
            "calibration_tokens": experiment["calibration_tokens"],
            "calibration_samples_per_module": experiment[
                "calibration_samples_per_module"
            ],
            "model": payload["model"]["path"],
            "num_attention_heads": topology["num_attention_heads"],
            "num_key_value_heads": topology["num_key_value_heads"],
            "uses_gqa": topology["uses_gqa"],
            "fake_quantization_only": (
                bool(experiment["fake_quantization_only"])
                and not is_baseline
                and not is_unquantized
            ),
            "packed_integer_kernel": experiment["packed_integer_kernel"],
        }
        rows.append(row)
    return rows


def latency_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    protocol = payload["protocol"]
    return [
        {
            **row,
            "dtype": protocol["dtype"],
            "group_size": protocol["group_size"],
            "warmup": protocol["warmup"],
            "repeats": protocol["repeats"],
            "reference_implementation": protocol["reference_implementation"],
            "fused_kernel": protocol["fused_kernel"],
            "gpu": payload["runtime"]["gpu"],
        }
        for row in payload["rows"]
    ]


def segment_robustness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity: dict[tuple[str, int, int, int], float] = {}
    for row in rows:
        if (
            row.get("scope") in {"q_proj_input", "q_proj_output_head"}
            and row.get("bits") is not None
            and row.get("transform") == "identity"
            and row.get("permutation") == "none"
            and int(row["eval_input_tokens"]) == 8192
        ):
            identity[
                (
                    str(row["scope"]),
                    int(row["bits"]),
                    int(row["eval_token_offset"]),
                    int(row["eval_input_tokens"]),
                )
            ] = float(row["perplexity"])

    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("scope") not in {"q_proj_input", "q_proj_output_head"}
            or row.get("bits") is None
            or int(row["eval_input_tokens"]) != 8192
        ):
            continue
        identity_ppl = identity.get(
            (
                str(row["scope"]),
                int(row["bits"]),
                int(row["eval_token_offset"]),
                int(row["eval_input_tokens"]),
            )
        )
        if identity_ppl is None:
            continue
        enriched = dict(row)
        enriched["perplexity_delta_vs_scope_identity"] = (
            float(row["perplexity"]) - identity_ppl
        )
        grouped[
            (
                str(row["scope"]),
                str(row["transform"]),
                int(row["bits"]),
                str(row["permutation"]),
            )
        ].append(enriched)

    output: list[dict[str, Any]] = []
    for (scope, transform, bits, permutation), members in sorted(grouped.items()):
        perplexities = [float(row["perplexity"]) for row in members]
        deltas = [
            float(row["perplexity_delta_vs_scope_identity"]) for row in members
        ]
        output.append(
            {
                "scope": scope,
                "transform": transform,
                "bits": bits,
                "permutation": permutation,
                "segment_count": len(members),
                "eval_token_offsets": ",".join(
                    str(value)
                    for value in sorted(
                        int(row["eval_token_offset"]) for row in members
                    )
                ),
                "perplexity_mean": statistics.fmean(perplexities),
                "perplexity_std": statistics.pstdev(perplexities),
                "delta_vs_scope_identity_mean": statistics.fmean(deltas),
                "delta_vs_scope_identity_std": statistics.pstdev(deltas),
                "segments_better_than_identity": sum(delta < 0.0 for delta in deltas),
            }
        )
    return output


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    numeric = float(value)
    if abs(numeric) < 1e-3 and numeric != 0.0:
        return f"{numeric:.3e}"
    return f"{numeric:.{digits}f}"


def markdown_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(fmt(row.get(field)) for field, _ in columns)
            + " |"
        )
    return "\n".join(lines)


def choose_rows(
    rows: list[dict[str, Any]],
    *,
    scopes: set[str],
    tokens: int | None = None,
    offset: int | None = 0,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("scope") in scopes
        and not row["unquantized"]
        and (tokens is None or int(row["eval_input_tokens"]) == tokens)
        and (offset is None or int(row["eval_token_offset"]) == offset)
    ]
    return sorted(
        selected,
        key=lambda row: (
            int(row["bits"]) if row["bits"] is not None else -1,
            str(row["scope"]),
            str(row["transform"]),
            str(row["permutation"]),
        ),
    )


def make_report(
    rows: list[dict[str, Any]],
    latency: list[dict[str, Any]],
    segment_robustness: list[dict[str, Any]],
) -> str:
    token_counts = sorted({int(row["eval_input_tokens"]) for row in rows})
    preferred_tokens = max(token_counts) if token_counts else None
    all_input = choose_rows(rows, scopes={"all_input"}, tokens=preferred_tokens)
    if not all_input:
        all_input = choose_rows(rows, scopes={"all_input"})
    one_sided = choose_rows(
        rows,
        scopes={"q_proj_input", "q_proj_output_head", "q_proj_two_sided"},
        tokens=preferred_tokens,
    )
    head_aware = choose_rows(
        rows,
        scopes={"attention_head", "qk_rope_pair"},
        tokens=preferred_tokens,
    )
    head_screen = choose_rows(
        rows,
        scopes={"attention_head", "qk_rope_pair"},
        tokens=2048,
    )
    baselines = [
        row
        for row in rows
        if row["baseline"]
        and row["eval_input_tokens"] == preferred_tokens
        and int(row["eval_token_offset"]) == 0
    ]
    latency_hidden = [
        row
        for row in latency
        if row["benchmark"] == "transform_forward"
        and row["shape"] == "hidden"
    ]
    latency_qproj = [
        row
        for row in latency
        if row["benchmark"] == "q_proj_online_input_rotation"
    ]

    def find_result(
        scope: str,
        transform: str,
        bits: int,
        *,
        permutation: str = "none",
    ) -> dict[str, Any] | None:
        for row in rows:
            if (
                row.get("scope") == scope
                and row.get("transform") == transform
                and row.get("bits") == bits
                and row.get("permutation") == permutation
                and int(row["eval_input_tokens"]) == preferred_tokens
                and int(row["eval_token_offset"]) == 0
            ):
                return row
        return None

    decisions: list[str] = []
    all_identity_q3 = find_result("all_input", "identity", 3)
    all_dct_q3 = find_result("all_input", "dct", 3)
    all_identity_q4 = find_result("all_input", "identity", 4)
    all_dct_q4 = find_result("all_input", "dct", 4)
    if all_identity_q3 and all_dct_q3 and all_identity_q4 and all_dct_q4:
        decisions.append(
            "统一 all-input 位置不支持 dense rotation："
            f"q3 Identity/DCT 为 {fmt(all_identity_q3['perplexity'])}/"
            f"{fmt(all_dct_q3['perplexity'])}，q4 为 "
            f"{fmt(all_identity_q4['perplexity'])}/"
            f"{fmt(all_dct_q4['perplexity'])}。"
        )

    input_identity = find_result("q_proj_input", "identity", 3)
    input_dct = find_result("q_proj_input", "dct", 3)
    input_hadamard = find_result("q_proj_input", "hadamard", 3)
    input_rdft = find_result("q_proj_input", "rdft", 3)
    output_dct = find_result("q_proj_output_head", "dct", 3)
    if all((input_identity, input_dct, input_hadamard, input_rdft, output_dct)):
        decisions.append(
            "DCT 的保留信号局限在 q_proj 单侧输入旋转："
            f"Identity/DCT/Hadamard/RDFT PPL 为 "
            f"{fmt(input_identity['perplexity'])}/"
            f"{fmt(input_dct['perplexity'])}/"
            f"{fmt(input_hadamard['perplexity'])}/"
            f"{fmt(input_rdft['perplexity'])}；输出侧 DCT 为 "
            f"{fmt(output_dct['perplexity'])}。"
        )

    spectral_input = find_result(
        "q_proj_input",
        "dct",
        3,
        permutation="spectral",
    )
    spectral_two_sided = find_result(
        "q_proj_two_sided",
        "dct",
        3,
        permutation="spectral",
    )
    if spectral_input and spectral_two_sided:
        decisions.append(
            "Permutation 没有稳定超过普通 DCT：输入侧 spectral-DCT 为 "
            f"{fmt(spectral_input['perplexity'])}，双侧 spectral-DCT 为 "
            f"{fmt(spectral_two_sided['perplexity'])}，后者还需 "
            f"{fmt(spectral_two_sided['metadata_bits'], 0)} metadata bits。"
        )

    head_identity = find_result("attention_head", "identity", 4)
    head_dct = find_result("attention_head", "dct", 4)
    rope_identity = find_result("qk_rope_pair", "identity", 4)
    rope_dct = find_result("qk_rope_pair", "dct", 4)
    if all((head_identity, head_dct, rope_identity, rope_dct)):
        decisions.append(
            "Head/RoPE-aware DCT 未获支持：attention-head q4 "
            f"Identity/DCT 为 {fmt(head_identity['perplexity'])}/"
            f"{fmt(head_dct['perplexity'])}，RoPE-pair q4 为 "
            f"{fmt(rope_identity['perplexity'])}/"
            f"{fmt(rope_dct['perplexity'])}。"
        )

    hidden_latency = {
        (row["transform"], row["permutation"]): row
        for row in latency_hidden
    }
    dct_latency = hidden_latency.get(("dct", "none"))
    hadamard_latency = hidden_latency.get(("hadamard", "none"))
    qproj_latency = {
        row["transform"]: row
        for row in latency_qproj
        if row["permutation"] == "none"
    }
    qproj_identity = qproj_latency.get("identity")
    qproj_dct = qproj_latency.get("dct")
    qproj_hadamard = qproj_latency.get("hadamard")
    if all(
        (
            dct_latency,
            hadamard_latency,
            qproj_identity,
            qproj_dct,
            qproj_hadamard,
        )
    ):
        decisions.append(
            "参考实现中 DCT 比 Hadamard 快，但不是零成本："
            f"{fmt(dct_latency['median_ms'])} ms 对 "
            f"{fmt(hadamard_latency['median_ms'])} ms；包含 FP16 GEMM 的 "
            "q_proj 路径中，Identity/DCT/Hadamard 为 "
            f"{fmt(qproj_identity['median_ms'])}/"
            f"{fmt(qproj_dct['median_ms'])}/"
            f"{fmt(qproj_hadamard['median_ms'])} ms。"
        )

    replicated_dct = next(
        (
            row
            for row in segment_robustness
            if row["scope"] == "q_proj_input"
            and row["transform"] == "dct"
            and row["bits"] == 3
            and row["permutation"] == "none"
            and row["segment_count"] >= 2
        ),
        None,
    )
    if replicated_dct:
        decisions.append(
            "跨测试区段复核中，q_proj 输入 DCT 相对同 scope Identity 的"
            f"平均 ΔPPL 为 "
            f"{fmt(replicated_dct['delta_vs_scope_identity_mean'])}，"
            f"{replicated_dct['segments_better_than_identity']}/"
            f"{replicated_dct['segment_count']} 个区段获胜。"
        )

    lines = [
        "# 合法模型旋转与 DCT 潜力复核（2026-07-17）",
        "",
        "## 证据边界",
        "",
        "- **已有工作**：2026-07-16 的 Llama-2-7B-chat 块级研究，证明原始 channel 顺序上的 DCT 低频裁剪不成立，并观察到 dense Hadamard/FFT 的 q3 重建优势；它不是端到端模型结果。",
        "- **本次新增尝试**：在 Llama-2-7B base 上，将 Identity、1D DCT、Hadamard 和实数 RDFT 放到相同、可逆的在线线性层位置；校准来自 WikiText-2 train，PPL 来自不重叠的 WikiText-2 test 前缀。",
        "- 权重是 fake-quant 后反量化到 FP16，通过普通浮点 GEMM 执行；本文不声称 packed INT3/INT4 存储或整数 kernel 加速。",
        "- 当前模型有 32 个 query heads 和 32 个 KV heads，不使用 GQA；实现按 KV-head 边界编写，但该 checkpoint 不能实证 GQA 收益。",
        "",
        "## 核心结论",
        "",
    ]
    lines.extend(f"- {decision}" for decision in decisions)
    if not decisions:
        lines.append("- 正式结果尚未完整生成。")
    lines.extend(
        [
            "",
        "## 统一 all-input 比较",
        "",
        ]
    )
    if baselines:
        lines.append(
            f"同协议 FP16 基线 PPL：{fmt(baselines[0]['perplexity'], 6)}。"
        )
        lines.append("")
    if all_input:
        lines.extend(
            [
                markdown_table(
                    all_input,
                    [
                        ("bits", "bits"),
                        ("transform", "transform"),
                        ("perplexity", "PPL"),
                        ("perplexity_delta_vs_fp16", "ΔPPL"),
                        (
                            "weight_relative_mse_parameter_weighted",
                            "weight rel-MSE",
                        ),
                        (
                            "calibration_output_relative_mse_parameter_weighted",
                            "calibration output rel-MSE",
                        ),
                        ("input_tokens_per_second", "tokens/s"),
                    ],
                ),
                "",
                "表中 tokens/s 来自各自独立的完整模型进程，只用于保留运行记录；变换之间的延迟比较以本文后面的 CUDA-event 专项基准为准。",
                "",
            ]
        )
    else:
        lines.extend(["统一 all-input 结果尚未生成。", ""])

    lines.extend(["## 单侧/双侧 1D 与 permutation", ""])
    if one_sided:
        lines.extend(
            [
                markdown_table(
                    one_sided,
                    [
                        ("scope", "scope"),
                        ("bits", "bits"),
                        ("transform", "transform"),
                        ("permutation", "permutation"),
                        ("perplexity", "PPL"),
                        (
                            "calibration_output_relative_mse_parameter_weighted",
                            "calibration rel-MSE",
                        ),
                        ("metadata_bits", "metadata bits"),
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["单侧结果尚未生成。", ""])

    lines.extend(["## Head / RoPE 边界", ""])
    if head_aware:
        lines.extend(
            [
                markdown_table(
                    head_aware,
                    [
                        ("scope", "scope"),
                        ("transform", "transform"),
                        ("bits", "bits"),
                        ("perplexity", "PPL"),
                        (
                            "calibration_output_relative_mse_parameter_weighted",
                            "calibration rel-MSE",
                        ),
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["Head-aware 结果尚未生成。", ""])
    if head_screen:
        lines.extend(
            [
                "q3 的 2,048-token 筛选如下；它只用于淘汰明显负方案：",
                "",
                markdown_table(
                    head_screen,
                    [
                        ("scope", "scope"),
                        ("transform", "transform"),
                        ("perplexity", "PPL"),
                    ],
                ),
                "",
            ]
        )

    lines.extend(["## 跨测试区段稳健性", ""])
    replicated = [
        row for row in segment_robustness if int(row["segment_count"]) >= 2
    ]
    if replicated:
        lines.extend(
            [
                markdown_table(
                    replicated,
                    [
                        ("scope", "scope"),
                        ("transform", "transform"),
                        ("permutation", "permutation"),
                        ("segment_count", "segments"),
                        ("perplexity_mean", "PPL mean"),
                        ("perplexity_std", "PPL std"),
                        (
                            "delta_vs_scope_identity_mean",
                            "mean Δ vs Identity",
                        ),
                        (
                            "segments_better_than_identity",
                            "wins",
                        ),
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["跨区段复核结果尚未生成。", ""])

    lines.extend(["## 参考变换延迟", ""])
    if latency_hidden:
        lines.extend(
            [
                "在线 transform（`[1,256,4096]`，group 128）：",
                "",
                markdown_table(
                    latency_hidden,
                    [
                        ("transform", "transform"),
                        ("permutation", "permutation"),
                        ("median_ms", "median ms"),
                        ("min_ms", "min ms"),
                        ("max_ms", "max ms"),
                    ],
                ),
                "",
            ]
        )
        if latency_qproj:
            lines.extend(
                [
                    "代表性 4096×4096 q_proj（在线输入 transform + FP16 GEMM）：",
                    "",
                    markdown_table(
                        latency_qproj,
                        [
                            ("transform", "transform"),
                            ("median_ms", "median ms"),
                            ("min_ms", "min ms"),
                            ("max_ms", "max ms"),
                        ],
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                "这些是未融合 PyTorch 参考实现的 CUDA-event 延迟；DCT/RDFT 内部使用 fp32 FFT，Hadamard 使用逐级张量操作。它们用于暴露在线成本，不代表优化后 kernel 排名。",
                "",
            ]
        )
    else:
        lines.extend(["正式延迟结果尚未生成。", ""])

    lines.extend(
        [
            "## 判定原则",
            "",
            "1. DCT 不再按“低频可裁剪”评价，而按 dense rotation 的量化质量、PPL 与在线成本评价。",
            "2. 单层或局部 MSE 改善若不能转化为同协议 PPL 改善，不作为继续投入依据。",
            "3. permutation 只有在收益超过元数据与 gather 成本时才保留。",
            "4. Head/RoPE-aware 方案必须用同 scope Identity 基线比较；未量化等价性只证明位置合法，不证明量化后有效。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = discover_results(args.runs_dir)
    rows = result_rows(results)
    latency = latency_rows(args.latency_json)
    segment_robustness = segment_robustness_rows(rows)
    provenance_by_variant = {
        str(payload["variant"]): payload.get("provenance", {})
        for _, payload in results
    }
    module_evidence_variants = {
        "all_input_identity_q3_8192",
        "all_input_dct_q3_8192",
        "all_input_identity_q4_8192",
        "all_input_dct_q4_8192",
        "qproj_input_identity_q3_8192",
        "qproj_input_dct_q3_8192",
        "qproj_input_hadamard_q3_8192",
        "qproj_input_rdft_q3_8192",
        "qproj_input_spectral_dct_q3_8192",
        "qproj_output_head_dct_q3_8192",
        "attention_head_identity_q4_8192",
        "attention_head_dct_q4_8192",
        "qk_rope_pair_identity_q4_8192",
        "qk_rope_pair_dct_q4_8192",
        "qproj_two_sided_spectral_dct_q3_8192",
    }
    module_evidence = {
        str(payload["variant"]): {
            "experiment": payload["experiment"],
            "model": payload["model"],
            "data": payload["data"],
            "aggregate": payload["aggregate"],
            "perplexity": payload["perplexity"],
            "modules": payload["modules"],
            "provenance": payload.get("provenance", {}),
        }
        for _, payload in results
        if str(payload["variant"]) in module_evidence_variants
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "model_rotation_results.csv", rows)
    write_csv(
        args.output_dir / "one_sided_rotation_results.csv",
        [
            row
            for row in rows
            if row.get("scope")
            in {"q_proj_input", "q_proj_output_head", "q_proj_two_sided"}
        ],
    )
    write_csv(args.output_dir / "transform_latency.csv", latency)
    write_csv(
        args.output_dir / "segment_robustness_results.csv",
        segment_robustness,
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_count": len(rows),
        "results": rows,
        "latency": latency,
        "segment_robustness": segment_robustness,
        "provenance_by_variant": provenance_by_variant,
    }
    write_json(args.output_dir / "model_rotation_summary.json", summary)
    write_json(
        args.output_dir / "model_rotation_module_metrics.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "variants": module_evidence,
        },
    )
    report = make_report(rows, latency, segment_robustness)
    (args.output_dir / "model_rotation_study.md").write_text(
        report,
        encoding="utf-8",
    )

    if args.publish:
        write_csv(ROOT / "docs" / "tables" / "model_rotation_results_20260717.csv", rows)
        write_csv(
            ROOT / "docs" / "tables" / "one_sided_rotation_results_20260717.csv",
            [
                row
                for row in rows
                if row.get("scope")
                in {"q_proj_input", "q_proj_output_head", "q_proj_two_sided"}
            ],
        )
        write_csv(
            ROOT / "docs" / "tables" / "transform_latency_20260717.csv",
            latency,
        )
        write_csv(
            ROOT / "docs" / "tables" / "segment_robustness_20260717.csv",
            segment_robustness,
        )
        write_json(
            ROOT / "docs" / "evidence" / "model_rotation_summary_20260717.json",
            summary,
        )
        write_json(
            ROOT
            / "docs"
            / "evidence"
            / "model_rotation_module_metrics_20260717.json",
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "variants": module_evidence,
            },
        )
        (ROOT / "docs" / "reports" / "model_rotation_study_20260717.md").write_text(
            report,
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "results": len(rows),
                "latency_rows": len(latency),
                "output_dir": str(args.output_dir),
                "published": args.publish,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
