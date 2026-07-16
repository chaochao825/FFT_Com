"""Functional online rotations and fake weight quantization for Llama linears."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .torch_transforms import (
    GroupedOrthogonalTransform,
    inverse_transform_linear_weight,
    transform_linear_weight,
)


LINEAR_PROJECTIONS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
ATTENTION_PROJECTIONS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
SUPPORTED_SCOPES = (
    "all_input",
    "attention_input",
    "q_proj_input",
    "q_proj_output_head",
    "q_proj_two_sided",
    "attention_head",
    "qk_rope_pair",
)


@dataclass(frozen=True)
class RotationPlan:
    targeted: bool
    input_transform: GroupedOrthogonalTransform | None
    output_transform: GroupedOrthogonalTransform | None
    boundary: str
    head_count: int | None = None
    kv_group_size: int | None = None


@dataclass(frozen=True)
class QuantizedLinearStats:
    weight_relative_mse: float
    transformed_weight_relative_mse: float
    metadata_bits: float
    parameter_count: int
    quantized_parameter_count: int
    bits: int | None
    quant_group_size: int


def _relative_mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.sum((reference.float() - estimate.float()) ** 2).double()
    denominator = torch.sum(reference.float() ** 2).double()
    if denominator.item() == 0.0:
        return 0.0 if numerator.item() == 0.0 else math.inf
    return float((numerator / denominator).item())


def fake_quantize_per_row_group(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric abs-max fake quantization over input-channel groups per row."""

    if bits < 2:
        raise ValueError("bits must be at least 2")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(
            f"weight input dimension {columns} is not divisible by {group_size}"
        )
    qmax = (1 << (bits - 1)) - 1
    grouped = weight.float().reshape(rows, columns // group_size, group_size)
    maxima = grouped.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(maxima > 0, maxima / qmax, torch.ones_like(maxima))
    codes = torch.clamp(torch.round(grouped / scales), -qmax, qmax)
    dequantized = (codes * scales).reshape_as(weight)
    return dequantized.to(weight.dtype), scales.squeeze(-1)


def _channel_vectors(
    weight: torch.Tensor,
    group_size: int,
    *,
    side: str,
) -> torch.Tensor:
    if side == "input":
        if weight.shape[1] % group_size:
            raise ValueError("input dimension must be divisible by group_size")
        return (
            weight.float()
            .reshape(weight.shape[0], -1, group_size)
            .permute(1, 2, 0)
            .contiguous()
        )
    if side == "output":
        if weight.shape[0] % group_size:
            raise ValueError("output dimension must be divisible by group_size")
        return weight.float().reshape(-1, group_size, weight.shape[1]).contiguous()
    raise ValueError("side must be 'input' or 'output'")


def adjacent_absolute_cosine(
    weight: torch.Tensor,
    group_size: int,
    *,
    side: str = "input",
    permutation: torch.Tensor | None = None,
) -> float:
    """Mean absolute cosine similarity of adjacent channels inside each group."""

    vectors = _channel_vectors(weight, group_size, side=side)
    vectors = F.normalize(vectors, dim=-1, eps=1e-12)
    if permutation is not None:
        values = permutation.to(vectors.device)
        if values.ndim == 1:
            vectors = vectors.index_select(1, values)
        else:
            if values.shape[:2] != vectors.shape[:2]:
                raise ValueError("permutation shape does not match channel groups")
            vectors = vectors.gather(
                1,
                values.unsqueeze(-1).expand_as(vectors),
            )
    similarity = torch.sum(vectors[:, :-1] * vectors[:, 1:], dim=-1).abs()
    return float(similarity.mean().item())


def spectral_channel_permutation(
    weight: torch.Tensor,
    group_size: int,
    *,
    side: str = "input",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Order channels with batched Fiedler-vector spectral seriation.

    The graph uses absolute cosine similarity, so sign-flipped but otherwise
    correlated channels remain adjacent. Each transform group gets its own
    permutation and therefore pays ``log2(group_size!)`` metadata bits.
    """

    vectors = _channel_vectors(weight, group_size, side=side)
    normalized = F.normalize(vectors, dim=-1, eps=1e-12)
    affinity = torch.bmm(normalized, normalized.transpose(1, 2)).abs()
    identity = torch.eye(
        group_size,
        dtype=affinity.dtype,
        device=affinity.device,
    ).unsqueeze(0)
    affinity = affinity * (1.0 - identity)
    laplacian = torch.diag_embed(affinity.sum(dim=-1)) - affinity
    _, eigenvectors = torch.linalg.eigh(laplacian)
    fiedler = eigenvectors[..., 1] if group_size > 1 else eigenvectors[..., 0]
    permutation = torch.argsort(fiedler, dim=-1)
    before = adjacent_absolute_cosine(weight, group_size, side=side)
    after = adjacent_absolute_cosine(
        weight,
        group_size,
        side=side,
        permutation=permutation,
    )
    diagnostics = {
        "adjacent_abs_cosine_before": before,
        "adjacent_abs_cosine_after": after,
        "adjacent_abs_cosine_delta": after - before,
        "permutation_metadata_bits": (
            permutation.shape[0]
            * math.lgamma(group_size + 1)
            / math.log(2.0)
        ),
    }
    return permutation, diagnostics


class OnlineRotatedLinear(nn.Module):
    """A dequantized linear with explicit online input/output transforms."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        input_transform: GroupedOrthogonalTransform | None = None,
        output_transform: GroupedOrthogonalTransform | None = None,
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed_input = (
            self.input_transform(values)
            if self.input_transform is not None
            else values
        )
        output = F.linear(transformed_input, self.weight, self.bias)
        if self.output_transform is not None:
            output = self.output_transform.inverse(output)
        return output

    def reconstructed_weight(self) -> torch.Tensor:
        return inverse_transform_linear_weight(
            self.weight,
            input_transform=self.input_transform,
            output_transform=self.output_transform,
        )


def build_online_rotated_linear(
    linear: nn.Linear,
    *,
    bits: int | None,
    quant_group_size: int,
    input_transform: GroupedOrthogonalTransform | None = None,
    output_transform: GroupedOrthogonalTransform | None = None,
) -> tuple[OnlineRotatedLinear, QuantizedLinearStats]:
    """Transform, fake-quantize, and wrap one linear layer."""

    device = linear.weight.device
    dtype = linear.weight.dtype
    if input_transform is not None:
        input_transform = input_transform.to(device=device)
    if output_transform is not None:
        output_transform = output_transform.to(device=device)

    with torch.no_grad():
        original = linear.weight.detach()
        work_dtype = (
            torch.float64 if original.dtype == torch.float64 else torch.float32
        )
        transformed = transform_linear_weight(
            original.to(work_dtype),
            input_transform=input_transform,
            output_transform=output_transform,
        )
        if bits is None:
            dequantized = transformed
        else:
            dequantized, _ = fake_quantize_per_row_group(
                transformed,
                bits,
                quant_group_size,
            )
        transformed_error = _relative_mse(transformed, dequantized)
        restored = inverse_transform_linear_weight(
            dequantized,
            input_transform=input_transform,
            output_transform=output_transform,
        )
        weight_error = _relative_mse(original, restored)

        transformed_bias = (
            linear.bias.detach().to(work_dtype)
            if linear.bias is not None
            else None
        )
        if transformed_bias is not None and output_transform is not None:
            transformed_bias = output_transform(transformed_bias.unsqueeze(0)).squeeze(0)

        wrapper = OnlineRotatedLinear(
            dequantized.to(dtype=dtype),
            transformed_bias.to(dtype=dtype) if transformed_bias is not None else None,
            input_transform=input_transform,
            output_transform=output_transform,
        )
        metadata_bits = 0.0
        if input_transform is not None:
            metadata_bits += input_transform.metadata_bits(linear.in_features)
        if output_transform is not None:
            metadata_bits += output_transform.metadata_bits(linear.out_features)
        stats = QuantizedLinearStats(
            weight_relative_mse=weight_error,
            transformed_weight_relative_mse=transformed_error,
            metadata_bits=metadata_bits,
            parameter_count=linear.weight.numel(),
            quantized_parameter_count=linear.weight.numel(),
            bits=bits,
            quant_group_size=quant_group_size,
        )
    return wrapper, stats


def _projection_name(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def _head_dim(config: Any) -> int:
    explicit = getattr(config, "head_dim", None)
    if explicit is not None:
        return int(explicit)
    hidden_size = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    if hidden_size % heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    return hidden_size // heads


def make_rotation_plan(
    module_name: str,
    linear: nn.Linear,
    model_config: Any,
    *,
    scope: str,
    transform_kind: str,
    permutation: torch.Tensor | None = None,
    input_permutation: torch.Tensor | None = None,
    output_permutation: torch.Tensor | None = None,
) -> RotationPlan:
    """Create a head/GQA/RoPE-aware plan for one named Llama projection."""

    if scope not in SUPPORTED_SCOPES:
        raise ValueError(f"unsupported scope {scope!r}")
    if permutation is not None:
        if input_permutation is not None or output_permutation is not None:
            raise ValueError(
                "use either permutation or side-specific permutations, not both"
            )
        input_permutation = permutation
        output_permutation = permutation
    projection = _projection_name(module_name)
    if projection not in LINEAR_PROJECTIONS:
        return RotationPlan(False, None, None, "not_targeted")

    head_dim = _head_dim(model_config)
    attention_heads = int(model_config.num_attention_heads)
    kv_heads = int(
        getattr(model_config, "num_key_value_heads", attention_heads)
    )
    if attention_heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    q_per_kv = attention_heads // kv_heads

    input_transform: GroupedOrthogonalTransform | None = None
    output_transform: GroupedOrthogonalTransform | None = None
    targeted = False
    boundary = "not_targeted"
    head_count: int | None = None

    if scope == "all_input" and projection in LINEAR_PROJECTIONS:
        targeted = True
        boundary = "input_channel_groups_128"
        input_transform = GroupedOrthogonalTransform(
            transform_kind,
            128,
            permutation=input_permutation,
            boundary=boundary,
        )
    elif scope == "attention_input" and projection in ATTENTION_PROJECTIONS:
        targeted = True
        boundary = "attention_input_groups_128"
        input_transform = GroupedOrthogonalTransform(
            transform_kind,
            128,
            permutation=input_permutation,
            boundary=boundary,
        )
    elif scope == "q_proj_input" and projection == "q_proj":
        targeted = True
        boundary = "q_input_groups_128"
        input_transform = GroupedOrthogonalTransform(
            transform_kind,
            128,
            permutation=input_permutation,
            boundary=boundary,
        )
    elif scope == "q_proj_output_head" and projection == "q_proj":
        targeted = True
        head_count = linear.out_features // head_dim
        boundary = "query_head_boundary"
        output_transform = GroupedOrthogonalTransform(
            transform_kind,
            head_dim,
            permutation=output_permutation,
            boundary=boundary,
        )
    elif scope == "q_proj_two_sided" and projection == "q_proj":
        targeted = True
        head_count = linear.out_features // head_dim
        if head_count != attention_heads:
            raise ValueError(
                f"{module_name} exposes {head_count} heads, "
                f"expected {attention_heads}"
            )
        boundary = "q_input_groups_128_and_query_head_boundary"
        input_transform = GroupedOrthogonalTransform(
            transform_kind,
            128,
            permutation=input_permutation,
            boundary="q_input_groups_128",
        )
        output_transform = GroupedOrthogonalTransform(
            transform_kind,
            head_dim,
            permutation=output_permutation,
            boundary="query_head_boundary",
        )
    elif scope == "attention_head" and projection in ATTENTION_PROJECTIONS:
        targeted = True
        if projection in {"q_proj", "k_proj", "v_proj"}:
            head_count = linear.out_features // head_dim
            expected_heads = attention_heads if projection == "q_proj" else kv_heads
            if head_count != expected_heads:
                raise ValueError(
                    f"{module_name} exposes {head_count} heads, expected {expected_heads}"
                )
            boundary = (
                "query_head_boundary"
                if projection == "q_proj"
                else "kv_head_boundary"
            )
            output_transform = GroupedOrthogonalTransform(
                transform_kind,
                head_dim,
                permutation=output_permutation,
                boundary=boundary,
            )
        else:
            head_count = linear.in_features // head_dim
            if head_count != attention_heads:
                raise ValueError(
                    f"{module_name} exposes {head_count} input heads, "
                    f"expected {attention_heads}"
                )
            boundary = "concatenated_query_head_boundary"
            input_transform = GroupedOrthogonalTransform(
                transform_kind,
                head_dim,
                permutation=input_permutation,
                boundary=boundary,
            )
    elif scope == "qk_rope_pair" and projection in {"q_proj", "k_proj"}:
        targeted = True
        head_count = linear.out_features // head_dim
        expected_heads = attention_heads if projection == "q_proj" else kv_heads
        if head_count != expected_heads:
            raise ValueError(
                f"{module_name} exposes {head_count} heads, expected {expected_heads}"
            )
        boundary = "llama_rope_split_half_pair"
        output_transform = GroupedOrthogonalTransform(
            transform_kind,
            2,
            layout="rope_pairs",
            head_dim=head_dim,
            boundary=boundary,
        )

    return RotationPlan(
        targeted=targeted,
        input_transform=input_transform,
        output_transform=output_transform,
        boundary=boundary,
        head_count=head_count,
        kv_group_size=q_per_kv,
    )


def replace_module(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    """Replace a named child module without mutating unrelated model state."""

    parent: nn.Module = root
    fields = qualified_name.split(".")
    for field in fields[:-1]:
        parent = getattr(parent, field)
    setattr(parent, fields[-1], replacement)


def model_topology_summary(config: Any) -> dict[str, int | bool]:
    """Return the attention topology fields relevant to legal rotations."""

    namespace = SimpleNamespace(
        hidden_size=int(config.hidden_size),
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(
            getattr(config, "num_key_value_heads", config.num_attention_heads)
        ),
        head_dim=_head_dim(config),
    )
    return {
        "hidden_size": namespace.hidden_size,
        "num_attention_heads": namespace.num_attention_heads,
        "num_key_value_heads": namespace.num_key_value_heads,
        "head_dim": namespace.head_dim,
        "queries_per_kv_head": (
            namespace.num_attention_heads // namespace.num_key_value_heads
        ),
        "uses_gqa": namespace.num_attention_heads != namespace.num_key_value_heads,
    }
