"""PyTorch orthogonal transforms for legal online model rotations.

The row-vector convention used throughout this module is

``forward(x) = x @ R.T`` and ``inverse(z) = z @ R``.

This makes an input-side linear rotation exact with ``C = W @ R.T`` and an
output-side rotation exact with ``C = R @ W`` followed by ``inverse``.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn


SUPPORTED_TRANSFORMS: Final[tuple[str, ...]] = (
    "identity",
    "dct",
    "hadamard",
    "rdft",
)

_DCT_PHASE_CACHE: dict[
    tuple[int, str, int | None, torch.dtype, bool],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def _require_positive_size(n: int) -> None:
    if n <= 0:
        raise ValueError(f"transform size must be positive, got {n}")


def _require_power_of_two(n: int) -> None:
    _require_positive_size(n)
    if n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")


def _fft_work_dtype(dtype: torch.dtype) -> torch.dtype:
    # CPU FFT does not support fp16/bf16, and explicit fp32 also avoids relying
    # on experimental complex-half behavior on CUDA.
    return torch.float64 if dtype == torch.float64 else torch.float32


def _dct_phase(
    n: int,
    reference: torch.Tensor,
    *,
    inverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    work_dtype = _fft_work_dtype(reference.dtype)
    device = reference.device
    key = (n, device.type, device.index, work_dtype, inverse)
    cached = _DCT_PHASE_CACHE.get(key)
    if cached is not None:
        return cached
    sign = 1.0 if inverse else -1.0
    angle = (
        torch.arange(n, device=device, dtype=work_dtype)
        * (sign * math.pi / (2.0 * n))
    )
    result = (torch.cos(angle), torch.sin(angle))
    _DCT_PHASE_CACHE[key] = result
    return result


def dct_ortho(values: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal DCT-II along the final dimension using one FFT."""

    n = values.shape[-1]
    _require_positive_size(n)
    original_dtype = values.dtype
    work = values.to(_fft_work_dtype(original_dtype))
    reordered = torch.cat(
        (work[..., ::2], work[..., 1::2].flip(dims=(-1,))),
        dim=-1,
    )
    spectrum = torch.fft.fft(reordered, dim=-1)
    cosine, sine = _dct_phase(n, work, inverse=False)
    coefficients = spectrum.real * cosine - spectrum.imag * sine
    coefficients = coefficients.clone()
    coefficients[..., 0] /= math.sqrt(n)
    if n > 1:
        coefficients[..., 1:] /= math.sqrt(n / 2.0)
    return coefficients.to(original_dtype)


def idct_ortho(values: torch.Tensor) -> torch.Tensor:
    """Invert :func:`dct_ortho` (orthonormal DCT-III synthesis)."""

    n = values.shape[-1]
    _require_positive_size(n)
    original_dtype = values.dtype
    work = values.to(_fft_work_dtype(original_dtype)).clone()
    work[..., 0] *= math.sqrt(n)
    if n > 1:
        work[..., 1:] *= math.sqrt(n / 2.0)

    cosine, sine = _dct_phase(n, work, inverse=True)
    imaginary_template = torch.cat(
        (
            torch.zeros_like(work[..., :1]),
            -work.flip(dims=(-1,))[..., :-1],
        ),
        dim=-1,
    )
    spectrum_real = work * cosine - imaginary_template * sine
    spectrum_imag = work * sine + imaginary_template * cosine
    reordered = torch.fft.ifft(
        torch.complex(spectrum_real, spectrum_imag),
        dim=-1,
    ).real

    output = torch.empty_like(reordered)
    even_count = n - n // 2
    output[..., ::2] = reordered[..., :even_count]
    output[..., 1::2] = reordered.flip(dims=(-1,))[..., : n // 2]
    return output.to(original_dtype)


def hadamard_ortho(values: torch.Tensor) -> torch.Tensor:
    """Apply a normalized fast Walsh-Hadamard transform on the last axis."""

    n = values.shape[-1]
    _require_power_of_two(n)
    output = values
    span = 1
    while span < n:
        grouped = output.reshape(*output.shape[:-1], -1, 2, span)
        first = grouped[..., 0, :]
        second = grouped[..., 1, :]
        output = torch.cat((first + second, first - second), dim=-1).reshape(
            *output.shape[:-1], n
        )
        span *= 2
    return output / math.sqrt(n)


def rdft_ortho(values: torch.Tensor) -> torch.Tensor:
    """Pack a real orthonormal FFT into ``n`` real orthogonal coordinates."""

    n = values.shape[-1]
    _require_positive_size(n)
    if n % 2:
        raise ValueError(f"RDFT currently requires an even size, got {n}")
    original_dtype = values.dtype
    work = values.to(_fft_work_dtype(original_dtype))
    spectrum = torch.fft.rfft(work, dim=-1, norm="ortho")
    output = torch.empty_like(work)
    output[..., 0] = spectrum[..., 0].real
    output[..., 1] = spectrum[..., -1].real
    if n > 2:
        scale = math.sqrt(2.0)
        output[..., 2::2] = scale * spectrum[..., 1:-1].real
        output[..., 3::2] = scale * spectrum[..., 1:-1].imag
    return output.to(original_dtype)


def irdft_ortho(values: torch.Tensor) -> torch.Tensor:
    """Invert :func:`rdft_ortho`."""

    n = values.shape[-1]
    _require_positive_size(n)
    if n % 2:
        raise ValueError(f"RDFT currently requires an even size, got {n}")
    original_dtype = values.dtype
    work = values.to(_fft_work_dtype(original_dtype))
    spectrum = torch.zeros(
        *work.shape[:-1],
        n // 2 + 1,
        dtype=torch.complex128 if work.dtype == torch.float64 else torch.complex64,
        device=work.device,
    )
    spectrum[..., 0] = torch.complex(work[..., 0], torch.zeros_like(work[..., 0]))
    spectrum[..., -1] = torch.complex(
        work[..., 1],
        torch.zeros_like(work[..., 1]),
    )
    if n > 2:
        scale = math.sqrt(2.0)
        spectrum[..., 1:-1] = torch.complex(
            work[..., 2::2] / scale,
            work[..., 3::2] / scale,
        )
    return torch.fft.irfft(spectrum, n=n, dim=-1, norm="ortho").to(original_dtype)


def apply_orthogonal_transform(
    values: torch.Tensor,
    kind: str,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply one supported real orthogonal transform on the final dimension."""

    normalized = kind.lower()
    if normalized not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"unsupported transform {kind!r}; expected one of {SUPPORTED_TRANSFORMS}"
        )
    if normalized == "identity":
        return values
    if normalized == "dct":
        return idct_ortho(values) if inverse else dct_ortho(values)
    if normalized == "hadamard":
        # The normalized Sylvester matrix is symmetric and self-inverse.
        return hadamard_ortho(values)
    return irdft_ortho(values) if inverse else rdft_ortho(values)


def _validate_permutation(permutation: torch.Tensor, group_size: int) -> None:
    if permutation.ndim not in (1, 2):
        raise ValueError("permutation must have shape [group_size] or [groups, group_size]")
    if permutation.shape[-1] != group_size:
        raise ValueError(
            f"permutation width {permutation.shape[-1]} != group size {group_size}"
        )
    expected = torch.arange(group_size, device=permutation.device)
    rows = permutation.reshape(-1, group_size)
    if not torch.equal(torch.sort(rows, dim=-1).values, expected.expand_as(rows)):
        raise ValueError("each permutation row must contain every group index once")


def _gather_grouped(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1:
        return values.index_select(-1, indices)
    if values.shape[-2] != indices.shape[0]:
        raise ValueError(
            f"permutation has {indices.shape[0]} groups, values have {values.shape[-2]}"
        )
    flat = values.reshape(-1, values.shape[-2], values.shape[-1])
    expanded = indices.unsqueeze(0).expand(flat.shape[0], -1, -1)
    return flat.gather(-1, expanded).reshape_as(values)


class GroupedOrthogonalTransform(nn.Module):
    """Grouped transform with optional fixed permutation or RoPE-pair layout."""

    def __init__(
        self,
        kind: str,
        group_size: int,
        *,
        permutation: torch.Tensor | None = None,
        layout: str = "contiguous",
        head_dim: int | None = None,
        boundary: str = "contiguous",
    ) -> None:
        super().__init__()
        normalized = kind.lower()
        if normalized not in SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"unsupported transform {kind!r}; expected one of {SUPPORTED_TRANSFORMS}"
            )
        _require_positive_size(group_size)
        if layout not in ("contiguous", "rope_pairs"):
            raise ValueError("layout must be 'contiguous' or 'rope_pairs'")
        if layout == "rope_pairs":
            if head_dim is None or head_dim <= 0 or head_dim % 2:
                raise ValueError("rope_pairs requires a positive even head_dim")
            if group_size != 2:
                raise ValueError("rope_pairs uses transform groups of size 2")
            if permutation is not None:
                raise ValueError("rope_pairs has a fixed algorithmic layout")

        self.kind = normalized
        self.group_size = int(group_size)
        self.layout = layout
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.boundary = boundary

        if permutation is None:
            self.register_buffer("permutation", None)
            self.register_buffer("inverse_permutation", None)
        else:
            values = torch.as_tensor(permutation, dtype=torch.long)
            _validate_permutation(values, self.group_size)
            self.register_buffer("permutation", values)
            self.register_buffer(
                "inverse_permutation",
                torch.argsort(values, dim=-1),
            )

    def extra_repr(self) -> str:
        return (
            f"kind={self.kind!r}, group_size={self.group_size}, "
            f"layout={self.layout!r}, boundary={self.boundary!r}"
        )

    def _contiguous(self, values: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        feature_dim = values.shape[-1]
        if feature_dim % self.group_size:
            raise ValueError(
                f"feature dimension {feature_dim} is not divisible by "
                f"group size {self.group_size}"
            )
        if self.kind == "identity" and self.permutation is None:
            return values
        grouped = values.reshape(
            *values.shape[:-1],
            feature_dim // self.group_size,
            self.group_size,
        )
        if inverse:
            grouped = apply_orthogonal_transform(grouped, self.kind, inverse=True)
            if self.inverse_permutation is not None:
                grouped = _gather_grouped(grouped, self.inverse_permutation)
        else:
            if self.permutation is not None:
                grouped = _gather_grouped(grouped, self.permutation)
            grouped = apply_orthogonal_transform(grouped, self.kind)
        return grouped.reshape_as(values)

    def _rope_pairs(self, values: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        assert self.head_dim is not None
        feature_dim = values.shape[-1]
        if feature_dim % self.head_dim:
            raise ValueError(
                f"feature dimension {feature_dim} is not divisible by "
                f"head_dim {self.head_dim}"
            )
        head_count = feature_dim // self.head_dim
        half = self.head_dim // 2
        if inverse:
            pairs = values.reshape(*values.shape[:-1], head_count, half, 2)
            restored = apply_orthogonal_transform(pairs, self.kind, inverse=True)
            heads = torch.cat((restored[..., 0], restored[..., 1]), dim=-1)
            return heads.reshape_as(values)
        heads = values.reshape(*values.shape[:-1], head_count, self.head_dim)
        pairs = torch.stack((heads[..., :half], heads[..., half:]), dim=-1)
        transformed = apply_orthogonal_transform(pairs, self.kind)
        return transformed.reshape_as(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.layout == "rope_pairs":
            return self._rope_pairs(values, inverse=False)
        return self._contiguous(values, inverse=False)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        if self.layout == "rope_pairs":
            return self._rope_pairs(values, inverse=True)
        return self._contiguous(values, inverse=True)

    def metadata_bits(self, feature_dim: int) -> float:
        """Permutation payload, excluding algorithmic head/RoPE layout metadata."""

        if self.permutation is None:
            return 0.0
        group_count = feature_dim // self.group_size
        encoded_groups = group_count if self.permutation.ndim == 2 else 1
        return encoded_groups * math.lgamma(self.group_size + 1) / math.log(2.0)


def transform_linear_weight(
    weight: torch.Tensor,
    *,
    input_transform: GroupedOrthogonalTransform | None = None,
    output_transform: GroupedOrthogonalTransform | None = None,
) -> torch.Tensor:
    """Return ``R_out @ W @ R_in.T`` under the row-vector convention."""

    transformed = weight
    if input_transform is not None:
        transformed = input_transform(transformed)
    if output_transform is not None:
        transformed = output_transform(transformed.transpose(0, 1)).transpose(0, 1)
    return transformed


def inverse_transform_linear_weight(
    weight: torch.Tensor,
    *,
    input_transform: GroupedOrthogonalTransform | None = None,
    output_transform: GroupedOrthogonalTransform | None = None,
) -> torch.Tensor:
    """Undo :func:`transform_linear_weight`."""

    restored = weight
    if output_transform is not None:
        restored = output_transform.inverse(restored.transpose(0, 1)).transpose(0, 1)
    if input_transform is not None:
        restored = input_transform.inverse(restored)
    return restored

