"""Compression and rate-estimation helpers."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from .metrics import relative_mse, squared_norm, symmetric_quantize
from .transforms import dct_matrix, hadamard_matrix, zigzag_indices


@dataclass(frozen=True)
class CompressionResult:
    reconstructed: np.ndarray
    bits_per_weight: float
    relative_mse: float
    details: dict[str, float | int | str]


def two_sided_transform(block: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left) @ np.asarray(block) @ np.asarray(right).T


def inverse_two_sided(coefficients: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left).T @ np.asarray(coefficients) @ np.asarray(right)


def dense_transform_quantize(
    block: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bits: int,
) -> CompressionResult:
    coefficients = two_sided_transform(block, left, right)
    quantized, _ = symmetric_quantize(coefficients, bits)
    reconstructed = inverse_two_sided(quantized, left, right)
    count = block.size
    bits_per_weight = bits + 32.0 / count
    return CompressionResult(
        reconstructed=reconstructed,
        bits_per_weight=bits_per_weight,
        relative_mse=relative_mse(block, reconstructed),
        details={"value_bits": bits, "scale_bits": 32, "payload": "dense"},
    )


def _masked_quantize(coefficients: np.ndarray, mask: np.ndarray, bits: int) -> np.ndarray:
    output = np.zeros_like(coefficients)
    selected = np.asarray(coefficients)[mask]
    quantized, _ = symmetric_quantize(selected, bits)
    output[mask] = quantized
    return output


def _topk_mask(values: np.ndarray, keep: int) -> np.ndarray:
    flat = np.abs(np.asarray(values).reshape(-1))
    keep = max(1, min(flat.size, keep))
    mask = np.zeros(flat.size, dtype=bool)
    if keep == flat.size:
        mask[:] = True
    else:
        chosen = np.argpartition(flat, flat.size - keep)[-keep:]
        mask[chosen] = True
    return mask.reshape(np.asarray(values).shape)


def _structured_dct_mask(n: int, keep: int) -> np.ndarray:
    mask = np.zeros(n * n, dtype=bool)
    mask[zigzag_indices(n)[:keep]] = True
    return mask.reshape(n, n)


def dct_hadamard_hybrid(
    block: np.ndarray,
    density: float = 0.125,
    base_bits: int = 8,
    residual_bits: int = 2,
    *,
    topk_base: bool = False,
) -> CompressionResult:
    """DCT sparse base plus dense Hadamard-quantized residual."""

    matrix = np.asarray(block, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hybrid currently expects square blocks")
    n = matrix.shape[0]
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")

    dct = dct_matrix(n)
    coefficients = two_sided_transform(matrix, dct, dct)
    keep = max(1, min(matrix.size, int(round(density * matrix.size))))
    mask = _topk_mask(coefficients, keep) if topk_base else _structured_dct_mask(n, keep)
    quantized_base = _masked_quantize(coefficients, mask, base_bits)
    base = inverse_two_sided(quantized_base, dct, dct)

    residual = matrix - base
    hadamard = hadamard_matrix(n)
    residual_coefficients = two_sided_transform(residual, hadamard, hadamard)
    quantized_residual, _ = symmetric_quantize(residual_coefficients, residual_bits)
    reconstructed = base + inverse_two_sided(quantized_residual, hadamard, hadamard)

    scale_bits = 64
    if topk_base:
        index_bits = keep * math.ceil(math.log2(matrix.size))
        metadata = "topk_indices"
    else:
        index_bits = 16
        metadata = "zigzag_keep_count"
    payload_bits = keep * base_bits + matrix.size * residual_bits + scale_bits + index_bits
    return CompressionResult(
        reconstructed=reconstructed,
        bits_per_weight=payload_bits / matrix.size,
        relative_mse=relative_mse(matrix, reconstructed),
        details={
            "base_bits": base_bits,
            "residual_bits": residual_bits,
            "base_density": density,
            "base_keep": keep,
            "index_bits": index_bits,
            "scale_bits": scale_bits,
            "payload": metadata,
        },
    )


@functools.lru_cache(maxsize=None)
def fft_hermitian_groups(n: int) -> tuple[tuple[int, int, int, int, bool], ...]:
    """Unique conjugate groups for a real ``n x n`` FFT."""

    groups: list[tuple[int, int, int, int, bool]] = []
    visited: set[tuple[int, int]] = set()
    for row in range(n):
        for col in range(n):
            index = (row, col)
            if index in visited:
                continue
            partner = ((-row) % n, (-col) % n)
            visited.add(index)
            visited.add(partner)
            groups.append((row, col, partner[0], partner[1], index == partner))
    return tuple(groups)


def fft_topk_energy(block: np.ndarray, density: float) -> float:
    """Energy retained by a conjugate-symmetric FFT payload at scalar budget."""

    matrix = np.asarray(block, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("FFT helper expects square blocks")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    n = matrix.shape[0]
    coefficients = np.fft.fft2(matrix, norm="ortho")
    candidates: list[tuple[float, float, int]] = []
    for group_index, (row, col, _, _, self_conjugate) in enumerate(fft_hermitian_groups(n)):
        cost = 1 if self_conjugate else 2
        energy = float(abs(coefficients[row, col]) ** 2) * cost
        candidates.append((energy / cost, energy, group_index))
    candidates.sort(reverse=True)

    budget = max(1, int(round(density * matrix.size)))
    used = 0
    retained = 0.0
    for _, energy, group_index in candidates:
        self_conjugate = fft_hermitian_groups(n)[group_index][4]
        cost = 1 if self_conjugate else 2
        if used + cost > budget:
            continue
        used += cost
        retained += energy
        if used >= budget:
            break
    total = squared_norm(coefficients)
    return 1.0 if total == 0.0 else retained / total


def fft_dense_quantize(block: np.ndarray, bits: int) -> CompressionResult:
    """Quantize the independent real degrees of a Hermitian 2-D FFT."""

    matrix = np.asarray(block, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("FFT helper expects square blocks")
    n = matrix.shape[0]
    coefficients = np.fft.fft2(matrix, norm="ortho")
    scalars: list[float] = []
    for row, col, _, _, self_conjugate in fft_hermitian_groups(n):
        value = coefficients[row, col]
        scalars.append(float(value.real))
        if not self_conjugate:
            scalars.append(float(value.imag))
    independent = np.asarray(scalars, dtype=np.float64)
    if independent.size != matrix.size:
        raise AssertionError("Hermitian representation must contain n^2 real scalars")
    quantized, _ = symmetric_quantize(independent, bits)

    rebuilt = np.zeros_like(coefficients)
    cursor = 0
    for row, col, partner_row, partner_col, self_conjugate in fft_hermitian_groups(n):
        real = quantized[cursor]
        cursor += 1
        imag = 0.0
        if not self_conjugate:
            imag = quantized[cursor]
            cursor += 1
        value = real + 1j * imag
        rebuilt[row, col] = value
        rebuilt[partner_row, partner_col] = np.conjugate(value)
    reconstructed = np.fft.ifft2(rebuilt, norm="ortho").real
    return CompressionResult(
        reconstructed=reconstructed,
        bits_per_weight=bits + 32.0 / matrix.size,
        relative_mse=relative_mse(matrix, reconstructed),
        details={
            "value_bits": bits,
            "scale_bits": 32,
            "payload": "Hermitian independent real scalars",
        },
    )
