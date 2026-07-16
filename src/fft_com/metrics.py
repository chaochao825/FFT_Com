"""Metrics and scalar quantizers for transform-compression experiments."""

from __future__ import annotations

import math

import numpy as np


def squared_norm(values: np.ndarray) -> float:
    array = np.asarray(values)
    return float(np.sum(np.abs(array) ** 2, dtype=np.float64))


def relative_mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return squared reconstruction error divided by reference energy."""

    denominator = squared_norm(reference)
    if denominator == 0.0:
        return 0.0 if squared_norm(estimate) == 0.0 else math.inf
    return squared_norm(np.asarray(reference) - np.asarray(estimate)) / denominator


def topk_energy(values: np.ndarray, density: float) -> float:
    """Fraction of energy retained by the largest-magnitude coefficients."""

    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    energy = np.abs(np.asarray(values).reshape(-1)) ** 2
    total = float(np.sum(energy, dtype=np.float64))
    if total == 0.0:
        return 1.0
    keep = max(1, min(energy.size, int(round(density * energy.size))))
    if keep == energy.size:
        return 1.0
    selected = np.partition(energy, energy.size - keep)[-keep:]
    return float(np.sum(selected, dtype=np.float64) / total)


def structured_energy(values: np.ndarray, flat_indices: np.ndarray, density: float) -> float:
    """Energy fraction in the first entries of a fixed flattened ordering."""

    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    flat = np.asarray(values).reshape(-1)
    keep = max(1, min(flat.size, int(round(density * flat.size))))
    denominator = squared_norm(flat)
    if denominator == 0.0:
        return 1.0
    return squared_norm(flat[np.asarray(flat_indices)[:keep]]) / denominator


def papr(values: np.ndarray) -> float:
    """Peak-to-average power ratio."""

    power = np.abs(np.asarray(values).reshape(-1)) ** 2
    average = float(np.mean(power, dtype=np.float64))
    if average == 0.0:
        return 0.0
    return float(np.max(power) / average)


def excess_kurtosis(values: np.ndarray) -> float:
    """Population excess kurtosis of real scalar components."""

    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.concatenate((array.real.reshape(-1), array.imag.reshape(-1)))
    else:
        array = array.reshape(-1)
    centered = array.astype(np.float64) - float(np.mean(array, dtype=np.float64))
    second = float(np.mean(centered**2, dtype=np.float64))
    if second == 0.0:
        return -3.0
    fourth = float(np.mean(centered**4, dtype=np.float64))
    return fourth / (second * second) - 3.0


def symmetric_quantize(values: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Quantize real or complex values with one symmetric abs-max scale."""

    if bits < 2:
        raise ValueError("bits must be at least 2 for signed symmetric quantization")
    array = np.asarray(values)
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(
        max(
            np.max(np.abs(array.real), initial=0.0),
            np.max(np.abs(array.imag), initial=0.0) if np.iscomplexobj(array) else 0.0,
        )
    )
    if max_abs == 0.0:
        return np.zeros_like(array), 0.0
    scale = max_abs / qmax

    def _quantize_real(component: np.ndarray) -> np.ndarray:
        codes = np.clip(np.rint(component / scale), -qmax, qmax)
        return codes * scale

    if np.iscomplexobj(array):
        quantized = _quantize_real(array.real) + 1j * _quantize_real(array.imag)
    else:
        quantized = _quantize_real(array)
    return quantized.astype(array.dtype, copy=False), scale


def batch_symmetric_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    """Per-matrix symmetric quantization for a ``[batch, rows, cols]`` array."""

    if bits < 2:
        raise ValueError("bits must be at least 2")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("values must have shape [batch, rows, cols]")
    qmax = (1 << (bits - 1)) - 1
    maxima = np.max(np.abs(array), axis=(1, 2), keepdims=True)
    scales = np.where(maxima > 0.0, maxima / qmax, 1.0)
    codes = np.clip(np.rint(array / scales), -qmax, qmax)
    return codes * scales


def batch_quantization_relative_mse(values: np.ndarray, bits: int) -> float:
    """Mean per-block relative quantization error."""

    array = np.asarray(values, dtype=np.float64)
    quantized = batch_symmetric_quantize(array, bits)
    numerator = np.sum((array - quantized) ** 2, axis=(1, 2), dtype=np.float64)
    denominator = np.sum(array**2, axis=(1, 2), dtype=np.float64)
    ratios = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return float(np.mean(ratios, dtype=np.float64))


def output_relative_mse(
    weights: np.ndarray,
    reconstructed: np.ndarray,
    inputs: np.ndarray,
) -> float:
    """Relative linear-output error for row-major samples in ``inputs``."""

    reference = np.asarray(inputs, dtype=np.float64) @ np.asarray(weights, dtype=np.float64).T
    estimate = np.asarray(inputs, dtype=np.float64) @ np.asarray(reconstructed, dtype=np.float64).T
    return relative_mse(reference, estimate)
