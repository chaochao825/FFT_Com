"""Reproducible probes for the historical FreqKV synthetic experiment.

The old prototype mixed three separate questions:

* whether its synthetic K/V tensors have low-frequency structure;
* whether low-frequency truncation reconstructs those tensors;
* whether an rFFT fallback that drops the imaginary part is numerically valid.

This module keeps those questions separate and exposes the measurements used by
``scripts/run_freqkv_reassessment.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .metrics import relative_mse, squared_norm
from .transforms import dct_matrix


@dataclass(frozen=True)
class RetentionMeasurement:
    requested_retention_ratio: float
    retained_components: int
    total_components: int
    actual_component_fraction: float
    selected_frequency_energy_retention: float
    reconstruction_energy_retention: float
    reported_energy_retention: float
    reconstruction_relative_mse: float
    reconstruction_mse: float
    metric_definition: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def apply_rope_like_rotation(values: np.ndarray) -> np.ndarray:
    """Apply the position-wise orthogonal rotation used by the old generator.

    ``values`` must have shape ``[heads, sequence, head_dim]``. The operation is
    orthogonal within every adjacent feature pair, so it preserves the energy
    of an independent Gaussian sample and does not create sequence smoothness.
    """

    output = np.asarray(values, dtype=np.float64).copy()
    if output.ndim != 3:
        raise ValueError("values must have shape [heads, sequence, head_dim]")
    _, sequence_length, head_dim = output.shape
    if head_dim % 2:
        raise ValueError("head_dim must be even")

    positions = np.arange(sequence_length, dtype=np.float64)
    for pair in range(head_dim // 2):
        theta = 10000.0 ** (-2.0 * pair / head_dim)
        angles = positions * theta
        cosine = np.cos(angles)[None, :]
        sine = np.sin(angles)[None, :]
        first = output[:, :, 2 * pair].copy()
        second = output[:, :, 2 * pair + 1].copy()
        output[:, :, 2 * pair] = first * cosine - second * sine
        output[:, :, 2 * pair + 1] = first * sine + second * cosine
    return output


def make_historical_synthetic_kv(
    sequence_length: int,
    *,
    head_dim: int = 64,
    heads: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Recreate the old independent-Gaussian K/V data model."""

    if sequence_length <= 0 or head_dim <= 0 or heads <= 0:
        raise ValueError("sequence_length, head_dim, and heads must be positive")
    rng = np.random.default_rng(seed)
    key = rng.standard_normal((heads, sequence_length, head_dim))
    value = rng.standard_normal((heads, sequence_length, head_dim))
    return {
        "K": apply_rope_like_rotation(key),
        "V": value,
    }


def make_smooth_positive_control_kv(
    sequence_length: int,
    *,
    head_dim: int = 64,
    heads: int = 4,
    seed: int = 0,
    decay: float = 0.20,
) -> dict[str, np.ndarray]:
    """Create a declared low-frequency positive control in an orthonormal DCT basis."""

    if decay <= 0.0:
        raise ValueError("decay must be positive")
    rng = np.random.default_rng(seed)
    dct = dct_matrix(sequence_length)
    envelope = np.exp(-decay * np.arange(sequence_length, dtype=np.float64))

    def draw() -> np.ndarray:
        coefficients = rng.standard_normal((heads, sequence_length, head_dim))
        coefficients *= envelope[None, :, None]
        return np.einsum("sf,hfd->hsd", dct.T, coefficients, optimize=True)

    return {"K": draw(), "V": draw()}


def _retained_count(ratio: float, total: int) -> int:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("retention ratio must be in (0, 1]")
    return max(1, min(total, int(ratio * total)))


def _rfft_parseval_weights(sequence_length: int) -> np.ndarray:
    frequency_count = sequence_length // 2 + 1
    weights = np.ones(frequency_count, dtype=np.float64)
    if sequence_length % 2 == 0:
        if frequency_count > 2:
            weights[1:-1] = 2.0
    elif frequency_count > 1:
        weights[1:] = 2.0
    return weights


def _weighted_rfft_energy(
    coefficients: np.ndarray,
    sequence_length: int,
) -> float:
    weights = _rfft_parseval_weights(sequence_length)
    return float(
        np.sum(
            np.abs(coefficients) ** 2 * weights[None, :, None],
            dtype=np.float64,
        )
    )


def measure_low_frequency_retention(
    values: np.ndarray,
    ratios: Iterable[float],
    *,
    transform_path: str,
) -> list[RetentionMeasurement]:
    """Measure low-frequency truncation under one explicit transform path."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("values must have shape [heads, sequence, head_dim]")
    sequence_length = data.shape[1]
    reference_energy = squared_norm(data)
    if reference_energy == 0.0:
        raise ValueError("zero-energy input is not supported")

    results: list[RetentionMeasurement] = []
    if transform_path == "dct_ii_ortho":
        dct = dct_matrix(sequence_length)
        coefficients = np.einsum("fs,hsd->hfd", dct, data, optimize=True)
        total_components = sequence_length
        total_frequency_energy = squared_norm(coefficients)
        for ratio in ratios:
            retained = _retained_count(float(ratio), total_components)
            masked = np.zeros_like(coefficients)
            masked[:, :retained, :] = coefficients[:, :retained, :]
            reconstructed = np.einsum(
                "sf,hfd->hsd", dct.T, masked, optimize=True
            )
            selected = squared_norm(masked) / total_frequency_energy
            reconstruction_energy = squared_norm(reconstructed) / reference_energy
            results.append(
                RetentionMeasurement(
                    requested_retention_ratio=float(ratio),
                    retained_components=retained,
                    total_components=total_components,
                    actual_component_fraction=retained / total_components,
                    selected_frequency_energy_retention=selected,
                    reconstruction_energy_retention=reconstruction_energy,
                    reported_energy_retention=reconstruction_energy,
                    reconstruction_relative_mse=relative_mse(data, reconstructed),
                    reconstruction_mse=float(np.mean((data - reconstructed) ** 2)),
                    metric_definition=(
                        "orthonormal DCT-II; time-domain reconstruction energy"
                    ),
                )
            )
        return results

    if transform_path not in {
        "rfft_parseval",
        "rfft_drop_imag_historical",
    }:
        raise ValueError(f"unknown transform path: {transform_path}")

    coefficients = np.fft.rfft(data, axis=1)
    total_components = coefficients.shape[1]
    total_parseval_energy = _weighted_rfft_energy(coefficients, sequence_length)
    total_unweighted_energy = squared_norm(coefficients)
    for ratio in ratios:
        retained = _retained_count(float(ratio), total_components)
        masked = np.zeros_like(coefficients)
        masked[:, :retained, :] = coefficients[:, :retained, :]
        selected = (
            _weighted_rfft_energy(masked, sequence_length) / total_parseval_energy
        )

        if transform_path == "rfft_parseval":
            reconstructed = np.fft.irfft(masked, n=sequence_length, axis=1)
            metric_definition = (
                "rFFT with Parseval endpoint weights; complex coefficients preserved"
            )
        else:
            # This mirrors ``tensor_freq.float()`` before ``torch.fft.irfft``:
            # the real part is preserved and the imaginary part is discarded.
            reconstructed = np.fft.irfft(
                masked.real.astype(np.complex128),
                n=sequence_length,
                axis=1,
            )
            metric_definition = (
                "historical rFFT fallback; imaginary part discarded before irFFT"
            )

        reconstruction_energy = squared_norm(reconstructed) / reference_energy
        roundtrip_frequency = np.fft.rfft(reconstructed, axis=1)
        if transform_path == "rfft_drop_imag_historical":
            reported_energy = (
                squared_norm(roundtrip_frequency) / total_unweighted_energy
            )
        else:
            reported_energy = reconstruction_energy
        results.append(
            RetentionMeasurement(
                requested_retention_ratio=float(ratio),
                retained_components=retained,
                total_components=total_components,
                actual_component_fraction=retained / total_components,
                selected_frequency_energy_retention=selected,
                reconstruction_energy_retention=reconstruction_energy,
                reported_energy_retention=reported_energy,
                reconstruction_relative_mse=relative_mse(data, reconstructed),
                reconstruction_mse=float(np.mean((data - reconstructed) ** 2)),
                metric_definition=metric_definition,
            )
        )
    return results


def frequency_threshold_rows(
    values: np.ndarray,
    thresholds: Iterable[float],
    *,
    energy_definition: str,
) -> list[dict[str, float | int | str]]:
    """Return required low-frequency counts and the old zero-based plot label.

    The historical plot used ``idx / component_count`` where ``idx`` was the
    zero-based result of ``searchsorted``. The true required count is
    ``idx + 1``. Both values are returned so the artifact can be interpreted
    without silently rewriting it.
    """

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("values must have shape [heads, sequence, head_dim]")
    sequence_length = data.shape[1]

    if energy_definition == "dct_ii_ortho":
        coefficients = np.einsum(
            "fs,hsd->hfd",
            dct_matrix(sequence_length),
            data,
            optimize=True,
        )
        component_energy = np.sum(coefficients**2, axis=(0, 2), dtype=np.float64)
    elif energy_definition in {
        "rfft_parseval",
        "rfft_historical_unweighted",
    }:
        coefficients = np.fft.rfft(data, axis=1)
        component_energy = np.sum(
            np.abs(coefficients) ** 2,
            axis=(0, 2),
            dtype=np.float64,
        )
        if energy_definition == "rfft_parseval":
            component_energy *= _rfft_parseval_weights(sequence_length)
    else:
        raise ValueError(f"unknown energy definition: {energy_definition}")

    total = float(np.sum(component_energy, dtype=np.float64))
    if total == 0.0:
        raise ValueError("zero-energy input is not supported")
    cumulative = np.cumsum(component_energy, dtype=np.float64) / total
    component_count = component_energy.size
    rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        value = float(threshold)
        if not 0.0 < value <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        index = min(int(np.searchsorted(cumulative, value)), component_count - 1)
        required = index + 1
        rows.append(
            {
                "energy_definition": energy_definition,
                "energy_threshold": value,
                "zero_based_index": index,
                "required_components": required,
                "total_components": component_count,
                "true_required_component_fraction": required / component_count,
                "historical_plot_label_fraction": index / component_count,
                "historical_label_understatement": 1.0 / component_count,
            }
        )
    return rows
