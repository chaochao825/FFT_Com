"""Kuramoto sanity probes and circular phase-codebook helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compression import dense_transform_quantize
from .metrics import relative_mse
from .transforms import paired_rotation_matrix


def wrap_phase(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def order_parameter(phases: np.ndarray) -> float:
    return float(abs(np.mean(np.exp(1j * np.asarray(phases)))))


def correlation_adjacency(vectors: np.ndarray) -> np.ndarray:
    """Absolute cosine-similarity graph with zero diagonal."""

    array = np.asarray(vectors, dtype=np.float64)
    centered = array - np.mean(array, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(
        centered,
        norms,
        out=np.zeros_like(centered),
        where=norms > 0.0,
    )
    adjacency = np.abs(normalized @ normalized.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def attractive_kuramoto(
    adjacency: np.ndarray,
    initial_phases: np.ndarray,
    *,
    coupling: float = 4.0,
    step_size: float = 0.05,
    steps: int = 400,
) -> np.ndarray:
    """Run an all-attractive, zero-natural-frequency Kuramoto system."""

    graph = np.asarray(adjacency, dtype=np.float64)
    phases = np.asarray(initial_phases, dtype=np.float64).copy()
    if graph.shape != (phases.size, phases.size):
        raise ValueError("adjacency and phase dimensions do not match")
    normalizer = np.sum(graph, axis=1)
    normalizer = np.where(normalizer > 0.0, normalizer, 1.0)
    for _ in range(steps):
        phase_difference = phases[None, :] - phases[:, None]
        velocity = coupling * np.sum(graph * np.sin(phase_difference), axis=1) / normalizer
        phases = wrap_phase(phases + step_size * velocity)
    return phases


@dataclass(frozen=True)
class KuramotoRotationProbe:
    initial_order_parameter: float
    final_order_parameter: float
    paired_angle_rms: float
    identity_relative_mse: float
    kuramoto_relative_mse: float
    error_ratio: float


def kuramoto_rotation_probe(
    block: np.ndarray,
    *,
    bits: int = 3,
    seed: int = 0,
) -> KuramotoRotationProbe:
    """Test a gauge-invariant paired rotation derived from synchronized phases.

    The construction is intentionally a sanity check. Purely attractive
    synchronization tends to equal phases; pairwise phase differences then
    tend to zero and the resulting real rotation tends back toward identity.
    """

    matrix = np.asarray(block, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("block must be square")
    if matrix.shape[0] % 2:
        raise ValueError("block dimension must be even")

    rng = np.random.default_rng(seed)
    row_initial = rng.uniform(-np.pi, np.pi, size=matrix.shape[0])
    col_initial = rng.uniform(-np.pi, np.pi, size=matrix.shape[1])
    row_final = attractive_kuramoto(correlation_adjacency(matrix), row_initial)
    col_final = attractive_kuramoto(correlation_adjacency(matrix.T), col_initial)

    row_angles = 0.5 * wrap_phase(row_final[0::2] - row_final[1::2])
    col_angles = 0.5 * wrap_phase(col_final[0::2] - col_final[1::2])
    left = paired_rotation_matrix(row_angles)
    right = paired_rotation_matrix(col_angles)

    identity = np.eye(matrix.shape[0], dtype=np.float64)
    identity_result = dense_transform_quantize(matrix, identity, identity, bits)
    kuramoto_result = dense_transform_quantize(matrix, left, right, bits)
    initial_order = 0.5 * (order_parameter(row_initial) + order_parameter(col_initial))
    final_order = 0.5 * (order_parameter(row_final) + order_parameter(col_final))
    angle_rms = float(
        np.sqrt(np.mean(np.concatenate((row_angles, col_angles)) ** 2, dtype=np.float64))
    )
    ratio = (
        kuramoto_result.relative_mse / identity_result.relative_mse
        if identity_result.relative_mse > 0.0
        else 1.0
    )
    return KuramotoRotationProbe(
        initial_order_parameter=initial_order,
        final_order_parameter=final_order,
        paired_angle_rms=angle_rms,
        identity_relative_mse=identity_result.relative_mse,
        kuramoto_relative_mse=kuramoto_result.relative_mse,
        error_ratio=ratio,
    )


def weighted_phase_template(
    phases: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted circular mean and concentration along the first axis."""

    phase_values = np.asarray(phases, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    if phase_values.shape != weight_values.shape:
        raise ValueError("phases and weights must have identical shapes")
    resultant = np.sum(weight_values * np.exp(1j * phase_values), axis=0)
    total_weight = np.sum(weight_values, axis=0)
    template = np.angle(resultant)
    concentration = np.divide(
        np.abs(resultant),
        total_weight,
        out=np.zeros_like(total_weight),
        where=total_weight > 0.0,
    )
    return template, concentration


def quantize_phase_uniform(
    phases: np.ndarray,
    bits: int,
    *,
    offset: np.ndarray | float = 0.0,
) -> np.ndarray:
    if bits <= 0:
        raise ValueError("bits must be positive")
    phase_values = np.asarray(phases, dtype=np.float64)
    offset_values = np.asarray(offset, dtype=np.float64)
    residual = wrap_phase(phase_values - offset_values)
    step = 2.0 * np.pi / (1 << bits)
    quantized = np.rint(residual / step) * step
    return wrap_phase(offset_values + quantized)


def learn_circular_codebook(
    phases: np.ndarray,
    weights: np.ndarray,
    levels: int,
    *,
    iterations: int = 20,
) -> np.ndarray:
    """Magnitude-weighted circular k-means codebook."""

    if levels <= 1:
        raise ValueError("levels must be greater than one")
    phase_values = np.asarray(phases, dtype=np.float64).reshape(-1)
    weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if phase_values.shape != weight_values.shape:
        raise ValueError("phases and weights must have identical shapes")
    centers = np.linspace(-np.pi, np.pi, levels, endpoint=False, dtype=np.float64)
    for _ in range(iterations):
        distance = np.abs(wrap_phase(phase_values[:, None] - centers[None, :]))
        assignment = np.argmin(distance, axis=1)
        updated = centers.copy()
        for level in range(levels):
            selected = assignment == level
            if not np.any(selected):
                continue
            resultant = np.sum(
                weight_values[selected] * np.exp(1j * phase_values[selected])
            )
            if abs(resultant) > 0.0:
                updated[level] = np.angle(resultant)
        if np.max(np.abs(wrap_phase(updated - centers))) < 1e-8:
            centers = updated
            break
        centers = updated
    return np.sort(wrap_phase(centers))


def quantize_phase_codebook(phases: np.ndarray, centers: np.ndarray) -> np.ndarray:
    phase_values = np.asarray(phases, dtype=np.float64)
    center_values = np.asarray(centers, dtype=np.float64)
    distance = np.abs(wrap_phase(phase_values[..., None] - center_values))
    assignment = np.argmin(distance, axis=-1)
    return center_values[assignment]


def phase_only_relative_mse(
    magnitudes: np.ndarray,
    phases: np.ndarray,
    quantized_phases: np.ndarray,
    total_block_energy: float,
) -> float:
    """Relative real-matrix error when only non-self-conjugate FFT phase changes."""

    reference = np.asarray(magnitudes) * np.exp(1j * np.asarray(phases))
    estimate = np.asarray(magnitudes) * np.exp(1j * np.asarray(quantized_phases))
    # Every stored value represents a two-element conjugate pair.
    error = 2.0 * float(np.sum(np.abs(reference - estimate) ** 2, dtype=np.float64))
    return 0.0 if total_block_energy == 0.0 else error / total_block_energy
