"""Orthogonal transforms used by the FFT_Com potential study."""

from __future__ import annotations

import functools
import math

import numpy as np


def _require_square_power_of_two(n: int) -> None:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"expected a positive power of two, got {n}")


@functools.lru_cache(maxsize=None)
def dct_matrix(n: int) -> np.ndarray:
    """Return the orthonormal DCT-II analysis matrix."""

    if n <= 0:
        raise ValueError("n must be positive")
    sample = np.arange(n, dtype=np.float64)[None, :]
    frequency = np.arange(n, dtype=np.float64)[:, None]
    matrix = np.cos(math.pi * (sample + 0.5) * frequency / n)
    matrix[0] *= math.sqrt(1.0 / n)
    if n > 1:
        matrix[1:] *= math.sqrt(2.0 / n)
    return matrix


@functools.lru_cache(maxsize=None)
def hadamard_matrix(n: int) -> np.ndarray:
    """Return a normalized Sylvester Hadamard matrix."""

    _require_square_power_of_two(n)
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / math.sqrt(n)


def randomized_hadamard_matrix(n: int, seed: int) -> np.ndarray:
    """Return ``H P D`` with deterministic permutation and sign diagonals."""

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    return hadamard_matrix(n)[:, permutation] * signs[None, :]


def haar_orthogonal_matrix(n: int, seed: int) -> np.ndarray:
    """Return a deterministic Haar-distributed orthogonal matrix."""

    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


@functools.lru_cache(maxsize=None)
def zigzag_indices(n: int) -> np.ndarray:
    """Return flattened square-matrix indices in JPEG-style zigzag order."""

    if n <= 0:
        raise ValueError("n must be positive")
    order: list[int] = []
    for diagonal in range(2 * n - 1):
        coordinates: list[tuple[int, int]] = []
        row_min = max(0, diagonal - n + 1)
        row_max = min(n - 1, diagonal)
        for row in range(row_min, row_max + 1):
            coordinates.append((row, diagonal - row))
        if diagonal % 2 == 0:
            coordinates.reverse()
        order.extend(row * n + col for row, col in coordinates)
    return np.asarray(order, dtype=np.int64)


@functools.lru_cache(maxsize=None)
def butterfly_pairs(n: int, stages: int | None = None) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return radix-2 butterfly pairings for a sequence of Givens stages."""

    _require_square_power_of_two(n)
    maximum = int(math.log2(n))
    if stages is None:
        stages = maximum
    if stages <= 0 or stages > maximum:
        raise ValueError(f"stages must be in [1, {maximum}], got {stages}")

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for stage in range(stages):
        half_span = 1 << stage
        span = half_span << 1
        first: list[int] = []
        second: list[int] = []
        for start in range(0, n, span):
            for offset in range(half_span):
                first.append(start + offset)
                second.append(start + offset + half_span)
        result.append(
            (
                np.asarray(first, dtype=np.int64),
                np.asarray(second, dtype=np.int64),
            )
        )
    return tuple(result)


def apply_butterfly_left(blocks: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Apply batched butterfly row rotations.

    ``blocks`` may be ``[n, n]`` or ``[batch, n, n]``. ``angles`` has shape
    ``[stages, n // 2]``.
    """

    values = np.asarray(blocks, dtype=np.float64)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("blocks must be square matrices or a batch of them")

    n = values.shape[1]
    angle_values = np.asarray(angles, dtype=np.float64)
    if angle_values.ndim != 2 or angle_values.shape[1] != n // 2:
        raise ValueError("angles must have shape [stages, n // 2]")
    pairings = butterfly_pairs(n, angle_values.shape[0])

    output = values.copy()
    for stage, (first, second) in enumerate(pairings):
        left = output[:, first, :].copy()
        right = output[:, second, :].copy()
        cosine = np.cos(angle_values[stage])[None, :, None]
        sine = np.sin(angle_values[stage])[None, :, None]
        output[:, first, :] = cosine * left + sine * right
        output[:, second, :] = -sine * left + cosine * right
    return output[0] if squeeze else output


def apply_butterfly_two_sided(
    blocks: np.ndarray,
    left_angles: np.ndarray,
    right_angles: np.ndarray,
) -> np.ndarray:
    """Apply ``R_left @ W @ R_right.T`` to a batch of square blocks."""

    left_applied = apply_butterfly_left(blocks, left_angles)
    transposed = np.swapaxes(left_applied, -1, -2)
    right_applied = apply_butterfly_left(transposed, right_angles)
    return np.swapaxes(right_applied, -1, -2)


def paired_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Return a block-diagonal real Givens rotation matrix."""

    values = np.asarray(angles, dtype=np.float64)
    n = values.size * 2
    matrix = np.eye(n, dtype=np.float64)
    for pair, angle in enumerate(values):
        first = pair * 2
        second = first + 1
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        matrix[first, first] = cosine
        matrix[first, second] = sine
        matrix[second, first] = -sine
        matrix[second, second] = cosine
    return matrix
