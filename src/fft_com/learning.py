"""Small CPU-only learned-rotation proxy based on butterfly Givens stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import batch_quantization_relative_mse, papr
from .transforms import apply_butterfly_two_sided


@dataclass(frozen=True)
class LearnedButterfly:
    left_angles: np.ndarray
    right_angles: np.ndarray
    train_objective: float
    initial_left_angles: np.ndarray
    initial_right_angles: np.ndarray
    initial_objective: float
    initial_source: str
    history: tuple[dict[str, float | int | str], ...]


def butterfly_quantization_objective(
    blocks: np.ndarray,
    left_angles: np.ndarray,
    right_angles: np.ndarray,
    bits: int,
) -> float:
    coefficients = apply_butterfly_two_sided(blocks, left_angles, right_angles)
    return batch_quantization_relative_mse(coefficients, bits)


def evaluate_butterfly(
    blocks: np.ndarray,
    left_angles: np.ndarray,
    right_angles: np.ndarray,
    bits: int,
) -> dict[str, float]:
    coefficients = apply_butterfly_two_sided(blocks, left_angles, right_angles)
    return {
        "relative_mse": batch_quantization_relative_mse(coefficients, bits),
        "papr_mean": float(np.mean([papr(block) for block in coefficients])),
    }


def learn_butterfly_rotation(
    train_blocks: np.ndarray,
    *,
    bits: int = 3,
    stages: int = 4,
    random_candidates: int = 12,
    init_scale: float = 0.45,
    coordinate_steps: tuple[float, ...] = (0.25, 0.10, 0.04),
    seed: int = 0,
) -> LearnedButterfly:
    """Fit a shared quantization-aware butterfly with derivative-free search.

    This is deliberately a small proxy, not an implementation of SpinQuant.
    It searches a restricted real orthogonal family and evaluates held-out
    blocks in the calling script.
    """

    blocks = np.asarray(train_blocks, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError("train_blocks must have shape [batch, n, n]")
    n = blocks.shape[1]
    if stages <= 0:
        raise ValueError("stages must be positive")
    rng = np.random.default_rng(seed)
    shape = (stages, n // 2)

    candidates: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    zeros = np.zeros(shape, dtype=np.float64)
    candidates.append(
        (
            butterfly_quantization_objective(blocks, zeros, zeros, bits),
            zeros.copy(),
            zeros.copy(),
            "identity_initialization",
        )
    )
    maximum_stages = int(np.log2(n))
    if stages == maximum_stages:
        hadamard_like = np.full(shape, np.pi / 4.0, dtype=np.float64)
        candidates.append(
            (
                butterfly_quantization_objective(
                    blocks, hadamard_like, hadamard_like, bits
                ),
                hadamard_like.copy(),
                hadamard_like.copy(),
                "hadamard_like_initialization",
            )
        )
    for candidate in range(random_candidates):
        left = rng.normal(0.0, init_scale, size=shape)
        right = rng.normal(0.0, init_scale, size=shape)
        objective = butterfly_quantization_objective(blocks, left, right, bits)
        candidates.append((objective, left, right, f"random_{candidate}"))
    objective, left_angles, right_angles, source = min(candidates, key=lambda item: item[0])
    left_angles = left_angles.copy()
    right_angles = right_angles.copy()
    initial_left_angles = left_angles.copy()
    initial_right_angles = right_angles.copy()
    initial_objective = float(objective)

    history: list[dict[str, float | int | str]] = [
        {
            "phase": "initial_selection",
            "source": source,
            "objective": float(objective),
            "candidates": len(candidates),
        }
    ]

    total_coordinates = left_angles.size + right_angles.size
    for sweep, step in enumerate(coordinate_steps):
        accepted = 0
        order = rng.permutation(total_coordinates)
        for coordinate in order:
            target = left_angles if coordinate < left_angles.size else right_angles
            local = coordinate if coordinate < left_angles.size else coordinate - left_angles.size
            row, col = np.unravel_index(local, target.shape)
            original = float(target[row, col])
            best_value = original
            best_objective = objective
            for delta in (-step, step):
                target[row, col] = original + delta
                candidate_objective = butterfly_quantization_objective(
                    blocks,
                    left_angles,
                    right_angles,
                    bits,
                )
                if candidate_objective < best_objective:
                    best_objective = candidate_objective
                    best_value = original + delta
            target[row, col] = best_value
            if best_objective < objective:
                objective = best_objective
                accepted += 1
        history.append(
            {
                "phase": "coordinate_sweep",
                "sweep": sweep,
                "step": float(step),
                "accepted": accepted,
                "objective": float(objective),
            }
        )

    return LearnedButterfly(
        left_angles=left_angles,
        right_angles=right_angles,
        train_objective=float(objective),
        initial_left_angles=initial_left_angles,
        initial_right_angles=initial_right_angles,
        initial_objective=initial_objective,
        initial_source=source,
        history=tuple(history),
    )
