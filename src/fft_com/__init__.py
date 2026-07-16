"""Reusable primitives for transform-aware matrix compression studies."""

from .compression import (
    dense_transform_quantize,
    dct_hadamard_hybrid,
    fft_dense_quantize,
    fft_topk_energy,
)
from .metrics import (
    papr,
    relative_mse,
    symmetric_quantize,
    topk_energy,
)
from .transforms import (
    dct_matrix,
    hadamard_matrix,
    randomized_hadamard_matrix,
)

__all__ = [
    "dct_matrix",
    "dense_transform_quantize",
    "dct_hadamard_hybrid",
    "fft_dense_quantize",
    "fft_topk_energy",
    "hadamard_matrix",
    "papr",
    "randomized_hadamard_matrix",
    "relative_mse",
    "symmetric_quantize",
    "topk_energy",
]
