from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fft_com.compression import (  # noqa: E402
    dense_transform_quantize,
    dct_hadamard_hybrid,
    fft_dense_quantize,
    fft_hermitian_groups,
    fft_topk_energy,
    inverse_two_sided,
    two_sided_transform,
)
from fft_com.kuramoto import (  # noqa: E402
    attractive_kuramoto,
    order_parameter,
    quantize_phase_uniform,
    weighted_phase_template,
)
from fft_com.metrics import relative_mse, topk_energy  # noqa: E402
from fft_com.transforms import (  # noqa: E402
    apply_butterfly_left,
    apply_butterfly_two_sided,
    dct_matrix,
    hadamard_matrix,
    randomized_hadamard_matrix,
    zigzag_indices,
)


class TransformTests(unittest.TestCase):
    def test_orthogonal_matrices_and_round_trip(self) -> None:
        rng = np.random.default_rng(3)
        block = rng.standard_normal((16, 16))
        for matrix in (
            dct_matrix(16),
            hadamard_matrix(16),
            randomized_hadamard_matrix(16, 4),
        ):
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(16), atol=1e-12)
            coefficients = two_sided_transform(block, matrix, matrix)
            reconstructed = inverse_two_sided(coefficients, matrix, matrix)
            np.testing.assert_allclose(reconstructed, block, atol=1e-11)

    def test_zigzag_is_permutation(self) -> None:
        order = zigzag_indices(8)
        self.assertEqual(order.size, 64)
        self.assertEqual(set(order.tolist()), set(range(64)))
        self.assertEqual(order[0], 0)

    def test_butterfly_preserves_energy(self) -> None:
        rng = np.random.default_rng(7)
        blocks = rng.standard_normal((5, 16, 16))
        left = rng.normal(0.0, 0.5, size=(3, 8))
        right = rng.normal(0.0, 0.5, size=(3, 8))
        transformed = apply_butterfly_two_sided(blocks, left, right)
        before = np.sum(blocks**2, axis=(1, 2))
        after = np.sum(transformed**2, axis=(1, 2))
        np.testing.assert_allclose(after, before, atol=1e-10)

    def test_full_pi_over_four_butterfly_is_hadamard_like(self) -> None:
        n = 16
        angles = np.full((4, n // 2), np.pi / 4.0)
        matrix = apply_butterfly_left(np.eye(n), angles)
        np.testing.assert_allclose(np.abs(matrix), np.full((n, n), 1.0 / np.sqrt(n)))
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(n), atol=1e-12)


class CompressionTests(unittest.TestCase):
    def test_fft_independent_scalar_count(self) -> None:
        for n in (4, 8, 16):
            scalar_count = sum(
                1 if group[4] else 2 for group in fft_hermitian_groups(n)
            )
            self.assertEqual(scalar_count, n * n)

    def test_fft_quantization_round_trip_at_high_bits(self) -> None:
        rng = np.random.default_rng(11)
        block = rng.standard_normal((16, 16))
        result = fft_dense_quantize(block, 16)
        self.assertLess(result.relative_mse, 1e-8)

    def test_fft_topk_energy_is_monotonic(self) -> None:
        rng = np.random.default_rng(13)
        block = rng.standard_normal((16, 16))
        low = fft_topk_energy(block, 0.125)
        high = fft_topk_energy(block, 0.25)
        self.assertLessEqual(low, high)
        self.assertGreater(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_dct_hybrid_improves_smooth_matrix_over_ternary(self) -> None:
        rng = np.random.default_rng(17)
        n = 16
        coefficients = np.zeros((n, n), dtype=np.float64)
        coefficients[:3, :3] = rng.standard_normal((3, 3))
        dct = dct_matrix(n)
        block = dct.T @ coefficients @ dct
        identity = np.eye(n)
        direct = dense_transform_quantize(block, identity, identity, 2)
        hybrid = dct_hadamard_hybrid(block, density=0.125)
        self.assertLess(hybrid.relative_mse, direct.relative_mse)

    def test_topk_energy_bounds(self) -> None:
        values = np.arange(16, dtype=np.float64).reshape(4, 4)
        self.assertGreaterEqual(topk_energy(values, 0.25), 0.25)
        self.assertLessEqual(topk_energy(values, 0.25), 1.0)
        self.assertEqual(relative_mse(values, values), 0.0)


class KuramotoTests(unittest.TestCase):
    def test_attractive_coupling_increases_order(self) -> None:
        rng = np.random.default_rng(19)
        phases = rng.uniform(-np.pi, np.pi, size=12)
        adjacency = np.ones((12, 12)) - np.eye(12)
        updated = attractive_kuramoto(adjacency, phases)
        self.assertGreater(order_parameter(updated), order_parameter(phases))
        self.assertGreater(order_parameter(updated), 0.99)

    def test_phase_template_and_uniform_quantizer(self) -> None:
        phases = np.asarray([[0.10, 1.0], [0.12, 1.1], [0.08, 0.9]])
        weights = np.ones_like(phases)
        template, concentration = weighted_phase_template(phases, weights)
        self.assertAlmostEqual(template[0], 0.10, places=6)
        self.assertGreater(concentration[0], 0.99)
        quantized = quantize_phase_uniform(phases, 3, offset=template)
        self.assertEqual(quantized.shape, phases.shape)


if __name__ == "__main__":
    unittest.main()
