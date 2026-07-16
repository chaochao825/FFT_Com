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
from fft_com.freqkv_audit import (  # noqa: E402
    apply_rope_like_rotation,
    frequency_threshold_rows,
    make_historical_synthetic_kv,
    make_smooth_positive_control_kv,
    measure_low_frequency_retention,
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


class FreqKVAuditTests(unittest.TestCase):
    def test_rope_like_rotation_preserves_pair_energy(self) -> None:
        rng = np.random.default_rng(23)
        values = rng.standard_normal((3, 32, 16))
        rotated = apply_rope_like_rotation(values)
        before = np.sum(values.reshape(3, 32, 8, 2) ** 2, axis=-1)
        after = np.sum(rotated.reshape(3, 32, 8, 2) ** 2, axis=-1)
        np.testing.assert_allclose(after, before, atol=1e-12)

    def test_gaussian_dct_energy_tracks_retained_fraction(self) -> None:
        values = make_historical_synthetic_kv(
            64, head_dim=32, heads=2, seed=29
        )["K"]
        result = measure_low_frequency_retention(
            values, (0.5,), transform_path="dct_ii_ortho"
        )[0]
        self.assertAlmostEqual(
            result.selected_frequency_energy_retention,
            0.5,
            delta=0.06,
        )
        self.assertAlmostEqual(
            result.reconstruction_relative_mse
            + result.reconstruction_energy_retention,
            1.0,
            places=12,
        )

    def test_historical_rfft_path_loses_imaginary_energy(self) -> None:
        values = make_historical_synthetic_kv(
            64, head_dim=32, heads=2, seed=31
        )["K"]
        correct = measure_low_frequency_retention(
            values, (1.0,), transform_path="rfft_parseval"
        )[0]
        historical = measure_low_frequency_retention(
            values, (1.0,), transform_path="rfft_drop_imag_historical"
        )[0]
        self.assertAlmostEqual(correct.reconstruction_energy_retention, 1.0, places=12)
        self.assertGreater(historical.reported_energy_retention, 0.40)
        self.assertLess(historical.reported_energy_retention, 0.60)

    def test_smooth_positive_control_is_low_frequency(self) -> None:
        values = make_smooth_positive_control_kv(
            64, head_dim=32, heads=2, seed=37
        )["K"]
        result = measure_low_frequency_retention(
            values, (0.25,), transform_path="dct_ii_ortho"
        )[0]
        self.assertGreater(result.selected_frequency_energy_retention, 0.99)

    def test_frequency_threshold_reports_true_count_and_old_label(self) -> None:
        n = 16
        coefficients = np.zeros((1, n, 1), dtype=np.float64)
        coefficients[:, -1, :] = 1.0
        values = np.einsum(
            "sf,hfd->hsd", dct_matrix(n).T, coefficients, optimize=True
        )
        result = frequency_threshold_rows(
            values, (0.9,), energy_definition="dct_ii_ortho"
        )[0]
        self.assertEqual(result["required_components"], n)
        self.assertEqual(result["true_required_component_fraction"], 1.0)
        self.assertEqual(result["historical_plot_label_fraction"], (n - 1) / n)


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
