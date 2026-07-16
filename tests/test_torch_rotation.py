from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on NumPy-only environments
    torch = None
    nn = None

if torch is not None:
    from fft_com.model_rotation import (
        adjacent_absolute_cosine,
        build_online_rotated_linear,
        make_rotation_plan,
        spectral_channel_permutation,
    )
    from fft_com.torch_transforms import (
        GroupedOrthogonalTransform,
        apply_orthogonal_transform,
        dct_ortho,
    )
    from fft_com.transforms import dct_matrix


@unittest.skipIf(torch is None, "PyTorch is optional for the NumPy-only study")
class TorchTransformTests(unittest.TestCase):
    def test_dct_matches_numpy_analysis_matrix(self) -> None:
        generator = torch.Generator().manual_seed(3)
        values = torch.randn(5, 16, generator=generator, dtype=torch.float64)
        actual = dct_ortho(values)
        expected = values.numpy() @ dct_matrix(16).T
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-11)

    def test_all_real_transforms_round_trip_and_preserve_energy(self) -> None:
        generator = torch.Generator().manual_seed(5)
        values = torch.randn(7, 16, generator=generator, dtype=torch.float64)
        reference_energy = torch.sum(values**2)
        for kind in ("dct", "hadamard", "rdft"):
            transformed = apply_orthogonal_transform(values, kind)
            restored = apply_orthogonal_transform(
                transformed,
                kind,
                inverse=True,
            )
            torch.testing.assert_close(restored, values, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(
                torch.sum(transformed**2),
                reference_energy,
                atol=1e-10,
                rtol=1e-10,
            )

    def test_group_permutation_round_trip(self) -> None:
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(3, 16, generator=generator, dtype=torch.float64)
        permutations = torch.stack(
            (
                torch.randperm(8, generator=generator),
                torch.randperm(8, generator=generator),
            )
        )
        transform = GroupedOrthogonalTransform(
            "dct",
            8,
            permutation=permutations,
        )
        torch.testing.assert_close(
            transform.inverse(transform(values)),
            values,
            atol=1e-10,
            rtol=1e-10,
        )
        self.assertGreater(transform.metadata_bits(16), 0.0)

    def test_rope_pair_layout_round_trip(self) -> None:
        values = torch.arange(32, dtype=torch.float64).reshape(2, 16)
        transform = GroupedOrthogonalTransform(
            "dct",
            2,
            layout="rope_pairs",
            head_dim=8,
        )
        torch.testing.assert_close(
            transform.inverse(transform(values)),
            values,
            atol=1e-10,
            rtol=1e-10,
        )


@unittest.skipIf(torch is None, "PyTorch is optional for the NumPy-only study")
class RotatedLinearTests(unittest.TestCase):
    def test_unquantized_input_and_output_rotations_are_exact(self) -> None:
        generator = torch.Generator().manual_seed(11)
        linear = nn.Linear(16, 16, bias=True, dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(
                torch.randn(linear.weight.shape, generator=generator, dtype=torch.float64)
            )
            linear.bias.copy_(
                torch.randn(linear.bias.shape, generator=generator, dtype=torch.float64)
            )
        values = torch.randn(4, 16, generator=generator, dtype=torch.float64)
        for kind in ("dct", "hadamard", "rdft"):
            wrapper, stats = build_online_rotated_linear(
                linear,
                bits=None,
                quant_group_size=8,
                input_transform=GroupedOrthogonalTransform(kind, 8),
                output_transform=GroupedOrthogonalTransform(kind, 8),
            )
            torch.testing.assert_close(
                wrapper(values),
                linear(values),
                atol=1e-10,
                rtol=1e-10,
            )
            self.assertLess(stats.weight_relative_mse, 1e-20)

    def test_rope_pair_output_rotation_is_exact_before_quantization(self) -> None:
        generator = torch.Generator().manual_seed(13)
        linear = nn.Linear(16, 16, bias=False, dtype=torch.float64)
        values = torch.randn(3, 16, generator=generator, dtype=torch.float64)
        transform = GroupedOrthogonalTransform(
            "dct",
            2,
            layout="rope_pairs",
            head_dim=8,
        )
        wrapper, _ = build_online_rotated_linear(
            linear,
            bits=None,
            quant_group_size=8,
            output_transform=transform,
        )
        torch.testing.assert_close(
            wrapper(values),
            linear(values),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_spectral_permutation_is_valid_and_improves_synthetic_adjacency(self) -> None:
        angles = torch.linspace(0.0, 2.0 * torch.pi, 16)
        vectors = torch.stack(
            (
                torch.cos(angles),
                torch.sin(angles),
                torch.cos(2.0 * angles),
                torch.sin(2.0 * angles),
            ),
            dim=0,
        )
        scrambled = torch.tensor([0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15])
        weight = vectors[:, scrambled]
        permutation, diagnostics = spectral_channel_permutation(
            weight,
            16,
            side="input",
        )
        self.assertEqual(set(permutation[0].tolist()), set(range(16)))
        self.assertGreater(
            diagnostics["adjacent_abs_cosine_after"],
            diagnostics["adjacent_abs_cosine_before"],
        )
        measured = adjacent_absolute_cosine(
            weight,
            16,
            permutation=permutation,
        )
        self.assertAlmostEqual(
            measured,
            diagnostics["adjacent_abs_cosine_after"],
            places=6,
        )

    def test_head_and_gqa_plan_respects_projection_boundaries(self) -> None:
        config = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        q_plan = make_rotation_plan(
            "model.layers.0.self_attn.q_proj",
            nn.Linear(32, 32, bias=False),
            config,
            scope="attention_head",
            transform_kind="dct",
        )
        k_plan = make_rotation_plan(
            "model.layers.0.self_attn.k_proj",
            nn.Linear(32, 16, bias=False),
            config,
            scope="attention_head",
            transform_kind="dct",
        )
        self.assertEqual(q_plan.head_count, 4)
        self.assertEqual(k_plan.head_count, 2)
        self.assertEqual(q_plan.kv_group_size, 2)
        self.assertEqual(k_plan.boundary, "kv_head_boundary")

        two_sided = make_rotation_plan(
            "model.layers.0.self_attn.q_proj",
            nn.Linear(32, 32, bias=False),
            config,
            scope="q_proj_two_sided",
            transform_kind="dct",
        )
        self.assertIsNotNone(two_sided.input_transform)
        self.assertIsNotNone(two_sided.output_transform)
        self.assertEqual(two_sided.head_count, 4)


if __name__ == "__main__":
    unittest.main()
