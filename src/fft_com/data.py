"""Data loading and deterministic block sampling for local model weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BlockRecord:
    dataset: str
    tensor_name: str
    layer: int
    row_start: int
    col_start: int
    values: np.ndarray


class SafeTensorStore:
    """Resolve and read tensors from a sharded Hugging Face checkpoint."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.weight_map: dict[str, str] = self.index["weight_map"]

    def load(self, tensor_name: str) -> np.ndarray:
        try:
            shard_name = self.weight_map[tensor_name]
        except KeyError as exc:
            raise KeyError(f"tensor not found in checkpoint: {tensor_name}") from exc
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required for real-model experiments"
            ) from exc
        shard_path = self.model_dir / shard_name
        with safe_open(str(shard_path), framework="numpy") as handle:
            return np.asarray(handle.get_tensor(tensor_name))


def _sample_positions(
    shape: tuple[int, int],
    block_size: int,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    row_blocks = shape[0] // block_size
    col_blocks = shape[1] // block_size
    if row_blocks == 0 or col_blocks == 0:
        raise ValueError(f"matrix shape {shape} is smaller than block size {block_size}")
    population = row_blocks * col_blocks
    chosen = rng.choice(population, size=min(count, population), replace=False)
    return [
        (int(index // col_blocks) * block_size, int(index % col_blocks) * block_size)
        for index in chosen
    ]


def collect_model_blocks(
    model_dir: str | Path,
    *,
    layers: tuple[int, ...],
    tensor_suffixes: dict[str, str],
    block_size: int,
    blocks_per_tensor: int,
    seed: int,
) -> list[BlockRecord]:
    """Collect aligned block coordinates across layers for each tensor family."""

    store = SafeTensorStore(model_dir)
    rng = np.random.default_rng(seed)
    records: list[BlockRecord] = []
    for dataset, suffix in tensor_suffixes.items():
        positions: list[tuple[int, int]] | None = None
        for layer in layers:
            tensor_name = f"model.layers.{layer}.{suffix}"
            matrix = store.load(tensor_name)
            if matrix.ndim != 2:
                raise ValueError(f"expected matrix tensor, got {tensor_name}: {matrix.shape}")
            if positions is None:
                positions = _sample_positions(
                    (int(matrix.shape[0]), int(matrix.shape[1])),
                    block_size,
                    blocks_per_tensor,
                    rng,
                )
            for row_start, col_start in positions:
                block = matrix[
                    row_start : row_start + block_size,
                    col_start : col_start + block_size,
                ].astype(np.float64)
                records.append(
                    BlockRecord(
                        dataset=dataset,
                        tensor_name=tensor_name,
                        layer=layer,
                        row_start=row_start,
                        col_start=col_start,
                        values=block,
                    )
                )
            del matrix
    return records


def load_layer0_embedding_activations(
    model_dir: str | Path,
    *,
    sample_count: int,
    seed: int,
    rms_epsilon: float = 1e-6,
) -> np.ndarray:
    """Sample real token embeddings and apply the first Llama RMSNorm."""

    store = SafeTensorStore(model_dir)
    embeddings = store.load("model.embed_tokens.weight")
    norm_weight = store.load("model.layers.0.input_layernorm.weight").astype(np.float64)
    rng = np.random.default_rng(seed)
    token_ids = rng.integers(0, embeddings.shape[0], size=sample_count)
    activations = embeddings[token_ids].astype(np.float64)
    variance = np.mean(activations**2, axis=-1, keepdims=True)
    activations = activations / np.sqrt(variance + rms_epsilon)
    activations *= norm_weight[None, :]
    return activations
