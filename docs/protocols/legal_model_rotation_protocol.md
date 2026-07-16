# Legal online model-rotation protocol

This protocol separates the 2026-07-16 block-level transform study from the
2026-07-17 model-level experiments.

## Evidence boundary

- The earlier study used sampled 64x64 blocks from Llama-2-7B-chat weights. It
  established coefficient statistics and block reconstruction error, not
  end-to-end perplexity or production-kernel latency.
- The model-level study uses the complete Llama-2-7B base checkpoint,
  WikiText-2 train activations for calibration, and a disjoint WikiText-2 test
  prefix for perplexity.
- Model weights and datasets remain server-local and are not published in this
  repository.

The reference run used Python 3.10.18, PyTorch 2.9.0+cu128, Transformers
4.57.1, Datasets 4.3.0, Accelerate 1.11.0, and one NVIDIA A800 80GB PCIe GPU.

## Staged-run provenance

The experiments were executed in batches while the evaluator gained the
two-sided q-projection scope and evaluation-offset orchestration. Every raw
result stores its own source SHA256 values, and the published summary preserves
them by variant. The core transform implementation has the same hash across all
valid runs; later two-sided and segment-replication runs use the final
evaluator/model-plan implementation. Earlier result JSON files were not
rewritten after execution.

## Linear algebra convention

Activations are row vectors and PyTorch linear layers compute

```text
y = x W^T.
```

For a real orthogonal transform `R`, an input-side rotation is

```text
C = W R^T
x' = x R^T
x' C^T = x W^T.
```

An output-side rotation is

```text
C = R W
z = x C^T = y R^T
y = z R.
```

The online transform or inverse is retained explicitly. This makes the
unquantized path functionally equivalent without assuming that DCT, Hadamard,
or FFT commutes with RoPE, residual connections, normalization, or nonlinear
attention.

## Compared transforms

- `identity`: no basis change.
- `dct`: orthonormal DCT-II analysis with DCT-III synthesis, implemented by a
  real-input FFT construction.
- `hadamard`: normalized Sylvester fast Walsh-Hadamard transform.
- `rdft`: an `N`-real-scalar orthogonal packing of the real FFT degrees of
  freedom: DC, Nyquist, and scaled real/imaginary components.

The RDFT representation is the fair real-valued FFT proxy. It does not require
complex linear weights or a separately stored Hermitian payload.

## Quantizer

The comparison uses signed symmetric abs-max fake quantization:

- one scale per weight row and per 128 input weights;
- q3 uses codes `[-3, 3]`;
- q4 uses codes `[-7, 7]`;
- quantized values are immediately dequantized to the model dtype.

Consequently, perplexity measures quantization quality and the measured model
time includes online transform overhead, but neither number represents a
packed INT3/INT4 kernel.

Weight relative MSE is computed on float32 dequantized coefficients after
inverse rotation, before the final model-dtype storage cast. Calibration-output
MSE and perplexity include that FP16 storage and online-transform roundoff.

## Real calibration activations

Calibration tokens come from the local WikiText-2 raw train Arrow file. A
single baseline-model forward pass records deterministic, evenly spaced input
vectors for every targeted linear. Local output relative MSE is then measured
against the original floating-point linear output before any model modules are
replaced.

Evaluation tokens come from the disjoint WikiText-2 raw test Arrow file. Text
rows are joined with blank lines before tokenization, matching the usual
WikiText concatenation convention; no per-row EOS token is inserted.

## Rotation scopes

- `all_input`: input-side groups of 128 for every q/k/v/o and gate/up/down
  projection. This is the primary same-location Identity/DCT/Hadamard/RDFT
  comparison.
- `q_proj_input`: `C = W R^T` for q projections only.
- `q_proj_output_head`: `C = R W` within each query-head boundary.
- `q_proj_two_sided`: q projections use both input groups and query-head output
  groups, allowing separate learned row and column permutations in
  `D_o P W Q D_i^T`.
- `attention_head`: q/k/v output rotations are grouped by query or KV head;
  o-projection input rotations are grouped by concatenated query heads.
- `qk_rope_pair`: q/k output coordinates are laid out as Llama split-half RoPE
  pairs, transformed in groups of two, and inverted immediately before RoPE.

For GQA-capable configurations, q uses query-head boundaries, k/v use KV-head
boundaries, and the query-per-KV ratio is recorded. Llama-2-7B has 32 query
heads and 32 KV heads, so this checkpoint does not empirically exercise GQA.

## Spectral permutation

Each transform group may learn a channel ordering from absolute cosine
similarity of weight-channel vectors:

1. build the dense affinity graph;
2. form its graph Laplacian;
3. sort channels by the Fiedler eigenvector;
4. apply the fixed permutation before DCT.

Every independently encoded group is charged

```text
log2(group_size!)
```

metadata bits. Runtime benchmarking separately measures the per-group gather.
An algorithmic head or RoPE-pair layout is not charged as a learned
permutation.

## Perplexity and latency reporting

- Perplexity uses deterministic non-overlapping contiguous windows. The first
  token of each window is not scored, and the exact input-token count and
  sequence length are stored in every JSON result.
- Screening runs use 2,048 test tokens; formal comparisons use 8,192.
- Promising q-projection variants are repeated on disjoint 8,192-token test
  spans beginning at offsets 0, 8,192, and 16,384.
- Transform latency uses CUDA events after warmup on shapes
  `[1, 256, 4096]` and `[1, 256, 11008]`.
- A representative 4096x4096 q projection measures floating-point GEMM plus
  the online input transform.
- All latency numbers are labeled as unfused PyTorch reference
  implementations.

## Decision rule

A direction is retained only if it improves same-protocol perplexity over the
Identity quantization baseline at the same scope and bit width. Weight MSE or
calibration-output MSE alone is insufficient. Permutation must additionally
justify metadata and gather cost, and head/RoPE-aware methods must be compared
against an Identity baseline targeting exactly the same projections.
