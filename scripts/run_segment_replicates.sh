#!/usr/bin/env bash
set -euo pipefail

unset PREFIX
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FFT_COM_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PY="${FFT_COM_PYTHON:-python}"
OUTDIR="$ROOT/runs/segment_replicates"

model_args=()
if [[ -n "${FFT_COM_MODEL_DIR:-}" ]]; then
  model_args+=(--model-dir "$FFT_COM_MODEL_DIR")
fi
if [[ -n "${FFT_COM_CALIBRATION_ARROW:-}" ]]; then
  model_args+=(--calibration-arrow "$FFT_COM_CALIBRATION_ARROW")
fi
if [[ -n "${FFT_COM_TEST_ARROW:-}" ]]; then
  model_args+=(--test-arrow "$FFT_COM_TEST_ARROW")
fi

cd "$ROOT"
mkdir -p "$OUTDIR"
: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

run_baseline() {
  local offset=$1
  local variant="fp16_baseline_offset${offset}"
  local output="$OUTDIR/${variant}.json"
  if [ -s "$output" ]; then
    echo "SKIP existing $output"
    return
  fi
  "$PY" scripts/run_model_rotation_eval.py \
    "${model_args[@]}" \
    --variant "$variant" \
    --baseline \
    --calibration-tokens 512 \
    --calibration-samples-per-module 32 \
    --eval-tokens 8192 \
    --eval-token-offset "$offset" \
    --sequence-length 2048 \
    --device cuda:0 \
    --output "$output"
}

run_variant() {
  local offset=$1
  local scope=$2
  local transform=$3
  local permutation=${4:-none}
  local suffix=$transform
  if [ "$permutation" != none ]; then
    suffix="${permutation}_${transform}"
  fi
  local variant="${scope}_${suffix}_q3_offset${offset}"
  local output="$OUTDIR/${variant}.json"
  if [ -s "$output" ]; then
    echo "SKIP existing $output"
    return
  fi
  "$PY" scripts/run_model_rotation_eval.py \
    "${model_args[@]}" \
    --variant "$variant" \
    --scope "$scope" \
    --transform "$transform" \
    --bits 3 \
    --permutation "$permutation" \
    --calibration-tokens 512 \
    --calibration-samples-per-module 32 \
    --eval-tokens 8192 \
    --eval-token-offset "$offset" \
    --sequence-length 2048 \
    --device cuda:0 \
    --output "$output"
}

for offset in 8192 16384; do
  run_baseline "$offset"
  for transform in identity dct hadamard rdft; do
    run_variant "$offset" q_proj_input "$transform"
  done
  run_variant "$offset" q_proj_input dct spectral
  run_variant "$offset" q_proj_output_head identity
  run_variant "$offset" q_proj_output_head dct
done
