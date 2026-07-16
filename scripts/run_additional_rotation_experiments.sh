#!/usr/bin/env bash
set -euo pipefail

unset PREFIX
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FFT_COM_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PY="${FFT_COM_PYTHON:-python}"
OUTDIR="$ROOT/runs/additional_8192"

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

common=(
  --calibration-tokens 512
  --calibration-samples-per-module 32
  --eval-tokens 8192
  --sequence-length 2048
  --device cuda:0
)

run_quantized() {
  local variant=$1
  local transform=$2
  local bits=$3
  local permutation=${4:-none}
  local output="$OUTDIR/${variant}.json"
  if [ -s "$output" ]; then
    echo "SKIP existing $output"
    return
  fi
  "$PY" scripts/run_model_rotation_eval.py \
    "${model_args[@]}" \
    --variant "$variant" \
    --scope q_proj_two_sided \
    --transform "$transform" \
    --bits "$bits" \
    --permutation "$permutation" \
    "${common[@]}" \
    --output "$output"
}

unquantized="$OUTDIR/qproj_two_sided_dct_unquantized_8192.json"
if [ ! -s "$unquantized" ]; then
  "$PY" scripts/run_model_rotation_eval.py \
    "${model_args[@]}" \
    --variant qproj_two_sided_dct_unquantized_8192 \
    --scope q_proj_two_sided \
    --transform dct \
    --unquantized \
    "${common[@]}" \
    --output "$unquantized"
fi

for bits in 3 4; do
  run_quantized "qproj_two_sided_identity_q${bits}_8192" identity "$bits"
  run_quantized "qproj_two_sided_dct_q${bits}_8192" dct "$bits"
  run_quantized \
    "qproj_two_sided_spectral_dct_q${bits}_8192" \
    dct "$bits" spectral
done

latency="$OUTDIR/latency.json"
if [ ! -s "$latency" ]; then
  "$PY" scripts/benchmark_transform_latency.py \
    --device cuda:0 \
    --dtype float16 \
    --batch-size 1 \
    --sequence-length 256 \
    --hidden-size 4096 \
    --intermediate-size 11008 \
    --group-size 128 \
    --warmup 20 \
    --repeats 100 \
    --output "$latency"
fi
