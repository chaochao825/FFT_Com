#!/usr/bin/env bash
set -euo pipefail

unset PREFIX
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FFT_COM_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PY="${FFT_COM_PYTHON:-python}"
OUTDIR="$ROOT/runs/formal_8192"

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

run_baseline() {
  local variant=fp16_baseline_8192
  local output="$OUTDIR/${variant}.json"
  if [ -s "$output" ]; then
    echo "SKIP existing $output"
    return
  fi
  "$PY" scripts/run_model_rotation_eval.py \
    "${model_args[@]}" \
    --variant "$variant" \
    --baseline \
    "${common[@]}" \
    --output "$output"
}

run_unquantized() {
  local variant=$1
  local scope=$2
  local transform=$3
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
    --unquantized \
    "${common[@]}" \
    --output "$output"
}

run_quantized() {
  local variant=$1
  local scope=$2
  local transform=$3
  local bits=$4
  local permutation=${5:-none}
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
    --bits "$bits" \
    --permutation "$permutation" \
    "${common[@]}" \
    --output "$output"
}

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

run_baseline
run_unquantized all_input_dct_unquantized_8192 all_input dct

for bits in 3 4; do
  for transform in identity dct hadamard rdft; do
    run_quantized \
      "all_input_${transform}_q${bits}_8192" \
      all_input "$transform" "$bits"
  done
done

for transform in identity dct hadamard rdft; do
  run_quantized \
    "qproj_input_${transform}_q3_8192" \
    q_proj_input "$transform" 3
done
run_quantized \
  qproj_input_spectral_dct_q3_8192 \
  q_proj_input dct 3 spectral

for transform in identity dct; do
  run_quantized \
    "qproj_input_${transform}_q4_8192" \
    q_proj_input "$transform" 4
done
run_quantized \
  qproj_input_spectral_dct_q4_8192 \
  q_proj_input dct 4 spectral

for transform in identity dct hadamard rdft; do
  run_quantized \
    "qproj_output_head_${transform}_q3_8192" \
    q_proj_output_head "$transform" 3
done
for transform in identity dct; do
  run_quantized \
    "qproj_output_head_${transform}_q4_8192" \
    q_proj_output_head "$transform" 4
done

for transform in identity dct hadamard rdft; do
  run_quantized \
    "attention_head_${transform}_q4_8192" \
    attention_head "$transform" 4
done

for transform in identity dct; do
  run_quantized \
    "qk_rope_pair_${transform}_q4_8192" \
    qk_rope_pair "$transform" 4
done

"$PY" - <<'PY'
import glob
import json
import os

for path in sorted(glob.glob("runs/formal_8192/*.json")):
    payload = json.load(open(path, encoding="utf-8"))
    aggregate = payload["aggregate"]
    print(
        os.path.basename(path),
        "ppl",
        round(payload["perplexity"]["perplexity"], 6),
        "wmse",
        aggregate.get("weight_relative_mse_parameter_weighted"),
        "cal",
        aggregate.get("calibration_output_relative_mse_parameter_weighted"),
        "tokens/s",
        round(payload["perplexity"]["input_tokens_per_second"], 2),
    )
PY
