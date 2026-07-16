#!/usr/bin/env bash
set -euo pipefail

unset PREFIX
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FFT_COM_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PY="${FFT_COM_PYTHON:-python}"
OUTDIR="$ROOT/runs/all_input_screen"

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

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

for bits in 3 4; do
  for transform in identity dct hadamard rdft; do
    variant="all_input_${transform}_q${bits}_screen"
    output="$OUTDIR/${variant}.json"
    if [ -s "$output" ]; then
      echo "SKIP existing $output"
      continue
    fi
    "$PY" scripts/run_model_rotation_eval.py \
      "${model_args[@]}" \
      --variant "$variant" \
      --scope all_input \
      --transform "$transform" \
      --bits "$bits" \
      --calibration-tokens 256 \
      --calibration-samples-per-module 16 \
      --eval-tokens 2048 \
      --sequence-length 2048 \
      --device cuda:0 \
      --output "$output"
  done
done

"$PY" - <<'PY'
import glob
import json
import os

for path in sorted(glob.glob("runs/all_input_screen/*.json")):
    payload = json.load(open(path, encoding="utf-8"))
    aggregate = payload["aggregate"]
    print(
        os.path.basename(path),
        "ppl",
        round(payload["perplexity"]["perplexity"], 6),
        "wmse",
        round(aggregate["weight_relative_mse_parameter_weighted"], 6),
        "cal",
        round(
            aggregate["calibration_output_relative_mse_parameter_weighted"],
            6,
        ),
        "seconds",
        round(payload["perplexity"]["elapsed_seconds"], 3),
    )
PY
