#!/usr/bin/env bash
# run_cutc_grid.sh — Lanza el GRID 3×3 de transferencia dialectal (Cut C, Sprint 4).
#   Lane A (Qwen3-1.7B) -> GPU 6 ; Lane B (Qwen3-4B) -> GPU 7.
# Matriz país-fuente × país-objetivo (ES/CR/PE), 3 semillas {42,43,44}.
# Orden: por SEMILLA (tras la 1ª pasada ya hay una matriz 3×3 completa que mirar),
# y dentro de cada semilla por (fuente, objetivo). Cada celda es REANUDABLE
# (omite si ya existe results/sprint4/<name>.json). Un fallo NO aborta el lane.
#
# Uso:
#   bash scripts/run_cutc_grid.sh                # ambos lanes en paralelo
#   GPU_A=6 GPU_B=7 bash scripts/run_cutc_grid.sh
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/jupyterlab_container/hf_cache}"
export TOKENIZERS_PARALLELISM=false
PY="${PY:-$HOME/jupyterlab_container/envs/ailab/bin/python}"
[ -x "$PY" ] || PY=python
CFG=configs/sprint4/grid
LOGDIR=results/checkpoints/grid_logs   # gitignored
mkdir -p "$LOGDIR" results/sprint4

GPU_A="${GPU_A:-6}"   # Qwen3-1.7B
GPU_B="${GPU_B:-7}"   # Qwen3-4B
SEEDS="42 43 44"
COUNTRIES="es cr pe"

build_list() {
  local tag="$1"
  local seed src tgt
  for seed in $SEEDS; do
    for src in $COUNTRIES; do
      for tgt in $COUNTRIES; do
        echo "$CFG/cutc_${tag}_${src}2${tgt}_s${seed}.yaml"
      done
    done
  done
}

run_lane() {
  local tag="$1" gpu="$2"
  local log="$LOGDIR/cutc_${tag}.log"
  echo "[lane $tag] GPU $gpu — $(date)" > "$log"
  while read -r cfg; do
    [ -f "$cfg" ] || { echo "[skip] no existe $cfg" >> "$log"; continue; }
    echo "=== $(basename "$cfg") $(date) ===" >> "$log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_lora_experiment.py \
      --config "$cfg" --results_dir results/sprint4/ >> "$log" 2>&1 \
      || echo "[FAIL] $cfg (continúo)" >> "$log"
  done < <(build_list "$tag")
  echo "[lane $tag] DONE $(date)" >> "$log"
}

run_lane qwen1.7b "$GPU_A" &
PID_A=$!
run_lane qwen4b "$GPU_B" &
PID_B=$!
wait "$PID_A" "$PID_B"
echo "GRID Cut C COMPLETO $(date)"
