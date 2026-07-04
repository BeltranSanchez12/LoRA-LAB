#!/usr/bin/env bash
# run_sprint3_grid.sh — Lanza el GRID de 3 semillas del Sprint 3 en 2 lanes de GPU.
#   Lane A (Qwen3-1.7B) -> GPU 6 ; Lane B (Qwen3-4B) -> GPU 7.
# Orden: fracciones ascendentes (rápidas primero), semillas 42/43/44, spot-checks
# del corte B al final. Cada celda es REANUDABLE (omite si ya existe
# results/<name>.json). Una celda que falle NO aborta el lane. Logs gitignored.
#
# Uso:
#   bash scripts/run_sprint3_grid.sh            # ambos lanes en paralelo
#   GPU_A=6 GPU_B=7 bash scripts/run_sprint3_grid.sh
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/jupyterlab_container/hf_cache}"
export TOKENIZERS_PARALLELISM=false
PY="${PY:-$HOME/jupyterlab_container/envs/ailab/bin/python}"
CFG=configs/sprint3/grid
LOGDIR=results/checkpoints/grid_logs   # gitignored
mkdir -p "$LOGDIR"

GPU_A="${GPU_A:-6}"   # Qwen3-1.7B
GPU_B="${GPU_B:-7}"   # Qwen3-4B
FRACS="4 7 10 16 25 50 100 250 500 1000 full"
SEEDS="42 43 44"

# Construye la lista ordenada de configs de un modelo (corte A) + spot-checks.
build_list() {
  local tag="$1"; shift
  local list=()
  local frac seed
  for frac in $FRACS; do
    for seed in $SEEDS; do
      list+=("$CFG/lora_${tag}_n${frac}_s${seed}.yaml")
    done
  done
  # spot-checks (corte B, 1 semilla) al final, si existen para este tag
  for extra in "$@"; do
    [ -f "$CFG/$extra" ] && list+=("$CFG/$extra")
  done
  printf '%s\n' "${list[@]}"
}

run_lane() {
  local gpu="$1"; shift
  local cfgs=("$@")
  local total="${#cfgs[@]}" i=0 done=0 skip=0 fail=0
  for cfg in "${cfgs[@]}"; do
    i=$((i+1))
    [ -f "$cfg" ] || { echo "[GPU $gpu] MISSING $cfg"; continue; }
    local name; name="$(basename "$cfg" .yaml)"
    if [ -f "results/$name.json" ]; then
      echo "[GPU $gpu] ($i/$total) SKIP $name (ya existe)"; skip=$((skip+1)); continue
    fi
    echo "[GPU $gpu] ($i/$total) >>> $name"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_lora_experiment.py --config "$cfg" \
      > "$LOGDIR/$name.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "[GPU $gpu] ($i/$total) <<< $name OK"; done=$((done+1));
    else echo "[GPU $gpu] ($i/$total) !!! $name FAILED (rc=$rc) — ver $LOGDIR/$name.log"; fail=$((fail+1)); fi
  done
  echo "[GPU $gpu] LANE DONE — ok=$done skip=$skip fail=$fail total=$total"
}

mapfile -t A_CELLS < <(build_list qwen1.7b qlora_qwen1.7b_nfull_s42.yaml full_ft_qwen1.7b_nfull_s42.yaml)
mapfile -t B_CELLS < <(build_list qwen4b qlora_qwen4b_nfull_s42.yaml)

echo "Lane A (GPU $GPU_A, 1.7B): ${#A_CELLS[@]} celdas"
echo "Lane B (GPU $GPU_B, 4B):  ${#B_CELLS[@]} celdas"

run_lane "$GPU_A" "${A_CELLS[@]}" &
PID_A=$!
run_lane "$GPU_B" "${B_CELLS[@]}" &
PID_B=$!
wait $PID_A; echo "Lane A terminado"
wait $PID_B; echo "Lane B terminado"
echo "GRID DONE"
