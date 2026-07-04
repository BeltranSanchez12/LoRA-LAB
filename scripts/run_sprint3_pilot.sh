#!/usr/bin/env bash
# run_sprint3_pilot.sh — Lanza el PILOTO de 1 seed del Sprint 3 paralelizando por GPU.
#   Grupo A (Qwen3-1.7B) -> GPU 0 ; Grupo B (Qwen3-4B) -> GPU 6.
# Cada celda es reanudable (omite si ya existe results/<name>.json). Una celda que
# falle no aborta el grupo. Resultados a results/ + all_results.csv.
#
# Uso:  bash scripts/run_sprint3_pilot.sh
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/jupyterlab_container/hf_cache}"
PY="${PY:-$HOME/jupyterlab_container/envs/ailab/bin/python}"
CFG=configs/sprint3/pilot
LOGDIR=results/checkpoints/pilot_logs   # gitignored
mkdir -p "$LOGDIR"

run_group() {
  local gpu="$1"; shift
  for cfg in "$@"; do
    [ -f "$cfg" ] || continue
    local name; name="$(basename "$cfg" .yaml)"
    echo "[GPU $gpu] >>> $name"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_lora_experiment.py --config "$cfg" \
      > "$LOGDIR/$name.log" 2>&1
    echo "[GPU $gpu] <<< $name (exit $?)"
  done
}

# Orden: fracciones ascendentes (rápidas primero) y spot-checks al final.
A_CELLS="$CFG/lora_qwen1.7b_n10_s42.yaml $CFG/lora_qwen1.7b_n16_s42.yaml $CFG/lora_qwen1.7b_n25_s42.yaml \
$CFG/lora_qwen1.7b_n50_s42.yaml $CFG/lora_qwen1.7b_n100_s42.yaml $CFG/lora_qwen1.7b_n250_s42.yaml \
$CFG/lora_qwen1.7b_n500_s42.yaml $CFG/lora_qwen1.7b_n1000_s42.yaml $CFG/lora_qwen1.7b_nfull_s42.yaml \
$CFG/qlora_qwen1.7b_nfull_s42.yaml $CFG/full_ft_qwen1.7b_nfull_s42.yaml"

B_CELLS="$CFG/lora_qwen4b_n10_s42.yaml $CFG/lora_qwen4b_n16_s42.yaml $CFG/lora_qwen4b_n25_s42.yaml \
$CFG/lora_qwen4b_n50_s42.yaml $CFG/lora_qwen4b_n100_s42.yaml $CFG/lora_qwen4b_n250_s42.yaml \
$CFG/lora_qwen4b_n500_s42.yaml $CFG/lora_qwen4b_n1000_s42.yaml $CFG/lora_qwen4b_nfull_s42.yaml \
$CFG/qlora_qwen4b_nfull_s42.yaml"

run_group 0 $A_CELLS &
PID_A=$!
run_group 6 $B_CELLS &
PID_B=$!
wait $PID_A; echo "Grupo A terminado"
wait $PID_B; echo "Grupo B terminado"
echo "PILOT DONE"
