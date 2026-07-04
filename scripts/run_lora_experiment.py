"""
run_lora_experiment.py — Lanza UN experimento PEFT desde un fichero de config.

Uso:
    python scripts/run_lora_experiment.py --config configs/sprint3/smoke_lora_qwen1.7b_n50.yaml
    CUDA_VISIBLE_DEVICES=6 python scripts/run_lora_experiment.py --config <cfg>

Un experimento = un YAML. Resultado en results/ + all_results.csv (vía save_result).
Reanudable: si ya existe results/<name>.json se omite (salvo experiment.overwrite=true).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_lora_experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Un experimento PEFT desde config YAML")
    parser.add_argument("--config", required=True, help="Ruta al YAML del experimento")
    parser.add_argument("--results_dir", default=None, help="Override del results_dir")
    args = parser.parse_args()

    from src.models.lora_finetune import load_experiment_config, run_experiment

    config = load_experiment_config(args.config)
    logger.info("Config: %s", args.config)

    summary = run_experiment(config, results_dir=args.results_dir)

    print("\n" + "=" * 78)
    print(f"EXPERIMENTO: {summary['experiment_name']}")
    print("-" * 78)
    print(f"  modelo            : {summary['model_name']}  ({summary['method']})")
    print(f"  fracción / seed   : {summary['data_fraction']} / {summary['seed']}")
    print(f"  loss (1ª -> últ.) : {summary.get('loss_first')} -> {summary.get('loss_last')}")
    print(f"  F1-macro          : {summary['f1_macro']:.4f}")
    print(f"  accuracy          : {summary['accuracy']:.4f}")
    print(f"  f1_per_class      : {summary['f1_per_class']}")
    print(f"  fallback_rate     : {summary['fallback_rate']:.4f}")
    print(f"  trainable_params  : {summary['trainable_params']:,}")
    print(f"  peak_vram_gb      : {summary['peak_vram_gb']}")
    print(f"  train_time_s      : {summary['train_time_s']}")
    print(f"  inference_lat_ms  : {summary['inference_latency_ms']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
