"""
run_prompting_baseline.py — Script autonomo para ejecutar el baseline de
prompting (T4) en Colab/Kaggle con GPU T4.

Uso:
    python scripts/run_prompting_baseline.py --model_id Qwen/Qwen3-1.7B
    python scripts/run_prompting_baseline.py --model_id Qwen/Qwen3-1.7B --dump-examples

El script:
1. Carga el protocolo de configs/prompting_protocol.yaml.
2. Carga Qwen3-1.7B con bitsandbytes 4-bit si hay GPU, float32 si no.
3. Para cada k en [0, 4, 8, 16]:
   - Genera predicciones sobre el test set completo de Cardiff ES.
   - Registra fallback_rate y latencia.
   - Crea un ExperimentResult y lo guarda con save_result.
4. Imprime tabla de resultados por k.

Flags:
    --model_id STR       ID del modelo HuggingFace (default: Qwen/Qwen3-1.7B)
    --config PATH        Ruta al YAML del protocolo (default: configs/prompting_protocol.yaml)
    --results_dir PATH   Directorio de resultados (default: results/)
    --seed INT           Semilla global (default: 42)
    --k_values INT...    Valores de k a evaluar (default: segun YAML)
    --dump-examples      Solo vuelca los ejemplos few-shot y sale (sin inferencia)
    --max_examples INT   Limitar test a N ejemplos (para pruebas rapidas)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Setup del path del proyecto
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline de prompting para Cardiff ES")
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen3-1.7B",
        help="ID del modelo HuggingFace (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "prompting_protocol.yaml"),
        help="Ruta al YAML del protocolo",
    )
    parser.add_argument(
        "--results_dir",
        default=str(PROJECT_ROOT / "results"),
        help="Directorio raiz de resultados",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla global",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=None,
        help="Valores de k a evaluar (default: segun YAML)",
    )
    parser.add_argument(
        "--dump-examples",
        action="store_true",
        help="Solo vuelca los ejemplos few-shot y sale (sin inferencia)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limitar test a N ejemplos (para pruebas rapidas)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str):
    """Carga el modelo y tokenizador.

    - GPU disponible: carga en 4-bit (bitsandbytes QLoRA) para ahorrar VRAM.
    - Sin GPU: carga en float32 (lento, solo para pruebas de codigo).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"

    logger.info("Cargando tokenizador: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if has_gpu:
        logger.info("GPU detectada. Cargando %s en 4-bit (QLoRA).", model_id)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        logger.warning(
            "Sin GPU. Cargando %s en float32 (muy lento). Solo para pruebas de codigo.",
            model_id,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to(device)

    logger.info("Modelo cargado en dispositivo: %s", device)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Semilla global
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Cargar protocolo
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f)["prompting_protocol"]

    k_values = args.k_values if args.k_values is not None else protocol["k_values"]
    logger.info("k_values: %s", k_values)

    # Cargar datos
    from datasets import load_from_disk
    data_dir = PROJECT_ROOT / "data" / "processed" / "cardiff_es"
    train_ds = load_from_disk(str(data_dir / "train"))
    test_ds = load_from_disk(str(data_dir / "test"))

    if args.max_examples is not None:
        test_ds = test_ds.select(range(min(args.max_examples, len(test_ds))))
        logger.info("Test limitado a %d ejemplos.", len(test_ds))

    # Importar clases del proyecto
    from src.models.prompting import PromptingBaseline, select_and_print_fixed_examples
    from src.models.evaluate import (
        ExperimentResult,
        compute_metrics,
        save_result,
        CostTracker,
        get_peak_vram_gb,
        reset_peak_vram,
        LABEL_NAMES,
    )

    # Instancia base del baseline (para seleccion de ejemplos)
    baseline = PromptingBaseline(config_path=config_path, k=0)

    # Pre-seleccionar ejemplos fijos para todos los k != 0
    fixed_examples: dict[int, list[dict]] = {}
    for k in k_values:
        if k > 0:
            fixed_examples[k] = baseline.select_examples(train_ds, k=k, seed=args.seed)

    # Dump de ejemplos y salida si se pide
    if args.dump_examples:
        print("\n=== EJEMPLOS FEW-SHOT FIJOS ===")
        for k, exs in fixed_examples.items():
            print(f"\n--- k={k} ---")
            for ex in exs:
                print(f"  [{ex['label']}] {ex['text'][:100]}")
        print("\nCopia estos ejemplos en configs/prompting_protocol.yaml -> fixed_examples")
        return

    # Cargar modelo
    model, tokenizer, device = load_model_and_tokenizer(args.model_id)

    # Etiquetas verdaderas (ints)
    y_true_ids = list(test_ds["label"])
    test_texts = list(test_ds["text"])

    # Resultados de la tabla resumen
    summary_rows: list[dict] = []

    results_dir = Path(args.results_dir)
    baselines_dir = results_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    for k in k_values:
        logger.info("=" * 60)
        logger.info("Evaluando k=%d (%s)...", k, "zero-shot" if k == 0 else "few-shot")

        b = PromptingBaseline(config_path=config_path, k=k)
        examples = fixed_examples.get(k, [])

        reset_peak_vram()
        t0 = time.perf_counter()

        preds_str, fallback_rate = b.predict_batch(
            test_texts,
            model,
            tokenizer,
            examples=examples,
            device=device,
        )

        elapsed_s = time.perf_counter() - t0
        peak_vram = get_peak_vram_gb()

        # Convertir predicciones string -> int
        label2id = {"negative": 0, "neutral": 1, "positive": 2}
        y_pred_ids = [label2id.get(p, 1) for p in preds_str]

        # Metricas de calidad
        metrics = compute_metrics(y_true_ids, y_pred_ids, label_names=LABEL_NAMES)

        # Latencia media por ejemplo
        inference_latency_ms = (elapsed_s / len(test_texts)) * 1000

        method_name = "prompting_zeroshot" if k == 0 else f"prompting_fewshot_k{k}"
        experiment_name = f"{method_name}_{args.model_id.replace('/', '_')}"

        # Contar parametros del modelo (todos en prompting, ningun parametro entrenado)
        total_params = sum(p.numel() for p in model.parameters())

        result = ExperimentResult(
            experiment_name=experiment_name,
            model_name=args.model_id,
            method=method_name,
            data_fraction="full",
            seed=args.seed,
            f1_macro=metrics["f1_macro"],
            accuracy=metrics["accuracy"],
            f1_per_class=metrics["f1_per_class"],
            trainable_params=0,  # prompting: no hay parametros entrenados
            peak_vram_gb=peak_vram,
            train_time_s=0.0,   # no hay entrenamiento
            inference_latency_ms=inference_latency_ms,
            fallback_rate=fallback_rate,
            notes=(
                f"Prompting {method_name}. k={k}. "
                f"Modelo: {args.model_id} ({total_params/1e6:.1f}M params). "
                f"Fallback: {fallback_rate:.1%}. "
                f"Tiempo total inferencia: {elapsed_s:.1f}s."
            ),
        )

        save_result(result, results_dir=str(results_dir))

        # JSON en baselines/ con nombre legible
        json_path = baselines_dir / f"{method_name}_{args.model_id.split('/')[-1]}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_json_dict(), f, ensure_ascii=False, indent=2)
        logger.info("JSON guardado: %s", json_path)

        summary_rows.append({
            "k": k,
            "method": method_name,
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            "fallback_rate": fallback_rate,
            "latency_ms": inference_latency_ms,
            "peak_vram_gb": peak_vram,
        })

    # Tabla resumen
    print("\n" + "=" * 75)
    print(f"{'k':>4}  {'method':<28}  {'F1-macro':>8}  {'Acc':>6}  {'Fallback':>8}  {'ms/ex':>7}")
    print("-" * 75)
    for row in summary_rows:
        print(
            f"{row['k']:>4}  {row['method']:<28}  "
            f"{row['f1_macro']:>8.4f}  {row['accuracy']:>6.4f}  "
            f"{row['fallback_rate']:>8.1%}  {row['latency_ms']:>7.1f}"
        )
    print("=" * 75)
    print(f"\nResultados guardados en: {results_dir}")


if __name__ == "__main__":
    main()
