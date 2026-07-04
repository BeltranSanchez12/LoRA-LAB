"""
run_encoder_baselines.py — Fine-tuning de encoders controlados sobre Cardiff ES.

Ejecutar desde la raíz del repo:
    python scripts/run_encoder_baselines.py

Genera:
    results/baselines/beto_cardiff_es.json     (sobreescribe el provisional de 1 época)
    results/baselines/xlmr_base_cardiff_es.json

Tiempo estimado en CPU: ~33 min por modelo (3 épocas).
Tiempo estimado en GPU T4: ~5-8 min por modelo.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_data import load_cardiff_es, DataConfig
from src.models.encoder_baseline import EncoderFineTuner
from src.models.evaluate import save_result

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


MODELS = [
    {
        "model_id": "dccuchile/bert-base-spanish-wwm-cased",
        "out_json": "results/baselines/beto_cardiff_es.json",
        "output_dir": "/tmp/beto_cardiff_es",
    },
    {
        "model_id": "xlm-roberta-base",
        "out_json": "results/baselines/xlmr_base_cardiff_es.json",
        "output_dir": "/tmp/xlmr_base_cardiff_es",
    },
]


def main() -> None:
    # Load data
    config = DataConfig(
        dataset_name="cardiffnlp/tweet_sentiment_multilingual",
        language="es",
    )
    logger.info("Loading Cardiff ES splits …")
    splits = load_cardiff_es(config=config)
    train_ds = splits["train"]
    val_ds = splits["validation"]
    test_ds = splits["test"]
    logger.info(
        "Splits loaded. Train=%d, Val=%d, Test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    results = {}
    for cfg in MODELS:
        model_id = cfg["model_id"]
        out_json = ROOT / cfg["out_json"]
        output_dir = cfg["output_dir"]

        logger.info("=" * 60)
        logger.info("Fine-tuning: %s", model_id)

        finetuner = EncoderFineTuner(model_id=model_id, epochs=3)
        train_time = finetuner.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=output_dir,
        )

        import torch
        peak_vram = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else None
        )

        result = finetuner.evaluate(
            test_ds,
            train_time_s=train_time,
            peak_vram_gb=peak_vram,
        )

        # Save JSON
        import json
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w") as f:
            json.dump(result.to_json_dict(), f, indent=2)
        logger.info("Saved: %s", out_json)

        # Also append to all_results.csv
        save_result(result, results_dir=str(ROOT / "results"))

        results[model_id] = result.f1_macro
        logger.info("%s — F1 macro = %.4f", model_id, result.f1_macro)

    logger.info("=" * 60)
    logger.info("SUMMARY:")
    for model_id, f1 in results.items():
        logger.info("  %-55s F1 macro = %.4f", model_id, f1)


if __name__ == "__main__":
    main()
