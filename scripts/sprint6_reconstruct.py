#!/usr/bin/env python
"""sprint6_reconstruct.py — Reconstrucción determinista de los pesos que el pipeline
nunca guardó (save_strategy="no"), para poder correr la robustez del Sprint 6 SOLO en
inferencia.

Reconstruye con datos LIMPIOS, semilla 42 y determinismo total (src.utils.seed.set_seed:
use_deterministic_algorithms + seeds de random/numpy/torch/cuda) los 4 modelos sin pesos:
  - LoRA Qwen3-1.7B (nfull, s42)   -> adaptador a  adapters/<name>/            (versionable)
  - LoRA Qwen3-4B   (nfull, s42)   -> adaptador a  adapters/<name>/            (versionable)
  - BETO  (encoder FT, 3 épocas)   -> pesos a      results/checkpoints/encoder_*  (local)
  - XLM-R (encoder FT, 3 épocas)   -> pesos a      results/checkpoints/encoder_*  (local)
Y re-infiere prompting k=4 (Qwen3-4B, 4-bit) — no tiene pesos que guardar.

NO se entrena sobre NADA perturbado. NO se toca la evidencia del paper: todo lo que este
script escribe va a results/sprint6/ (JSON/CSV) y a adapters/ + results/checkpoints/.

Sanity check: cada F1 reconstruido se compara con la banda del paper. Criterio (usuario):
  - LoRA:     media_paper ± std_por_semilla (±0.006 en 1.7B, ±0.022 en 4B)
  - Encoders/prompting (semilla única, deterministas): ±0.005
Si TODOS caen en banda -> baseline válido para la robustez. Si alguno se sale -> el script
lo marca FAIL y hay que PARAR (no seguir con un baseline malo).

Uso:
    CUDA_VISIBLE_DEVICES=0 HF_HOME=~/jupyterlab_container/hf_cache \
        python scripts/sprint6_reconstruct.py
    (--smoke: prueba de plomería rápida y barata, resultados a results/sprint6_smoke/)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sprint6_reconstruct")

# --- Números del paper (paper/main.tex) + valor original de semilla 42 (results/*.json) ---
# paper_mean = cifra publicada; band = tolerancia acordada; orig_s42 = valor almacenado s42.
PAPER = {
    "lora_1.7b":    {"label": "LoRA Qwen3-1.7B (nfull)",  "paper_mean": 0.698, "band": 0.006, "orig_s42": 0.6922061633056852},
    "lora_4b":      {"label": "LoRA Qwen3-4B (nfull)",    "paper_mean": 0.707, "band": 0.022, "orig_s42": 0.7219585769659487},
    "prompt_k4_4b": {"label": "Prompting k=4 (Qwen3-4B)", "paper_mean": 0.652, "band": 0.005, "orig_s42": 0.6522459597615115},
    "beto":         {"label": "BETO (encoder FT)",        "paper_mean": 0.661, "band": 0.005, "orig_s42": 0.6612994350282486},
    "xlmr":         {"label": "XLM-R (encoder FT)",       "paper_mean": 0.646, "band": 0.005, "orig_s42": 0.6464889915810161},
}
ORDER = ["lora_1.7b", "lora_4b", "prompt_k4_4b", "beto", "xlmr"]


def _cuda_cleanup():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def _lora_config(base_model: str, name: str, results_dir: str, adapters_dir: str,
                 fraction, min_steps: int) -> dict:
    """Espejo EXACTO de configs/sprint3/grid/lora_qwen*_nfull_s42.yaml + persistencia."""
    return {
        "experiment": {"name": name, "seed": 42, "overwrite": True, "gpu_group": "A"},
        "model": {
            "base_model": base_model,
            "method": "lora",
            "peft": {
                "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
        },
        "data": {"processed_dir": "data/processed/cardiff_es", "train_fraction": fraction},
        "training": {
            "epochs": 5, "min_steps": min_steps, "batch_size": 8, "grad_accum": 1,
            "learning_rate": 0.0002, "warmup_ratio": 0.1, "max_length": 256,
            "lr_scheduler_type": "cosine", "eval_points": 4,
        },
        "eval": {"max_examples": None},
        "output": {"results_dir": results_dir, "save_adapter_dir": adapters_dir},
    }


def reconstruct_lora(key: str, base_model: str, name: str, results_dir: str,
                     adapters_dir: str, fraction, min_steps: int) -> float:
    from src.utils.seed import set_seed
    from src.models.lora_finetune import run_experiment
    set_seed(42)  # determinismo total (use_deterministic_algorithms + cudnn.deterministic)
    logger.info("[%s] Reconstruyendo %s (fraction=%s) ...", key, base_model, fraction)
    cfg = _lora_config(base_model, name, results_dir, adapters_dir, fraction, min_steps)
    summary = run_experiment(cfg, results_dir=str(ROOT / results_dir))
    f1 = float(summary["f1_macro"])
    logger.info("[%s] F1-macro reconstruido = %.4f (adaptador en %s/%s)",
                key, f1, adapters_dir, name)
    _cuda_cleanup()
    return f1


# ---------------------------------------------------------------------------
# Encoders (BETO, XLM-R)
# ---------------------------------------------------------------------------

def reconstruct_encoder(key: str, model_id: str, short: str, results_dir: str,
                        epochs: int) -> float:
    from src.utils.seed import set_seed
    from src.data.load_data import load_cardiff_es, DataConfig
    from src.models.encoder_baseline import EncoderFineTuner
    set_seed(42)
    logger.info("[%s] Reconstruyendo encoder %s (epochs=%d) ...", key, model_id, epochs)
    splits = load_cardiff_es(
        config=DataConfig(dataset_name="cardiffnlp/tweet_sentiment_multilingual", language="es")
    )
    ft = EncoderFineTuner(model_id=model_id, epochs=epochs)
    ckpt_dir = ROOT / "results" / "checkpoints" / f"encoder_{short}"
    tt = ft.train(
        train_dataset=splits["train"],
        val_dataset=splits["validation"],
        output_dir=str(ckpt_dir / "_hf_trainer"),
    )
    res = ft.evaluate(splits["test"], train_time_s=tt)
    ft.save(ckpt_dir)  # pesos finales (mejor época) para re-inferencia
    out_json = ROOT / results_dir / f"recon_encoder_{short}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    f1 = float(res.f1_macro)
    logger.info("[%s] F1-macro reconstruido = %.4f (pesos en %s)", key, f1, ckpt_dir)
    _cuda_cleanup()
    return f1


# ---------------------------------------------------------------------------
# Prompting k=4 (Qwen3-4B, 4-bit) — reutiliza el script canónico en subproceso aislado
# ---------------------------------------------------------------------------

def reconstruct_prompting_k4_4b(results_dir: str, max_examples: int | None) -> float:
    out_dir = ROOT / results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_prompting_baseline.py"),
        "--model_id", "Qwen/Qwen3-4B",
        "--k_values", "4",
        "--results_dir", str(out_dir),
        "--seed", "42",
    ]
    if max_examples is not None:
        cmd += ["--max_examples", str(max_examples)]
    logger.info("[prompt_k4_4b] Re-inferencia prompting k=4 (subproceso): %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    jpath = out_dir / "baselines" / "prompting_fewshot_k4_Qwen3-4B.json"
    data = json.loads(jpath.read_text(encoding="utf-8"))
    f1 = float(data["f1_macro"])
    logger.info("[prompt_k4_4b] F1-macro reconstruido = %.4f", f1)
    _cuda_cleanup()
    return f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstrucción determinista de pesos (Sprint 6)")
    ap.add_argument("--smoke", action="store_true", help="Plomería rápida y barata")
    args = ap.parse_args()

    if args.smoke:
        results_dir, adapters_dir = "results/sprint6_smoke", "adapters_smoke"
        fraction, min_steps, epochs, max_ex = "50", 20, 1, 40
        name_suffix = "_smoke"
    else:
        results_dir, adapters_dir = "results/sprint6", "adapters"
        fraction, min_steps, epochs, max_ex = "full", 80, 3, None
        name_suffix = ""

    (ROOT / results_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    results: dict[str, float] = {}
    errors: dict[str, str] = {}

    plan = [
        ("lora_1.7b", lambda: reconstruct_lora(
            "lora_1.7b", "Qwen/Qwen3-1.7B", f"lora_qwen1.7b_nfull_s42{name_suffix}",
            results_dir, adapters_dir, fraction, min_steps)),
        ("lora_4b", lambda: reconstruct_lora(
            "lora_4b", "Qwen/Qwen3-4B", f"lora_qwen4b_nfull_s42{name_suffix}",
            results_dir, adapters_dir, fraction, min_steps)),
        ("beto", lambda: reconstruct_encoder(
            "beto", "dccuchile/bert-base-spanish-wwm-cased", "beto_cardiff_es",
            results_dir, epochs)),
        ("xlmr", lambda: reconstruct_encoder(
            "xlmr", "xlm-roberta-base", "xlmr_base_cardiff_es", results_dir, epochs)),
        ("prompt_k4_4b", lambda: reconstruct_prompting_k4_4b(results_dir, max_ex)),
    ]

    for key, fn in plan:
        try:
            results[key] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] FALLÓ: %s", key, exc)
            errors[key] = f"{type(exc).__name__}: {exc}"

    # --- Sanity table ---
    rows = []
    all_pass = True
    for key in ORDER:
        ref = PAPER[key]
        lo, hi = ref["paper_mean"] - ref["band"], ref["paper_mean"] + ref["band"]
        if key in results:
            f1 = results[key]
            in_band = (lo - 1e-9) <= f1 <= (hi + 1e-9)
            status = "PASS" if in_band else "FAIL"
            if not in_band:
                all_pass = False
            delta_s42 = f1 - ref["orig_s42"]
        else:
            f1, in_band, status, delta_s42 = float("nan"), False, "ERROR", float("nan")
            all_pass = False
        rows.append({
            "key": key, "label": ref["label"], "reconstructed_f1": f1,
            "paper_mean": ref["paper_mean"], "band": ref["band"],
            "band_lo": lo, "band_hi": hi, "orig_s42": ref["orig_s42"],
            "delta_vs_orig_s42": delta_s42, "in_band": in_band, "status": status,
        })

    # CSV
    import csv
    csv_path = ROOT / results_dir / "sanity_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Print
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 92)
    print(f"SANITY CHECK — baseline LIMPIO reconstruido (semilla 42)   [{elapsed/60:.1f} min]")
    print("-" * 92)
    print(f"{'modelo':<26} {'recon':>8} {'paper':>7} {'banda':>16} {'Δ vs s42':>10}  {'estado':>6}")
    print("-" * 92)
    for r in rows:
        band = f"[{r['band_lo']:.3f},{r['band_hi']:.3f}]"
        recon = "  ERROR " if r["status"] == "ERROR" else f"{r['reconstructed_f1']:.4f}"
        dlt = "     n/a" if r["status"] == "ERROR" else f"{r['delta_vs_orig_s42']:+.4f}"
        print(f"{r['label']:<26} {recon:>8} {r['paper_mean']:>7.3f} {band:>16} {dlt:>10}  {r['status']:>6}")
    print("=" * 92)
    if errors:
        print("ERRORES:")
        for k, v in errors.items():
            print(f"  - {k}: {v}")
    print(f"\nCSV: {csv_path}")
    print("RESULTADO GLOBAL:", "TODOS EN BANDA ✓ — baseline válido, seguir a Gate 1"
          if all_pass else "FUERA DE BANDA ✗ — PARAR y revisar")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
