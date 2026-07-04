#!/usr/bin/env python
"""sprint6_robustness.py — Prueba de robustez del Sprint 6 (T2), SOLO inferencia.

Carga los pesos CONGELADOS (adaptadores LoRA reconstruidos + encoders guardados) y re-evalúa
los 5 modelos clave sobre el test de Cardiff ES LIMPIO y sobre cada uno de los 7 conjuntos
perturbados (taxonomía aprobada) + code_switching (opcional, aparte). CERO reentrenamiento:
lo único que cambia es el texto de entrada en inferencia.

Para cada modelo × condición mide: Macro-F1, F1 por clase (neg/neu/pos), y —frente al limpio—
la caída absoluta y relativa (agregada y por clase). Guarda todo en
    results/robustness_results.csv

Los números limpios deben reproducir el baseline reconstruido (semilla 42):
    LoRA-1.7B 0.6922 · LoRA-4B 0.7220 · Prompting-k4 0.6522 · BETO 0.6613 · XLM-R 0.6465
(validación impresa por modelo). El paper NO se toca.

Uso:
    CUDA_VISIBLE_DEVICES=0 HF_HOME=~/jupyterlab_container/hf_cache \
        python scripts/sprint6_robustness.py            # completo (~30-35 min)
        python scripts/sprint6_robustness.py --smoke    # wiring: limpio+1 pert, 40 ej, 5 modelos
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("sprint6_robustness")

from src.utils.seed import set_seed
from src.data.perturbations import PERTURBATIONS, OPTIONAL_PERTURBATIONS, apply_perturbation
from src.models.evaluate import compute_metrics, LABEL_NAMES

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

EXPECTED_CLEAN = {
    "lora_1.7b": 0.6922061633056852,
    "lora_4b": 0.7219585769659487,
    "prompt_k4_4b": 0.6522459597615115,
    "beto": 0.6612994350282486,
    "xlmr": 0.6464889915810161,
}
MODEL_LABELS = {
    "lora_1.7b": "LoRA Qwen3-1.7B", "lora_4b": "LoRA Qwen3-4B",
    "prompt_k4_4b": "Prompting k=4 (Qwen3-4B)", "beto": "BETO", "xlmr": "XLM-R",
}
CORE = list(PERTURBATIONS.keys())              # 7 nucleares
OPTIONAL = list(OPTIONAL_PERTURBATIONS.keys())  # code_switching

CSV_COLUMNS = [
    "model", "model_label", "method", "perturbation", "optional", "n_test", "fallback_rate",
    "f1_macro", "f1_negative", "f1_neutral", "f1_positive",
    "f1_macro_clean", "delta_abs", "delta_rel_pct",
    "delta_f1_negative", "delta_f1_neutral", "delta_f1_positive",
]


def _cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_test():
    from datasets import load_from_disk
    ds = load_from_disk(str(ROOT / "data" / "processed" / "cardiff_es" / "test"))
    return list(ds["text"]), [int(x) for x in ds["label"]]


def cond_texts(cond: str, texts: list[str]) -> list[str]:
    if cond == "limpio":
        return list(texts)
    return apply_perturbation(cond, texts, seed=42)


# ---------------------------------------------------------------------------
# Cargadores de modelos -> devuelven (predict_fn, method)
#   predict_fn(texts) -> (y_pred_ids: list[int], fallback_rate: float | None)
# ---------------------------------------------------------------------------

def load_lora(base_model: str, adapter_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from src.models.prompting import PromptingBaseline

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    model.config.use_cache = True
    pb = PromptingBaseline(config_path=ROOT / "configs" / "prompting_protocol.yaml", k=0)

    def predict(texts):
        preds_str, fb = pb.predict_batch(texts, model, tok, examples=[], device="cuda")
        return [LABEL2ID.get(p, 1) for p in preds_str], fb

    return predict, "lora", (model, base)


def load_prompting_4b():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_from_disk
    from src.models.prompting import PromptingBaseline

    model_id = "Qwen/Qwen3-4B"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    train_ds = load_from_disk(str(ROOT / "data" / "processed" / "cardiff_es" / "train"))
    base = PromptingBaseline(config_path=ROOT / "configs" / "prompting_protocol.yaml", k=0)
    examples = base.select_examples(train_ds, k=4, seed=42)
    pb = PromptingBaseline(config_path=ROOT / "configs" / "prompting_protocol.yaml", k=4)

    def predict(texts):
        preds_str, fb = pb.predict_batch(texts, model, tok, examples=examples, device="cuda")
        return [LABEL2ID.get(p, 1) for p in preds_str], fb

    return predict, "prompting_fewshot_k4", (model,)


def load_encoder(ckpt_dir: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir)).to("cuda").eval()

    def predict(texts, batch_size=16, max_length=128):
        y = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b = texts[i:i + batch_size]
                enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                enc = {k: v.to("cuda") for k, v in enc.items()}
                logits = model(**enc).logits
                y.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        return y, None

    return predict, "encoder_finetuned", (model,)


LOADERS = {
    "lora_1.7b": lambda: load_lora("Qwen/Qwen3-1.7B", ROOT / "adapters" / "lora_qwen1.7b_nfull_s42"),
    "lora_4b": lambda: load_lora("Qwen/Qwen3-4B", ROOT / "adapters" / "lora_qwen4b_nfull_s42"),
    "prompt_k4_4b": load_prompting_4b,
    "beto": lambda: load_encoder(ROOT / "results" / "checkpoints" / "encoder_beto_cardiff_es"),
    "xlmr": lambda: load_encoder(ROOT / "results" / "checkpoints" / "encoder_xlmr_base_cardiff_es"),
}


def evaluate_model(key: str, conditions: list[str], texts, y_true) -> list[dict]:
    set_seed(42)
    logger.info("=== Modelo %s: cargando pesos congelados ===", key)
    predict, method, handles = LOADERS[key]()

    # 1) limpio primero (para deltas + validación)
    per_cond = {}
    for cond in conditions:
        tx = cond_texts(cond, texts)
        y_pred, fb = predict(tx)
        m = compute_metrics(y_true, y_pred, label_names=LABEL_NAMES)
        per_cond[cond] = {"m": m, "fb": fb}
        logger.info("  [%s] %-16s F1=%.4f", key, cond, m["f1_macro"])

    clean = per_cond["limpio"]["m"]
    f1c = clean["f1_macro"]
    exp = EXPECTED_CLEAN[key]
    ok = abs(f1c - exp) < 1e-6
    logger.info("  [%s] LIMPIO reconstruido=%.6f esperado=%.6f  %s",
                key, f1c, exp, "OK (bit-exacto)" if ok else f"WARN Δ={f1c-exp:+.6f}")

    rows = []
    for cond in conditions:
        m = per_cond[cond]["m"]
        fpc, fpc_c = m["f1_per_class"], clean["f1_per_class"]
        delta = m["f1_macro"] - f1c
        rows.append({
            "model": key, "model_label": MODEL_LABELS[key], "method": method,
            "perturbation": cond, "optional": cond in OPTIONAL, "n_test": len(y_true),
            "fallback_rate": per_cond[cond]["fb"],
            "f1_macro": m["f1_macro"], "f1_negative": fpc["negative"],
            "f1_neutral": fpc["neutral"], "f1_positive": fpc["positive"],
            "f1_macro_clean": f1c, "delta_abs": delta,
            "delta_rel_pct": 100.0 * delta / f1c if f1c else 0.0,
            "delta_f1_negative": fpc["negative"] - fpc_c["negative"],
            "delta_f1_neutral": fpc["neutral"] - fpc_c["neutral"],
            "delta_f1_positive": fpc["positive"] - fpc_c["positive"],
        })

    for h in handles:
        del h
    del predict
    _cleanup()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/robustness_results.csv")
    args = ap.parse_args()

    texts, y_true = load_test()
    if args.smoke:
        texts, y_true = texts[:40], y_true[:40]
        conditions = ["limpio", "sin_tildes"]
        out_path = ROOT / "results" / "sprint6" / "robustness_smoke.csv"
    else:
        conditions = ["limpio"] + CORE + OPTIONAL
        out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Test=%d ejemplos, condiciones=%s", len(y_true), conditions)

    all_rows = []
    # Escritura incremental: cabecera ya, filas por modelo (sobrevive a fallos)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
    for key in ["lora_1.7b", "lora_4b", "prompt_k4_4b", "beto", "xlmr"]:
        rows = evaluate_model(key, conditions, texts, y_true)
        all_rows.extend(rows)
        with out_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writerows(rows)
        logger.info("  [%s] %d filas escritas en %s", key, len(rows), out_path)

    # Resumen en consola: caída agregada por modelo × perturbación (solo nucleares)
    print("\n" + "=" * 100)
    print("ROBUSTEZ — Δ Macro-F1 (perturbado − limpio) por modelo × perturbación (nucleares)")
    print("-" * 100)
    hdr = f"{'perturbación':<16}" + "".join(f"{MODEL_LABELS[k][:14]:>15}" for k in EXPECTED_CLEAN)
    print(hdr)
    print("-" * 100)
    by = {(r["model"], r["perturbation"]): r for r in all_rows}
    for cond in [c for c in conditions if c != "limpio" and c not in OPTIONAL]:
        line = f"{cond:<16}"
        for k in EXPECTED_CLEAN:
            r = by.get((k, cond))
            line += f"{r['delta_abs']:>+15.4f}" if r else f"{'—':>15}"
        print(line)
    print("=" * 100)
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()
