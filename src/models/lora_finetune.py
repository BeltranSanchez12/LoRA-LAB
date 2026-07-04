"""
lora_finetune.py — Bucle de fine-tuning PEFT parametrizado por config (Sprint 3, T1).

Un experimento = un fichero YAML = {modelo} × {método} × {fracción} × {semilla}.
Métodos soportados: lora | qlora | full_ft.

El modelo generativo se entrena por SFT para emitir la etiqueta (negative/neutral/
positive) dado el MISMO prompt de chat que usa el baseline de prompting (system
prompt + answer cue + chat template, thinking off). La loss se aplica solo sobre
los tokens de la etiqueta (el prompt se enmascara con -100).

Política de entrenamiento justo (Sprint 3):
  - Mínimo de pasos en fracciones pequeñas: max_steps = max(min_steps, epochs*steps/época),
    para que 10/16/25/50 ejemplos no se queden cortos.
  - Se selecciona el MEJOR checkpoint por F1-macro de VALIDACIÓN (no el último),
    evaluado con el mismo harness generativo.

El eval final reutiliza EXACTAMENTE el harness de los baselines, sobre el test COMPLETO:
  - genera + parsea con PromptingBaseline (k=0)  -> mismo parser, mismo fallback
  - compute_metrics -> F1-macro / accuracy / F1 por clase
  - coste: trainable_params, peak_vram_gb, train_time_s, inference_latency_ms

Reanudable: si ya existe el JSON del experimento, se omite.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_experiment_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fraction_dir_name(fraction: Any) -> str:
    """Mapea la fracción (10/16/.../"full") al nombre de carpeta en disco."""
    if str(fraction).lower() in ("full", "all", "1.0", "1"):
        return "n_full"
    return f"n_{int(fraction)}"


def _is_full_fraction(fraction: Any) -> bool:
    return str(fraction).lower() in ("full", "all", "1.0", "1")


def _resolve_fraction_path(processed: Path, fraction: Any, seed: int) -> Path:
    """Resuelve la ruta al subconjunto de train para (fracción, semilla).

    Layout por semilla del grid de 3 semillas:
        train_fractions/n_{frac}/s{seed}   (fracciones pequeñas, re-muestreadas)
        train_fractions/n_full             (semilla-agnóstico: todo el train)

    Compatibilidad: si no existe la ruta por-semilla, cae al layout plano
    `train_fractions/n_{frac}` del piloto de 1 semilla.
    """
    root = processed / "train_fractions"
    if _is_full_fraction(fraction):
        return root / "n_full"
    per_seed = root / f"n_{int(fraction)}" / f"s{int(seed)}"
    if per_seed.exists():
        return per_seed
    legacy = root / f"n_{int(fraction)}"
    if legacy.exists():
        logger.warning(
            "No hay subconjunto por-semilla en %s; uso layout plano %s (sin varianza de muestreo).",
            per_seed, legacy,
        )
        return legacy
    raise FileNotFoundError(
        f"No existe subconjunto para fracción={fraction}, seed={seed}: "
        f"ni {per_seed} ni {legacy}. Ejecuta scripts/build_fraction_subsets.py."
    )


# ---------------------------------------------------------------------------
# Datos: construir ejemplos tokenizados (prompt enmascarado, loss en la etiqueta)
# ---------------------------------------------------------------------------

def build_tokenized_dataset(train_ds, prompting_baseline, tokenizer, max_length: int):
    """Convierte (text,label) -> {input_ids, labels} con el prompt enmascarado.

    input_ids = prompt_ids + label_ids + [eos]
    labels    = [-100]*len(prompt_ids) + label_ids + [eos]
    El prompt se construye con el MISMO render que el baseline de prompting.
    """
    eos_id = tokenizer.eos_token_id
    examples = []
    texts = list(train_ds["text"])
    labels = list(train_ds["label"])
    for text, lab in zip(texts, labels):
        prompt_str = prompting_baseline.render_prompt(text, [], tokenizer)
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        label_word = ID2LABEL[int(lab)]
        target_ids = tokenizer(label_word, add_special_tokens=False)["input_ids"] + [eos_id]

        input_ids = (prompt_ids + target_ids)[:max_length]
        n_prompt = min(len(prompt_ids), len(input_ids))
        label_field = [-100] * n_prompt + input_ids[n_prompt:]
        examples.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": label_field,
        })
    return examples


@dataclass
class PadCollator:
    """Colación dinámica: pad de input_ids/attention_mask y labels (-100)."""
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(base_model: str, method: str, peft_cfg: dict):
    """Carga el modelo según el método y le aplica LoRA si procede.

    method: "lora" | "qlora" | "full_ft"
    Retorna (model, tokenizer, trainable_params).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common = dict(trust_remote_code=True)
    if method == "qlora":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map={"": 0}, **common,
        )
        model = prepare_model_for_kbit_training(model)
    else:  # lora | full_ft
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map={"": 0}, **common,
        )

    if method in ("lora", "qlora"):
        lcfg = LoraConfig(
            r=peft_cfg.get("r", 16),
            lora_alpha=peft_cfg.get("lora_alpha", 32),
            lora_dropout=peft_cfg.get("lora_dropout", 0.05),
            target_modules=peft_cfg.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tokenizer, int(trainable)


# ---------------------------------------------------------------------------
# Eval generativo (mismo harness que los baselines)
# ---------------------------------------------------------------------------

def _eval_generative(pb, model, tokenizer, texts, y_true, device="cuda"):
    """Genera+parsea con PromptingBaseline y calcula métricas. Devuelve (metrics, fallback)."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.evaluate import compute_metrics, LABEL_NAMES

    preds_str, fallback = pb.predict_batch(texts, model, tokenizer, examples=[], device=device)
    y_pred = [LABEL2ID.get(p, 1) for p in preds_str]
    metrics = compute_metrics(y_true, y_pred, label_names=LABEL_NAMES)
    return metrics, fallback


def _make_best_val_callback(pb, tokenizer, val_texts, val_true, eval_steps, device="cuda"):
    """Crea un TrainerCallback que evalúa F1-macro de validación y guarda el mejor
    estado (parámetros entrenables) en CPU. Restaurable tras el entrenamiento."""
    from transformers import TrainerCallback

    class BestValF1Callback(TrainerCallback):
        def __init__(self):
            self.best_f1 = -1.0
            self.best_step = -1
            self.best_state = None
            self.history: list[tuple[int, float, float]] = []

        def _evaluate(self, model, step):
            import torch
            was_training = model.training
            prev_cache = getattr(model.config, "use_cache", True)
            model.config.use_cache = True
            model.eval()
            metrics, fb = _eval_generative(pb, model, tokenizer, val_texts, val_true, device)
            f1 = metrics["f1_macro"]
            self.history.append((step, f1, fb))
            logger.info("  [val] step=%d F1-macro=%.4f fallback=%.3f", step, f1, fb)
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_step = step
                self.best_state = {
                    n: p.detach().cpu().clone()
                    for n, p in model.named_parameters() if p.requires_grad
                }
            model.config.use_cache = prev_cache
            if was_training:
                model.train()

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % eval_steps == 0:
                self._evaluate(kwargs["model"], state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            # Asegura una evaluación en el último paso.
            if not self.history or self.history[-1][0] != state.global_step:
                self._evaluate(kwargs["model"], state.global_step)

    return BestValF1Callback()


def _restore_best(model, best_state):
    import torch
    if not best_state:
        return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n].to(p.device))


# ---------------------------------------------------------------------------
# Experimento completo: train + eval + coste
# ---------------------------------------------------------------------------

def run_experiment(config: dict, results_dir: Optional[str] = None) -> dict:
    """Ejecuta un experimento PEFT completo y persiste el resultado."""
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    from datasets import load_from_disk

    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.prompting import PromptingBaseline
    from src.models.evaluate import (
        ExperimentResult, save_result,
        reset_peak_vram, get_peak_vram_gb, measure_inference_latency,
    )

    exp = config["experiment"]
    mcfg = config["model"]
    dcfg = config["data"]
    tcfg = config["training"]
    ecfg = config.get("eval", {})
    seed = int(exp.get("seed", 42))
    method = mcfg["method"]
    base_model = mcfg["base_model"]
    fraction = dcfg.get("train_fraction", "full")
    peft_cfg = mcfg.get("peft", config.get("peft", {}))
    results_dir = results_dir or config.get("output", {}).get("results_dir", "results/")

    experiment_name = exp["name"]
    json_path = Path(results_dir).resolve() / f"{experiment_name}.json"
    if json_path.exists() and not exp.get("overwrite", False):
        logger.info("Experimento ya existe (%s); se omite (reanudable).", json_path)
        return json.loads(json_path.read_text())

    set_seed(seed)

    # --- Datos ---
    # processed_dir = dominio FUENTE (train + validation para selección de checkpoint).
    # eval_dir (opcional) = dominio OBJETIVO sobre cuyo split `test` se evalúa.
    # Si eval_dir no se indica, fuente y objetivo coinciden (comportamiento del Corte A/B).
    # Esto habilita la matriz de transferencia del Corte C: train en país A, eval en país B.
    processed = PROJECT_ROOT / dcfg.get("processed_dir", "data/processed/cardiff_es")
    eval_processed = PROJECT_ROOT / dcfg["eval_dir"] if dcfg.get("eval_dir") else processed
    frac_path = _resolve_fraction_path(processed, fraction, seed)
    logger.info("Subconjunto de train: %s (fracción=%s, seed=%d)", frac_path, fraction, seed)
    logger.info("Validación (selección de checkpoint): %s", processed / "validation")
    logger.info("Evaluación (test objetivo): %s", eval_processed / "test")
    train_ds = load_from_disk(str(frac_path))
    val_ds = load_from_disk(str(processed / "validation"))
    test_ds = load_from_disk(str(eval_processed / "test"))
    eval_max = ecfg.get("max_examples")
    if eval_max:
        test_ds = test_ds.select(range(min(int(eval_max), len(test_ds))))

    proto_path = PROJECT_ROOT / "configs" / "prompting_protocol.yaml"
    pb = PromptingBaseline(config_path=proto_path, k=0)

    # --- Modelo ---
    model, tokenizer, trainable_params = load_model_and_tokenizer(base_model, method, peft_cfg)
    logger.info("Método=%s, params entrenables=%d", method, trainable_params)

    max_length = int(tcfg.get("max_length", 256))
    train_examples = build_tokenized_dataset(train_ds, pb, tokenizer, max_length)

    # --- Política de pasos: mínimo en fracciones pequeñas ---
    batch = int(tcfg.get("batch_size", 8))
    grad_accum = int(tcfg.get("grad_accum", 1))
    epochs = float(tcfg.get("epochs", 5))
    min_steps = int(tcfg.get("min_steps", 80))
    steps_per_epoch = max(1, math.ceil(len(train_examples) / (batch * grad_accum)))
    max_steps = max(min_steps, int(math.ceil(epochs * steps_per_epoch)))
    eval_points = int(tcfg.get("eval_points", 5))
    eval_steps = max(1, max_steps // eval_points)
    logger.info("n_train=%d steps/época=%d -> max_steps=%d (min_steps=%d), eval cada %d pasos",
                len(train_examples), steps_per_epoch, max_steps, min_steps, eval_steps)

    grad_ckpt = bool(tcfg.get("gradient_checkpointing", method == "qlora"))
    out_dir = Path(results_dir).resolve() / "checkpoints" / experiment_name
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.1)),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        max_steps=max_steps,
        logging_steps=max(1, eval_steps),
        save_strategy="no",
        seed=seed,
        bf16=True,
        report_to=[],
        gradient_checkpointing=grad_ckpt,
    )
    if grad_ckpt:
        model.config.use_cache = False

    # Callback de mejor-checkpoint por F1-macro de validación
    val_texts = list(val_ds["text"])
    val_true = [int(x) for x in val_ds["label"]]
    best_cb = _make_best_val_callback(pb, tokenizer, val_texts, val_true, eval_steps)

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_examples,
        data_collator=collator,
        callbacks=[best_cb],
    )

    reset_peak_vram()
    train_out = trainer.train()
    peak_vram = get_peak_vram_gb()

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    loss_first = losses[0] if losses else None
    loss_last = losses[-1] if losses else None
    train_time_s = float(train_out.metrics.get("train_runtime", 0.0))

    # Restaurar el MEJOR checkpoint por F1-macro de validación
    _restore_best(model, best_cb.best_state)
    logger.info("Restaurado mejor checkpoint: step=%d val_F1=%.4f",
                best_cb.best_step, best_cb.best_f1)

    # Persistencia opcional del adaptador final (mejor-val). save_strategy="no" evita
    # checkpoints intermedios; aquí guardamos SOLO los pesos finales cuando el config lo pide
    # (output.save_adapter_dir). Necesario para re-inferencia sin reentrenar (Sprint 6).
    save_adapter_dir = config.get("output", {}).get("save_adapter_dir")
    if save_adapter_dir and method in ("lora", "qlora"):
        adapter_path = Path(save_adapter_dir)
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        adapter_path = adapter_path / experiment_name
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_path))
        logger.info("Adaptador final guardado en %s", adapter_path)

    # --- Eval final sobre el test COMPLETO con el harness de baselines ---
    model.config.use_cache = True
    model.eval()
    test_texts = list(test_ds["text"])
    y_true = [int(x) for x in test_ds["label"]]
    metrics, fallback_rate = _eval_generative(pb, model, tokenizer, test_texts, y_true)

    def _predict_one(batch_texts):
        out, _ = pb.predict_batch(batch_texts, model, tokenizer, examples=[], device="cuda")
        return out
    latency_ms = measure_inference_latency(_predict_one, test_texts[: min(60, len(test_texts))], n_warmup=3)

    result = ExperimentResult(
        experiment_name=experiment_name,
        model_name=base_model,
        method=method,
        data_fraction=fraction if str(fraction) != "full" else "full",
        seed=seed,
        f1_macro=metrics["f1_macro"],
        accuracy=metrics["accuracy"],
        f1_per_class=metrics["f1_per_class"],
        trainable_params=trainable_params,
        peak_vram_gb=peak_vram,
        train_time_s=train_time_s,
        inference_latency_ms=latency_ms,
        fallback_rate=fallback_rate,
        notes=(
            f"PEFT {method} sobre {base_model}, fracción={fraction}, seed={seed}. "
            f"r={peft_cfg.get('r')}, lr={tcfg.get('learning_rate')}, "
            f"max_steps={max_steps} (min_steps={min_steps}). "
            f"best_val_F1={best_cb.best_f1:.4f}@step{best_cb.best_step}. "
            f"loss {loss_first}->{loss_last}. "
            f"n_train={len(train_examples)}, n_test={len(test_texts)}. "
            f"train_dir={processed.name}, eval_dir={eval_processed.name}."
        ),
    )
    save_result(result, results_dir=results_dir)

    summary = result.to_json_dict()
    summary["loss_first"] = loss_first
    summary["loss_last"] = loss_last
    summary["best_val_f1"] = best_cb.best_f1
    summary["best_val_step"] = best_cb.best_step
    summary["val_history"] = best_cb.history
    return summary
