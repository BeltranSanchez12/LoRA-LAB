# Diario — Sprint 3: La rejilla central (cortes A y B)

**Fecha de inicio:** 2026-06-22
**Sprint:** 3 — Barrido PEFT (LoRA/QLoRA/full-FT) → figuras A (eficiencia de datos) y B (Pareto)
**Presupuesto del sprint:** ~40 h de 180 h totales

---

## Objetivo del sprint

Lanzar la rejilla `{modelo} × {método} × {fracción} × {semilla}` que produce las dos
figuras estrella (A: curva de eficiencia de datos con punto de cruce; B: frontera de
Pareto calidad–coste generativos-vs-encoders), reutilizando el harness del Sprint 2.

---

## T1 — Bucle de entrenamiento PEFT parametrizado por config + SMOKE TEST

**Estado:** infraestructura construida y validada con smoke test. **Parado en el
checkpoint humano antes de lanzar el grid completo** (disciplina del spec).

### Diseño

- **Un experimento = un YAML** (`configs/sprint3/...`): modelo, método (`lora|qlora|full_ft`),
  fracción de datos, semilla, hiperparámetros PEFT y de entrenamiento.
- **Entrenamiento** (`src/models/lora_finetune.py`): SFT en el que el modelo generativo
  aprende a emitir la etiqueta (`negative|neutral|positive`) dado el **mismo prompt de chat**
  que el baseline de prompting (system prompt + answer cue + chat template, thinking off).
  La loss se aplica **solo sobre los tokens de la etiqueta** (el prompt se enmascara a -100).
  `transformers.Trainer` + colación dinámica propia; LoRA/QLoRA vía `peft`; full-FT sin adapter.
- **Eval = MISMO harness que los baselines**: se reutiliza `PromptingBaseline(k=0)` para
  generar+parsear sobre el test, y `compute_metrics` para F1-macro/accuracy/F1 por clase.
  Esto garantiza comparabilidad directa con prompting y encoders, y un `fallback_rate` medido
  con el mismo parser robusto.
- **Coste**: `trainable_params`, `peak_vram_gb` (`reset/get_peak_vram`), `train_time_s`
  (del propio Trainer) y `inference_latency_ms` (`measure_inference_latency`).
- **Reanudable**: si ya existe `results/<name>.json` el experimento se omite; checkpoints a
  `results/checkpoints/` (gitignored).

### Smoke test (checkpoint del punto 1)

Config `configs/sprint3/smoke_lora_qwen1.7b_n50.yaml`: Qwen3-1.7B, LoRA (r=16,
q/k/v/o_proj), fracción **n=50**, 1 seed (42), `max_steps=30`, eval sobre 200 ejemplos de
test. Ejecutado en una H200 (GPU 0). Resultados:

| Señal | Valor |
|---|---|
| Adapter LoRA entrena | sí — `trainable_params` = 6 422 528 (~0.37 % de 1.7B) |
| La loss baja | 2.06 → 0.03 (train_loss medio 0.54) |
| Eval con el harness de baselines | F1-macro **0.624**, acc 0.625 |
| Parseable / fallback | `fallback_rate` = **0.000** |
| VRAM pico | 8.2 GB |
| Tiempo de entrenamiento | 3.5 s (30 pasos, n=50) |
| Latencia de inferencia | 55.6 ms/ejemplo |

El pipeline end-to-end queda validado: entrena, la loss baja, el eval es parseable con
fallback 0 y un F1-macro razonable (ya en el rango de los encoders), y se registran las
cuatro métricas de coste. **Se entregan estos números al checkpoint humano antes de
lanzar el grid completo.**

*Entrada añadida: 2026-06-22 — claude-opus-4-8 (ml-dev)*
