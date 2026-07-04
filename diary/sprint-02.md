# Diario — Sprint 2: Infraestructura y Baselines

**Fecha de inicio:** 2026-06-04
**Fecha de cierre:** 2026-06-05 (cierre provisional — T4 y xlm-roberta-base pendientes de ejecución en Colab)
**Sprint:** 2 — Datos, baselines y harness de evaluación
**Presupuesto del sprint:** ~35 h de 180 h totales
**Elaborado por:** memoir-writer (claude-sonnet-4-6)

---

## Objetivo del sprint

Tener el pipeline de datos operativo, el harness de evaluación completo, y todos los
baselines ejecutados (o en estado bloqueado-por-GPU documentado). Al cerrar, la rejilla
experimental del Sprint 3 debe poder poblarse con resultados reales en cuanto lleguen
los runs de Colab.

**Estado al cierre:** Parcialmente cerrado — checkpoint, NO cierre definitivo. T1–T3
completadas; T5 parcial (BETO ejecutado a 1 época CPU provisional, xlm-roberta-base
pendiente); T4 bloqueado por falta de GPU local. T6 (qa-validator) ejecutado tras añadir
`tests/test_pipeline.py` y corregir 3 asserts: **89 passed, 1 skipped** (el único skip es
`test_xlmr_trainable_params_not_none`, condicionado al placeholder pendiente de Colab —
pasará a PASS automáticamente con el JSON definitivo). T7 completo.

---

## Estado de cada tarea

| # | Tarea | Agente | Estado |
|---|-------|--------|--------|
| T1 | Pipeline de datos Cardiff ES + fracciones | ml-dev | **Completo** |
| T2 | Harness de evaluación (ExperimentResult, métricas, CostTracker) | ml-dev | **Completo** |
| T3 | Baseline TF-IDF + LR | ml-dev | **Completo** — F1 macro = 0.5665 |
| T4 | Prompting baseline zero-shot + few-shot k=4,8,16 (Qwen3-1.7B) | ml-dev | **Pendiente GPU** (código listo, ejecución en Colab) |
| T5 | Encoder baselines fine-tuned: BETO + xlm-roberta-base | ml-dev | **Parcial**: BETO 1 época CPU (provisional); xlm-roberta-base pendiente Colab |
| T6 | qa-validator: tests del pipeline, harness, anti-leakage | qa-validator | **Completo** — 89 passed, 1 skipped |
| T7 | Diario + actualizar secciones del paper | memoir-writer | **Completo** (este fichero) |

### Nota de uso de IA

Todos los agentes son instancias de **claude-sonnet-4-6** (Anthropic) operando como
subagentes dentro del sistema de orquestación de Claude Code. Uso declarado de forma
explícita por sprint (política: ilimitado con declaración de integridad).

| Agente | Rol en este sprint |
|--------|--------------------|
| ml-dev | Pipeline de datos, harness, baselines TF-IDF, encoder y prompting |
| research-scout | Verificación de fuga XLM-T, corrección .bib (ustun2020udapter, eliminación garciapablos2024qlora) |
| qa-validator | Auditoría de código y tests (pytest: 89 passed, 1 skipped — añadido `test_pipeline.py` y corregidos 3 asserts con nombres de clave incorrectos) |
| memoir-writer | Diario sprint-02.md + secciones Dataset, Evaluation Protocol y Baselines en paper |

---

## Resultados disponibles al cierre

| Modelo / Método | F1 macro (Cardiff ES test) | Notas |
|---|---|---|
| TF-IDF + LR | **0.5665** | Definitivo. Suelo de la curva calidad-coste. |
| BETO fine-tuned (1 época, CPU) | **0.6216** | Provisional — re-ejecutar 3 épocas en Colab |
| pysentimiento/robertuito off-the-shelf | **0.7582** | Out-of-domain (TASS 2020). Referencia, NO en la Pareto B. |
| Qwen3-1.7B zero-shot / few-shot k=4,8,16 | PENDING | Código listo; ejecución en Colab. |
| xlm-roberta-base fine-tuned | PENDING | Placeholder en disco; ejecución en Colab. |

---

## Decisiones tomadas

### Dataset: Cardiff ES como corpus principal (no TASS)

Cardiff NLP `tweet_sentiment_multilingual` (ES) elegido por: acceso libre inmediato (CC BY 4.0),
splits fijos predefinidos, y uso establecido en la literatura. Tamaños reales del dataset:
- **Train:** 1 839 ejemplos (613 negative, 613 neutral, 613 positive — perfectamente balanceado)
- **Validation:** 324 ejemplos (108 por clase)
- **Test:** 870 ejemplos (290 por clase)

El dataset es más pequeño de lo estimado (~7k). Esto refuerza la pertinencia del corte A:
con 1 839 ejemplos de train, las fracciones n=50,100,250,500,1000 cubren 2.7% a 54.4%.

TASS 2020 se mantiene como candidato para validación cruzada futura (si llega la licencia SEPLN).

### Exclusión de cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

Fuga de distribución confirmada por research-scout (ver `docs/research/xlmt_leakage_check.md`):
el modelo fue fine-tuned sobre el train+dev del mismo dataset que usamos como test. Excluido.
Sustituido por xlm-roberta-base afinado por nosotros.

### Encoders controlados para la Pareto B

- **BETO** (`dccuchile/bert-base-spanish-wwm-cased`): fine-tuned sobre Cardiff ES train.
- **xlm-roberta-base**: fine-tuned sobre Cardiff ES train.
- **pysentimiento/robertuito**: referencia out-of-domain (TASS 2020), NO en la Pareto.

### Protocolo de prompting congelado

Definido en `configs/prompting_protocol.yaml`: Qwen3-1.7B, zero-shot y few-shot k=4,8,16,
fallback="neutral", registro de fallback_rate. El mismo protocolo se aplica a Qwen3-4B
en Sprint 3.

### .bib: correcciones del research-scout

- `ustun2022udapter` → `ustun2020udapter` (año y venue EMNLP 2020 confirmados).
- `garciapablos2024qlora` → eliminado (no verificable sin acceso web; reemplazar cuando
  se encuentre la referencia exacta en la revista PLN/IberLEF 2024).

---

## Riesgos activos

| Riesgo | Impacto | Estado |
|---|---|---|
| TASS InterTASS (licencia SEPLN) | ALTO — corte C | Usuario inició trámite 2026-06-05; sin noticias |
| BETO provisional (1 época) | MEDIO — Pareto B | Pendiente re-ejecución en Colab (3 épocas) |
| T4 prompting sin ejecutar | MEDIO — corte A | Pendiente Colab |
| Confirmación David Contreras | ALTO — integridad | Pendiente del Sprint 1 |

---

## Artefactos producidos en el sprint

| Fichero | Agente | Descripción |
|---|---|---|
| `src/data/load_data.py` | ml-dev | Pipeline Cardiff ES + fracciones |
| `data/processed/cardiff_es/` | ml-dev | Splits Arrow + metadata.json |
| `src/models/evaluate.py` | ml-dev | Harness completo |
| `src/models/baselines.py` | ml-dev | TF-IDF + LR |
| `src/models/encoder_baseline.py` | ml-dev | EncoderFineTuner + OffTheShelfEncoder |
| `src/models/prompting.py` | ml-dev | PromptingBaseline |
| `configs/prompting_protocol.yaml` | ml-dev | Protocolo congelado |
| `scripts/run_encoder_baselines.py` | ml-dev | Script para BETO+xlmr en Colab |
| `notebooks/sprint2_gpu_baselines.ipynb` | ml-dev | Notebook Colab para T4+T5 |
| `notebooks/README.md` | qa-validator | Instrucciones de lanzamiento Colab (notebook y scripts) + checklist de cierre |
| `tests/test_pipeline.py` | qa-validator | Tests de fracciones, anti-solapamiento, harness y metadata.json |
| `tests/test_encoder_baseline.py` | ml-dev | Tests encoder (mock + tiny model) |
| `tests/test_prompting.py` | ml-dev | Tests prompting (mock) |
| `results/baselines/tfidf_lr.json` | ml-dev | TF-IDF+LR — F1=0.5665 |
| `results/baselines/beto_cardiff_es.json` | ml-dev | BETO 1 época CPU — F1=0.6216 (provisional) |
| `results/baselines/robertuito_offtheshelf.json` | ml-dev | Robertuito OOD — F1=0.7582 |
| `results/baselines/xlmr_base_cardiff_es.json` | ml-dev | Placeholder (ejecución Colab) |
| `results/baselines/prompting_zeroshot_PENDING.json` | ml-dev | Placeholder prompting (Colab) |
| `docs/research/xlmt_leakage_check.md` | research-scout | Evidencia fuga XLM-T |
| `docs/research/references.bib` | research-scout | ustun corregido, garciapablos eliminado |
| `diary/sprint-02.md` | memoir-writer | Este fichero |
| `paper/main.tex` | memoir-writer | Dataset, Evaluation Protocol y Baselines actualizados |

---

## Próximos pasos — cierre definitivo

**Para cerrar Sprint 2:**
1. Usuario ejecuta `notebooks/sprint2_gpu_baselines.ipynb` en Colab T4.
2. Copia los JSONs de vuelta a `results/baselines/`.
3. qa-validator hace pase final (xlmr JSON + resultados prompting).
4. memoir-writer actualiza tablas en `paper/main.tex` con F1 definitivos.
5. Commit de cierre + push.

**Sprint 3 (tras cierre):** LoRA/QLoRA sobre Qwen3-1.7B y 4B, curva de aprendizaje
(corte A), frontera de Pareto (corte B), redacción de resultados.

---

*Cierre provisional: 2026-06-05 — memoir-writer (claude-sonnet-4-6)*

---

## Fijación de entorno — 2026-06-09 (post-sprint, auditoría de reproducibilidad)

**Problema detectado:** La DGX tenía `datasets==4.1.1` instalado. A partir de la serie
4.x, los datasets con script de carga propio (como `cardiffnlp/tweet_sentiment_multilingual`)
dejan de ejecutarse sin `trust_remote_code=True` explícito. El portátil funcionaba porque
`requirements.txt` usaba `datasets>=2.19.0` y la instalación local había resuelto una
versión anterior.

**Cambios aplicados:**

| Fichero | Cambio |
|---|---|
| `requirements.txt` | `datasets>=2.19.0` → `datasets==4.1.1` (pin exacto) |
| `src/data/load_data.py` | `load_dataset(…)` → añadido `trust_remote_code=True` |

**Motivación (reproducibilidad):** Fijar la versión exacta de `datasets` garantiza que
portátil, DGX y cualquier CI futura resuelvan el mismo comportamiento del loader. El
parámetro `trust_remote_code=True` es requerido por datasets ≥ 3.x para scripts de carga
de terceros; sin él la llamada devuelve un `DatasetNotFoundError` silencioso en 4.x.

*Entrada añadida: 2026-06-09 — claude-sonnet-4-6*

---

## Loader sin script: migración al export Parquet del Hub — 2026-06-20 (corrección definitiva)

**Problema detectado:** El parche del 2026-06-09 (fijar `datasets==4.1.1` + `trust_remote_code=True`)
**dejó de funcionar**. En `datasets` 4.x el soporte de scripts de carga se eliminó por completo
y `trust_remote_code` ya no se acepta: cargar `cardiffnlp/tweet_sentiment_multilingual` aborta con

```
RuntimeError: Dataset scripts are no longer supported, but found tweet_sentiment_multilingual.py
```

Es decir, el dataset es de los basados en script y el enfoque anterior (pin de versión +
`trust_remote_code`) era una solución muerta: cualquier `datasets` ≥ 4.0 lo rompe.

**Arreglo definitivo:** Cargar el **export Parquet automático** que el Hub mantiene para todos
los datasets en la rama `refs/convert/parquet`, en lugar del script original. Es un origen sin
código ejecutable y, por construcción, contiene los mismos ejemplos.

**Cambios aplicados:**

| Fichero | Cambio |
|---|---|
| `src/data/load_data.py` | `load_dataset(name, subset, trust_remote_code=True)` → `load_dataset("parquet", data_files={split: "hf://datasets/<name>@refs/convert/parquet/<subset>/<split>/*.parquet"})`. Eliminado `trust_remote_code`. |
| `src/data/load_data.py` | `DataConfig` gana el campo `revision="refs/convert/parquet"` (configurable para enchufar otros datasets/ramas). |
| `requirements.txt` | `datasets==4.1.1` → `datasets>=2.18.0` (ya no hace falta fijar versión; el loader es agnóstico y compatible con 4.x). |

**Por qué Parquet del Hub (opción a) y no los ficheros de GitHub xlm-t (opción b):** el export
Parquet es el **mismo dato** que el script generaba, re-servido por el propio Hub, así que
preserva splits, orden de clases y ejemplos sin reconstruir nada. Los ficheros crudos de
`cardiffnlp/xlm-t` habrían exigido re-mapear etiquetas y re-derivar splits, con riesgo de
desviarse de los baselines ya calculados.

**Validación (equivalencia exacta con lo previo — los baselines siguen válidos):**

| Split | n esperado | n obtenido | Reparto de clases (neg/neu/pos) |
|---|---|---|---|
| train | 1839 | **1839** | 613 / 613 / 613 ✓ |
| validation | 324 | **324** | 108 / 108 / 108 ✓ |
| test | 870 | **870** | 290 / 290 / 290 ✓ |

- Etiquetas restauradas como `ClassLabel(['negative','neutral','positive'])` → 0/1/2 (idéntico a `LABEL2ID`).
- `make_fraction_subsets` sigue produciendo n=50/100/250/500/1000 (+full) estratificados.
- qa-validator añadió `tests/test_qa_cardiff_parquet_loader.py` (37 tests sobre datos reales del Hub, con skip elegante si no hay red): tamaños exactos, distribución por clase, dtype int, orden de ClassLabel, fracciones y anti-leakage entre splits. **37/37 passed**.
- Suite completa: **122 passed, 5 skipped** (los skips son los tests de `metadata.json`, que requieren datos procesados en disco).

*Entrada añadida: 2026-06-20 — claude-opus-4-8 (ml-dev)*
