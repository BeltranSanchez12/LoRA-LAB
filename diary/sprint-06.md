# Diario — Sprint 6: Robustez al ruido del español de redes (extensión de defensa)

**Fecha de cierre:** 2026-07-01
**Rama:** `main`
**Sprint:** 6 — Extensiones (material de apéndice/defensa, no cuerpo del paper)
**Agente principal:** Claude Code (claude-opus-4-8); QA de cierre por qa-validator (rol)

---

## Objetivo del sprint

Sin tocar el paper congelado, generar un **hallazgo nuevo y defendible** sobre la robustez de
cada método ante el ruido real del español de redes (T1–T3), entregado como apéndice/defensa.
El explorador interactivo (T4) y la demo (T5) quedan **aparte / pendientes** (ver abajo).

## Regla transversal (auto-auditada en cada gate)

- **El paper NO se toca** en todo el sprint (`git diff` contra `paper/` = vacío; también intactos
  `results/all_results.csv`, `results/baselines/`, `summary_corte*.csv` y los JSON del grid).
- **Re-inferencia pura:** cero reentrenamiento sobre texto perturbado. Lo único que cambia es el
  texto de entrada en inferencia; los pesos quedan congelados.
- **Cero datos inventados:** cada cifra de la figura y del análisis sale de un CSV del repo.
- **Honestidad por encima del titular:** si el prompting/encoders aguantan mejor, se reporta tal cual.

---

## Dependencia — el blocker de los pesos y cómo se resolvió

Al arrancar, la premisa del sprint ("cargar los adaptadores y re-inferir") **no se sostenía**:

- `src/models/lora_finetune.py` entrenaba con `save_strategy="no"` y **nunca persistía el adaptador**
  (entrena → evalúa en memoria → escribe el JSON de métricas → descarta los pesos). Los directorios
  `results/checkpoints/lora_*` estaban **vacíos**.
- Los encoders (BETO/XLM-R) entrenaban a `/tmp/...` (ya purgado): **tampoco había checkpoints**.
- De los 5 modelos, solo **prompting** era re-inferible (carga el Qwen base cacheado).

**Parada + decisión del usuario → reconstrucción.** Solución (`scripts/sprint6_reconstruct.py`):

1. **Fix del guardado** (opt-in, sin cambiar el comportamiento del grid): `lora_finetune.py` guarda
   el adaptador final (mejor-val) si `output.save_adapter_dir`; `encoder_baseline.py` gana un
   método `save()`.
2. **Reconstrucción determinista** (semilla 42, `set_seed` → `use_deterministic_algorithms` +
   `cudnn.deterministic`) con **datos LIMPIOS** de los 4 modelos sin pesos, y re-inferencia de
   prompting k=4 (4-bit). Escribe SOLO en `results/sprint6/` (evidencia del paper intacta).
3. **Sanity check:** los 5 reproducen el F1 limpio del paper **BIT-EXACTO** (Δ = +0.000000):
   LoRA-1.7B 0.6922 · LoRA-4B 0.7220 · Prompting-k4 0.6522 · BETO 0.6613 · XLM-R 0.6465
   (`results/sprint6/sanity_check.csv`).

Como los adaptadores **regeneran bit-exactos** desde el script, no se versionan los binarios
(25M+47M): quedan **gitignored con el comando de regeneración documentado** en `.gitignore`
(`python scripts/sprint6_reconstruct.py`). Encoders (420M/1.1G) locales.

---

## GATE 1 — Taxonomía de perturbaciones (T1) [PARADA + validación]

`src/data/perturbations.py`: 7 perturbaciones nucleares + 1 opcional, cada una **aislada** y
**determinista** (semilla 42; las de ruido usan un RNG sembrado por-texto, independiente del orden).
Cobertura sobre el test (n=870), reproducible con `python -m src.data.perturbations`:

| Perturbación | % afect. | | Perturbación | % afect. |
|---|---|---|---|---|
| sin_tildes | 65.9 % | | alargamientos | 95.4 % |
| sin_emojis | 3.1 % | | abrev_chat | 62.5 % |
| minusculas | 78.2 % | | sin_puntuacion | 71.6 % |
| mayusculas | 100.0 % | | code_switching *(opc.)* | 37.2 % |

**Decisiones aprobadas:** (1) las 7 nucleares tal cual; (2) `sin_puntuacion` **preserva** `@user`/
`#hashtags` (placeholders vistos en train; quitarlos metería un cambio de distribución ajeno);
(3) `code_switching` **opcional, fuera de la figura**, reportado aparte; (4) baseline = limpio
**semilla-42** reconstruido; (5) adaptadores gitignored (regeneran).

## GATE 2 — Prueba de robustez + figura + análisis (T2, T3)

- **T2** (`scripts/sprint6_robustness.py`): re-evalúa los 5 modelos (**pesos congelados**) en los 7
  sets perturbados + `code_switching` (aparte). El limpio reproduce el baseline **bit-exacto** en los
  5 → las caídas son fiables. `results/robustness_results.csv` (45 filas = 5 × 9): Macro-F1, **F1 por
  clase (neg/neu/pos)**, Δ absoluta y relativa vs limpio, y **Δ por clase**.
- **T3** (`scripts/sprint6_robustness_figure.py` → `results/figures/figF_robustness.{png,pdf}`):
  heatmap Δ Macro-F1 (7 pert × 5 modelos + fila media) de **un solo panel**. El ángulo por-clase
  se reporta en `results/sprint6/robustness_analysis.md` (donde el artefacto de la neutral se
  etiqueta sin ambigüedad visual). Ajuste tras revisión: se eliminó el panel por-clase porque su
  celda neutral de BETO (+0.006) daba en verde la falsa impresión de "mejora".

### Hallazgos (honestos)

- **Titular — colapso *cased* ante mayúsculas.** El simple mayusculado hunde a los codificadores:
  **BETO −0.184**, **XLM-R −0.110**, mientras los generativos apenas se inmutan (LoRA ≈ −0.011;
  prompting **+0.004**, inmune). Explicación mecánica: BETO es `...-cased`; TODO EN MAYÚSCULAS es
  fuertemente OOD para su tokenización.
- **Fragilidad concentrada, no difusa.** La media pésima de los encoders está dominada por ese
  único evento: **sin mayúsculas, BETO queda en +0.003** (robusto al resto) y XLM-R en −0.011.
- **"LoRA más robusto" se sostiene EN MEDIA** (LoRA-4B −0.006, LoRA-1.7B −0.007 > prompting −0.014
  ≫ BETO −0.024 ≈ XLM-R −0.025) **pero no es la historia**; el prompting es **más frágil que LoRA**
  a `sin_tildes`/`minusculas`/`sin_puntuacion`.
- **Hipótesis de la neutral REFUTADA.** El ruido no ataca preferentemente a la neutral; la clase más
  frágil es la **positiva**. El +0.006 de la neutral en BETO es un **artefacto** del colapso
  (bajo mayúsculas: positiva 0.732→0.375 = −0.357, negativa 0.702→0.516, neutral casi plana
  0.550→0.541): el clasificador degradado vierte predicciones al cajón "neutral".
- **code_switching (opcional, aparte):** no degrada a la mayoría (LoRA +0.006/+0.002, XLM-R +0.005,
  BETO −0.001); solo el few-shot prompting cae (−0.023).

## GATE 3 — QA + cierre (T6, este documento)

**Veredicto qa-validator: PASS.** Checklist verificado contra `results/robustness_results.csv`:

1. **Trazabilidad** — titulares OK: BETO mayúsculas −0.1841, XLM-R −0.1099; artefacto por clase de
   BETO (pos 0.7322→0.3750, neg 0.7017→0.5159, neu 0.5500→0.5408); medias con/sin mayúsculas
   (BETO −0.0239/+0.0028; XLM-R −0.0248/−0.0106). Cada cifra del análisis y de figF sale del CSV.
2. **Reproducibilidad** — sanity limpio **bit-exacto** en los 5 (`sanity_check.csv` Δ=0.0, in_band);
   el clean de T2 reproduce bit-exacto; adaptadores gitignored + comando de regeneración en
   `.gitignore`.
3. **Paper intacto** — `git diff 2a7b59f..HEAD -- paper/` = **vacío**; evidencia canónica sin cambios.
4. **code_switching** — 5 filas, todas `optional=True`, y **fuera** de la figura (solo 7 nucleares).

---

## Pendientes explícitos (fuera de este cierre)

- **T4 — Explorador interactivo (HTML):** 3 vistas cableadas a los CSVs reales. Lo monta Beltrán +
  Claude **en paralelo**; no forma parte de T6.
- **T5 — Demo en vivo:** opcional, va después.

## Trazabilidad de uso de IA (declaración de integridad, CE37/RA5)

- **Reconstrucción de pesos y robustez:** Claude Code (claude-opus-4-8) mediante scripts deterministas
  (`sprint6_reconstruct.py`, `sprint6_robustness.py`, `sprint6_robustness_figure.py`); gates con
  parada y validación humana en la dependencia (decisión de reconstruir), la taxonomía (T1) y los
  resultados (T2/T3), incluidos ajustes de mensaje pedidos por el usuario.
- **Cifras:** todas trazadas a `results/robustness_results.csv` (cero datos inventados); baseline
  limpio reproducido bit-exacto contra el paper antes de medir caídas.
- **QA de cierre:** rol qa-validator (auditoría read-only; no modifica el paper ni el código fuente).
- El **paper no se alteró**: este sprint es material de apéndice/defensa derivado de re-inferencia.

## Estado final

T1–T3 cerrados; T6 (QA + diario) cerrado. Commits del sprint: `644ff88` (reconstrucción + sanity +
taxonomía), `dca1c65` (identidad git DGX), `b2827d8` (T2+T3), `5257ab0` (ajustes T3) + este diario.
`git` limpio, push a `origin/main`. **Paper CONGELADO e intacto.** T4/T5 pendientes explícitos.
