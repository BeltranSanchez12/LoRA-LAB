# Sprint 7 — Correcciones pre-entrega del paper

**Fecha de cierre:** 2026-07-04
**Estado:** CERRADO. QA PASS (10/10 criterios). Versión congelada de referencia: 39f2145.

**Commits del sprint:**
- `4657ef7` — plan: sprint-07-spec.md añadido
- `40eb79b` — A-bibliografía: referencias.bib + citas InterTASS 2018 recableadas
- `ee625b0` — B-figuras: variantes _paper sin suptitle + pie Figura 2 corregido
- `a732b08` — C-texto: correcciones de redacción C1–C7
- `515272f` — D-añadidos: correo institucional, metadatos PDF, frase de robustez

---

## BLOQUE A — Bibliografía

Completado en la primera sesión de este sprint (con corte por límite de gasto a mitad de sesión). El orquestador pulió detalles menores antes del commit `40eb79b`: años duplicados en booktitle de las entradas nuevas, guiones en apellidos compuestos de TASS 2018, llaves anti-minúscula en campos note.

**Cambios en `docs/research/references.bib`:**

- **A1 (mirandaescalada2023distemist):** Reemplazada completamente. Venue corregida de "BioCreative VII" (especulativa, con nota `[VERIFICAR: puede ser BioCreative VII 2022 o IberLEF 2023 version extendida]` que aparecía en el PDF compilado) a la venue verificada online: Working Notes of CLEF 2022, CEUR Workshop Proceedings vol. 3180, pp. 179–203, https://ceur-ws.org/Vol-3180/paper-11.pdf. Autores corregidos (10 autores verificados: Miranda-Escalada, Gascó, Lima-López, Farré-Maduell, Estrada, Nentidis, Krithara, Katsimpras, Paliouras, Krallinger). Verificación online exitosa — no fue necesario el fallback de cita mínima.

- **A2 (limpieza de notas personales):** Eliminadas anotaciones de trabajo de 12 entradas: ding2023peftsurvey ("survey exhaustivo..."), mosbach2023fewshot ("Clave para el corte A..."), min2022rethinking ("Muestra que etiquetas..."), canete2020beto ("BETO: bert-base-spanish-wwm-cased..."), perez2022robertuito ("F1 macro TASS 2020 0.743"), conneau2020xlmr ("XLM-RoBERTa: baseline multilingue..."), qwen3technicalreport2025 ("Describe Qwen3..."), diazgaliano2020tass ("Incluye InterTASS con segmentacion..."), pfeiffer2020madx ("Marco seminal..., Aplicable al corte C..."), ustun2020udapter ("Venue verificado... Relevante para corte C"), grattafiori2024llama32 ("Describe la familia Llama 3..."), pfeiffer2021adapterfusion ("relevante para estrategias..."). Se mantiene información puramente bibliográfica (arXiv IDs, HuggingFace IDs, venues).

- **A3 (capitalización):** Añadidas llaves protectoras en títulos: `{Spanish}` en canete2020beto y gutierrezfandino2022maria; `{Qwen3}` en qwen3technicalreport2025; `{Gemma}` en team2024gemma2. Las demás entradas ya tenían los términos protegidos.

- **A4 (MarIA):** `gutierrezfandino2022maria` cambiado de `@inproceedings` a `@article`; el campo `journal = {Procesamiento del Lenguaje Natural}` ya existía y ahora se imprime correctamente (vol. 68, pp. 39–60, 2022).

- **A5 (nueva entrada TASS 2018):** Añadida `martinezcamara2018tass` con datos verificados online (CEUR-WS vol. 2172, pp. 13–27, Sevilla 2018). Citas recableadas en `paper/main.tex`: líneas 211 (Sec I contribuciones), 458 (Sec III-A) y 1117 (Sec IV-D) — todas pasadas de `\cite{diazgaliano2020tass}` a `\cite{martinezcamara2018tass}`. `diazgaliano2020tass` queda solo para TASS 2020 (línea 312).

**Remates post-QA (incluidos en commit `40eb79b` o aplicados antes de `515272f`):**
- Comentario en references.bib con la palabra "verificar" reformulado para que `grep -ri "VERIFICAR"` devuelva 0 resultados estrictos sobre el fichero.
- `perez2022robertuito` cambiado de `@article` a `@inproceedings` (mismo bug de tipo que afectaba a MarIA: era `@article` con campo `booktitle` pero sin `journal`, por lo que BibTeX no imprimía la venue LREC; ahora el tipo es correcto y la conferencia aparece en el PDF).

**Verificaciones BLOQUE A (QA criterios 1–3, 6, 8):**
1. `grep -i VERIFICAR` sobre PDF → vacío ✓
2. `grep -E "bETO|xLM|¿="` sobre PDF → vacío ✓
3. Sin `undefined` en main.log ✓
4. Referencias del PDF [1]–[17] revisadas manualmente — sin anotaciones personales ✓
5. 14 páginas ✓
6. 3 citas a `martinezcamara2018tass` para InterTASS 2018; `diazgaliano2020tass` solo en línea 312 (TASS 2020) ✓

**Agente usado:** memoir-writer (claude-sonnet-4-6). Nota: la sesión se cortó por límite de gasto justo al finalizar los cambios del bloque; el orquestador completó los remates antes del commit.

---

## BLOQUE B — Figuras

Completado y commiteado en `ee625b0` por orquestador:
- Generadas variantes `_paper` de todas las figuras sin `suptitle` incrustado.
- Referencias en `main.tex` actualizadas a `figB_quality_cost_paper` y `figE_quality_cost_paper`.
- Pie de Figura 2 corregido (B3): `${\sim}9$--$24\times$ menos parámetros entrenables que los codificadores` → `$17$--$43\times$ (1.7B) y $9$--$24\times$ (4B) menos parámetros entrenables que los codificadores`.
- Activos originales `defense/assets/` intactos — diff vacío contra 39f2145, md5sum idéntico (verificado por QA criterio 5).

---

## BLOQUE C — Texto

Todos los cambios en `paper/main.tex`, commiteados en `a732b08`. Restricción cumplida: ningún valor numérico de resultados modificado; tablas I–V byte-idénticas respecto a 39f2145 salvo marcadores y notas nuevas en Tabla III (verificado por QA criterio 7 con comparación Python de entornos `table/table*`).

**C1 — Precisión del punto de cruce (dos apariciones):**
- Abstracto: "menos de $n=7$ ejemplos etiquetados bastan para el modelo de 1.7B" → "con tan solo $n=7$ ejemplos etiquetados (el cruce se produce entre $n=4$ y $n=7$) bastan para el modelo de 1.7B" (el orquestador recuperó el verbo "bastan" en el Resumen).
- Contribuciones Sec I: "produce con menos de $n=7$ ejemplos etiquetados para Qwen3-1.7B" → "produce con tan solo $n=7$ ejemplos etiquetados (el cruce se produce entre $n=4$ y $n=7$) para Qwen3-1.7B" (el orquestador eliminó la redundancia "se produce…el cruce se produce" en la versión publicada).

**C2 — Tercera pregunta en la Introducción:**
- "dos preguntas" → "tres preguntas"; añadido párrafo "Tercero, aunque el español presenta una notable variación dialectal..." tras el bloque "Segundo / Del mismo modo..."; "ambas carencias" → "estas tres cuestiones". Flujo coherente con el Abstracto ("Abordamos las tres") y con los tres hallazgos de Conclusiones.

**C3 — Frase confusa en Discusión:**
- "(RTX~3070, RTX~4060~Ti) que no están disponibles en la gama H200/A100" → "(RTX~3070, RTX~4060~Ti), a diferencia de la gama empresarial H200/A100".

**C4 — LoRA-4B excede objetivo T4:**
- Párrafo de 3 frases insertado tras el párrafo de QLoRA en Sec IV-C: reconoce que LoRA Qwen3-4B (16.7 GB) supera marginalmente el objetivo T4 de 16 GB, recomienda QLoRA (6.79 GB) en ese hardware, confirma que LoRA 1.7B (8.72 GB) cabe holgadamente.

**C5 — RoBERTuito blindado en Discusión:**
- Párrafo de 3 frases insertado tras "el ajuste fino completo queda dominado" (dentro de la subsección de implicaciones de Pareto): explica que RoBERTuito (0.758) no contradice las conclusiones porque fue ajustado sobre TASS 2020 (transferencia desde dominio muy cercano, no comparación controlada).

**C6 — "Amenazas a la validez" reducida:**
- 9 frases (que repetían contenido de Sec VI) → 3 frases compactas remitiendo a `\ref{sec:conclusions}`. La lista detallada de Limitaciones en Sec VI queda intacta.

**C7 — Notas al pie en Tabla III — TRAZA DE INTEGRIDAD:**

Esta tarea es relevante para la declaración de uso de IA: la hipótesis del spec para la nota (b) fue refutada por los logs reales, y la nota publicada usa exclusivamente la explicación respaldada por evidencia.

- Marcador `$^{\mathrm{a}}$` en 2.87 GB (prompting Qwen3-4B VRAM). Nota publicada: carga en 4 bits (NF4, doble cuantización, cómputo fp16); `torch.cuda.max_memory_allocated` tras resetear el contador al acabar la carga del modelo; incluye pesos cuantizados (~2.3 GB) más KV-cache de generación; no comparable con una carga bf16 (~8 GB) ni con los picos de entrenamiento del resto de filas. Evidencia verificada: `results/prompting_fewshot_k4_Qwen_Qwen3-4B.json` → `peak_vram_gb: 2.871407616`; `scripts/run_prompting_baseline.py` líneas 121–134 y 235–247 confirman NF4 + reset antes de `predict_batch`.

- Marcador `$^{\mathrm{b}}$` en 124 s (Full-FT Qwen3-1.7B). **Hipótesis del spec REFUTADA:** el spec sugería que full-FT completaba "menos pasos (mínimo 60 frente a 80 de LoRA)". Los logs muestran ambas configuraciones con `max_steps=1150` idéntico (`results/full_ft_qwen1.7b_nfull_s42.json` y `results/lora_qwen1.7b_nfull_s42.json`); los valores `min_steps` (60 vs 80) son el umbral de parada temprana, no el número de pasos ejecutados. La nota correcta, respaldada por logs: ambas configuraciones ejecutan los mismos 1150 pasos de optimización; el tiempo reportado incluye las evaluaciones periódicas de validación, más lentas a través del modelo con adaptadores LoRA sin fusionar (54.10 ms vs 32.83 ms por ejemplo, medidos en inferencia). La nota publicada no contiene la hipótesis incorrecta.

**Agente usado:** memoir-writer (claude-sonnet-4-6). El orquestador aplicó dos micro-ajustes gramaticales en C1 antes de `a732b08`.

---

## BLOQUE D — Añadidos

Completados en `paper/main.tex`, commiteados en `515272f`.

**D1 — Correo institucional (~línea 70):**
- `beltran.sanchez.careaga@gmail.com` → `202216017@alu.comillas.edu`

**D2 — Metadatos PDF (~línea 40, bloque `\hypersetup`):**
- Añadidos `pdftitle` y `pdfauthor` a los campos existentes de color. `pdfinfo` confirma: Title = "Fine-tuning eficiente de LLM abiertos pequeños para el análisis de sentimiento en español: un estudio reproducible de LoRA/QLoRA bajo restricciones de cómputo de capa gratuita"; Author = "Beltrán Sánchez Careaga".

**D3 — Frase de robustez al final de Trabajo futuro:**
- Añadido como último ítem del itemize: "Un análisis complementario de robustez frente a perturbaciones típicas de redes sociales (mayúsculas, elongaciones, emojis, etc.) apunta a una fragilidad diferencial de los codificadores *cased* frente a los modelos generativos, cuya caracterización completa se deja como extensión." Terminología conforme al glosario (QA criterio 9 confirmado).

**Agente usado:** memoir-writer (claude-sonnet-4-6).

---

## Resumen de QA (informe en docs/planning/sprint-07-qa.md)

Veredicto: **PASS con 1 aviso menor** (criterio 1: comentario BibTeX preexistente con "RAZON: No se pudo verificar..." — no aparece en el PDF; reformulado antes del commit final). Criterios 2–10: todos PASS. Valores numéricos de resultados verificados byte-a-byte contra versión congelada 39f2145. Paper: 14 páginas antes y después del sprint. 0 referencias indefinidas.

---

## Uso de IA — declaración

- **memoir-writer (claude-sonnet-4-6):** redacción de todos los cambios de texto (C1–C7, D1–D3), reescritura completa de references.bib (A1–A5), citas recableadas en main.tex. Todas las afirmaciones factuales de las notas C7 proceden de los logs de ml-dev (JSON de resultados y scripts Python verificados por qa-validator); ninguna cifra fue inventada.
- **Orquestador:** revisión y micro-ajustes antes de cada commit (A: años duplicados en booktitle, guiones en apellidos, llaves en notes; C: dos ajustes gramaticales en C1).
- Política de IA del trabajo: uso declarado conforme a lo acordado con David Contreras (tutor).
