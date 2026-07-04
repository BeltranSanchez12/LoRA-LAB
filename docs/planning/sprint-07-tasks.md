# Sprint 7 — Tareas descompuestas

**Objetivo:** corregir bibliografía, figuras y texto del paper antes de entrega, sin alterar ningún valor numérico de resultados.
**Presupuesto estimado:** ~9 h. Bloques en orden A→B→C→D; push solo tras QA.
**Commits:** uno por bloque (`sprint7/A-bibliografia`, `sprint7/B-figuras`, `sprint7/C-texto`, `sprint7/D-anadidos`).

---

## Pre-trabajo (paralelo, antes de los bloques)

Estas tareas no generan commit propio; su evidencia es consumida por los bloques que las siguen.

| # | Tarea | Agente | Est. | Consumida por |
|---|-------|--------|------|---------------|
| PRE-1 | Revisar logs/configs de entrenamiento e inferencia para (a) el 2.87 GB de prompting Qwen3-4B y (b) el 124 s de full-FT vs 169 s de LoRA. Documentar qué se midió exactamente; no inferir. Si no hay evidencia concluyente para (a), anotarlo así. | ml-dev | 0.5 h | C7 |
| PRE-2 | Verificar online la venue real de Miranda-Escalada et al. (DisTEMIST overview): CEUR-WS / BioASQ-CLEF 2022, BioCreative VII 2022 o IberLEF 2023. Anotar URL o DOI encontrado. Si no se puede verificar, documentarlo. | memoir-writer | 0.25 h | A1 |
| PRE-3 | Verificar online los datos exactos de Martínez-Cámara et al., TASS 2018 overview (actas TASS 2018 / SEPLN, CEUR-WS vol. 2172 o equivalente). Anotar autores, título, año, vol. y URL definitivos. | memoir-writer | 0.25 h | A5 |

---

## Bloque A — Bibliografía

**Archivo:** `docs/research/references.bib`
**Dependencias previas:** PRE-2 (para A1), PRE-3 (para A5).

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| A1 | Ref DisTEMIST [15]: eliminar el `note` con el texto "VERIFICAR: ...". Fijar venue según resultado de PRE-2; si no fue verificable, dejar solo autores + título + año + URL sin nota especulativa. | memoir-writer | 0.25 h |
| A2 | Eliminar los textos de `note` en refs [3], [4], [5], [6], [7], [14] y [16] listados en la spec. Verificar que el campo `note` resultante quede vacío o ausente en cada una; no dejar fragmentos residuales. | memoir-writer | 0.5 h |
| A3 | Proteger con llaves en los campos `title` del `.bib`: `{BETO}`, `{XLM-RoBERTa}`, `{QLoRA}`, `{LoRA}`, `{GPT-3}`, `{NLP}`, `{DisTEMIST}` y cualquier otro acrónimo que aparezca capitalizado en el PDF actual. Verificar tras recompilar que no hay des-capitalización. | memoir-writer | 0.5 h |
| A4 | Ref MarIA [13]: el campo `journal` ya existe pero la entrada es `@inproceedings` (BibTeX ignora `journal` en ese tipo). Cambiar el tipo de la entrada a `@article` para que el campo `journal = {Procesamiento del Lenguaje Natural}` tenga efecto. | memoir-writer | 0.25 h |
| A5 | Añadir entrada nueva para Martínez-Cámara et al. TASS 2018 con los datos verificados en PRE-3. En `main.tex`, sustituir la cita de [12] por la nueva referencia en todas las menciones de InterTASS 2018 (Secciones I, II-B, III-A, IV-D). Mantener [12] solo para TASS 2020 / serie TASS. | memoir-writer | 0.5 h |
| **COMMIT** | `sprint7/A-bibliografia` — solo cambios en `.bib` y las citas de `main.tex` modificadas por A5. | — | — |

---

## Bloque B — Figuras

**Archivos:** `scripts/make_figures.py`, `scripts/make_figures_sprint5.py`, `scripts/aggregate_cutc.py`; figuras nuevas en `results/figures/`; `paper/main.tex` (referencias a los assets).
**Dependencias previas:** ninguna (paralelo a A).

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| B1 | En los tres scripts de figuras, añadir una rama de generación con sufijo `_paper` que omita completamente `ax.set_title()` y `fig.suptitle()`. Los assets originales A–E de `results/figures/` (sin sufijo) no se modifican; los usa la defensa. Regenerar las cinco variantes `_paper` y guardarlas en `results/figures/`. | ml-dev | 1.5 h |
| B2 | Comprobar que las variantes `_paper` quedan cubiertas por la excepción de `.gitignore` ya existente. Si no, añadir la excepción mínima necesaria. Verificar también que los assets A–E originales permanecen intactos (`git diff` limpio sobre ellos). | ml-dev | 0.25 h |
| B3 | En `main.tex`, actualizar las cinco referencias de figura para apuntar a las variantes `_paper`. Corregir el pie de la Figura 2: sustituir "∼9–24× menos parámetros entrenables que los codificadores" por "17–43× (1.7B) y 9–24× (4B) menos parámetros entrenables que los codificadores". | memoir-writer | 0.25 h |
| **COMMIT** | `sprint7/B-figuras` — scripts modificados, variantes `_paper` de figuras, y cambios en `main.tex` de B3. | — | — |

---

## Bloque C — Texto

**Archivo:** `paper/main.tex`
**Dependencias previas:** PRE-1 (para C7). El resto de tareas de C son independientes entre sí.

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| C1 | Resumen e Introducción (contribuciones): sustituir "menos de n = 7 ejemplos etiquetados bastan" por "con tan solo n = 7 ejemplos etiquetados (el cruce se produce entre n = 4 y n = 7)". Verificar que aparece exactamente en los dos lugares. | memoir-writer | 0.25 h |
| C2 | Introducción: cambiar "dos preguntas de relevancia práctica" por "tres preguntas de relevancia práctica" y añadir el párrafo de transferencia dialectal de la spec tras el segundo punto. Ajustar transiciones. | memoir-writer | 0.5 h |
| C3 | Discusión: sustituir la frase "GPU de consumo de 8 GB (RTX 3070, RTX 4060 Ti) que no están disponibles en la gama H200/A100" por "GPU de consumo de 8 GB (RTX 3070, RTX 4060 Ti), a diferencia de la gama empresarial H200/A100". | memoir-writer | 0.25 h |
| C4 | Sección IV-C o Discusión: insertar el párrafo sobre LoRA-4B (16.7 GB) excediendo el objetivo T4 de 16 GB, con la recomendación de usar QLoRA en ese hardware. Texto exacto en la spec. | memoir-writer | 0.5 h |
| C5 | Sección V: insertar el párrafo que explica por qué RoBERTuito (0.758) supera los resultados del estudio sin contradecir las conclusiones. Texto exacto en la spec. | memoir-writer | 0.5 h |
| C6 | Consolidar limitaciones duplicadas: reducir "Amenazas a la validez" (Sección V) a 2–3 frases con remisión a Sección VI; dejar la lista detallada solo en VI. Elegir la edición que menos altere la maquetación; verificar que el documento sigue en ≥ 10 páginas. | memoir-writer | 0.5 h |
| C7 | Insertar dos notas al pie en la Tabla III basadas ÚNICAMENTE en la evidencia de PRE-1. (a) Si la evidencia explica el 2.87 GB, añadir nota al pie con esa explicación; si no, marcar el número para revisión manual. (b) Si los logs confirman los 60 pasos vs 80, añadir la nota al pie del texto exacto de la spec. No redactar explicaciones sin respaldo de logs. | memoir-writer | 0.25 h |
| **COMMIT** | `sprint7/C-texto` — solo cambios de texto en `main.tex` (C1–C7); ningún valor numérico de resultados modificado. | — | — |

---

## Bloque D — Añadidos

**Archivo:** `paper/main.tex`
**Dependencias previas:** ninguna.

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| D1 | Sustituir el Gmail del autor por `202216017@alu.comillas.edu` en el campo de afiliación/email del `.tex`. | memoir-writer | 0.25 h |
| D2 | Añadir en el preámbulo `\hypersetup{pdftitle={...título exacto del paper...}, pdfauthor={Beltrán Sánchez Careaga}}`. Verificar en el PDF compilado que los metadatos aparecen. | memoir-writer | 0.25 h |
| D3 | Sección de Trabajo futuro: añadir al final una frase sobre el análisis de robustez frente a perturbaciones de redes sociales (mayúsculas, elongaciones, emojis) y la fragilidad diferencial de los codificadores *cased* frente a los generativos. Ajustar al glosario establecido. | memoir-writer | 0.25 h |
| **COMMIT** | `sprint7/D-anadidos` — D1, D2 y D3 en `main.tex`. | — | — |

---

## QA — Validación final

**Dependencias:** todos los bloques A, B, C, D completos y compilación limpia del PDF.
**Push:** solo después de que los 10 criterios estén verificados.

| # | Criterio | Verificación |
|---|----------|--------------|
| QA-1 | `grep -ri "VERIFICAR"` sobre `.tex` y `.bib` → 0 resultados. | Automatizable |
| QA-2 | Ninguna anotación personal en refs [3]–[16] del PDF compilado. | Visual, ref por ref |
| QA-3 | "bETO", "xLM" y "¿=" no aparecen en el PDF. | grep / búsqueda en PDF |
| QA-4 | Ningún gráfico contiene "Figura A/B/C/D/E" incrustada; pies van de Figura 1 a Figura 5 en orden. | Visual en PDF |
| QA-5 | Assets A–E de `defense/assets/` (o `results/figures/`) intactos; `git diff` limpio sobre ellos; el explorer sigue funcionando. | git diff + abrir explorer.html |
| QA-6 | Todas las menciones de InterTASS 2018 citan la nueva ref TASS 2018; [12] queda solo para TASS 2020 / serie. | grep + lectura de secciones I, II-B, III-A, IV-D |
| QA-7 | Diff numérico contra commit `1f0a13e`: ningún valor de resultados cambia en Tablas I–V (solo el pie nuevo de Tabla III y el pie corregido de Figura 2). | diff de .tex |
| QA-8 | `pdflatex` sin errores ni `undefined references`; PDF ≥ 10 páginas. | Log de compilación |
| QA-9 | Los párrafos nuevos (C2, C4, C5, D3) respetan el glosario: "parámetros entrenables", "ajuste fino", "codificador", "frontera de Pareto"; punto decimal. | Lectura dirigida |
| QA-10 | Las notas al pie de C7 están respaldadas por evidencia de logs (adjuntar en informe de QA). Si (a) no pudo verificarse, el número está marcado para revisión manual, no explicado sin evidencia. | Adjuntar evidencia de PRE-1 |

**Agente:** qa-validator para todos los criterios.
**Estimación:** 1 h.

---

## Resumen de dependencias críticas

```
PRE-1 (ml-dev)  ──────────────────────────────► C7
PRE-2 (memoir-writer) ──► A1
PRE-3 (memoir-writer) ──► A5
                              A completo ──► B ──► C ──► D ──► QA ──► push
```

**Total estimado: ~9 h** (PRE: 1 h; A: 2 h; B: 2 h; C: 2.75 h; D: 0.75 h; QA: 1 h).
Presupuesto global del proyecto: 180 h. Sprints 1–6 consumieron ~171 h según roadmap. Este sprint consume ~9 h → total ~180 h. Sin margen; no añadir alcance.
