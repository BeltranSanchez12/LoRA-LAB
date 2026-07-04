# Diario — Sprint 5: Pulido editorial, figuras nuevas y traducción al español

**Fecha de cierre:** 2026-06-27
**Rama:** `main`
**Sprint:** 5 — Fase 6 (pulido + traducción EN→ES del paper)
**Agente principal:** Claude Code (claude-opus-4-8); QA final por qa-validator

---

## Objetivo del sprint

Dejar el paper listo para entrega: unificar institución, eliminar anotaciones de
procedencia, corregir todos los desbordes de maquetación, añadir 1–2 figuras de alto
valor desde datos existentes, y traducir el documento completo al español. Al cerrar:
compila exit 0 sin desbordes que afecten a la lectura, cada cifra trazada a su tabla, e
idioma (texto + figuras) coherente de punta a punta.

**Orden de ejecución (clave):** la traducción reflota toda la maquetación a dos columnas
(el español ocupa ~8 % más), por eso los arreglos de maquetación (T5) van DESPUÉS de
traducir (T3) e integrar las figuras nuevas (T4), no antes. Ejecución por gates con
parada y validación humana en cada uno.

## Regla transversal (auto-auditada en cada gate)

Paper congelado salvo por estas tareas. La traducción y ediciones tocan SOLO texto:
NUNCA números, datos de tablas, fórmulas, `\cite`, `\ref`, `\label` ni la bibliografía
(los títulos de las referencias se quedan en su idioma original). Cada cifra sigue
trazada a su tabla.

---

## GATE 1 — T1 + T2 (institución + limpieza)

- **T1.** `\author`: afiliación fijada a *Universidad Pontificia de Comillas -- ICAI*
  (sustituye `[Institution TBD]`).
- **T2.** Eliminadas las 4 anotaciones `(source: configs/...)` del cuerpo (Metodología:
  LoRA config, Full-FT config, Training schedule, SFT objective). Sin paréntesis huérfanos
  ni puntuación colgando. Recompila exit 0.
- Commit: `T1+T2: institución (Comillas-ICAI) + limpieza de anotaciones de procedencia`.

## GATE 2 — T3a (muestra de registro) [PARADA + validación]

Traducida SOLO la sección Discusión como muestra. Se fijó el glosario y se validó el
registro con el usuario antes de continuar.

### Glosario / decisiones de registro (aprobadas)

| Inglés | Español |
|---|---|
| trainable parameters | parámetros entrenables |
| fine-tuning | **fine-tuning** (anglicismo, invariable) |
| full fine-tuning | **ajuste fino completo** (etiqueta `Full-FT` se mantiene en tablas/figuras) |
| prompting / few-shot / LoRA / QLoRA | anglicismos, invariables |
| in-context learning | aprendizaje en contexto |
| encoder | codificador |
| Pareto frontier | frontera de Pareto |
| quality–cost trade-off | compromiso calidad-coste |
| crossover (point) | punto de cruce |
| stratified sampling | muestreo estratificado |
| seed | semilla |
| Cut A/B/C | **Corte A/B/C** (global) |
| wall-clock time | **tiempo de ejecución** |
| ms/example | **ms/ej.** (también en Tabla III) |
| harness | arnés (generativo) |
| fallback | (etiqueta de) reserva |

**Decisiones finas:**
- `prior`: revisado caso por caso. NO había ningún "prior" bayesiano; el único uso era
  de sentido llano → "conocimiento previo".
- **Decimales con PUNTO en TODO el documento** (0.707, 2.6σ), dentro y fuera de `$...$` y
  en tablas. Separadores de millar (1,839 / 10,000) intactos (no se tocan números).
- Etiquetas literales de clase que el modelo emite en inglés (`\textit{negative/neutral/
  positive}` en el objetivo SFT y el fallback) se mantienen en inglés porque el protocolo
  las define "in English"; los nombres conceptuales de clase en prosa van en español
  (positivo/negativo/neutral).
- Babel: `\usepackage[spanish,es-tabla,es-nodecimaldot]{babel}` → "Tabla"/"Figura"/
  "Resumen"/"Palabras clave" automáticos, y el punto decimal se mantiene inerte en modo
  matemático (sin mezcla punto/coma). `\IEEEkeywordsname` → "Palabras clave".

## GATE 3 — T3b (traducción completa) + T4 (figuras nuevas)

### T3b — Traducción EN→ES (commit por bloque)
Bloques: título · resumen · `\IEEEkeywords` · Introducción · Trabajo relacionado ·
Metodología · Resultados (Cortes A/B/C, tablas y pies) · Discusión · Conclusiones/
limitaciones/trabajo futuro. Texto 100 % español; ningún número/`\cite`/`\ref`/`\label`/
dato alterado. Compila exit 0, 0 refs/citas indefinidas.

### T4 — Figuras nuevas D y E (en español, desde datos existentes) [borradores validados]
Se mostraron borradores y se aprobaron antes de integrar.
- **Figura D — F1 por clase** (barras agrupadas neg/neu/pos para LoRA-4B, LoRA-1.7B,
  BETO, XLM-R). Mensaje: la clase neutral es el cuello de botella. Fuente: `f1_per_class`
  de los JSON; **LoRA = media de 3 semillas {42,43,44}**. Verificada celda a celda contra
  `tab:perclass`.
- **Figura E — Compromiso calidad/coste (dispersión 4D)** (elección del usuario frente al
  parallel-coordinates): F1(y) × VRAM(x), tamaño del marcador ∝ parámetros entrenables,
  latencia anotada. Valores exactos de Tabla III (`tab:cutB` / `summary_corteB.csv`).
- Exportadas PNG+PDF a `results/figures/` (ruta versionada de figA/B/C). Las figuras
  estaban gitignoradas: se añadió excepción en `.gitignore` para figA–figE (PNG+PDF) y se
  trackearon. Script reproducible: `scripts/make_figures_sprint5.py`.
- Integradas con `\includegraphics` + pie + referencia en prosa (D en análisis por clase,
  E en Corte B junto a figB).

## Arreglo de contenido (entre GATE 3 y 4)

Barrido "dos cortes / two cuts / two complementary" en todo el documento → **1 ocurrencia**
(Metodología): "dos cortes complementarios" → "tres cortes" (el grid son A, B y C; residual
de cuando C era trabajo futuro). Resto de menciones ya correctas. Commit aparte.

## GATE 4 — T5 (maquetación) [PARADA + validación]

Tras traducir e integrar D/E, arreglados TODOS los desbordes (6 overfull → 0):

| Problema | Arreglo |
|---|---|
| Tabla III (Corte B, 7 col) se salía por la derecha (82pt) | `table*` (doble columna, nítida, sin escalar) |
| Tablas IV y V colisionaban (33pt / 57pt) | `\resizebox{\columnwidth}` cada una (V refluye a otra página) |
| Tabla I (2.4pt) | `\resizebox{\columnwidth}` |
| Paths `\texttt{}` largos sin partir (model names, 96pt+5pt) | `\allowbreak` tras `/`, `-`, `_` |
| Pie de Figura E rozaba el texto | resuelto por reflujo |

`table*`/`\resizebox`/`\allowbreak` son puramente de maquetación (cero cambios de cifras).
Todas las figuras A–E renderizan. Residual cosmético documentado: **0 overfull**;
**26 underfull `\hbox`** (10 con badness 10000) + 1 underfull `\vbox` — espaciado holgado
por justificación en columnas estrechas IEEE con palabras españolas largas; no afecta a la
lectura.

## GATE 5 — T6 (QA final + cierre)

QA por qa-validator sobre el checklist de cierre: integridad de cada cifra trazada a su
tabla (abstract, Resultados, Discusión, pies, figuras D/E); 10.000× solo en §II-A (Hu et
al.); 0 `(source:...)` en PDF; institución correcta; texto 100 % español; figuras A–E en
español; compila exit 0, 0 refs/citas indefinidas, 0 overfull; figuras + script trackeados
en origin.

**Veredicto qa-validator: PASS en los 4 ítems** (suite `tests/test_sprint5_qa_validation.py`,
134/134). Verificadas contra sus fuentes: Corte A (22 pares media±std), Corte B/Tabla III
(F1/params/VRAM/tiempo/latencia), Tabla IV+Fig D (per-clase, LoRA=media 3 semillas), las 18
medias de Tabla V, σ de transferencia, factores 17–43×/9–24×/268×, splits, 10.000× una vez.

**2 avisos de cifras PREEXISTENTES (errores de redondeo anteriores al sprint), corregidos
con autorización del usuario tras confirmar la convención del paper (std con ddof=0):**
1. TF-IDF+LR macro-F1: `0.567` → `0.566` (fuente `tfidf_lr_cardiff_es_full.json`,
   f1_macro=0.5664891). En §III-B y Tabla I.
2. Tabla V, Qwen3-4B ES→ES: `±0.006` → `±0.007` (std ddof=0 de las 3 semillas = 0.006525;
   convención ddof=0 confirmada contra celdas correctas: 1.7B ES→ES=0.007, 4B CR→CR=0.018,
   1.7B PE→CR=0.005). Commit aparte `fix: redondeo`.

---

## Trazabilidad de uso de IA (declaración de integridad, CE37/RA5)

- **Traducción y maquetación:** Claude Code (claude-opus-4-8) sobre `paper/main.tex`, por
  bloques con compilación y verificación tras cada uno; gates con validación humana del
  registro (GATE 2), de los borradores de figuras (GATE 3/T4) y de la maquetación (GATE 4).
- **Figuras D/E:** generadas por `scripts/make_figures_sprint5.py` desde datos ya en el
  repo (cero datos inventados); cada barra/punto trazado a su tabla de origen.
- **QA de cierre:** agente qa-validator (auditoría read-only, sin modificar el paper).
- El contenido científico (números, tablas, hallazgos) NO se alteró: este sprint es
  editorial (idioma + maquetación) + 2 visualizaciones derivadas de datos existentes.

## Estado final

Paper en español, ≥10 páginas (14), compila exit 0, 0 refs/citas indefinidas, 0 overfull,
0 `(source:...)`, institución Comillas–ICAI, figuras A–E en español y trackeadas. Paper
CONGELADO de nuevo tras el push.
