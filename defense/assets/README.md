# `assets/` — Bundle consolidado para el deck de defensa

Todo lo que **Claude Design** necesita para montar las **11 slides** (6 bloques; el cierre es una
**demo en vivo**), en un solo sitio. El brief
(`04_design_brief.md`) es la instrucción slide a slide; este README es el **manifiesto de
conexión** + el índice de la carpeta.

- **Con conexión al repo:** Claude Design lee los ficheros por su **ruta en origin** (columna
  *Origin* de la tabla de abajo).
- **Sin conexión:** descarga esta carpeta; contiene una **copia** de todo lo que las slides usan
  (mismos nombres de fichero). La única excepción es `paper/main.tex` (fuente de verdad numérica):
  no se copia aquí por tamaño; se lee de origin.

---

## A. MANIFIESTO DE CONEXIÓN — ficheros a leer del repo

Lista completa de lo que Claude Design debe leer de `origin/main` para montar las slides. **Cada
fichero verificado como trackeado** (`git ls-files --error-unmatch`, ✓).

| # | Origin (ruta en el repo) | Propósito | Slide |
|---|--------------------------|-----------|-------|
| 1 | `defense/03_guion.tex` (+ `.pdf`) | Guion congelado: mensajes, tiempos, arco. Espina dorsal. | todas |
| 2 | `defense/04_design_brief.md` | Brief slide a slide (este bundle). | todas |
| 3 | `results/figures/figA_f1_vs_n.png` (+ `.pdf`) | Figura A — F1 vs nº ejemplos (cruce prompting→LoRA). | 4 |
| 4 | `results/figures/figB_quality_cost.png` (+ `.pdf`) | Figura B — Pareto calidad vs parámetros. | 5 |
| 5 | `results/figures/figD_per_class.png` (+ `.pdf`) | Figura D — F1 por clase (neutral = cuello de botella). | 7 |
| 6 | `results/figures/figC_transfer.png` (+ `.pdf`) | Figura C — heatmaps 3×3 transferencia dialectal. | 8 |
| 7 | `results/figures/figE_quality_cost.png` (+ `.pdf`) | Figura E — dispersión 4D. **RESERVA Q&A**, no en las slides del discurso. | Q&A |
| 8 | `results/summary_corteB.csv` | Datos Corte B → **mini-tabla LoRA vs QLoRA** (F1, VRAM). | 6 |
| 9 | `results/summary_corteA.csv` | Datos Corte A (f1_mean/std por n). Respaldo. | 4 |
| 10 | `results/sprint4/all_results.csv` | Datos Corte C (transferencia). Respaldo. | 8 |
| 11 | `paper/main.tex` (+ `paper/main.pdf`) | **Fuente de verdad (español)**: Tablas I–V, captions, cifras autorizadas. *(No copiado aquí; leer de origin.)* | todas |
| 12 | `results/figures/figF_robustness.png` (+ `.pdf`) | Heatmap de robustez (Sprint 6). **Asset de Q&A/respaldo**, NO slide del discurso. | Q&A · 11-B |
| 13 | `results/robustness_results.csv` | Datos de robustez (5 modelos × 8 condiciones). Respaldo de figF y de la **Slide 11-B** (colapso cased: MAYÚSCULAS BETO −0.184 / XLM-R −0.110). | 11-B · Q&A |
| 14 | `defense/assets/explorer.html` (+ `chart.umd.min.js`) | Explorador interactivo offline (Pareto / cruce / transferencia). **Asset de Q&A**, NO slide. | Q&A |
| 15 | `demo/app.py` (+ `demo/README.md`) | **Demo en vivo** (Slide 11): 5 modelos en GPU / `--cpu` fallback; DGX vía túnel SSH. | 11 |

> **Sprint 6:** el guion (`03_guion.tex`/`.pdf`) y el brief (`04_design_brief.md`) de esta carpeta
> están **sincronizados con origin** (cierre = demo en vivo, 11 slides, Corte C en titular).

> Ruta canónica de las figuras = `results/figures/` (trackeada desde Sprint 5). La antigua
> `defense/assets/` (solo A/B/C) queda **superada**: no usar.

---

## B. Contenido de esta carpeta (copias offline)

| Fichero en `assets/` | Copia de (origin) | Para qué slide |
|---------------------------|-------------------|----------------|
| `03_guion.tex` / `03_guion.pdf` | `defense/03_guion.tex` / `.pdf` | guion base — todas |
| `04_design_brief.md` | `defense/04_design_brief.md` | brief slide a slide — todas |
| `figA_f1_vs_n.png` / `.pdf` | `results/figures/figA_f1_vs_n.*` | Slide 4 (Corte A) |
| `figB_quality_cost.png` / `.pdf` | `results/figures/figB_quality_cost.*` | Slide 5 (Corte B Pareto) |
| `figD_per_class.png` / `.pdf` | `results/figures/figD_per_class.*` | Slide 7 (F1 por clase) |
| `figC_transfer.png` / `.pdf` | `results/figures/figC_transfer.*` | Slide 8 (Corte C) |
| `figE_quality_cost.png` / `.pdf` | `results/figures/figE_quality_cost.*` | **Reserva Q&A** (no slide) |
| `summary_corteB.csv` | `results/summary_corteB.csv` | Slide 6 — mini-tabla QLoRA (**usado**) · Explorer Vista 1 |
| `summary_corteA.csv` | `results/summary_corteA.csv` | Slide 4 — respaldo · Explorer Vista 2 |
| `cutC_all_results.csv` | `results/sprint4/all_results.csv` | Slide 8 — respaldo Corte C |
| `transfer_matrix_qwen1.7b.csv` | `results/sprint4/transfer_matrix_qwen1.7b.csv` | Explorer Vista 3 (Corte C) |
| `transfer_matrix_qwen4b.csv` | `results/sprint4/transfer_matrix_qwen4b.csv` | Explorer Vista 3 (Corte C) |
| `explorer.html` | **generado** por `scripts/build_explorer.py` | Explorador interactivo T4 (3 vistas, offline) |
| `chart.umd.min.js` | vendorizado de `chart.js@4.4.4` (jsdelivr) | Chart.js local; se incrusta inline en `explorer.html` (sin CDN) |
| `figF_robustness.png` / `.pdf` | `results/figures/figF_robustness.*` | **Q&A** — heatmap de robustez (Sprint 6) |
| `robustness_results.csv` | `results/robustness_results.csv` | Slide **11-B** (respaldo del colapso cased) + figF |

---

## C. Cobertura slide → asset (verificación)

Toda slide que necesita un asset lo tiene en esta carpeta:

| Slide | Asset que pide el brief | ¿Presente aquí? |
|-------|-------------------------|-----------------|
| 1 Portada | — | n/a |
| 2 Problema | — (opcional, imagen del autor) | n/a |
| 3 Método | esquema compuesto por Design (sin imagen en repo) | n/a (conceptual) |
| 4 Corte A | `figA_f1_vs_n.png` (+ `summary_corteA.csv`) | ✓ |
| 5 Corte B Pareto | `figB_quality_cost.png` | ✓ |
| 6 Corte B QLoRA | mini-tabla ← `summary_corteB.csv` | ✓ |
| 7 F1 por clase | `figD_per_class.png` | ✓ |
| 8 Corte C (titular) | `figC_transfer.png` (+ `cutC_all_results.csv`) | ✓ |
| 9 Límites + robustez | — (texto) | n/a |
| 10 Contribución | — (texto) | n/a |
| 11 Demo en vivo (cierre) | demo `demo/app.py` (live) + **11-B** ← `robustness_results.csv` | ✓ |
| Q&A | `figE` (reserva) · `figF` (robustez) · `explorer.html` | ✓ |

Ninguna slide del brief pide un asset ausente de esta carpeta/manifiesto.

---

## E. Explorador interactivo (T4, Sprint 6)

`explorer.html` — HTML autocontenido y **OFFLINE** (sin build; **Chart.js v4.4.4 vendorizado
localmente en `chart.umd.min.js` e incrustado inline** — sin CDN, abre con doble clic sin
internet: 0 cargas de red externas) con 3 vistas en pestañas, **cableado 100 % a CSVs reales
del repo** (embebidos verbatim; cero cifras inventadas). Generado por
`scripts/build_explorer.py`, que **valida** cada valor (F1∈[0,1], sin celdas vacías, los 8 puntos
reproducen la Tabla III) antes de escribir; si algo no cuadra, **para y no genera**.

| Vista | Qué muestra | CSV que consume |
|-------|-------------|-----------------|
| 1 · Frontera de Pareto (Corte B) | F1 vs coste, toggle params ↔ VRAM, frontera recalculada por eje, RoBERTuito 0.758 como referencia fuera de dominio | `results/summary_corteB.csv` (+ `results/baselines/robertuito_offtheshelf.json`) |
| 2 · Cruce F1-vs-n (Corte A) | curvas LoRA 1.7B/4B con banda ±std, suelos de prompting y anclas de encoders, punto de cruce calculado del dato (1.7B n=7, 4B n=50) | `results/summary_corteA.csv` (+ suelos/anclas de `summary_corteB.csv`) |
| 3 · Transferencia dialectal (Corte C) | heatmap 3×3 train×eval por modelo, diagonal marcada, columna ES resaltada (objetivo penalizado), tooltip media±std | `results/sprint4/transfer_matrix_qwen1.7b.csv` y `..._qwen4b.csv` |

LoRA/QLoRA = medias de 3 semillas {42,43,44} donde aplica (Cortes A y C). Regenerar:
`python scripts/build_explorer.py`.

## D. Notas

- **Figura E** es **asset de reserva para Q&A**, no entra en las slides del discurso (demasiado densa
  para 15 min). Anotado igual en el brief.
- **Figura F** (robustez), **`explorer.html`** y la **demo** (`demo/app.py`) son **respaldo de Q&A /
  cierre en vivo**, no slides del discurso. La **Slide 11-B** (respaldo del Plan B) la compone Design
  con las cifras de `results/robustness_results.csv` (MAYÚSCULAS: BETO −0,184 / XLM-R −0,110).
- **Regla intacta:** Claude Design **diseña**, no inventa datos ni regenera figuras. Las cifras son
  las del repo; ante duda numérica, manda `paper/main.tex` (español, congelado).
- El script que generó las figuras D/E es `scripts/make_figures_sprint5.py` (trazabilidad; **no** se
  ejecuta para montar el deck).
