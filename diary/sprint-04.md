# Diario — Sprint 4: Corte C (transferencia dialectal)

**Fecha de cierre:** 2026-06-26
**Rama:** `sprint-04-cut-c`
**Sprint:** 4 — Transferencia dialectal LoRA entre variedades del español (InterTASS 2018)
**Agente principal:** memoir-writer (claude-sonnet-4-6)

---

## Objetivo del sprint

Incorporar el Corte C (transferencia dialectal) como resultado empírico presente en el
paper, sustituyendo el placeholder `\todo` de la sección IV-C y el párrafo de "trabajo
futuro" de Conclusions. Actualizar Abstract, Introduction y Conclusions para reflejar
tres cortes. Compilar sin errores ni citas rotas.

---

## T1 — Lectura del estado previo del paper

**Agente:** memoir-writer (lectura de ficheros).

Se leyó `paper/main.tex` completo para inventariar:
- Estructura de secciones y etiquetas (`\label`).
- Convenciones de figuras: `\begin{figure}[t]`, `\includegraphics[width=\columnwidth]{...}`,
  pie `\caption{...}`, `\label{fig:...}`. Figura sin extensión; `\graphicspath` apunta a
  `../results/figures/`.
- Convenciones de tablas: `booktabs`, caption encima, `\label{tab:...}`.
- Claves de citas activas y `.bib` disponible en `docs/research/references.bib`.
- Placeholder Cut C: `\todo[inline]{...}` en la subsección IV-C.
- Párrafo "Dialectal transfer (Cut C)" en Future Work: marcado como extensión contingente
  al acceso a InterTASS.

Se verificó que `results/figures/figC_transfer.pdf` existe (producido por ml-dev/qa-validator).
Se verificó que los datos auditados por qa-validator cubren 54 celdas en `results/sprint4/`.

---

## T2 — Decisiones metodológicas a declarar en el texto

**Agente:** memoir-writer (razonamiento sobre el diseño).

Decisiones críticas que el texto debe declarar honestamente:

1. **Evaluación sobre development, no test.** El gold del split de test de
   InterTASS 2018 se distribuye en un `.qrel` que devuelve HTTP 404. El split de
   development NO se usó en selección de checkpoint (que usa un hold-out estratificado
   del 15 % del train). Es una decisión metodológica honesta, no un cherry-picking.

2. **Descarte de NONE.** La etiqueta NONE denota "ausencia de objetivo de opinión",
   no "sentimiento neutro". Fundirla con NEU contaminaría el esquema de polaridad.
   Se declara explícitamente en el texto.

3. **Criterio de 2σ, no test estadístico.** El texto es explícito: "no formal hypothesis
   test has been conducted", "no correction for multiple comparisons". Los valores en σ
   son una medida de magnitud respecto al ruido de semillas, no p-valores.

4. **Excepción no replicada (CR→PE en 1.7B, 3.9σ).** Se menciona en el texto; el patrón
   robusto replicado en ambos modelos es únicamente la columna ES.

5. **Neutral minoritario.** ES y CR tienen ~15 % de neutral; PE ~26 %. Se declara en
   Limitations.

---

## T3 — Ediciones realizadas en paper/main.tex

**Agente:** memoir-writer (edición de fichero).

Ediciones por zona del documento:

### Preámbulo
- Añadido `\usepackage{multirow}` (necesario para la tabla de transferencia 3×3).
- Actualizado Index Terms: añadido `dialectal transfer`.

### Abstract
- Ampliado para reflejar tres preguntas de investigación (antes eran dos).
- Añadida frase sobre el hallazgo de asimetría dialectal (penalización hacia ES,
  0.031–0.039 puntos, 2.6–4.6σ, replicado en ambos modelos).

### Introduction
- Párrafo del "roadmap" actualizado: "Cuts A and B" → "Cuts A, B, and C".
- Nuevo bullet de contribución: **Dialectal transfer asymmetry (Cut C)**, con
  descripción del hallazgo principal.

### Methodology → Dataset
- Sustituido el párrafo tentativo ("if a licence is obtained...") por descripción
  completa del corpus InterTASS 2018: variedades, tamaños tras descartar NONE,
  hold-out 15 %, evaluación sobre development, inaccesibilidad del test gold,
  independencia respecto a Cardiff ES.
- Añadido bloque **Cut C** en la subsección de Experimental Design con el conteo
  de celdas (54) e indicación de hiperparámetros heredados de Cut B.

### Results → IV-C (de placeholder a resultado completo)
- Reemplazado `\todo[inline]{...}` con la subsección completa:
  - Corpus y setup (variedades, descarte NONE, hold-out, evaluación en dev).
  - Nota metodológica explícita sobre inaccesibilidad del test gold.
  - Descripción del diseño de la matriz 3×3.
  - Tabla `\ref{tab:cutC}` con las 18 celdas (media±desv, 3 semillas, dos modelos),
    diagonal en negrita.
  - Figura `\ref{fig:cutC}` con `figC_transfer` (sin extensión, igual que figA/figB).
  - Análisis de resultados: no penalización hacia CR/PE; penalización robusta hacia ES;
    excepción no replicada CR→PE en 1.7B; tamaño sólo sube el techo, no cambia el patrón.
  - Párrafo explícito de rigor: "no formal hypothesis test", "no correction for multiple
    comparisons", el criterio 2σ es heurística de magnitud.

### Discussion
- "two research questions" → "three research questions".
- Nueva subsección **(3)** en el recapitulatorio de preguntas de investigación:
  asimetría dialectal y su implicación práctica (hacia CR/PE libre; hacia ES requiere
  datos específicos o adaptador ES).
- Amenaza a la validez extendida: Cut C usa development splits e InterTASS (pequeño),
  con puntero a Conclusions para los detalles.

### Conclusions
- Párrafo de apertura: "two research questions" → "three research questions".
- Nuevo **Finding 3 (Cut C)**: resumen del hallazgo de asimetría, magnitud, limitación
  sobre el criterio de 2σ.
- Bloque **Limitations** ampliado con cuatro nuevos items específicos de Cut C:
  corpus pequeño/clase escasa, cobertura dialectal limitada (sólo 3 variedades),
  evaluación en development, y ausencia de test estadístico formal.
- **Future Work**: reescrito el antiguo bullet "Dialectal transfer (Cut C)" como
  "Broader dialectal coverage" (más variedades, test real, test estadístico formal).
  Eliminado el bullet "Dialectal corpus construction" (ya no procede: el corpus se obtuvo).

---

## T4 — Compilación

**Agente:** memoir-writer (Bash, uso autorizado por ser compilación LaTeX).

Entorno no tenía pdflatex instalado; se instalaron `texlive-base`,
`texlive-latex-base`, `texlive-latex-extra`, `texlive-fonts-recommended`,
`texlive-publishers` (que incluye IEEEtran.cls).

Secuencia de compilación:
```
pdflatex -interaction=nonstopmode main.tex   # pase 1 → 12 páginas
bibtex main                                  # 2 warnings de campo vacío en .bib preexistentes
pdflatex -interaction=nonstopmode main.tex   # pase 2
pdflatex -interaction=nonstopmode main.tex   # pase 3 (referencias estables)
```

Resultado:
- **12 páginas** (superando el mínimo de 10).
- **0 "undefined reference"** en el log.
- **0 "Citation undefined"** en el log.
- Warnings presentes pero preexistentes (campo journal vacío en `perez2022robertuito`,
  campo booktitle vacío en `gutierrezfandino2022maria`; ambos heredados del sprint anterior).
- PDF producido: `paper/main.pdf` (364 139 bytes).

---

## Uso de IA — Declaración de integridad

Esta sesión fue íntegramente operada por el agente **memoir-writer** (claude-sonnet-4-6),
instrumento del sistema multi-agente de IA Lab.

**Qué hizo memoir-writer:**
- Leer y analizar la estructura de `paper/main.tex` antes de editar.
- Redactar la subsección IV-C completa (Corte C: transferencia dialectal).
- Actualizar Abstract, Introduction (contribuciones + roadmap), Methodology (dataset +
  diseño experimental), Discussion (pregunta 3 + amenazas a la validez), y Conclusions
  (Finding 3 + Limitations + Future Work).
- Compilar el documento con pdflatex/bibtex y verificar cero referencias rotas.
- Redactar este diario.

**Qué NO hizo memoir-writer:**
- Inventar resultados o cifras: todos los números provienen del briefing del qa-validator
  (auditado externamente).
- Tocar `src/`, datos, tests, ni ningún fichero fuera de `paper/` y `diary/`.
- Hacer commit ni merge a main (explícitamente excluido por el usuario).

**Datos numéricos:** provistos por qa-validator; memoir-writer los transcribió sin
modificación.

**Política de IA del trabajo:** consultar con David Contreras qué partes del paper
pueden haber sido generadas/asistidas por IA y cómo citarlo en la declaración de
integridad académica.

---

## Estado al cierre del sprint

| Item                              | Estado        |
|-----------------------------------|---------------|
| Subsección IV-C redactada         | Completo      |
| Abstract actualizado (3 cortes)   | Completo      |
| Introduction actualizada (3 cortes + bullet Cut C) | Completo |
| Methodology (InterTASS + Cut C design) | Completo |
| Discussion (pregunta 3 + validity) | Completo     |
| Conclusions (Finding 3 + Limitations + FW) | Completo |
| Figura figC_transfer referenciada | Completo      |
| Tabla tab:cutC en LaTeX           | Completo      |
| Compilación limpia (0 errores)    | Completo      |
| Páginas >= 10                     | 12 páginas    |
| Commit a main                     | Pendiente (lo hace el usuario) |
