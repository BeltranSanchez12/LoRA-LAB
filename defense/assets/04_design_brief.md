# Brief para Claude Design — Deck de defensa (15 min)

> **Qué es esto.** Las instrucciones, slide a slide, para que Claude Design monte el deck de la
> defensa a partir de los archivos YA versionados en este repo. Derivado del guion congelado
> (`defense/03_guion.tex`), mapeado contra sus 6 bloques y sus **13:45** de contenido.
>
> **Actualizado (Sprint 5):** el paper está **en español** (`paper/main.tex`, congelado) y las
> figuras viven en su **ruta versionada real** `results/figures/` (figA–figE, ya trackeadas). Para
> montar el deck con la conexión al repo, lee de ahí; si la conexión falla, todo está duplicado en
> `defense/assets/` (mismo nombre de fichero).
>
> **Actualizado (Sprint 6):** el **cierre pasa a ser una DEMO EN VIVO** (Slide 11 nueva) que remata
> clasificando una frase y mostrando el **colapso cased** con el botón de MAYÚSCULAS (los encoders
> voltean, los LoRA aguantan). La **robustez** se menciona de viva voz en Límites y se **demuestra en
> vivo** (sin slide de heatmap dedicada). El **explorador** (`explorer.html`) y el **heatmap de
> robustez** (`figF`) quedan como **assets de Q&A**, no como slides. El total sigue en **13:45**
> (§4): se **comprime Corte C a su titular** para hacer sitio a la demo.

---

## ⚠️ Regla no negociable: Claude Design DISEÑA, no inventa datos

- **Usa las figuras tal cual están en el repo** (`results/figures/`). No las regeneres, no las
  recolorees cambiando su significado, no alteres sus cifras.
- **No inventes números.** Toda cifra en pantalla sale del guion, de las figuras, de los CSV de
  `results/` o de las tablas del paper (`paper/main.tex`). Si falta un dato para una slide,
  **déjalo en blanco y avísalo** — no lo rellenes.
- Las figuras del deck (A, B, C, D, E) **ya están en español y con sus valores definitivos**, todas
  exportadas a PNG+PDF y commiteadas. (Script: `scripts/make_figures_sprint5.py`, para trazabilidad;
  Design NO lo ejecuta.) La **Figura F** (heatmap de robustez, Sprint 6) también existe en PNG+PDF,
  pero es **asset de Q&A**, no una slide del discurso.
- El único elemento sin asset de imagen es el **esquema de la Slide 3** (los 3 cortes): es un
  diagrama conceptual que Design compone con cajas/flechas a partir del texto — sin datos numéricos.

---

## 1. Manifiesto de archivos (todo en `origin`, rama `main`)

Claude Design debe leer estos archivos del repo. Rutas exactas (ruta versionada real):

| # | Archivo (ruta en el repo) | Para qué sirve |
|---|---------------------------|----------------|
| 1 | `defense/03_guion.tex` (+ `defense/03_guion.pdf`) | **Guion**: mensajes clave, marcas de tiempo y arco. Es la espina dorsal del deck. |
| 2 | `results/figures/figA_f1_vs_n.png` (+ `.pdf`) | **Figura A** — F1-macro vs nº de ejemplos (cruce prompting→LoRA). Slide 4. |
| 3 | `results/figures/figB_quality_cost.png` (+ `.pdf`) | **Figura B** — Pareto calidad vs parámetros entrenables. Slide 5. |
| 4 | `results/figures/figD_per_class.png` (+ `.pdf`) | **Figura D** — F1 por clase (neg/neu/pos) para LoRA-1.7B/4B, BETO, XLM-R. **Slide 7** (cuello de botella neutral). |
| 5 | `results/figures/figC_transfer.png` (+ `.pdf`) | **Figura C** — dos heatmaps 3×3 de transferencia dialectal. Slide 8. |
| 6 | `results/figures/figE_quality_cost.png` (+ `.pdf`) | **Figura E** — dispersión 4D calidad/coste. **Asset de RESERVA para Q&A**, NO va en las slides del discurso (demasiado densa para 15 min). |
| 7 | `results/summary_corteB.csv` | Datos de Corte B (método, f1_macro, params, peak_vram_gb, latencia…). **Usado** para la mini-tabla de la Slide 6. |
| 8 | `results/summary_corteA.csv` | Datos de Corte A (por modelo y n: f1_mean/std, params, vram…). Respaldo de la Slide 4. |
| 9 | `results/sprint4/all_results.csv` | Datos de Corte C (transferencia). Respaldo de la Slide 8. |
| 10 | `results/all_results.csv` | Agregado general; respaldo de cualquier cifra. |
| 11 | `paper/main.tex` (+ `paper/main.pdf`) | **Fuente de verdad (en español)**: Tablas I–V, captions y números autorizados. Ante cualquier duda numérica, manda esto. |
| 12 | `defense/02_banco.tex` | *Opcional*: por si se quiere una slide-backup oculta de Q&A. No es obligatorio para el deck principal. |
| 13 | `results/figures/figF_robustness.png` (+ `.pdf`) | **Figura F** — heatmap de robustez (Sprint 6). **Asset de Q&A**, NO en las slides. |
| 14 | `defense/assets/explorer.html` | **Explorador** interactivo offline (Pareto / cruce / transferencia). **Asset de Q&A**, NO en las slides. |
| 15 | `demo/app.py` (+ `demo/README.md`) | **Demo en vivo** (Slide 11): 5 modelos en GPU / `--cpu` fallback. Cifras del colapso cased → Slide 11-B. |

> **Figuras:** ruta canónica = `results/figures/` (ahí las genera el repo y están trackeadas).
> La carpeta `defense/assets/` contiene una **copia** de todas para descarga offline (mismos
> nombres). La antigua `defense/assets/` (solo A/B/C) queda **superada**: no la uses.
> Formato: usa el **PNG** para colocar en slide; el **PDF** es el vectorial por si se necesita
> reescalar sin pérdida.

---

## 2. Principios de diseño

- **Las slides acompañan al discurso, no lo reproducen.** Nada de párrafos; el contenido lo digo yo.
- **Texto mínimo en pantalla** (en **español**, coherente con el guion): un titular + 3–5
  palabras/cifras clave por slide como máximo.
- **Cifras grandes y legibles**: los números (0,707; 17–43×; −0,031…) son los protagonistas visuales.
- **Registro académico serio**: sobrio, sin animaciones llamativas ni iconografía decorativa.
- **Color con significado, no decorativo**: mantén la convención de las figuras (1.7B en azul, 4B en
  rojo). Usa un color de acento para destacar LoRA en el Pareto; reserva el rojo/alerta solo para la
  penalización de Corte C. No introduzcas paletas nuevas que choquen con las figuras.
- **Coherencia con las figuras**: si una figura ya usa azul=1.7B / rojo=4B, los textos de esa slide
  deben respetarlo. (Nota: las figuras usan **punto** decimal —0.707—; en los textos de pantalla en
  español puedes usar coma —0,707—; mantén una sola convención dentro de cada slide.)

---

## 3. Brief slide a slide (11 slides · arco de 6 bloques · 13:45)

> **Cambio Sprint 5:** se añade la **Figura D como slide propia** (Slide 7).
>
> **Cambio Sprint 6:** el **Cierre** deja de ser una slide de texto y pasa a ser una **DEMO EN VIVO**
> (Slide 11, con respaldo oculto 11-B para el Plan B). La Contribución queda sola en la Slide 10. El
> deck pasa a **11 slides**. Para no alargar el tiempo, el **Corte C (Slide 8) se cuenta en titular**
> (~0:45); su figura no cambia.

### Slide 1 — Portada
- **Rol**: identidad y título. (Bloque 1, apertura · **00:00**)
- **En pantalla**: título del trabajo (en español) · autor (Beltrán Sánchez Careaga) · institución
  (**Universidad Pontificia de Comillas – ICAI**) · fecha.
- **Figura/tabla**: ninguna (o un acento gráfico sobrio).

### Slide 2 — El problema
- **Rol**: enganchar con el problema. (Bloque 1 · **00:00 → 01:45**)
- **Mensaje clave**: *Adaptar un modelo era caro; la pregunta es cuándo y a qué coste deja de serlo.*
- **En pantalla**: titular tipo «Análisis de sentimiento en español… barato» + tres etiquetas
  (positivo / neutral / negativo).
- **Figura/tabla**: ninguna obligatoria. (Si YO aporto una imagen de tuits de ejemplo, colócala; no
  la inventes.)

### Slide 3 — Método y diseño
- **Rol**: marco del estudio. (Bloque 2 · **01:45 → 04:00**)
- **Mensaje clave**: *Estudio reproducible en hardware gratuito: LoRA/QLoRA sobre 2 Qwen3 vs encoders, en 3 cortes.*
- **En pantalla**: esquema con tres ramas — **Corte A** (¿cuándo afinar gana al prompting?),
  **Corte B** (¿LoRA vs encoders contando coste?), **Corte C** (¿transfiere entre variedades?).
  Añadir, pequeño: modelos (Qwen3-1.7B / 4B), datasets (Cardiff, InterTASS), métrica (macro-F1).
- **Figura/tabla**: **esquema compuesto por Design** (cajas/flechas). No hay imagen en el repo para
  esto; es un diagrama conceptual, sin cifras.

### Slide 4 — Resultado 1 · Corte A (punto de cruce)
- **Rol**: primer resultado. (Bloque 3a · **04:00 → 05:15**)
- **Mensaje clave**: *El punto en que afinar supera al prompting depende del tamaño del modelo.*
- **En pantalla**: titular + dos cifras grandes → «**1.7B: < 7 ejemplos**» · «**4B: ≈ 50 ejemplos**».
- **Figura**: `results/figures/figA_f1_vs_n.png` (a pantalla grande). Respaldo de datos:
  `results/summary_corteA.csv`.

### Slide 5 — Resultado 2 · Corte B (frontera de Pareto)  ← slide central
- **Rol**: el resultado estrella. (Bloque 3b · **05:15 → 08:00**)
- **Mensaje clave**: *LoRA domina la frontera calidad-coste: más calidad que los encoders entrenando una fracción de los parámetros.*
- **En pantalla**: titular + cifras grandes → «**4B 0,707 · 1.7B 0,698**» vs «**BETO 0,661**» ·
  «**17–43× (1.7B) / 9–24× (4B) menos parámetros**».
- **Figura**: `results/figures/figB_quality_cost.png`.
- **Pre-empción W1 (RoBERTuito)**: lo digo de viva voz. **RoBERTuito NO aparece en la figura** (está
  fuera de Pareto a propósito). Si se quiere reforzar, añade una nota al pie discreta:
  «RoBERTuito 0,758 = out-of-domain, fuera de la frontera». No lo metas como punto en el Pareto.

### Slide 6 — Resultado 2b · Corte B (QLoRA: coste-memoria)
- **Rol**: cierre del trade-off de coste. (Bloque 3b, tramo final · **≈ 07:00 → 08:00**)
- **Mensaje clave**: *QLoRA retiene ~97–99 % de la calidad a ~mitad de VRAM → el 4B cabe en 8 GB.*
- **En pantalla**: 2–3 cifras grandes → «**97–99 % F1**» · «**6,79 GB (4B)**» · «**cabe en 8 GB de consumo**».
- **Figura/tabla**: **mini-tabla** que Design maqueta a partir de `results/summary_corteB.csv` /
  Tabla III del paper — filas LoRA vs QLoRA (F1, VRAM pico). Sin inventar columnas.

### Slide 7 — Resultado 2c · F1 por clase (la clase neutral es el cuello de botella)  ← NUEVA
- **Rol**: matiz visual del Corte B que abre la honestidad del bloque de límites. (Bloque 3b→4 ·
  **≈ 08:00**, transición a Límites)
- **Mensaje clave**: *La clase neutral es el cuello de botella para todos; los encoders se hunden ahí (0,50–0,55) y LoRA la sostiene (~0,61–0,63).*
- **En pantalla**: titular + la idea «**neutral = clase difícil**» + dos cifras de contraste →
  «**encoders 0,50–0,55**» vs «**LoRA ~0,61–0,63**».
- **Figura**: `results/figures/figD_per_class.png` (barras agrupadas neg/neu/pos; la columna neutral,
  más baja, es el argumento). Respaldo de datos: Tabla IV del paper (`tab:perclass`).

### Slide 8 — Resultado 3 · Corte C (transferencia dialectal)
- **Rol**: tercer resultado. (Bloque 3c · **08:00 → 09:30**)
- **Mensaje clave**: *La transferencia es casi gratis salvo al apuntar al español de España; y se replica en los dos modelos.*
- **En pantalla**: titular + «**CR/PE: sin penalización**» · «**ES: −0,031 a −0,039 F1**».
- **Figura**: `results/figures/figC_transfer.png` (los dos heatmaps; la columna ES, más oscura, es la
  penalización). Respaldo: `results/sprint4/all_results.csv`.
- **Sprint 6 (tiempo)**: el discurso de este corte se **comprime a su titular** (~0:45) para dejar
  sitio a la demo. La slide y la figura **no cambian**; el detalle 3×3 se remite al **explorador,
  Vista 3**, en Q&A.

### Slide 9 — Límites honestos
- **Rol**: credibilidad + pre-empción W4. (Bloque 4 · **09:30 → 11:45**)
- **Mensaje clave**: *Los límites están declarados; varios «negativos» prueban que el experimento medía lo que decía.*
- **En pantalla**: lista mínima (4 ítems) → «dev-split (test 404)» · «σ = magnitud, no test formal» ·
  «**H200 → la VRAM es la del target T4 (4,9–8,7 GB)**» · «1 dataset (A/B)».
- **Pre-empción W4**: el ítem de hardware en pantalla + lo desarrollo de viva voz. Color sobrio
  (es honestidad, no alarma).
- **Figura/tabla**: ninguna; es una slide de texto mínimo. (Enlaza con la Slide 7: la dificultad de
  la clase neutral es intrínseca, no un fallo del método.)

### Slide 10 — Contribución
- **Rol**: condensar el valor sin overclaim, antes de la demo de cierre. (Bloque 5 · **11:15 → 12:15**)
- **Mensaje clave**: *La aportación no es un récord de F1, sino un mapa reproducible de cuándo, cuánto y a qué coste.*
- **En pantalla**: 3 líneas (una por contribución: cruce / Pareto / asimetría dialectal) + la palabra
  «**reproducible**» destacada.
- **No** afirmar más que el paper (RoBERTuito sigue fuera de Pareto; no escribir «el mejor modelo»).
- **Figura/tabla**: ninguna. (El «Gracias» y el remate van en la Slide 11, con la demo.)

### Slide 11 — Demo en vivo (cierre)  ← NUEVA (Sprint 6)
- **Rol**: rematar en directo; el hallazgo de robustez se **ve**, no se cuenta. (Bloque 6 · **12:15 → 13:45**)
- **Mensaje clave**: *El adaptador clasifica en vivo y aguanta el ruido real; los encoders cased voltean con el mayúsculado.*
- **En pantalla (holding slide, sobria)**: titular «**Demo en vivo — 5 modelos**» + la frase que se
  teclea («*pues no está nada mal*») + «Gracias» + contacto. La atención está en la ventana de la
  demo (navegador), no en la slide.
- **Demo (lo que se ve en pantalla)**: la ventana de `demo/app.py` (modo GPU, DGX vía túnel SSH)
  muestra la **tabla de los 5 modelos** (predicción + probabilidad por clase) y el **botón
  MAYÚSCULAS**; al pulsarlo, se reevalúa la misma frase y se ve el colapso cased (BETO/XLM-R voltean,
  LoRA aguantan).
- **Slide 11-B (RESPALDO, oculta) — Plan B obligatorio**: por si la demo falla. **Tabla antes/después
  que compone Design** (no inventa), en dos niveles, ambos trazables:
  1. *Titular cuantitativo* (fuente autoritativa: `results/robustness_results.csv`, fila
     `mayusculas`): con MAYÚSCULAS el macro-F1 cae **BETO −0,184 · XLM-R −0,110**, mientras
     **LoRA ≈ −0,01** y prompting **+0,004**. Este es el dato que respalda el Plan B.
  2. *Ilustración de una frase* (de la **ejecución de la demo**, márcala como tal, **no** del CSV):
     para «PUES NO ESTÁ NADA MAL» → BETO positivo→negativo, XLM-R positivo→negativo, LoRA-1.7B/4B y
     prompting estables. Diséñala como «antes → después» por modelo.
  *Opcional*: el autor puede sustituir la ilustración por una **captura real** de la demo antes de la
  defensa.
- **Figura/tabla**: ninguna en la holding; la **11-B** es la tabla de respaldo (fuente:
  `results/robustness_results.csv`, también copiada en `assets/`).

---

## 4. Mapa rápido slide → figura/tabla → tiempo

| Slide | Bloque guion | Tiempo | Asset principal |
|-------|--------------|--------|-----------------|
| 1 Portada | 1 | 00:00 | — |
| 2 Problema | 1 | 00:00–01:45 | — |
| 3 Método | 2 | 01:45–04:00 | esquema (compuesto) |
| 4 Corte A | 3a | 04:00–05:15 | `figA_f1_vs_n.png` |
| 5 Corte B Pareto | 3b | 05:15–08:00 | `figB_quality_cost.png` |
| 6 Corte B QLoRA | 3b | ~07:00–08:00 | mini-tabla ← `summary_corteB.csv` |
| 7 F1 por clase (neutral) | 3b→4 | ~08:00 | `figD_per_class.png` |
| 8 Corte C (titular) | 3c | 08:00–08:45 | `figC_transfer.png` |
| 9 Límites (W4) + robustez | 4 | 08:45–11:15 | — (texto) |
| 10 Contribución | 5 | 11:15–12:15 | — |
| 11 Demo en vivo (cierre) | 6 | 12:15–13:45 | demo `demo/app.py` (+ 11-B respaldo) |

**Assets de reserva (Q&A, NO en el deck principal):**
- `results/figures/figE_quality_cost.png` — dispersión 4D (F1 × VRAM × parámetros × latencia): trade-off de coste completo.
- `results/figures/figF_robustness.png` — heatmap de robustez (7 perturbaciones × 5 modelos): si preguntan por robustez más allá del mayúsculado. Tabla: `results/robustness_results.csv`.
- `defense/assets/explorer.html` — explorador interactivo offline (Vista 1 Pareto / Vista 2 cruce / Vista 3 transferencia).
- `demo/app.py` — la demo en vivo sigue abierta para probar otras frases (o `--cpu` en el portátil).

---

**Recordatorio final para Claude Design:** las figuras y cifras son las del repo; tu trabajo es la
composición visual (jerarquía, tipografía, color con significado, legibilidad), **no** la generación
de datos. Si algo no está en los archivos del manifiesto, pregunta; no lo inventes.
