# Inventario del proyecto para la defensa

> **Qué es esto.** El mapa priorizado de todo lo que hay que dominar para la defensa, antes de
> escribir una sola explicación. Tres bloques: (1) lo matemáticamente/conceptualmente denso,
> (2) las decisiones de diseño y sus análisis, (3) las debilidades defendibles mapeadas a su
> sección del paper con su contramedida.
>
> **Estado del feedback.** Aún no me has pasado el feedback real de tus tutores. El Bloque 3 son
> debilidades que **yo infiero** del propio paper. Cuando me des el feedback, lo cruzo con esta
> lista: lo que coincida sube de prioridad, y lo nuevo se añade.
>
> **Fuente.** Todo trazado a `paper/main.tex` por sección (§) y tabla/figura. Las cifras de este
> inventario salen de las Tablas I–V del paper. Lo que es conocimiento externo (no está en el
> paper) va marcado como **[externo]**.

---

## Cómo está montado el proyecto (para situarse en 30 segundos)

Un estudio empírico reproducible: ¿merece la pena hacer *fine-tuning* eficiente (LoRA/QLoRA) de
LLMs generativos abiertos y pequeños (Qwen3-1.7B y 4B) para clasificar el sentimiento de tuits en
español, en hardware gratuito (T4 16 GB)? Se responde con tres "cortes" (cuts) experimentales:

- **Cut A — ¿Cuándo el fine-tuning gana al prompting?** (eficiencia de datos / punto de cruce)
- **Cut B — ¿Gana LoRA a los encoders clásicos (BETO, XLM-R) en calidad-coste?** (frontera de Pareto)
- **Cut C — ¿Un adaptador entrenado en una variedad dialectal transfiere a otras?** (transferencia)

Datos: Cardiff NLP tweet sentiment (ES) para A y B; InterTASS 2018 (ES/CR/PE) para C.

---

# BLOQUE 1 — Densidad matemática / conceptual (priorizada)

Prioridad = (probabilidad de que el tribunal pregunte) × (dificultad de reconstruirlo sin saltos).
**P1 = imprescindible dominar; P2 = importante; P3 = de apoyo.**

| # | Concepto | Prioridad | Dónde (§ / tabla) | Por qué es denso / qué hay que saber reconstruir |
|---|----------|-----------|-------------------|---------------------------------------------------|
| 1 | **LoRA: ΔW = AB** (rango r, escala α, r≪d, ~10.000× menos params) | **P1** | §III-C "LoRA Configuration"; §II-A; Tabla IV | Hay que poder derivar por qué AB con r=16 reduce parámetros, qué es α/r como factor de escala, por qué no añade latencia (se fusiona en W al inferir), y de dónde sale el "~10.000×". |
| 2 | **QLoRA: NF4 + doble cuantización + cómputo bf16** | **P1** | §III-C "QLoRA Configuration" | Qué es cuantizar a 4 bits, qué es NF4 **[externo: detalle de Dettmers]**, qué es la doble cuantización, por qué baja la VRAM 1.8–2.5× pero sube el tiempo (dequantización al vuelo). `prepare_model_for_kbit_training` + gradient checkpointing. |
| 3 | **Macro-F1** (definición, por qué macro y no accuracy/micro) | **P1** | §III-D; Tablas I, III, V | Reconstruir F1 = 2PR/(P+R) por clase y promediado sin pesos. Por qué macro castiga el fallo en la clase difícil (neutral) y por eso es "la cifra más informativa". |
| 4 | **Objetivo SFT con enmascarado de prompt (−100)** | **P1** | §III-C "Model Selection and SFT Objective" | Solo el token de etiqueta (+EOS) contribuye a la cross-entropy; los tokens del prompt se enmascaran con −100. Es lo que hace que prompting y fine-tuning solo difieran en "pesos actualizados". |
| 5 | **Caracterización en σ del nivel de semilla (2σ = heurístico, NO significancia)** | **P1** | §IV-C; Discusión | Δ/σ_seed; el 2σ es magnitud relativa al ruido entre semillas, **no** un p-valor. Punto crítico de honestidad estadística: hay que saber decir qué NO afirma. |
| 6 | **Punto de cruce y su mecanismo** (suelo de prompting ∝ conocimiento preentrenado) | **P1** | §IV-A; §II-C; Discusión; Tabla II | Por qué el cruce se desplaza a la derecha con el tamaño: 1.7B cruza en n<7, 4B en n≈50. Mecanismo de Min et al. **[externo]**: las demostraciones dan formato/espacio de etiquetas, no el mapeo; suelo alto ⇒ más señal para superarlo. |
| 7 | **Submuestreo estratificado y disjunto por semilla** | **P1** | §III-A; §IV-A | Cada semilla extrae un subconjunto **independiente y disjunto** ⇒ la std refleja varianza de muestreo de datos, no solo de inicialización. Estratificación: en n=4 → neg=2, neu=1, pos=1. |
| 8 | **Frontera de Pareto / dominación** | **P2** | §IV-B; Fig. 2; Tabla IV | Qué significa "dominar el frente": ser mejor o igual en todos los ejes (calidad y los 4 de coste) y estrictamente mejor en al menos uno. Por qué RoBERTuito (mayor F1) NO está en el frente. |
| 9 | **Los 4 ejes de coste** (params entrenables, VRAM pico, tiempo, latencia) | **P2** | §III-D; Tabla IV | `torch.cuda.max_memory_allocated()` para VRAM pico; latencia ms/ejemplo; por qué params **entrenables** ≠ totales (clave para el "10.000×"). |
| 10 | **max_steps = max(80, 5·⌈n/8⌉)** y schedule (cosine + warmup 0.1) | **P2** | §III-C "Training Schedule" | Por qué un suelo de 80 pasos: que las fracciones diminutas (n=4) no queden infraentrenadas. lr=2e-4 LoRA/QLoRA vs 1e-5 full-FT. |
| 11 | **Selección por mejor val-F1 (no último paso)** | **P2** | §III-C; `BestValF1Callback` | Se evalúa val macro-F1 4 veces; se queda el mejor checkpoint, no el final. Evita overfitting tardío. |
| 12 | **Desbalance de clases en Cut C + descarte de NONE** | **P2** | §III-A; §IV-C | NONE = "sin objetivo", conceptualmente distinto de neutral; fusionarlo contaminaría el esquema. PE tiene 26% neutral vs 15% ES/CR ⇒ más varianza. |
| 13 | **Por qué neutral es la clase difícil** (ausencia de señal, no patrón) | **P3** | §IV-B "Per-Class" ; Tabla V | Detectar "no-sentimiento" es intrínsecamente más difícil; ranking pos≥neg>neu idéntico en todos los modelos ⇒ propiedad de la tarea, no artefacto. |
| 14 | **Familia de PEFT y alternativas** (adapters, DoRA, IA³) | **P3** | §II-A; §VI | Contexto: de dónde viene LoRA (Houlsby adapters) y qué hay más allá (DoRA descompone magnitud/dirección). **[externo]** |

---

# BLOQUE 2 — Decisiones de diseño y sus análisis (qué / por qué / alternativa descartada)

Para la defensa, cada decisión es una pregunta potencial. Formato: **decisión → porqué → alternativa descartada**.

| # | Decisión | Por qué (justificación en el paper) | Alternativa descartada | § / fuente |
|---|----------|--------------------------------------|------------------------|-----------|
| D1 | Modelos Qwen3-1.7B y 4B | Buen soporte multilingüe, Apache 2.0, compatibles con QLoRA en 16 GB | Modelos ≥7B (no caben en free-tier T4) | §III-B |
| D2 | **Excluir** `cardiffnlp/twitter-xlm-roberta-...` de los baselines | Fue afinado sobre el mismo dataset que nuestro test → fuga de distribución; comparación injusta | Usarlo como "baseline fuerte" (rechazado por fuga) | §III-B; `docs/research/xlmt_leakage_check.md` |
| D3 | RoBERTuito **fuera** del frente de Pareto | Afinado en TASS 2020 (otro corpus); es referencia out-of-domain, no comparación controlada | Ponerlo en el frente (sería comparar peras con manzanas) | §III-B; Tabla I |
| D4 | Cut B con **una sola semilla (42)** | Varianza de semilla despreciable a full-data (±0.006 F1 en 1.7B) | 3 semillas en todo (coste de cómputo) — sí se hizo en full-data LoRA | §III-C; §IV-B |
| D5 | Target modules solo `{q,k,v,o}_proj` | Convención LoRA en atención; mantiene params bajos | Incluir MLP/all-linear (más params) | §III-C |
| D6 | r=16, α=32, dropout 0.05 | Fijos en TODAS las celdas para aislar el efecto de datos/variedad | Barrido de r (no es el objeto del estudio) | §III-C |
| D7 | Evaluar Cut C en **dev split** (no test) | El gold de test de InterTASS 2018 daba HTTP 404 (inaccesible) | Test split (imposible); dev no se usó en train/selección | §III-A; §IV-C |
| D8 | Macro-F1 como métrica primaria | Trata las 3 clases por igual; robusta a dificultad por clase | Accuracy/micro-F1 (ocultarían el fallo en neutral) | §III-D |
| D9 | Tamaños n con valores diminutos (4, 7) | Localizar el cruce LoRA-vs-prompting **por debajo de n=10** | Empezar en n=10 (perdería el cruce del 1.7B) | §III-A |
| D10 | Correr en **H200** declarando target **T4 16 GB** | VRAM se mide por celda como cota superior de viabilidad en T4; tiempos sí difieren | Correr en T4 real (mucho más lento para el barrido) | §III-C; §V |
| D11 | Full-FT con lr=1e-5 (vs 2e-4 LoRA) | Evitar olvido catastrófico al mover todos los pesos | Mismo lr que LoRA (inestable) | §III-C |
| D12 | Harness generativo único (parse + fallback neutral) | Que prompting y fine-tuning compartan protocolo de evaluación; única variable = pesos | Evaluadores distintos (confundiría protocolo con calidad) | §III-D |
| D13 | Descartar etiqueta NONE en Cut C | NONE = ausencia de objetivo, no es neutral; fusionar contaminaría polaridad | Mapear NONE→neutral (contaminación conceptual) | §III-A; §IV-C |

---

# BLOQUE 3 — Debilidades defendibles (inferidas del paper) → sección + contramedida

> Estas son las que **yo** anticipo. Cada una enmarcada como *motivación del diseño*, no como fallo.
> Prioridad P1 = casi seguro que la preguntan. Ordenadas por riesgo.

| # | Debilidad / objeción probable | Prio | § / tabla | Encuadre defensivo (resumen — se desarrolla en el banco Q&A) |
|---|-------------------------------|------|-----------|---------------------------------------------------------------|
| W1 | **RoBERTuito saca 0.758, MÁS que todos tus modelos (0.707).** ¿No te gana un modelo de serie? | **P1** | Tabla I | Es out-of-domain (TASS 2020), evaluado zero-shot; **no** es comparación controlada. Mide otra cosa: lo que ya existe entrenado en otro corpus. Tus baselines controlados (BETO 0.661) sí comparten protocolo. Honestidad: se reporta, pero con † y fuera del frente. |
| W2 | **Cut C evaluado en dev, no en test.** | **P1** | §III-A, §IV-C, §VI | Decisión honesta forzada por HTTP 404 del gold de test, no elección de modelado. Dev nunca se usó en train ni selección (esa usó un 15% held-out aparte). Se avisa explícitamente que no es comparable con resultados de test publicados. |
| W3 | **2σ–4.6σ no es significancia estadística; sin test formal ni corrección por comparaciones múltiples.** | **P1** | §IV-C, §VI | El paper YA lo dice de forma explícita: el 2σ es heurístico de magnitud, no p-valor. Es una elección de honestidad, no un descuido. Future work: test formal en corpus mayores. |
| W4 | **Corriste en H200 pero el título dice T4/free-tier.** | **P1** | §III-C, §V | Lo que importa para "free-tier" es la **VRAM pico** (medida por celda) como cota superior; eso valida la viabilidad en T4. Solo difieren los tiempos de reloj, y se declara como limitación. |
| W5 | **Cut B con una sola semilla.** | P2 | §III-C, §IV-B | Justificado: varianza de semilla a full-data = ±0.006 (1.7B). La excepción (full-data LoRA) sí lleva 3 semillas. No se oculta. |
| W6 | **Todo A/B sale de un único dataset (Cardiff ES).** | P2 | §V | Limitación declarada. El diseño prioriza profundidad/control sobre amplitud; los tres cortes ya dan 3 ejes distintos. Future work: otros dominios/tareas. |
| W7 | **El "10.000×" compara params entrenables, no totales.** | P2 | §IV-B, Tabla IV | Correcto y es el punto: en PEFT lo que se almacena/reentrena por tarea son los params entrenables (6–12 M), no el backbone congelado. El eje de coste relevante es ese. |
| W8 | **Corpus de Cut C muy pequeños (539–738) → posible infraentrenamiento.** | P2 | §IV-C, §VI | Reconocido como limitación; explica la mayor varianza y el neutral bajo en PE. No invalida la asimetría, que **replica** en ambos tamaños de modelo. |
| W9 | **La afirmación de que Qwen capta mejor neutral por "conocimiento más rico que BETO" es causal y especulativa.** | P2 | §IV-B Per-Class | Es una interpretación, marcada como "suggests". El dato duro (LoRA > encoders en neutral, +0.06/+0.11) es sólido; la causa es hipótesis razonada, no demostrada. |
| W10 | **Los umbrales de cruce (n<7, n≈50) ¿son universales?** | P2 | §IV-A, §V | No, y el paper lo recalca: dependen del suelo de prompting del modelo. Lo universal/reproducible es la **existencia** del desplazamiento con el tamaño, no el número. |
| W11 | **n=4 tiene varianza enorme (±0.089 en 4B).** | P3 | Tabla II | Esperable: adaptar desde 1–2 ejemplos/clase. Se contrae para n=25. Refuerza por qué se necesitan 3 semillas disjuntas. |
| W12 | **Encoders también con una sola semilla.** | P3 | Tabla I | Misma lógica que D4; son anclas de referencia controladas, no el objeto central. |
| W13 | **Excepción CR→PE (3.9σ en 1.7B) que no replica en 4B.** | P3 | §IV-C | Se reporta abiertamente como NO replicada (0.3σ en 4B). Solo la columna ES replica. Es honestidad, y refuerza que ES es el patrón real. |

---

## "Preguntas que NO quieres que te hagan" (semilla para el banco)

1. ¿Por qué RoBERTuito te gana y aun así dices que LoRA "domina"? (W1 — la más peligrosa)
2. Si no hay test estadístico, ¿cómo sabes que la asimetría de Cut C no es ruido? (W3)
3. ¿Tu hardware no contradice la premisa "free-tier" del título? (W4)
4. ¿Evaluar en dev no infla tus resultados de Cut C? (W2)
5. ¿Una sola semilla en Cut B no es frágil para sacar conclusiones de coste? (W5)

---

## Mapa rápido figuras/tablas → uso en defensa

| Recurso | Contenido | Cut | Slide candidata |
|---------|-----------|-----|-----------------|
| Tabla I (`tab:baselines`) | Baselines (TF-IDF, BETO, XLM-R, RoBERTuito†) | B | Contexto/anclas |
| Tabla II (`tab:cutA`) | F1 vs n, ambos modelos + prompting | A | Resultado A |
| Fig. A (`figA_f1_vs_n.pdf`) | Curva F1 vs n con cruce | A | Resultado A (visual) |
| Tabla IV (`tab:cutB`) | Calidad + 4 costes | B | Resultado B |
| Fig. B (`figB_quality_cost.pdf`) | Pareto calidad-params | B | Resultado B (visual) |
| Tabla V (`tab:perclass`) | F1 por clase | B | Análisis neutral |
| Tabla III/`tab:cutC` | Matriz 3×3 transferencia | C | Resultado C |
| Fig. C (`figC_transfer.pdf`) | Heatmaps de transferencia | C | Resultado C (visual) |

---

## Auto-auditoría de este inventario

- **Cifras trazadas**: todas las del inventario (0.567, 0.661, 0.646, 0.758, 0.707, 0.698, 0.681, 0.701, 0.696, ±0.006, ±0.089, 0.031–0.039, 2.6–4.6σ, 6–12 M, ~10.000×, VRAM 4.93/6.79/8.72/15.84/16.7, tiempos, n diminutos 4/7) salen de Tablas I–V y §III–§V de `paper/main.tex`. ✔
- **[externo] marcado**: NF4/Dettmers, mecanismo de Min et al., DoRA/IA³, adapters de Houlsby → señalados como conocimiento externo, no como página del paper. ✔
- **Feedback real**: AÚN NO incorporado. El Bloque 3 es inferencia mía; pendiente de cruzar con el feedback de tutores. ⚠ (acordado contigo)
- **Posible punto a verificar contra fuente antes de afirmarlo en la defensa**: en Tabla II, el 4B en n=50 da 0.666 y el "cruce ≈50" se apoya en que el mejor prompting 4B es 0.652 (k=4); confirmar en la defensa que se cita el *mejor* prompting, no el 0-shot (0.633). Ya verificado aquí contra Tabla II. ✔
- **Incoherencia aparente detectada y resuelta**: VRAM de 4B LoRA aparece como 16.7 GB en Tabla IV pero el texto dice que QLoRA "trae el 4B dentro de 8 GB"; NO es contradicción: el que baja a 6.79 es QLoRA, el 16.7 es LoRA sin cuantizar. Coherente. ✔

---

### Siguiente paso (Gate 0→1)

Si validas este inventario, paso a **Fase 1 (Contenido)**: empiezo por la sección *"Cómo leer este proyecto desde cero"* y luego, en la calibración, te entrego el **primer ítem fácil** (probablemente **Macro-F1**, #3) y el **primer ítem difícil** (probablemente **LoRA: ΔW=AB**, #1) para que ajustes nivel y registro antes de seguir de corrido.

**Para mejorar el Bloque 3: pásame el feedback real de tus tutores cuando puedas.**
