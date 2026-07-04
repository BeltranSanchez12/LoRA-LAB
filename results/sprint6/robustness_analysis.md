# Robustez al ruido del español de redes — análisis (Sprint 6, T2–T3)

**Material de apéndice/defensa. El paper NO se toca.**
Fuente única de todas las cifras: [`results/robustness_results.csv`](../robustness_results.csv).
Figura: [`results/figures/figF_robustness.png`](../figures/figF_robustness.png).

## Método
Re-inferencia **pura** (pesos congelados, cero reentrenamiento): se perturba únicamente el
texto de entrada del test limpio de Cardiff ES (n=870, semilla 42) y se re-evalúan los 5
modelos clave. El baseline limpio reconstruido reproduce el del paper **bit-exacto**
(0.6922 / 0.7220 / 0.6522 / 0.6613 / 0.6465 para LoRA-1.7B / LoRA-4B / prompting k=4 /
BETO / XLM-R), por lo que las caídas Δ = F1(perturbado) − F1(limpio) son interpretables.
Cada perturbación se aplica **aislada** y **determinista** (semilla 42). `code_switching`
es opcional y se reporta aparte (no entra en la figura).

## Hallazgo 1 — En media, LoRA es el más robusto… pero por poco y no es la historia
El orden de robustez (media de Δ Macro-F1 sobre las 7 perturbaciones nucleares, ↑ mejor) es
**LoRA-4B −0.0057 ≈ LoRA-1.7B −0.0065 > prompting k=4 −0.0138 ≫ BETO −0.0239 ≈ XLM-R −0.0248**.
Es decir, *"LoRA es más robusto"* **se sostiene en media**, pero (a) el margen sobre el
prompting es pequeño en las perturbaciones suaves y (b) la media está dominada por un único
efecto brutal en los encoders (siguiente párrafo). Perturbaciones prácticamente inocuas para
todos: `sin_emojis` (Δ≈0 en los 5; solo el 3.1 % del test lleva emoji), `sin_tildes` y
`minusculas` (efecto casi nulo, incluso **positivo** en BETO: +0.016 y +0.004).

## Hallazgo 2 — El efecto dominante: el mayusculado hunde a los encoders *cased*
La celda que manda en el heatmap es `MAYÚSCULAS`: **BETO −0.184** y **XLM-R −0.110**,
frente a un LoRA prácticamente indiferente (−0.011 / −0.012) y un prompting **inmune** (+0.004).
Tiene explicación mecánica: BETO es `bert-base-spanish-wwm-**cased**` y XLM-R también es
sensible a la caja; un tweet TODO EN MAYÚSCULAS es fuertemente fuera-de-distribución para su
tokenización, mientras que los generativos (Qwen, BPE, preentrenamiento masivo) lo absorben.
**Este es el hallazgo defendible del sprint:** la fragilidad no está donde el titular sugería
(no es "LoRA vs prompting"), sino en que **los codificadores *cased* colapsan ante el ruido de
caja más trivial y los modelos generativos no.**

**Fragilidad concentrada, no difusa (importante para no exagerar).** La media pésima de los
encoders (−0.024 BETO / −0.025 XLM-R) está **dominada por un único evento**: si se excluye
`MAYÚSCULAS`, la media de las otras 6 perturbaciones es **+0.003 en BETO** (robusto al resto) y
**−0.011 en XLM-R** (leve). Es decir, el codificador *cased* tiene **un punto de fallo específico
y explicable**, no una fragilidad general: aguanta bien tildes, emojis, minúsculas, alargamientos,
abreviaturas y quitar puntuación. En contraste, la media de LoRA apenas cambia al excluir
mayúsculas (−0.007 → −0.006): su leve sensibilidad es difusa y pequeña. No debe leerse la fila
"media (7)" como "los encoders son peores en todo" — lo son en **un** eje concreto (la caja).

## Hallazgo 3 — El prompting tiene un perfil de fragilidad distinto (honestidad)
El few-shot k=4, pese a aguantar las mayúsculas, es el **más frágil** ante tres perturbaciones
concretas: `sin_tildes` −0.019, `minusculas` −0.030 y `sin_puntuacion` −0.030 (peor que
cualquier LoRA en las tres). Su peor media la arrastran precisamente estas: quitar señales de
formato al prompt degrada más al few-shot que al adaptador entrenado. `alargamientos` castiga
sobre todo a XLM-R (−0.044) y a LoRA-4B (−0.022); `abrev_chat` es un golpe pequeño y bastante
uniforme (~−0.003 a −0.013). Conclusión honesta: **según la perturbación, el prompting cae más
que LoRA; no hay un ganador único de robustez perturbación a perturbación.**

## Ángulo por clase — la neutral NO es la que se hunde (hipótesis refutada)
Contra la hipótesis de partida, el ruido **no** ataca preferentemente a la clase neutral
(desglose por clase en `results/robustness_results.csv`, columnas `delta_f1_*`). En los
generativos la clase **positiva** es la más frágil (media sobre 7 pert.: LoRA-1.7B −0.010,
LoRA-4B −0.009, prompting −0.020); en XLM-R la caída se reparte pareja (neutral −0.027 ≈
positiva −0.028).

**El +0.006 de la neutral en BETO es un ARTEFACTO del colapso, no una mejora.** Bajo
`MAYÚSCULAS`, el F1 de la clase **positiva** de BETO se desploma de 0.732 a **0.375 (−0.357)** y
el de la **negativa** de 0.702 a 0.516 (−0.186), mientras la **neutral** apenas se mueve
(0.550 → 0.541, −0.009). No es que el modelo entienda mejor la neutral: un clasificador degradado
**vierte sus predicciones al cajón "neutral"** (su clase de refugio), lo que sostiene
artificialmente el recall/F1 de la neutral mientras las otras dos se hunden; promediado sobre las
7 perturbaciones eso deja ese **+0.006 engañoso**. Debe leerse así, y **no** como que la neutral
"aguanta mejor" — eso contradiría en falso a la **Figura D** del paper, donde la neutral es la
clase **más débil** en limpio (F1 0.550, la más baja de las tres). Conclusión honesta:
**si alguna clase es sistemáticamente la más frágil es la positiva, no la neutral.**

## Nota — code_switching (opcional, fuera de la figura)
El swap parcial ES→EN (50 % de palabras funcionales, 37.2 % del test afectado) **no degrada** a
la mayoría: LoRA-1.7B +0.006, LoRA-4B +0.002, XLM-R +0.005, BETO −0.001 (todos con base
multilingüe robusta). Solo el few-shot prompting cae de forma apreciable (**−0.023**). Se
reporta como nota y no se incluye en el heatmap principal por editar palabras de contenido
(posible confusión con la señal de sentimiento).

## Cobertura de la taxonomía (test Cardiff ES, n=870; reproducible con `python -m src.data.perturbations`)
| Perturbación | % afectados | | Perturbación | % afectados |
|---|---|---|---|---|
| sin_tildes | 65.9 % | | alargamientos | 95.4 % |
| sin_emojis | 3.1 % | | abrev_chat | 62.5 % |
| minusculas | 78.2 % | | sin_puntuacion | 71.6 % |
| mayusculas | 100.0 % | | code_switching *(opc.)* | 37.2 % |
