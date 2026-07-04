# Sprint 4 — Corte C: transferencia dialectal (fase 4, ampliación)

**Presupuesto:** ~25 h. **Estatus: ampliación/colchón.** Es la primera que flexiona o pasa a trabajo futuro si las horas o los datos aprietan.

## Decisión de entrada (gate — léela antes de invertir horas)
Confirma que tienes datos etiquetados **por variedad** de español (TASS InterTASS u otro: subconjuntos por país, p. ej. ES, PE, CR, UY, MX…). Si la licencia no ha llegado o los datos son insuficientes → **C pasa a "trabajo futuro"**: se documenta en el paper como limitación + línea futura y saltas directo al Sprint 5. No fabriques datos dialectales (ni con GPT-4): rompería la validez del corte.

## Objetivo
Medir cuánto se transfiere un adaptador LoRA **entre variedades** del español (y, opcional, desde el inglés), usando el mejor modelo pequeño del Sprint 3. Figura estrella: **heatmap de transferencia** (entrena en X → evalúa en Y).

## Diseño
- **Un solo modelo** (el mejor generativo pequeño del Sprint 3 con LoRA). Aquí NO se barren modelos ni métodos: C va de transferencia, no de comparar arquitecturas.
- Un adaptador LoRA **por variedad disponible**.
- **Evaluación cruzada:** cada adaptador sobre el test de cada variedad → matriz N×N.
- Referencias: un adaptador entrenado sobre "todas las variedades juntas" (techo) y, opcional, inglés→español (transferencia cross-lingual).
- Varias semillas (los conjuntos por variedad son pequeños, ~200-400 de train).

## Tareas

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| T1 | Integrar los datos por variedad vía la interfaz enchufable del Sprint 2: splits train/test por variedad, sin solapamiento. | ml-dev | 4 h |
| T2 | Entrenar un adaptador LoRA por variedad sobre el mejor modelo del S3 (en Colab/GPU). | ml-dev | 7 h |
| T3 | Evaluación cruzada: cada adaptador × test de cada variedad → matriz; + referencia "all-varieties" y opcional EN→ES. | ml-dev | 6 h |
| T4 | Heatmap de transferencia (calidad de publicación), reproducible desde script. | ml-dev | 3 h |
| T5 | qa-validator: matriz completa, splits por variedad sin solapamiento, varianza/significancia entre semillas. | qa-validator | 3 h |
| T6 | memoir-writer: diario + subsección de resultados "transferencia dialectal" con el heatmap. | memoir-writer | 2 h |

**Total: ~25 h.**

## Criterios de aceptación
- Matriz de transferencia completa (todas las celdas), con semillas, en `results/`.
- Heatmap generado y reproducible.
- Resultados con media ± desviación; afirmaciones firmadas por qa-validator.
- Subsección del paper redactada.
- (Si C → trabajo futuro: en su lugar, un párrafo en el paper documentando la limitación y la línea futura, y el sprint se cierra ahí.)

## Riesgos y notas
- **Datos escasos por variedad:** con 200-400 ejemplos hay ruido; por eso varias semillas y reporte honesto. Si una variedad tiene muy pocos datos, exclúyela y dilo.
- **Si la transferencia sale pobre o ruidosa, ES un hallazgo** ("los adaptadores no transfieren bien entre variedades"). Repórtalo; no fuerces la figura.
- Cómputo en Colab: scripts reanudables y resultados persistidos.

## Dependencias
- Gate de datos por variedad (TASS u otro) + el mejor modelo y el harness del Sprint 3.
