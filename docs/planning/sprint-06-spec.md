# Sprint 6 — Extensiones: robustez a ruido real + artefactos interactivos de defensa

**Presupuesto:** ~16 h (~13 h sin la demo opcional).

## Objetivo
Sin tocar el paper congelado, (1) generar un hallazgo nuevo y defendible sobre la robustez de cada método ante el ruido real del español de redes, y (2) convertir los resultados que ya existen en artefactos que el tribunal pueda tocar en la defensa. Todo se entrega como apéndice / material de defensa, no en el cuerpo del paper.

## Prioridad (el deck y el ensayo van antes que esto)
Este sprint son mejoras para la defensa, no sustituyen ensayar. Si el tiempo aprieta, el orden de rendimiento es: T4 (explorador) es lo que más suma a la defensa; T2–T3 (robustez) es el hallazgo nuevo; T5 (demo) es opcional.

## Tareas

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| T1 | **Taxonomía de perturbaciones.** Definir un conjunto controlado y determinista (semilla fija) de perturbaciones lingüísticas del español informal: sin tildes (á→a, ñ→n), sin emojis, minúsculas/MAYÚSCULAS, alargamientos ("holaaaa"), abreviaturas de chat (q/x/tmb/xq), sin puntuación, y opcional code-switching. Para cada una: función exacta + % de ejemplos del test afectados. **Gate: aprobar antes de correr.** | ml-dev + Beltrán | 2 h |
| T2 | **Prueba de robustez (re-inferencia).** Para cada perturbación, re-evaluar SOLO en inferencia (cero reentrenamiento) los modelos clave: LoRA-1.7B, LoRA-4B, prompting (k=4), BETO, XLM-R. Medir Macro-F1, caída absoluta y relativa vs limpio, y caída por clase (¿se hunde la neutral?). Guardar en `results/robustness_results.csv`. | ml-dev | 5 h |
| T3 | **Figura de robustez + mini-análisis.** Barras o heatmap: Δ F1 por perturbación × método. Mensaje: quién aguanta el ruido real. Mini-análisis de 2–3 párrafos para apéndice/defensa (qué duele más a quién y por qué), en español, cada cifra trazada al CSV. | ml-dev + memoir-writer | 2 h |
| T4 | **Explorador interactivo (HTML).** HTML autocontenido con 3 vistas cableadas a los CSVs reales: (a) frontera de Pareto calidad-coste con toggle parámetros/VRAM (prototipo ya hecho por Claude), (b) curva de cruce F1-vs-n con slider de tamaño de modelo, (c) matriz de transferencia dialectal interactiva. Sin dependencias externas salvo CDN permitido; datos = CSVs del repo, cero inventado. | Beltrán + Claude | 3 h |
| T5 | **Demo en vivo (opcional, puesta en producción de fase 5).** App mínima (Gradio/Streamlit ligero o CLI) que carga el mejor adaptador y clasifica una frase en español en tiempo real, con etiqueta y probabilidad, al lado del baseline de prompting. Sin sobreingeniería. Corre en T4 o CPU para el 1.7B. | ml-dev | 3 h |
| T6 | **QA + integración.** Verificar: `robustness_results.csv` trazable; figura de robustez coherente; `explorer.html` abre y las 3 vistas cargan; (si se hizo T5) la demo lanza y clasifica; todo trackeado en origin; el paper intacto. Diario del sprint. | qa-validator | 1.5 h |

**Total: ~16 h (~13 h sin T5).**

## Criterios de aceptación
- Prueba de robustez completa: caída de F1 por perturbación × método, con caída por clase.
- Figura de robustez + mini-análisis (apéndice/defensa), cada cifra trazada a `robustness_results.csv`.
- Explorador interactivo funcional: 3 vistas, datos reales del repo, cero dato inventado.
- (Si se hace T5) demo que carga el adaptador y clasifica en vivo.
- Paper congelado intacto; todo lo nuevo es material de defensa/apéndice, no cuerpo del paper.
- Todo versionado y en origin; working tree limpio.

## Riesgos y notas
- **Re-inferencia, NO reentrenamiento.** Si tienta reentrenar sobre texto perturbado, eso es otro estudio y queda fuera de alcance.
- **Honestidad por encima del titular.** Si la robustez NO respalda "LoRA es más robusto", se reporta tal cual. Puede que el prompting aguante mejor el ruido: sería un hallazgo igual de válido y defendible.
- **Sanity check obligatorio antes de perturbar:** reproducir el F1 en el test limpio y confirmar que coincide con el paper (0.698 / 0.707 / 0.652 / 0.661 / 0.646). Sin baseline limpio correcto, las caídas no significan nada.
- **Cero datos inventados** en figuras ni en el explorador: todo sale de un CSV del repo.
- El paper NO se toca en todo el sprint.

## Dependencias
- Adaptadores/checkpoints de los modelos clave guardados y cargables (LoRA-1.7B, LoRA-4B).
- Test de Cardiff ES y los CSVs de Cut A/B/C ya en origin (para robustez y para el explorador).
