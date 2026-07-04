# Sprint 3 — La rejilla central: cortes A y B (fase 4, parte 2)

**Presupuesto del sprint:** ~40 h (el más pesado en cómputo). Corre en Colab/GPU.

## Objetivo
Lanzar la rejilla de experimentos PEFT que, de una sola tubería, produce las **dos figuras estrella** del proyecto:
- **Figura A:** curva de eficiencia de datos con **punto de cruce** (F1 macro vs. nº de ejemplos), una línea por método/tamaño.
- **Figura B:** **frontera de Pareto** calidad–coste con generativos (LoRA/QLoRA/full-FT) y encoders en el mismo plano.

## Diseño de la rejilla (el corazón del sprint)
Un experimento = `{modelo} × {método} × {fracción de datos} × {semilla}`, definido por un fichero de config.

- **Modelos generativos (confirmados):** Qwen3-1.7B (caballo de batalla) y Qwen3-4B (secundario). Qwen3-8B vía QLoRA como **punto extra** solo si sobra tiempo.
- **Métodos:** LoRA, QLoRA, y *full fine-tuning* donde quepa (en el 1.7B); más el **prompting** cuyo protocolo se congeló en el Sprint 2 (zero/few-shot k=0/4/8/16).
- **Fracciones de datos (para A):** las del Sprint 2 — 50 / 100 / 250 / 500 / 1000 / full, estratificadas.
- **Semillas:** 3 por celda (para hablar de significancia).

## Lo que entra de los Sprints anteriores
- Del **Sprint 2**: el harness (calidad + coste idénticos para todo), las fracciones de datos, y los baselines ya medidos.
- **Para la figura B**, los puntos de comparación ya son:
  - Suelo clásico: TF-IDF + LR.
  - Encoders **controlados** (afinados por nosotros en Cardiff ES train, van EN la Pareto): BETO y xlm-roberta-base.
  - Referencia **off-the-shelf** (robertuito, entrenado en TASS): se dibuja como línea/punto de referencia out-of-domain, NO como competidor controlado en la Pareto.
- **Para la figura A**, el rival de LoRA es el prompting del Sprint 2, aplicado **a los dos tamaños** (1.7B y 4B) para completar el eje "tamaño".

## Tareas

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| T1 | Bucle de entrenamiento PEFT parametrizado por config (YAML): un experimento = un fichero; loguea calidad y coste en `results/`. Reanudable (para sobrevivir a desconexiones de Colab). | ml-dev | 8 h |
| T2 | **Barrido de A:** LoRA por fracción de datos × {Qwen3-1.7B, Qwen3-4B} × semillas; recoge F1 vs. nº de ejemplos. Lanza el prompting del Sprint 2 en los mismos puntos y tamaños como rival. | ml-dev | 10 h |
| T3 | **Barrido de B:** a datos completos, LoRA/QLoRA/full-FT en los generativos, recogiendo calidad y coste (params/VRAM/latencia). Coloca también los encoders controlados y la referencia off-the-shelf en el mismo plano. | ml-dev | 8 h |
| T4 | **Figuras de publicación:** (A) curva con punto de cruce; (B) frontera de Pareto generativos-vs-encoders. Reproducibles desde script. | ml-dev | 5 h |
| T5 | **qa-validator:** rejilla completa (sin celdas vacías), varianza razonable entre semillas, significancia (media ± desviación / test), y splits coherentes entre A y B. | qa-validator | 5 h |
| T6 | **memoir-writer:** diario + Metodología (diseño de la rejilla) y primer borrador de Resultados con las figuras A y B. | memoir-writer | 4 h |

**Total: ~40 h.**

## Criterios de aceptación
- Rejilla completa: todas las celdas planificadas corridas con sus semillas; resultados en `results/`.
- Figura A (punto de cruce) y Figura B (Pareto generativos-vs-encoders) generadas, con calidad de publicación y reproducibles desde script.
- Resultados con media ± desviación entre semillas; cada afirmación respaldada (firma del qa-validator).
- Borrador de Metodología y Resultados (con A y B) en el paper.

## Riesgos y notas
- **Cómputo:** es el sprint más caro. En GPU gratuita, planifica las tandas, guarda *checkpoints* y persiste resultados tras cada run. Si el reloj aprieta, recorta en este orden: primero Qwen3-8B (es "punto extra"), luego de 3 a 2 semillas. **Nunca** elimines un baseline ni mezcles splits.
- **Desconexiones de Colab/Kaggle:** scripts reanudables; nada de perder horas de entrenamiento por un *timeout*.
- **Si A no muestra un cruce claro, eso TAMBIÉN es un hallazgo** (p. ej. "el prompting nunca alcanza al fine-tuning en este rango" o "se cruza muy pronto"). Repórtalo honesto; no fuerces la figura.

## Dependencias
- Requiere el **Sprint 2 cerrado** (pipeline, harness y los baselines definitivos, incluido el prompting con su protocolo congelado).
- El Sprint 4 (corte C) reutiliza esta maquinaria y **el mejor modelo pequeño que salga de aquí**.