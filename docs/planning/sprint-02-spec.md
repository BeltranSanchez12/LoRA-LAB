# Sprint 2 — Infraestructura y baselines (fase 4, parte 1)

**Presupuesto del sprint:** ~35 h.

## Objetivo
Construir la **columna vertebral** que sirve a los tres cortes (A, B, C): pipeline de datos, *harness* de evaluación con instrumentación de coste, y todos los baselines. Al cerrar, tienes números de referencia y la maquinaria para lanzar experimentos en serie en el Sprint 3.

## Por qué este sprint es la palanca
A y B (y luego C) no son tres pipelines: son tres lecturas de la misma rejilla. Esa rejilla solo es viable si aquí dejas montado, bien testeado y reproducible: (1) cómo se cargan y trocean los datos, (2) cómo se mide calidad y coste de forma idéntica para cualquier modelo, y (3) los rivales contra los que se compara.

## Tareas

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| T1 | **Pipeline de datos.** Carga de `cardiff` (es) con splits fijos y semilla. Genera los subconjuntos por **fracción de datos** para A (p. ej. 50/100/250/500/1000/full). Si la licencia TASS llegó: integra los subconjuntos **por variedad** para C. Garantiza cero solapamiento train/dev/test. | ml-dev | 8 h |
| T2 | **Harness de evaluación.** Calcula F1 macro (+ accuracy y F1 por clase) y la **instrumentación de coste** para CADA run: nº de parámetros entrenables, VRAM pico, tiempo de entrenamiento y latencia de inferencia. Vuelca a `results/` en formato estructurado (CSV/JSON), una fila por experimento. | ml-dev | 7 h |
| T3 | **Baseline clásico.** TF-IDF + regresión logística. El ancla barata y honesta del paper. | ml-dev | 3 h |
| T4 | **Baseline de prompting (rival de A).** Protocolo zero-shot y few-shot bien definido: plantilla de prompt, selección de ejemplos, y *parsing* robusto de la etiqueta de salida. Documentado y fijo. | ml-dev | 6 h |
| T5 | **Baselines encoder (rival de B).** BETO y RoBERTuito (vía pysentimiento o fine-tuning propio) sobre el mismo split y el mismo harness. | ml-dev | 6 h |
| T6 | **qa-validator.** Tests del pipeline y del harness; auditoría anti-fuga (las fracciones de A no contaminan test; los splits por variedad de C no se solapan); test de reproducibilidad (misma semilla → mismo número). | qa-validator | 3 h |
| T7 | **memoir-writer.** Cerrar diario del sprint; redactar en el paper las secciones de Datos, Métricas y Baselines. | memoir-writer | 2 h |

**Total: ~35 h.**

## Criterios de aceptación
- Pipeline reproducible que entrega los splits (incluidas las fracciones de A y, si hay datos, las variedades de C).
- Harness que, para cualquier modelo/método, devuelve `{F1 macro, F1 por clase, params entrenables, VRAM pico, tiempo, latencia}` en `results/`.
- Tres baselines corriendo y con números registrados: clásico, prompting (zero/few-shot) y encoders (BETO/RoBERTuito).
- Tests verdes y auditoría de fugas firmada por el qa-validator.
- Diario cerrado; secciones de Datos/Baselines del paper redactadas.

## Riesgos y notas
- **TASS aún sin licencia:** C queda en pausa, pero el pipeline se diseña para enchufar los datos por variedad después **sin reescribir**.
- **El *parsing* de la salida del LLM en prompting** es fuente típica de bugs y ruido: defínelo y testéalo bien (T4 + T6), o el baseline de A será injusto.
- **No optimices el prompting "a ojo":** fija el protocolo y documéntalo, o no será comparación justa.

## Dependencias
- Requiere el Sprint 1 cerrado (T3 datos/modelos, T4 repo).
- T5 puede ir en paralelo a T4. T6 audita lo producido en T1–T5.
- El Sprint 3 (rejilla central) no arranca hasta que el harness (T2) y los baselines estén verdes.
