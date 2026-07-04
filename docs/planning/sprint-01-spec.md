# Sprint 1 — Arranque (fases 1, 2 y 3)  ·  [afilado]

**Proyecto:** Adaptación eficiente (LoRA/QLoRA) de LLMs abiertos pequeños para análisis de sentimiento en español — estudio empírico unificado de **calidad, coste y robustez** bajo presupuesto de cómputo fijo (GPU gratuita).

**Presupuesto del sprint:** ~25 h (de 180 h totales).

## Objetivo
Dejar el proyecto definido y el andamiaje técnico listo para desarrollar en el Sprint 2. Al cerrar, cualquiera debería poder clonar el repo, leer las specs y saber qué se construye, con qué datos y cómo se mide.

## Pregunta de investigación (tres cortes de un mismo experimento)
Para sentimiento en español, y bajo un presupuesto de cómputo fijo:
- **(A) Punto de cruce datos–método.** ¿A partir de cuántos ejemplos etiquetados un LLM pequeño con LoRA supera al mismo modelo en *few-shot prompting*, y cómo depende del tamaño del modelo?
- **(B) Generativos vs. encoders.** ¿Los LLM generativos pequeños con LoRA superan a los encoders establecidos en español (BETO/RoBERTuito) en la frontera calidad–coste, o solo los igualan a mayor coste?
- **(C) Robustez dialectal.** ¿Cuánto se transfiere un adaptador LoRA entre variedades del español (y desde el inglés)?

**Contribución:** un estudio reproducible, bajo presupuesto *free-tier*, que entrega tres figuras de una sola tubería: (A) curva de eficiencia de datos con punto de cruce, (B) frontera de Pareto calidad–coste generativos-vs-encoders, (C) mapa de transferencia entre dialectos.

## Alcance y prioridad (importante)
- **Núcleo comprometido: A + B.** Comparten la rejilla central del Sprint 3.
- **Ampliación planificada: C.** Reutiliza la maquinaria con un solo modelo (Sprint 4). Es el colchón de horas: la primera en flexionar/recortarse si el presupuesto aprieta. Depende de conseguir datos etiquetados por variedad.

## Datos (bloqueados)
- **Principal (A y B), sin fricción:** `cardiffnlp/tweet_sentiment_multilingual`, subconjunto español. 3 clases, splits listos, descarga directa con `datasets`.
- **Para C — etiquetado por variedad:** TASS (SEPLN), cuyos subconjuntos InterTASS van por país (ES, PE, CR, UY, MX…). Requiere firmar licencia → **arrancar el trámite YA** (T3). Si no llega, C pasa a trabajo futuro.
- **Baseline publicado (para B):** pysentimiento (modelos robertuito/beto entrenados con TASS 2020).
- **Extensión opcional:** emoción (TASS 2020 / EmoEvent) solo si sobra todo lo demás.

> Tarea núcleo de A y B: polaridad de 3 clases. Métrica de calidad: **F1 macro**. Métricas de coste: parámetros entrenables, VRAM pico, tiempo de entrenamiento, latencia de inferencia.

## Modelo(s) base candidatos
- **Caballo de batalla (1–4B):** variante pequeña de Qwen3 o Llama-3.x multilingüe, para correr muchas configuraciones en GPU gratuita.
- **Punto extra (si da tiempo):** un 7–8B vía QLoRA (p. ej. Llama-3.1-8B-Instruct).
- El research-scout confirma el mejor LLM pequeño multilingüe actual (T2/T3).

## Tareas

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| T1 | Redactar `specs.md`: las tres preguntas (A/B/C), alcance, prioridad (A+B núcleo, C colchón) y métricas. | tú + project-planner | 3 h |
| T2 | SOTA inicial por corte: PEFT en LLMs, sentimiento en español, panorama de baselines encoder, robustez dialectal de PEFT; anchor clínico catalán; ≥10 refs al `.bib`; delimitar el hueco de A, B y C (una frase cada uno). | research-scout | 8 h |
| T3 | Decisión de datos y modelos: confirmar `cardiff` (A/B); **iniciar la licencia TASS** y confirmar corpus por variedad (C); fijar modelos base candidatos. | research-scout + tú | 4 h |
| T4 | Repo y estructura (`src/`, `data/`, `configs/`, `results/`, `tests/`, `paper/`, `diary/`, `docs/`) + entorno reproducible (requirements/uv, semillas, README). | ml-dev | 4 h |
| T5 | Instalar agentes en `.claude/agents/` y probar el flujo orquestador → subagentes. | tú | 2 h |
| T6 | Fase 3: analizar los casos reales de profesionales y anotar la conexión con la motivación. | tú + memoir-writer | 2 h |
| T7 | Abrir el diario y registrar el sprint; crear el esqueleto del paper IEEE. | memoir-writer | 2 h |

**Total: ~25 h.**

## Criterios de aceptación
- `specs.md` con las tres preguntas, prioridad y métricas (F1 macro + coste).
- Estado del arte inicial en `docs/research/` con el hueco de A, B y C en una frase cada uno y `.bib` con ≥10 referencias verificadas.
- Datos decididos: `cardiff` confirmado; **trámite de licencia TASS iniciado** (o alternativa dialectal identificada para C).
- Modelos base candidatos fijados.
- Repo reproducible que arranca; agentes funcionando.
- Diario del Sprint 1 cerrado y esqueleto del paper creado.

## Riesgos y notas
- **Integridad / alcance:** confirmar con David Contreras (a) uso de IA permitido, (b) nivel de novedad esperado, (c) idioma del paper. Cerrar antes de avanzar.
- **C depende de datos por variedad:** si TASS no llega, C → trabajo futuro (previsto y legítimo). El pipeline del Sprint 2 se diseña para enchufar esos datos después sin reescribir.
- **A + B + C es ambicioso aun compartiendo tubería:** el project-planner vigila las 180 h; C es el colchón que se recorta primero, después el modelo de 8B, después semillas (de 3 a 2). Nunca se recorta el rigor (baselines, no mezclar splits).
