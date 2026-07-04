# Diario — Sprint 1: Arranque

**Fecha de cierre:** 2026-06-04
**Sprint:** 1 — Arranque (fases 1, 2 y 3)
**Duración estimada:** ~25 h de 180 h totales
**Elaborado por:** memoir-writer (claude-sonnet-4-6)

---

## Objetivo del sprint

Dejar el proyecto definido y el andamiaje técnico listo para desarrollar en el Sprint 2. Al cerrar, cualquiera debería poder clonar el repo, leer las specs y saber qué se construye, con qué datos y cómo se mide.

**Estado al cierre:** Objetivo alcanzado en su mayor parte. El andamiaje técnico está operativo y la investigación inicial está documentada. Quedan dos bloqueos abiertos que no dependen de los agentes sino de aprobaciones externas: (1) la propuesta de datasets y modelos base pendiente de aprobación del usuario, y (2) la licencia TASS pendiente de tramitar con SEPLN. Ver sección "Decisiones pendientes".

---

## Tareas completadas

| # | Tarea | Agente | Artefacto producido | Estado |
|---|-------|--------|---------------------|--------|
| T1 | Redactar specs: preguntas A/B/C, alcance, prioridad y métricas | project-planner | `docs/planning/sprint-01-spec.md` | Completo |
| T2 | SOTA inicial: PEFT en LLMs, sentimiento en español, robustez dialectal, anchor clínico; ≥10 refs al .bib | research-scout | `docs/research/sota.md`, `docs/research/references.bib` | Completo (20 refs verificadas conceptualmente) |
| T3 | Propuesta de datos y modelos: Cardiff confirmado, evaluación de candidatos TASS y modelos 1-4B | research-scout | `docs/research/proposal.md` | Completo — pendiente aprobación del usuario |
| T4 | Repo y estructura + entorno reproducible (requirements, semilla, README) | ml-dev | `README.md`, `requirements.txt`, `src/utils/seed.py`, `configs/example_experiment.yaml`, estructura de directorios | Completo |
| T5 | Instalar agentes en `.claude/agents/` y probar flujo orquestador → subagentes | project-planner / usuario | `.claude/agents/` | Completo |
| T6 | Fase 3: casos reales y conexión con la motivación del proyecto | memoir-writer / usuario | (pendiente confirmación de entrega separada) | No documentado como artefacto explícito |
| T7 | Abrir el diario y crear el esqueleto del paper IEEE | memoir-writer | `diary/sprint-01.md`, `paper/main.tex` | Completo (este fichero) |

### Nota sobre uso de IA

Todos los agentes implicados en este sprint son instancias del modelo **claude-sonnet-4-6** (Anthropic), operando como subagentes especializados dentro del sistema de orquestación de Claude Code:

- **project-planner:** redacción de la spec del sprint, definición de las preguntas de investigación y estructura de métricas.
- **research-scout:** revisión del estado del arte, evaluación de datasets y modelos candidatos, mantenimiento del fichero `.bib`.
- **ml-dev:** estructura del repositorio, ficheros de entorno reproducible (`requirements.txt`, `seed.py`, YAML de ejemplo), README.
- **memoir-writer:** cierre del diario y esqueleto del paper (este agente).

El uso de IA en este proyecto está declarado conforme a la política pendiente de confirmar con David Contreras (ver "Decisiones pendientes"). Todo el contenido generado por IA está identificado por agente en este diario.

---

## Decisiones tomadas

### Idioma del paper
**Inglés.** La asignatura lo permite y la audiencia objetivo (venues internacionales de NLP) requiere inglés. Decisión tomada por el usuario al iniciar el sprint.

### Política de uso de IA
**Uso ilimitado de IA permitido** en este proyecto, con obligación de declaración explícita. Esta política está recogida en el diario de cada sprint. Pendiente confirmación formal con David Contreras antes del cierre del Sprint 2 (ver "Riesgos activos").

### Política de commits
Los commits del repositorio no llevan atribución explícita de IA en el mensaje de commit (por decisión del usuario). La trazabilidad se mantiene en este diario.

### Métricas principales
- **Calidad:** F1 macro (3 clases: positivo, negativo, neutro).
- **Coste:** parámetros entrenables, VRAM pico (GB), tiempo de entrenamiento (s), latencia de inferencia (ms).

### Alcance y prioridad
- **Núcleo comprometido:** cortes A (punto de cruce datos-método) y B (generativos vs. encoders).
- **Ampliación planificada (colchón):** corte C (robustez dialectal). Primera en recortarse si el presupuesto de cómputo aprieta.

### Reproducibilidad
- Semilla global: **42**. Tres semillas por celda en la rejilla principal para reportar media ± desviación.
- Un fichero YAML por experimento en `configs/`.
- Dependencias fijadas en `requirements.txt` (Python 3.11).

### Dataset configurado en el andamiaje (provisional)
El fichero `configs/example_experiment.yaml` ya referencia `cardiffnlp/tweet_sentiment_multilingual` (ES) como dataset, pero el campo `base_model` está marcado como `"TBD"` hasta la aprobación del usuario.

---

## Decisiones pendientes de aprobación del usuario

Las siguientes decisiones han sido propuestas por research-scout en `docs/research/proposal.md` pero **no son finales** hasta que el usuario las apruebe explícitamente:

| Componente | Propuesta | Acción requerida |
|---|---|---|
| Dataset principal (cortes A y B) | `cardiffnlp/tweet_sentiment_multilingual` ES | Aprobar o modificar |
| Dataset validación cruzada (corte B) | TASS 2020 (si se obtiene acceso SEPLN) | Aprobar o modificar |
| Dataset dialectal (corte C) | TASS InterTASS — solicitar acceso SEPLN | Aprobar e iniciar trámite |
| Modelo caballo de batalla principal | Qwen3-1.7B (Apache 2.0) | Aprobar o modificar |
| Modelo caballo de batalla secundario | Qwen3-4B (Apache 2.0) | Aprobar o modificar |
| Modelo "punto extra" | Qwen3-8B vía QLoRA | Aprobar o modificar |
| Baselines encoder | pysentimiento/robertuito + XLM-T (cardiffnlp) + BETO fine-tuned | Aprobar o modificar |
| Método PEFT | LoRA + QLoRA (librería PEFT de Hugging Face + TRL) | Aprobar o modificar |

**El Sprint 2 no puede arrancar hasta que el usuario apruebe (o modifique) la propuesta de datasets y modelos.**

---

## Riesgos activos

### Riesgo 1 — Licencia TASS (impacto ALTO en corte C)
- **Descripción:** El corpus TASS InterTASS (necesario para el corte C de robustez dialectal) requiere firma de licencia académica con SEPLN. El acceso puede tardar 1-3 días hábiles.
- **Acción requerida:** Iniciar el trámite de solicitud YA en http://tass.sepln.org/tass_data/download.php con email académico.
- **Plan de contingencia:** Si TASS no llega en 2 semanas desde la solicitud, el corte C pasa a trabajo futuro (previsto en la spec) o se construye un mini-corpus de validación dialectal semi-automático (5 variedades × 200 tweets, etiquetado con GPT-4 + revisión humana).
- **Estado:** NO INICIADO — bloqueado hasta aprobación del usuario.

### Riesgo 2 — Confirmación con David Contreras (impacto ALTO en integridad académica)
- **Descripción:** La spec del sprint indica que hay que confirmar con David Contreras: (a) uso de IA permitido, (b) nivel de novedad esperado, (c) idioma del paper.
- **Acción requerida:** El usuario debe confirmar estas tres preguntas con David Contreras antes del cierre del Sprint 2.
- **Estado:** PENDIENTE. El proyecto avanza con la hipótesis de que el uso de IA es permitido con declaración explícita (como se recoge en este diario), inglés como idioma del paper, y el nivel de novedad definido por los tres cortes A/B/C.
- **Riesgo si no se confirma:** Podría haber que cambiar el idioma del paper, ajustar el nivel de novedad esperado, o modificar la política de declaración de IA.

### Riesgo 3 — Presupuesto de cómputo (impacto MEDIO en corte C y modelo 8B)
- **Descripción:** El presupuesto total es de ~180 h de GPU (T4 16 GB, free-tier Colab/Kaggle). Los cortes A+B son el núcleo; el corte C y el modelo de 8B son los colchones que se recortan primero.
- **Estado:** Bajo control en Sprint 1. Revisable al inicio de Sprint 3 con datos reales de tiempo de entrenamiento.

### Riesgo 4 — Referencias con DOI no verificado
- **Descripción:** Varios trabajos en `references.bib` tienen la nota `[NO-DOI-VERIFICADO]` o indicaciones de verificación pendiente (p. ej., `garciapablos2024qlora`, `mirandaescalada2023distemist`, `ustun2022udapter`). Algunos trabajos del sota.md sobre LLMs en sentimiento en español tampoco tienen venue exacto verificado.
- **Acción requerida:** En el Sprint 2, research-scout debe verificar los DOIs y venues antes de que memoir-writer los cite en el paper.
- **Estado:** Documentado; no bloquea el Sprint 1 pero sí afectará la sección Related Work del paper.

---

## Artefactos producidos en el sprint

| Fichero | Agente | Descripción |
|---|---|---|
| `docs/planning/sprint-01-spec.md` | project-planner | Spec completa del sprint: preguntas, alcance, prioridad, métricas |
| `docs/research/sota.md` | research-scout | Estado del arte: PEFT, sentimiento ES, robustez dialectal, anchor clínico |
| `docs/research/proposal.md` | research-scout | Propuesta razonada de datasets y modelos (para aprobación) |
| `docs/research/references.bib` | research-scout | 20 referencias BibTeX verificadas conceptualmente |
| `README.md` | ml-dev | Documentación del proyecto y estructura del repo |
| `requirements.txt` | ml-dev | Dependencias fijadas (Python 3.11, torch, transformers, peft, trl, etc.) |
| `src/utils/seed.py` | ml-dev | Fixture de semilla global (42) para reproducibilidad |
| `configs/example_experiment.yaml` | ml-dev | Plantilla de config para experimentos (dataset, modelo, PEFT, training) |
| `diary/sprint-01.md` | memoir-writer | Este fichero |
| `paper/main.tex` | memoir-writer | Esqueleto del paper en plantilla IEEE |

---

## Próximos pasos — Sprint 2

El Sprint 2 puede arrancar una vez el usuario apruebe la propuesta de datasets y modelos.

Tareas previstas para Sprint 2 (sujeto a planificación detallada de project-planner):

1. Aprobar propuesta de datasets/modelos (usuario) — bloqueante.
2. Iniciar trámite de licencia TASS (usuario) — bloqueante para corte C.
3. Confirmar política de uso de IA con David Contreras (usuario) — bloqueante para integridad.
4. Implementar pipeline de datos: descarga Cardiff ES, preprocesado, splits para curva de aprendizaje (corte A).
5. Implementar pipeline de entrenamiento: LoRA/QLoRA sobre modelo aprobado, integrado con configs YAML.
6. Implementar evaluación: F1 macro + métricas de coste, guardado estructurado en `results/`.
7. Experimentos piloto: un punto de la curva A, un baseline encoder B, para validar el pipeline de extremo a extremo.
8. Actualizar el diario (Sprint 2) y las secciones Methodology y Related Work del paper con la información ya disponible.

---

*Diario cerrado: 2026-06-04 — memoir-writer (claude-sonnet-4-6)*
