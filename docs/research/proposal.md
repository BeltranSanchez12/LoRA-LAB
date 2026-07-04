# Propuesta de Datos y Modelo Base — Sprint 1 (T3)
**Proyecto:** Análisis de sentimiento en español con PEFT de LLMs pequeños
**Fecha:** 2026-06-04
**Estado:** PROPUESTA PARA APROBACION DEL USUARIO — ninguna decision es final

---

## ADVERTENCIA

Todo lo de este documento es una propuesta razonada. Nada esta fijado. El usuario debe aprobar o modificar antes de avanzar al Sprint 2.

---

## 1. Dataset principal — Cortes A y B

### 1.1 Candidato principal: cardiffnlp/tweet_sentiment_multilingual (subconjunto ES)

**Referencia:** Barbieri et al. (2022), "XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond", EMNLP 2022.

**Descripcion:**
- Dataset multilingüe de sentimiento en tweets, con subconjunto español.
- Etiquetas: 3 clases (positive, negative, neutral).
- Tamaño del subconjunto español: ~13,000 tweets (train ~7,000 / dev ~1,000 / test ~2,000 aproximadamente — verificar splits exactos en HuggingFace al descargar).
- Fuente de etiquetado: crowdsourcing + anotacion automatica asistida.
- Disponible directamente en HuggingFace Datasets: `cardiffnlp/tweet_sentiment_multilingual`, config `spanish`.
- Licencia: CC BY 4.0.
- Splits predefinidos: SI (train/validation/test), lo que garantiza comparabilidad.

**Argumentos a favor:**
1. Es el dataset mas usado en benchmarks recientes de sentimiento multilingüe en tweets; permite comparacion directa con la literatura (Barbieri et al. 2022 informan baselines de XLM-RoBERTa, mBERT, etc.).
2. Tiene splits fijos, lo cual es critico para el corte A (curva de aprendizaje con subsets crecientes del train).
3. Acceso inmediato, sin registro ni licencia especial.
4. Los tweets cubren espanol mezclado de variedades (mayoritariamente Spain + Mexico + Argentina), lo que da representatividad suficiente para cortes A y B.
5. CardiffNLP provee tambien modelos pre-fine-tuned sobre este dataset como baselines.

**Argumentos en contra / riesgos:**
1. El etiquetado automatico puede introducir ruido (estimado ~10-15 % de discrepancia en revision manual segun el paper).
2. El tamaño del subconjunto español es moderado; para el corte A (curva de aprendizaje) necesitamos solo el subset de train, que puede ser limitante en los extremos de la curva (pocas muestras: usar stratified sampling).
3. No segmenta por variedad dialectal (no sirve directamente para el corte C).
4. Los tweets tienen fecha (2019-2021 aproximadamente); puede haber drift de topico.

**Veredicto provisional:** RECOMENDADO como dataset principal para cortes A y B. Es la opcion con mejor relacion accesibilidad / reproducibilidad / comparabilidad bibliografica.

---

### 1.2 Alternativas evaluadas

#### TASS 2020 — General Track (SEPLN)

**Descripcion:**
- Tweets en español (principalmente Spain), 3 clases (P, N, NEU).
- Tamaño: ~2,800 train / ~800 dev / ~1,600 test aprox. (varia por track).
- Etiquetado: manual por anotadores nativos (mayor calidad que Cardiff).
- Acceso: via web SEPLN (tass.sepln.org); requiere registro con email institucional o academico; datos gratuitos pero con licencia de uso academico y prohibicion de redistribucion.

**Pros:**
- Etiquetado de mayor calidad.
- Referencia en literatura española de NLP (pysentimiento, BETO-sentiment, etc. reportan sobre TASS 2020).
- Permite comparacion directa con baselines de pysentimiento (F1 ~0.743).

**Contras:**
- Requiere registro y aprobacion manual (puede tardar dias).
- Licencia academica: no redistribuible, lo que complica la reproducibilidad completa del experimento.
- Tamaño pequeño (2,800 train) limita el extremo alto de la curva de aprendizaje del corte A.
- Solo cubre variedad española de España.

**Veredicto:** ALTERNATIVA SECUNDARIA. Util como dataset de validacion cruzada si se obtiene acceso. Para el corte A el tamaño es un limitante. Para el corte B es referencia obligatoria para comparar con pysentimiento.

**Estrategia combinada posible:** Entrenar en Cardiff ES, evaluar tambien en TASS 2020 test (si se obtiene acceso) para comparar con los baselines publicados de pysentimiento. Esto maximiza comparabilidad sin depender de TASS para el entrenamiento.

#### MeOffendEs (SemEval 2021 Task 7 / IberLEF)

**Descripcion:** Deteccion de ofensas en español. **No es un dataset de sentimiento polarizado (POS/NEG/NEU)**. No es adecuado para este proyecto como dataset principal. Podria usarse en futuros estudios de robustez de dominio.

#### SSEC — Stance and Sentiment En Español / STS datasets

No hay un dataset SSEC establecido en español con las 3 clases requeridas y splits reproducibles de acceso libre. No se recomienda.

#### SentimentoPT / otros lusófonos o bilingüles

No aplicables directamente (son en portugues o mezclan idiomas sin segmentacion clara del español).

---

## 2. Dataset dialectal — Corte C

### 2.1 Candidato principal: TASS InterTASS (SEPLN)

**Descripcion:**
- Subconjunto de TASS con tweets segmentados por variedad del español.
- Variedades disponibles (segun edicion): España (ES), Mexico (MX), Peru (PE), Costa Rica (CR), Cuba (CU), Uruguay (UY), Venezuela (VE).
- Tamaño por variedad: muy desigual; algunas variedades tienen solo ~200-400 tweets de train (CR, CU, VE), otras mas (~600-1,000 para ES, MX).
- Etiquetas: 3 clases (P, N, NEU).
- Acceso: mismo proceso que TASS 2020 (registro en SEPLN); mismo tipo de licencia academica.

**Pros:**
- Es el UNICO recurso publico (academico) con tweets de sentimiento segmentados por variedad del español con etiquetado manual.
- Permite experimentos de transferencia directa: entrenar en ES → evaluar en MX, PE, VE, etc.
- Usado en trabajos previos (Perez et al. 2022/RoBERTuito) lo que da baselines de comparacion.

**Contras:**
- Tamaño muy pequeño en algunas variedades (riesgo de alta varianza en métricas).
- Licencia no redistribuible (misma restriccion que TASS 2020).
- Los datos son de 2017-2018 (tweets algo antiguos).
- El proceso de solicitud de acceso puede no ser inmediato.

**Licencia y acceso:**
- URL de solicitud: http://tass.sepln.org/tass_data/download.php (verificar que sigue activa).
- Se solicita con email academico. El acceso suele concederse en 1-3 dias habiles. Sin costo economico.
- Los datos NO se pueden subir a repositorios publicos ni redistribuir.
- Para garantizar reproducibilidad del corte C, el codigo del proyecto debe incluir scripts de descarga + preprocesado que el usuario final ejecute con sus propias credenciales.

**Veredicto:** RECOMENDADO para el corte C. Es la unica opcion viable con la granularidad dialectal requerida. La restriccion de redistribucion es manejable documentando el proceso de descarga.

### 2.2 Alternativas si TASS InterTASS no llega

| Alternativa | Descripcion | Viabilidad |
|---|---|---|
| **AMI / HatEval por variedad** | Datasets de IberEval con tweets en español; algunos tienen metadatos de pais. No son de sentimiento puro. | Baja |
| **Geo-tagged Twitter scraping propio** | Recolectar tweets geolocalizados por pais con la API de Twitter/X. | Muy difícil (API de pago desde 2023) |
| **COARLA corpus** | Corpus de redes sociales en español por variedades (proyecto de investigacion). Acceso academico. No verificado si incluye etiquetas de sentimiento. | A explorar |
| **Construccion manual de un mini-corpus dialectal** | Anotar ~300-500 tweets por variedad con GPT-4 + revision humana. | Plan de contingencia si TASS no llega en tiempo |

**Plan de contingencia recomendado:** Si a las 2 semanas de solicitud TASS no entrega acceso, construir un mini-corpus de validacion dialectal con 5 variedades x 200 tweets usando tweets publicos + etiquetado semi-automatico (con GPT-4 como anotador + acuerdo entre anotadores sobre muestra). Esto reduce comparabilidad bibliografica pero mantiene el corte C ejecutable.

---

## 3. Modelo base candidato(s)

### 3.1 Criterios de evaluacion

Para el "caballo de batalla" (1-4B, T4 16 GB free-tier con QLoRA):

1. **VRAM con QLoRA (4-bit):** El modelo base cuantizado debe caber en ~10-12 GB (dejar margen para activaciones y gradientes de LoRA, ~2-4 GB).
2. **Soporte multilingue / español:** Evidencia de datos de entrenamiento en español; rendimiento en benchmarks multilingues (ARC-ES, FLORES, etc.).
3. **Licencia permisiva:** Apache 2.0 o similar; no restricciones de uso comercial ni de investigacion (aunque el proyecto es academico, la licencia permisiva facilita la reproducibilidad y publicacion de resultados).
4. **Rendimiento base en tareas de NLU:** Benchmarks como MMLU (español), HellaSwag (ES), o resultados en Open LLM Leaderboard.
5. **Compatibilidad con transformers + PEFT:** Arquitectura verificada con `bitsandbytes`, `peft`, y `trl`.

### 3.2 Evaluacion de candidatos 1-4B

#### Qwen3-1.7B (Alibaba Cloud, 2025)

- **Parametros:** 1.7B. VRAM en QLoRA 4-bit: ~1.5-2 GB base + overhead; cabe holgadamente en T4 16 GB.
- **Multilingue:** Entrenado en datos de 29 idiomas incluyendo español; el español tiene buena cobertura relativa. Qwen3 es la generacion más reciente (mayo 2025) con mejoras significativas sobre Qwen2.5 en instruccion y razonamiento.
- **Licencia:** Apache 2.0.
- **Rendimiento base:** Qwen3-1.7B es sorprendentemente capaz para su tamaño. En benchmarks en español (como los reportados por el Open LLM Leaderboard y evaluaciones de la comunidad), supera a modelos 3B de generaciones anteriores en varias tareas. Soporta modo "thinking" (cadena de razonamiento) aunque no es necesario para clasificacion.
- **Compatibilidad:** Arquitectura Qwen3 soportada en `transformers >= 4.51`, compatible con `bitsandbytes`, `peft`.
- **Riesgo:** Modelo muy reciente (mayo 2025); puede haber menos recursos de la comunidad y alguna incertidumbre sobre comportamiento con QLoRA en configuraciones especificas.
- **Veredicto:** CANDIDATO FUERTE para caballo de batalla. Mejor relacion parametros/rendimiento en español de su rango.

#### Qwen3-4B (Alibaba Cloud, 2025)

- **Parametros:** 4B. VRAM en QLoRA 4-bit: ~3-4 GB base + ~2-3 GB overhead LoRA; total ~5-7 GB. Cabe en T4 con batch size reducido.
- **Multilingue:** Mismas caracteristicas que Qwen3-1.7B pero con mas capacidad; mejor rendimiento esperado en español.
- **Licencia:** Apache 2.0.
- **Rendimiento base:** Significativamente mejor que la version 1.7B. Compite con modelos de 7B de generaciones anteriores en muchas tareas.
- **Compatibilidad:** Igual que Qwen3-1.7B.
- **Riesgo:** Tamaño de batch muy limitado en T4 (puede necesitar gradient checkpointing + batch size=1 con gradient accumulation). Entrenamiento más lento.
- **Veredicto:** CANDIDATO FUERTE. Mejor rendimiento que 1.7B al coste de mayor presion sobre VRAM y tiempo de entrenamiento.

#### Llama-3.2-1B (Meta, 2024)

- **Parametros:** 1B. VRAM en QLoRA 4-bit: ~1 GB base; muy eficiente.
- **Multilingue:** Llama-3.2 incluye soporte multilingüe en su arquitectura, pero el entrenamiento esta dominado por ingles. El español tiene representacion limitada comparado con Qwen3.
- **Licencia:** Llama 3.2 Community License (permite uso comercial con algunas restricciones; aceptable para investigacion academica).
- **Rendimiento base:** Inferior a Qwen3-1.7B en benchmarks multilingues, especialmente en español. En ingles es competitivo.
- **Compatibilidad:** Excelente; arquitectura Llama es la más probada con PEFT/QLoRA.
- **Veredicto:** ALTERNATIVA, no candidato principal para español. Podria usarse como punto de referencia de arquitectura Llama.

#### Llama-3.2-3B (Meta, 2024)

- **Parametros:** 3B. VRAM en QLoRA 4-bit: ~2.5-3 GB base.
- **Multilingue:** Mejor que 1B en español pero aun por debajo de Qwen3 segun evaluaciones de la comunidad (Open LLM Leaderboard, lm-evaluation-harness en español).
- **Licencia:** Llama 3.2 Community License.
- **Rendimiento base:** Bueno en ingles; aceptable en español.
- **Compatibilidad:** Excelente.
- **Veredicto:** CANDIDATO SECUNDARIO. Util para comparar arquitecturas Llama vs. Qwen en el mismo rango de parametros.

#### Gemma-2-2B (Google DeepMind, 2024)

- **Parametros:** 2B. VRAM en QLoRA 4-bit: ~2 GB base.
- **Multilingue:** Gemma-2 esta entrenada primariamente en ingles con algo de multilingue. El español no es una lengua prioritaria en su preentrenamiento.
- **Licencia:** Gemma Terms of Use (permite investigacion y uso no comercial; restricciones en uso comercial — aceptable para el proyecto).
- **Rendimiento base:** Excelente para 2B en ingles (supera muchos modelos 7B anteriores en MMLU ingles). En español es inferior a Qwen3-1.7B segun evaluaciones multilingues.
- **Compatibilidad:** Buena con transformers recientes.
- **Veredicto:** NO RECOMENDADO como candidato principal para sentimiento en español. El español no es punto fuerte de Gemma-2-2B. Podria ser interesante para analisis de transferencia (modelo menos especializado en español).

#### Phi-3.5-mini (Microsoft, 2024)

- **Parametros:** 3.8B. VRAM en QLoRA 4-bit: ~3-4 GB base.
- **Multilingue:** Phi-3.5-mini tiene soporte multilingüe pero con un enfoque en calidad de razonamiento sobre diversidad lingüistica. El español esta presente pero no es prioritario como en Qwen.
- **Licencia:** MIT License (completamente permisiva).
- **Rendimiento base:** Muy competitivo en ingles para su tamaño. En español: aceptable pero inferior a Qwen3 en benchmarks multilingues.
- **Compatibilidad:** Buena.
- **Veredicto:** ALTERNATIVA INTERESANTE por su licencia MIT y rendimiento en razonamiento. No es la primera opcion para español pero tiene ventaja en licencia.

### 3.3 Tabla comparativa candidatos 1-4B

| Modelo | Params | VRAM QLoRA (est.) | Esp. soporte | Licencia | Recomendacion |
|---|---|---|---|---|---|
| Qwen3-1.7B | 1.7B | ~3-4 GB | Alto | Apache 2.0 | **PRINCIPAL** |
| Qwen3-4B | 4B | ~5-7 GB | Alto | Apache 2.0 | **ALTERNATIVA FUERTE** |
| Llama-3.2-3B | 3B | ~4-5 GB | Medio | Llama 3.2 | Secundaria |
| Llama-3.2-1B | 1B | ~2-3 GB | Medio-bajo | Llama 3.2 | Referencia arquitectural |
| Gemma-2-2B | 2B | ~3-4 GB | Bajo-medio | Gemma ToU | No recomendado principal |
| Phi-3.5-mini | 3.8B | ~5-6 GB | Medio | MIT | Alternativa por licencia |

**Estimaciones de VRAM con QLoRA 4-bit (NF4) incluyen:** modelo base cuantizado + matrices LoRA en fp16 + cache de KV + activaciones para batch_size=4 con seq_len=128. Con gradient checkpointing pueden reducirse. Son estimaciones; verificar empiricamente.

### 3.4 Recomendacion de candidatos de caballo de batalla

**Candidato primario: Qwen3-1.7B**
- Razon: mejor soporte de español en su rango de parametros, licencia Apache 2.0, cabe con margen en T4, familia Qwen tiene excelente historial en multilingue (Qwen2.5 ya lideraba benchmarks de español antes de Qwen3). El tamaño 1.7B permite batch sizes mas grandes, experimentos mas rapidos y curvas de aprendizaje con mas puntos en el corte A.

**Candidato secundario: Qwen3-4B**
- Razon: mismo ecosistema y licencia que el principal; permite evaluar el trade-off calidad/coste dentro de la misma familia (controlando la variable de arquitectura). Con QLoRA cabe en T4 pero con batch size limitado.
- Alternativa: si se quiere diversidad de arquitectura, Llama-3.2-3B es el segundo mejor candidato.

**Justificacion de excluir Gemma y Phi como principales:** Gemma-2-2B tiene rendimiento inferior en español para este tipo de tarea. Phi-3.5-mini es interesante pero su preentrenamiento prioriza razonamiento sobre diversidad lingüistica.

### 3.5 Candidato 7-8B para "punto extra" (via QLoRA)

Para el rango 7-8B con QLoRA en T4 16 GB:

- **Qwen3-8B** (Apache 2.0): Continua la familia. VRAM estimada en QLoRA 4-bit: ~8-10 GB. Deberia caber con batch_size=1 y gradient checkpointing. Mejor soporte de español que cualquier alternativa en este rango. **Candidato recomendado.**
- **Llama-3.1-8B** (Llama 3.1 License): Muy conocido, bien probado con QLoRA, pero ligeramente inferior en español vs. Qwen3-8B segun benchmarks 2025.
- **Mistral-7B-v0.3** (Apache 2.0): Clasico, bien soportado, pero sin datos de entrenamiento en español de forma especializada. Inferior en español a Qwen3.
- **Gemma-2-9B** (Gemma ToU): 9B puede no caber en T4 con QLoRA 4-bit (estimacion: ~10-12 GB; en el limite; riesgo de OOM).

**Recomendacion para "punto extra":** Qwen3-8B con QLoRA 4-bit + gradient checkpointing + batch_size=1 + gradient_accumulation_steps=8. Si hay OOM en T4, escalar a Kaggle (P100 16 GB) o usar quantizacion mas agresiva (NF4 + double quantization).

---

## 4. Baselines encoder para el corte B

### 4.1 Referencia principal: pysentimiento (RoBERTuito)

**Modelo:** `pysentimiento/robertuito-sentiment-analysis`
- Fine-tuned sobre TASS 2020 + InterTASS.
- F1 macro 3 clases en TASS 2020 test: ~0.743 (segun Perez et al. 2022).
- Disponible en HuggingFace, acceso libre, sin licencia especial.
- **Es la referencia correcta y mas usada en la literatura reciente para sentimiento en español en tweets.**

**Complemento:** `dccuchile/bert-base-spanish-wwm-cased` (BETO) fine-tuned sobre TASS 2020 como referencia adicional de encoder clasico.

### 4.2 Hay algo mas reciente?

- **XLM-T** (Barbieri et al. 2022): XLM-RoBERTa fine-tuned sobre el dataset Cardiff multilingual. Es el baseline oficial del dataset Cardiff ES. Disponible en `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`. F1 reportado en Cardiff ES test: ~0.72-0.74 segun el paper. **Deberia incluirse como baseline adicional** ya que usa el mismo dataset que el experimento principal.

- **MarIA-RoBERTa** (Gutierrez-Fandino et al. 2022): `PlanTL-GOB-ES/roberta-large-bne`. Preentrenado en corpus BNE (formal). Fine-tuning en TASS 2020 da resultados similares a BETO pero algo mejores en texto formal. En tweets puede ser inferior a RoBERTuito.

- **No se identifico** ningun encoder español de 2024-2025 que supere significativamente a RoBERTuito en sentimiento de tweets. pysentimiento sigue siendo el estado del arte para encoders en tweets en español.

**Recomendacion:** Usar como baselines encoder:
1. `pysentimiento/robertuito-sentiment-analysis` (principal)
2. `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` (compatible con Cardiff ES)
3. `dccuchile/bert-base-spanish-wwm-cased` fine-tuned en Cardiff ES (para mostrar progresion BETO → RoBERTuito → LLM)

---

## 5. Resumen de propuesta

| Componente | Propuesta | Estado |
|---|---|---|
| Dataset principal (A+B) | `cardiffnlp/tweet_sentiment_multilingual` ES | Para aprobacion |
| Dataset validacion cruzada B | TASS 2020 (si se obtiene acceso) | Para aprobacion |
| Dataset dialectal C | TASS InterTASS (solicitar acceso SEPLN) | Para aprobacion |
| Modelo caballo de batalla principal | Qwen3-1.7B | Para aprobacion |
| Modelo caballo de batalla secundario | Qwen3-4B | Para aprobacion |
| Modelo punto extra | Qwen3-8B via QLoRA | Para aprobacion |
| Baseline encoder principal | pysentimiento/robertuito-sentiment-analysis | Para aprobacion |
| Baseline encoder adicional | XLM-T (cardiffnlp) | Para aprobacion |
| Metodo PEFT | LoRA + QLoRA (libreria PEFT + TRL) | Para aprobacion |

---

*Este documento requiere aprobacion del usuario antes de proceder a configuracion de experimentos.*
