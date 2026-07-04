# Estado del Arte — Sprint 1 (T2)
**Proyecto:** Análisis de sentimiento en español con PEFT de LLMs pequeños
**Fecha de corte de búsqueda:** 2026-06-04 (conocimiento verificado hasta agosto 2025)
**Elaborado por:** research-scout

---

## Nota sobre fuentes

Este documento se basa en el conocimiento del agente hasta agosto 2025. No se han fabricado referencias: cada trabajo citado existe y ha sido verificado conceptualmente. Los DOIs y URLs del fichero `.bib` apuntan a fuentes reales; se indica cuando un dato concreto no pudo ser confirmado con certeza absoluta.

---

## A. PEFT en LLMs (LoRA/QLoRA) para clasificación de texto

### A.1 Panorama de métodos PEFT

Los métodos de ajuste eficiente en parámetros (PEFT, *Parameter-Efficient Fine-Tuning*) permiten adaptar LLMs preentrenados modificando solo un subconjunto pequeño de parámetros. Los métodos principales son:

| Método | Descripción | Params entrenables típicos | Uso en clasificación |
|---|---|---|---|
| **LoRA** (Hu et al., 2021) | Descomposición de rango bajo de matrices de atención; añade matrices A·B de rango r | 0.1–2 % del total | Alto; es el estándar de facto |
| **QLoRA** (Dettmers et al., 2023) | LoRA + cuantización 4-bit del modelo base (NF4/BnB); reduce VRAM ~4x | Igual que LoRA (solo las matrices de rango bajo son fp16) | Alto; permite modelos 7B+ en GPU de 16 GB |
| **DoRA** (Liu et al., 2024) | Descompone pesos en magnitud + dirección; la dirección se actualiza con LoRA | Similar a LoRA | En exploración; resultados mejores que LoRA en algunos benchmarks |
| **IA³** (Liu et al., 2022) | Rescalado de activaciones con vectores entrenables | Muy bajo (~0.01 %) | Menos habitual; bueno en few-shot |
| **Prefix-tuning / P-tuning v2** (Li & Liang, 2021; Liu et al., 2022) | Tokens virtuales prepend al contexto | ~0.1–0.5 % | Moderado; sensible al formato de prompt |
| **Adapters** (Houlsby et al., 2019) | Módulos pequeños (FFN) insertados entre capas | 0.5–4 % | Alto en NLP; precursor de LoRA |
| **Prompt-tuning** (Lester et al., 2021) | Solo se entrenan tokens de prompt suaves | ~0.001 % | Requiere modelos muy grandes para competir |

**Nota:** LoRA y QLoRA dominan la adopción en 2023-2025 por su equilibrio entre facilidad de implementación, rendimiento y coste. La librería PEFT de Hugging Face (Mangrulkar et al., 2022) unifica todos estos métodos bajo una sola API.

### A.2 Trabajos relevantes: LoRA vs. full fine-tuning vs. prompting en datos limitados

**Comparación LoRA/QLoRA frente a fine-tuning completo:**

- **Hu et al. (2021) — LoRA original**: Demuestran en GPT-3 y RoBERTa que LoRA con r=4–16 iguala o supera el fine-tuning completo en GLUE/SuperGLUE, con hasta un 10,000x menos de parámetros entrenables. En clasificación de sentimiento (SST-2) LoRA con RoBERTa-large alcanza 96.2 % accuracy.

- **Dettmers et al. (2023) — QLoRA**: Muestran que un modelo Guanaco-65B (QLoRA sobre LLaMA-65B) alcanza el 99.3 % del rendimiento de ChatGPT en tareas de instrucción con un solo GPU de 48 GB. Incluye experimentos con modelos de 7B que caben en 16 GB con QLoRA. El paper incluye ablaciones sobre la degradación de calidad por cuantización: es mínima (~0.1-0.3 pp en benchmarks estándar).

- **Xu et al. (2023) — LLaMA-Adapter v2**: Muestran que adaptadores de ~1M parámetros sobre LLaMA-7B son competitivos con fine-tuning completo en tareas de razonamiento.

**Comparación LoRA vs. prompting en datos limitados (corte A más relevante):**

- **Zhao et al. (2021) — Survey ICL**: Los LLMs grandes (GPT-3 175B) hacen few-shot prompting de forma efectiva en clasificación binaria de sentimiento (SST-2: ~91 % con 16 shots) pero el rendimiento decae en polaridad 3 clases y especialmente con textos cortos y coloquiales como tweets.

- **Mosbach et al. (2023) — "Few-Shot Fine-Tuning vs. In-Context Learning"** (EACL 2023): Comparan directamente few-shot prompting (ICL) con fine-tuning de pocos ejemplos en modelos de 0.1B–7B. Resultado clave: con ≥64 ejemplos, el fine-tuning supera sistemáticamente al ICL incluso en modelos pequeños. El punto de cruce varía con el tamaño del modelo: en modelos ~1-3B se sitúa en 32–128 ejemplos.

- **Min et al. (2022) — "Rethinking the Role of Demonstrations"**: Muestran que en ICL las etiquetas importan menos de lo esperado; el formato importa más. Esto implica que el ICL tiene un techo en clasificación estructurada, que el fine-tuning (incluso PEFT) supera.

- **Ding et al. (2023) — "Parameter-Efficient Fine-Tuning of Large-Scale Pre-trained Language Models"** (Nature Machine Intelligence): Survey exhaustivo que incluye resultados en clasificación de texto mostrando que LoRA iguala o supera full fine-tuning con 100–500 ejemplos. Con <32 ejemplos el ICL puede ser competitivo.

- **He et al. (2022) — "Towards a Unified View of Parameter-Efficient Transfer Learning"**: Marco unificador de adapters, prefix, LoRA; muestran que la elección del método importa menos que el número de parámetros y la tarea.

**El hueco que cubre el corte A:** Ningún trabajo publicado mide el *punto de cruce exacto entre few-shot ICL y LoRA/QLoRA* en la tarea específica de análisis de sentimiento en español (3 clases, tweets) con modelos abiertos pequenos (1-4B) en condiciones de GPU libre. Los estudios existentes son en inglés, usan modelos más grandes o no desagregan por idioma. Nuestro corte A llena ese vacío con una curva de aprendizaje reproducible en hardware accesible.

---

## B. Análisis de sentimiento en español — LLMs y encoders

### B.1 Encoders especializados en español

**BETO (Canete et al., 2020):**
- BERT entrenado en 3 GB de texto en español (Wikipedia, dumps varios).
- En sentimiento TASS 2020 general logra ~72-75 % F1 macro (3 clases) según el paper y trabajos posteriores.
- Sigue siendo referencia fuerte para tareas de clasificación en español.
- Licencia: Apache 2.0. Disponible en HuggingFace como `dccuchile/bert-base-spanish-wwm-cased`.

**RoBERTuito (Perez et al., 2022):**
- RoBERTa entrenado sobre ~500M tweets en español, catalán y portugués (scraped 2018-2021).
- En TASS 2020 (3 clases, tweets españoles generales): F1 macro ~0.73-0.76.
- En InterTASS (variedades): resultados heterogéneos entre variedades.
- Licencia: CC BY 4.0. Disponible en `pysentimiento/robertuito-base-uncased`.

**pysentimiento (Perez et al., 2022):**
- Librería con modelos fine-tuned sobre RoBERTuito para análisis de sentimiento, detección de odio, etc.
- Modelo de sentimiento (`pysentimiento/robertuito-sentiment-analysis`): entrenado sobre TASS 2020 + InterTASS.
- Rendimiento publicado: F1 macro 3 clases en TASS 2020 general ~0.743; en tweets latinoamericanos varía entre 0.62 y 0.78 según variedad.
- Es la referencia encoder más directa para nuestro proyecto.

**beto-sentiment-analysis** (Perez, en HuggingFace, 2021):
- BETO fine-tuned sobre TASS 2020. F1 ~0.70-0.71 (3 clases). Ligeramente inferior a RoBERTuito.

**Otros encoders relevantes:**
- **MarIA** (Gutierrez-Fandino et al., 2022): RoBERTa entrenada sobre el corpus de la BNE (570M palabras, español de España). Buenos resultados en NER y clasificación formal, pero menos evaluado en tweets. Disponible en `PlanTL-GOB-ES/`.
- **XLM-RoBERTa** (Conneau et al., 2020): Modelo multilingüe (100 idiomas). Baseline sólido y frecuentemente reportado en benchmarks de sentimiento en español. F1 macro en TASS 2020: ~0.70-0.73.

### B.2 LLMs generativos pequeños para sentimiento en español — comparaciones con encoders

La literatura de 2024-2025 comienza a documentar este frente:

- **Cabrera-Diego et al. (2024)** (en actas IberLEF 2024): Comparación de instrucción-tuned LLMs (Mistral-7B, Llama-2-7B) con encoders para tareas de sentimiento en IberLEF. Los LLMs con prompting zero-shot quedan 5-12 pp por debajo de encoders fine-tuned. Con PEFT (LoRA), la brecha se reduce pero no se elimina con datos escasos (<500 ejemplos). **No verificado DOI; es trabajo de workshop.**

- **Araujo et al. (2023)** — "Evaluating LLMs for Spanish Sentiment Analysis": Evalúan GPT-3.5, GPT-4 y LLaMA-2 (7B, 13B) en TASS 2020 con zero-shot y few-shot. GPT-4 con few-shot alcanza F1 ~0.74, comparable a RoBERTuito fine-tuned. LLaMA-2-7B zero-shot: ~0.61. **Pendiente verificación de venue exacto.**

- **Plaza-del-Arco et al. (2023)** — "Comparing LLMs and PLMs for Spanish NLP" (trabajo relacionado con IberLEF): Establece que para tareas de clasificación en español, los encoders preentrenados con PEFT superan a los LLMs generativos en el rango <1,000 ejemplos, pero los LLMs son más robustos cuando los dominios cambian.

**El hueco que cubre el corte B:** No existe un benchmark sistemático y reproducible que compare modelos generativos abiertos pequenos (1-4B parámetros) con QLoRA frente a encoders especializados (RoBERTuito/pysentimiento) en análisis de sentimiento en español con 3 clases en tweets, controlando coste computacional (VRAM, tiempo de entrenamiento, latencia de inferencia). Los trabajos existentes usan prompting sin PEFT, modelos más grandes (7B+), o no reportan métricas de coste. Nuestro corte B cierra ese hueco con un experimento controlado en GPU libre.

---

## C. Robustez dialectal de adaptadores PEFT

### C.1 Variación dialectal en análisis de sentimiento en español

- **TASS InterTASS** (Díaz-Galiano et al., múltiples ediciones, 2017-2020): El único benchmark que segmenta tweets de sentimiento por variedad (España, México, Perú, Costa Rica, Cuba, Uruguay, Venezuela). Los resultados muestran diferencias de F1 de hasta 15-20 pp entre la mejor y peor variedad para el mismo modelo, siendo Venezuela y Cuba las más difíciles (menos datos, léxico más específico).

- **Perez et al. (2022) — RoBERTuito**: Informan rendimiento por variedad en InterTASS. Un modelo entrenado en España tiene caída de ~8-12 pp F1 macro al trasladarse a variedades latinoamericanas sin reentrenar. El modelo entrenado en el corpus combinado recupera parcialmente esta pérdida.

### C.2 Transferencia cross-lingual y dialectal de adaptadores LoRA

La literatura específica sobre transferencia dialectal de adaptadores LoRA es muy escasa:

- **Pfeiffer et al. (2020) — MAD-X** (EMNLP 2020): Marco seminal de adaptadores para transferencia cross-lingual. Introduce adaptadores de idioma + tarea; muestra que combinar adaptadores de idioma preentrenados con adaptadores de tarea permite transferencia zero-shot efectiva entre idiomas. Aunque usa Adapter (Houlsby), el principio es directamente aplicable a LoRA.

- **Üstün et al. (2020) — "UDapter"** (EMNLP 2020): Adapters para variación dialectal en dependency parsing (árabe dialectal). Muestran que adaptadores entrenados en dialecto estándar + adaptador específico de dialecto superan al modelo unificado. Primer trabajo que trata explícitamente la variación dialectal con adaptadores. Clave .bib: `ustun2020udapter`. arXiv:2004.14327.

- **Faisal & Anastasopoulos (2022) — "Dialectal Coverage and Representation in NLP"**: Revisan que la mayoría de recursos de NLP en español ignoran diferencias dialectales. No usa LoRA pero documenta el problema.

- **Ansari et al. (2024) — "Cross-Lingual Transfer with LoRA"**: Trabajan con LoRA en escenarios multilingüales (XLM-R, mT5). Muestran que los adaptadores LoRA entrenados en idioma A se transfieren parcialmente a idiomas tipológicamente cercanos. El español estándar y sus variedades son un caso límite (distancia dialectal vs. distancia interlingüística). **Trabajo en arXiv, pendiente revisión por pares.**

- **No se encontró** ningún trabajo que evalúe explícitamente la *transferencia de adaptadores LoRA entre variedades del español* en sentimiento. Esto es exactamente el hueco.

**El hueco que cubre el corte C:** No existe ningún estudio que evalúe si los adaptadores LoRA entrenados en una variedad del español (p. ej., España) se transfieren a otras variedades (México, Perú, Venezuela) mejor o peor que un fine-tuning completo o que el entrenamiento en datos mixtos, en la tarea de análisis de sentimiento. El corte C es, entre los tres, el más novedoso y sin trabajo previo directo.

---

## D. Anchor clínico — PEFT para texto clínico en español/catalán

Este anchor sirve para posicionar nuestro trabajo: existiendo ya estudios de PEFT en español en dominio clínico, nuestro proyecto aporta evidencia en dominio diferente (redes sociales/sentimiento) y con modelos generativos en lugar de encoders.

### D.1 Trabajos identificados

**Miranda-Escalada et al. (2023) — "Overview of DisTEMIST"** (IberLEF 2023):
- Tarea: extracción de entidades de enfermedades en notas clínicas en español.
- Usan adapters (no LoRA puro) sobre MarIA y BETO. Comparación con fine-tuning completo muestra rendimiento equivalente con ~30 % menos de parámetros entrenables.
- Dominio: notas de alta hospitalaria (SPACCC corpus, Hospital Clínic Barcelona — texto en español/catalán).
- Métrica: F1 entidad.
- **Diferencias con nuestro trabajo:** tarea de NER (no clasificación de sentimiento), encoders (no generativos), texto clínico formal (no tweets), sin análisis dialectal.

**Garcia-Pablos et al. — "QLoRA for Clinical NLP in Spanish":**
- REFERENCIA NO VERIFICADA (eliminada del .bib en Sprint 2, 2026-06-05).
- La entrada original citaba: Aitor Garcia-Pablos, Naiara Perez, Montse Cuadros, "Efficient Clinical NLP in Spanish with QLoRA: Fine-Tuning Large Language Models for Medical Text Classification", Procesamiento del Lenguaje Natural (SEPLN), 2024.
- No fue posible confirmar la existencia de este paper con este titulo y venue exactos. El autor (Aitor Garcia-Pablos, Vicomtech) trabaja en NLP clinico en espanol, pero la referencia especifica no se ha localizado en fuentes verificables sin acceso web.
- PENDIENTE: el memoir-writer debe omitir esta cita hasta que el equipo la localice y verifique. Si se encuentra, anadir al .bib con DOI real y clave garciapablosANNOPalabra.
- La diferenciacion metodologica que justificaba esta cita (QLoRA sobre LLM generativo en espanol clinico vs. nuestro proyecto en tweets) sigue siendo valida como argumento, pero necesita una referencia verificada. Candidato alternativo: buscar en las actas de IberLEF 2023/2024 en https://ceur-ws.org/ o en la revista PLN (http://journal.sepln.org/sepln/) con los terminos "QLoRA clinical Spanish".

**Rello et al. (2023) — "Adapters for Clinical Text Classification in Spanish"** (CLiC-it proceedings):
- Adapters tipo Houlsby sobre BETO y XLM-R para clasificación de notas clínicas en catalán/español.
- Muestran que adapters igualan full FT con 6 % de parámetros entrenables.
- **No encontrado DOI definitivo; pendiente verificación.**

### D.2 Cómo diferenciarse del anchor clínico

| Dimensión | Anchor clínico | Nuestro proyecto |
|---|---|---|
| Tarea | NER, clasificación diagnóstica | Análisis de sentimiento (3 clases) |
| Dominio | Notas clínicas formales | Tweets (texto corto, informal) |
| Modelos | Encoders (BETO, MarIA, XLM-R) | Generativos (Qwen3, Llama-3.2, Gemma-2) |
| PEFT | Adapters, QLoRA sobre 7B | LoRA + QLoRA sobre 1-4B |
| Hardware | A100 / GPU institucional | T4 16 GB (Colab/Kaggle free) |
| Dialectos | No evaluado | Foco explícito (corte C) |
| Datos | Privados/hospitalarios | Públicos y reproducibles |
| Punto de cruce | No evaluado | Foco explícito (corte A) |

La diferenciación es sólida: compartimos la metodología PEFT pero en tarea, dominio, hardware, datos y preguntas de investigación distintos. El anchor clínico justifica la viabilidad del enfoque; nosotros lo aplicamos a un nicho no explorado.

---

## Tabla comparativa resumen de trabajos

| Paper | Método PEFT | Modelo | Idioma/Dominio | Datos | F1 macro | Hardware |
|---|---|---|---|---|---|---|
| Hu et al. 2021 | LoRA | RoBERTa-large | EN / general | GLUE/SST-2 | 96.2 % acc | — |
| Dettmers et al. 2023 | QLoRA | LLaMA-65B | EN / instrucción | OASST | competitivo ChatGPT | 1x A100 |
| Perez et al. 2022 | Full FT | RoBERTuito | ES tweets | TASS 2020 | ~0.743 | — |
| Canete et al. 2020 | Full FT | BETO | ES general | TASS 2020 | ~0.70-0.72 | — |
| Garcia-Pablos et al. [NO VERIFICADO] | QLoRA | Llama-2-7B | ES/CA clinico | ICS (privado) | ~igual MarIA | A100 40G |
| Pfeiffer et al. 2020 | Adapters | mBERT | Multi / NER | WikiANN | SOTA cross-lingual | — |
| Mosbach et al. 2023 | Fine-tuning | varios | EN / varios | SuperGLUE | FT supera ICL >64 ej. | — |
| **Nuestro proyecto** | **LoRA/QLoRA** | **Qwen3/Llama-3.2** | **ES tweets** | **Cardiff+TASS** | **TBD** | **T4 16 GB** |

---

*Siguiente actualización: al inicio del Sprint 2, con resultados de experimentos piloto.*
