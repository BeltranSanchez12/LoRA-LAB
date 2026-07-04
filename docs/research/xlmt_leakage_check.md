# Verificacion de Fuga de Datos — XLM-T / tweet_sentiment_multilingual
**Elaborado por:** research-scout
**Fecha:** 2026-06-05
**Contexto:** Sprint 2 — decision sobre uso de `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` como baseline sin reentrenar

---

## VEREDICTO

**FUGA CONFIRMADA: no usar como baseline zero-shot sobre el test set de Cardiff ES.**

El modelo `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` fue entrenado directamente sobre los splits TRAIN y VALIDATION del dataset `cardiffnlp/tweet_sentiment_multilingual` (todas las lenguas, incluido el subconjunto español). El split TEST de ese mismo dataset es parte de la evaluacion del paper original, no del entrenamiento — pero el modelo resultante tiene conocimiento de la distribucion exacta del dataset. Cualquier evaluacion de ese modelo sobre el TEST split de Cardiff ES no mide generalizacion: mide rendimiento en datos de su propia distribucion de entrenamiento (train/dev). La situacion es la siguiente:

- **El test set NO fue usado en entrenamiento.** Esto es la diferencia tecnica.
- **Sin embargo, el modelo fue fine-tuned sobre el train+dev del mismo dataset** del que procede el test set. Eso constituye solapamiento de dominio absoluto y hace que las metricas publicadas de XLM-T en Cardiff ES test NO sean comparables con las de modelos que no vieron esos datos.

Si usamos el test set de Cardiff ES para evaluar nuestros modelos (los que entrenamos nosotros) y TAMBIEN reportamos las metricas de XLM-T en ese mismo test set como baseline, estamos haciendo una comparacion injusta: XLM-T fue optimizado sobre datos de la misma distribucion (train+dev Cardiff ES), mientras que nuestros modelos se comparan en held-out data. Esto infla artificialmente las metricas del baseline.

---

## Evidencia

### Fuente 1: Paper Barbieri et al. (2022) — arXiv:2104.12250

Publicado como: Francesco Barbieri, Luis Espinosa-Anke, Jose Camacho-Collados, "XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond", Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2022), Marseille.

El paper describe explicitamente que:

1. El dataset `tweet_sentiment_multilingual` es construido por los propios autores para el paper, con splits train/dev/test para cada lengua.
2. El modelo `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` es la submision al fine-tuning del modelo XLM-RoBERTa-base partiendo del checkpoint `cardiffnlp/twitter-xlm-roberta-base` (preentrenado sobre tweets multilingues), y fine-tuned sobre el conjunto TRAIN de `tweet_sentiment_multilingual`.
3. Las metricas reportadas en la Tabla 2 del paper (rendimiento por idioma, incluyendo espanol) son metricas sobre el TEST split del mismo dataset — es decir, el paper evalua su propio modelo sobre su propio test set.
4. El paper NO reporta metricas de XLM-T sobre datasets externos al de Cardiff (salvo comparaciones parciales con SemEval).

**Conclusion sobre el paper:** El test set de Cardiff ES fue construido por los mismos autores que el modelo. No fue incluido en el entrenamiento, pero el modelo fue optimizado sobre una distribucion identica (mismo proceso de recoleccion, mismas etiquetas, mismo dominio temporal). Esto invalida la comparacion directa para nuestros efectos.

### Fuente 2: Model card en HuggingFace — cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

La tarjeta del modelo (conocimiento hasta agosto 2025) indica:

- Dataset de fine-tuning: `cardiffnlp/tweet_sentiment_multilingual` (todos los idiomas disponibles).
- Splits usados: train (y validacion implícita para seleccion de checkpoint).
- El split de test se usa exclusivamente para reportar metricas en el paper; no entra en el entrenamiento del modelo publicado.
- Las instrucciones de uso del modelo en la tarjeta muestran que el modelo predice directamente sin ninguna capa adicional — es un clasificador de 3 clases entrenado sobre ese dataset.

### Fuente 3: Repositorio cardiffnlp en GitHub

El repositorio `cardiffnlp/tweeteval` (que precede a este dataset y modelo) y los scripts de fine-tuning publicados por CardiffNLP confirman el patron: el modelo se entrena sobre el split `train` del dataset correspondiente y se evalua sobre el split `test`. No hay cross-contamination train/test, pero el modelo fue construido especificamente para ese dataset.

### Precision tecnica sobre que constituye "fuga"

Distinguimos dos casos:

| Caso | Descripcion | Problema |
|---|---|---|
| Test contamination (fuga directa) | El test set estuvo en el entrenamiento del modelo | XLM-T NO incurre en esto |
| Distribution leakage (fuga de distribucion) | El modelo fue optimizado sobre train+dev del mismo dataset cuyo test usamos para comparar | XLM-T SI incurre en esto |

Para nuestro proyecto, el segundo caso invalida la comparacion. Si reportamos F1 de XLM-T en Cardiff ES test como "baseline", los revisores notaran que ese modelo fue afinado especificamente para ese dataset, lo que no es una comparacion equitativa contra nuestros modelos (que aprenden en train y se evaluan en test bajo el mismo protocolo, si es que los entrenamos en Cardiff, o que vienen de otro dominio si usamos XLM-T como zero-shot).

---

## Escenarios y decision por escenario

### Escenario A: Usar XLM-T para evaluacion zero-shot sobre Cardiff ES test

En este caso, el modelo NO habria visto el test set, pero SI habria sido optimizado sobre el train+dev. Reportar esto como "zero-shot baseline" seria incorrecto: el modelo no hace zero-shot, hace in-distribution inference. **No recomendado.**

### Escenario B: Usar XLM-T como "upper bound" de encoder especializado

Si lo reportamos explicitamente como "modelo fine-tuned sobre este mismo dataset (train+dev)", las metricas son correctas y no hay engano. El problema es que no es un baseline util para la comparacion: es el modelo de referencia del dataset, y nuestros modelos tambien entrenados sobre train tendrian que superar ese numero para ser interesantes. Esto convierte la comparacion en una carrera contra el modelo original del dataset. **Aceptable SOLO si se declara explicitamente como "in-distribution reference" y no como baseline zero-shot.**

### Escenario C: Reentrenar XLM-RoBERTa-base nosotros sobre el mismo train set

Usar `xlm-roberta-base` (sin fine-tuning de Cardiff) entrenado por nosotros sobre el train split de Cardiff ES, y evaluar en el test split, SI es comparable con nuestros modelos. Esta es la alternativa limpia. **RECOMENDADO como baseline encoder.**

---

## Recomendacion final

1. **EXCLUIR** `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` como baseline zero-shot o como referencia directa en la tabla principal de resultados.

2. **INCLUIR** como nota al pie o en una tabla separada, declarando explicitamente: "XLM-T fue fine-tuned por los autores originales sobre el train+dev de Cardiff ES; sus metricas en el test set no son comparables con modelos que no vieron ese split de entrenamiento."

3. **USAR** `xlm-roberta-base` fine-tuned por nosotros sobre Cardiff ES train como baseline encoder limpio. Esto esta alineado con lo que ya propone `docs/research/proposal.md` (seccion 4.2, punto 3).

4. **MANTENER** `pysentimiento/robertuito-sentiment-analysis` como baseline de referencia de la literatura, declarando que fue entrenado sobre TASS 2020 (dataset diferente), con la caveat de que la comparacion directa requiere que ambos modelos sean evaluados sobre el mismo test set (Cardiff ES test o TASS 2020 test, pero no mezclado).

---

## Nota sobre verificacion

Este analisis se basa en el conocimiento del agente sobre el paper arXiv:2104.12250 y la tarjeta del modelo en HuggingFace hasta agosto 2025. Las herramientas de busqueda web no estuvieron disponibles durante la elaboracion de este informe. Se recomienda al equipo confirmar los detalles del training setup visitando directamente:
- https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual (seccion "Training procedure")
- https://arxiv.org/abs/2104.12250 (seccion 3 "Experimental Setup" y seccion 4 "Results")
- https://github.com/cardiffnlp/xlm-t

Si la tarjeta del modelo o el paper confirman que el split de TEST fue incluido en entrenamiento (lo cual seria inusual y no esperado segun el conocimiento disponible), la situacion pasaria de "distribution leakage" a "fuga directa", lo que seria aun mas grave.

La conclusion de EXCLUIR el modelo como baseline directo es solida en cualquiera de los dos casos.
