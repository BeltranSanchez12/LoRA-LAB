# Sprint 7 — Correcciones pre-entrega del paper (hallazgos de revisión externa)

## Contexto y restricciones globales

El paper (14 págs, IEEE dos columnas, español) está congelado desde el Sprint 5. Una revisión externa ha detectado errores de bibliografía, figuras y redacción que deben corregirse ANTES de la entrega. Restricciones no negociables:

- **NO modificar ningún valor numérico de resultados** (tablas, F1, VRAM, tiempos, σ). Solo se tocan bibliografía, títulos internos de figuras, un pie de figura y texto de redacción.
- Mantener el glosario establecido: "parámetros entrenables", "ajuste fino", "codificador", "frontera de Pareto"; LoRA/QLoRA/prompting/fine-tuning se mantienen como anglicismos; punto decimal en todo el documento.
- Disciplina git: `git pull` antes de empezar (dos orígenes: repo principal y clon en DGX). Commit por bloque (A, B, C, D), push solo al final tras pasar QA.
- Las figuras del paper deben quedar explícitamente fuera de `.gitignore` (ya existe la excepción; verificar que las regeneradas la cumplen).
- qa-validator valida los criterios de aceptación del final antes de cerrar el sprint.

---

## BLOQUE A — Bibliografía (crítico)

**A1. Ref [15] (DisTEMIST) — eliminar TODO olvidado.** El campo `note` contiene literalmente "[VERIFICAR: puede ser BioCreative VII 2022 o IberLEF 2023 version extendida]" y se ha colado en el PDF. Verificar la venue real del overview de DisTEMIST de Miranda-Escalada et al. (comprobar online: probablemente working notes de BioASQ/CLEF 2022 en CEUR-WS, no BioCreative VII), fijar UNA venue correcta y eliminar la nota. Si no puede verificarse online desde el contenedor, dejar la cita mínima verificable (autores, título, año, arXiv/URL) sin la nota especulativa.

**A2. Limpiar anotaciones personales del `.bib`.** Los campos `note` de las refs [3], [4], [5], [6], [7], [14] y [16] contienen notas de trabajo que aparecen en el PDF. Eliminar exactamente estos textos (y cualquier otro similar):
- [3]: "survey exhaustivo de metodos PEFT con resultados en clasificacion"
- [4]: "Clave para el corte A: FT supera ICL con ¿=64 ejemplos" (nótese el "¿=" que es un "≥" mal codificado)
- [5]: "Muestra que etiquetas importan menos que formato en ICL"
- [6]: "bETO: bert-base-spanish-wwm-cased..." (la parte de nota; el ID de HuggingFace puede quedarse si se reformatea limpio)
- [7]: "F1 macro TASS 2020 0.743" y similares
- [14]: "xLM-RoBERTa: baseline multilingue de referencia en benchmarks de espanol"
- [16]: notas descriptivas redundantes ("Describe Qwen3 (0.6B...)...")

**A3. Capitalización BibTeX.** "bETO" y "xLM-RoBERTa" aparecen en el PDF por la des-capitalización automática de BibTeX. Proteger con llaves en el `.bib`: `{BETO}`, `{XLM-RoBERTa}`, `{QLoRA}`, `{LoRA}`, `{GPT-3}`, `{NLP}`, `{DisTEMIST}`, etc. Revisar todo título tras recompilar.

**A4. Ref [13] (MarIA) — falta la revista.** Aparece como "vol. 68, 2022, pp. 39–60" sin publicación. Añadir `journal = {Procesamiento del Lenguaje Natural}`.

**A5. Cita correcta para InterTASS 2018.** Actualmente [12] (overview de TASS **2020**) se cita cada vez que se habla de InterTASS **2018** (Secciones I, II-B, III-A, IV-D). Añadir una nueva referencia: Martínez-Cámara et al., "Overview of TASS 2018: Opinions, Health and Emotions", actas del workshop TASS 2018 / SEPLN (CEUR-WS, vol. 2172) — **verificar los datos exactos online antes de fijarlos**. Citar la nueva ref en todas las menciones de InterTASS 2018; mantener [12] solo para las menciones de TASS 2020 y de la serie TASS en general (p. ej. el 0.743 de RoBERTuito).

---

## BLOQUE B — Figuras

**B1. Eliminar la doble numeración.** Los gráficos tienen títulos incrustados "Figura A", "Figura B", "Figura E", "Figura D" (nombres de assets de defensa) que chocan con los pies IEEE "Figura 1–5", y además aparecen desordenados (la E sale antes que la D). Modificar los scripts de generación para producir **variantes del paper sin `suptitle`** (p. ej. sufijo `_paper`) y usarlas en el `.tex`. NO tocar los assets originales A–E de `defense/assets/` (los usan el explorer y la defensa).

**B2. Verificar `.gitignore`.** Las variantes nuevas de figuras deben quedar versionadas (comprobar la excepción de `.gitignore` conocida del proyecto).

**B3. Pie de la Figura 2.** Dice "∼9–24× menos parámetros entrenables que los codificadores" para ambas configuraciones LoRA, pero ese rango es solo del 4B. Corregir a: "17–43× (1.7B) y 9–24× (4B) menos parámetros entrenables que los codificadores".

---

## BLOQUE C — Texto

**C1. Precisión del punto de cruce (Resumen + contribuciones de la Introducción).** "menos de n = 7 ejemplos etiquetados bastan" es técnicamente incorrecto (el cruce se produce entre n=4 y n=7; n=7 basta). Sustituir en ambos sitios por: "con tan solo n = 7 ejemplos etiquetados (el cruce se produce entre n = 4 y n = 7)".

**C2. Tercera pregunta en la Introducción.** El párrafo de motivación dice "dos preguntas de relevancia práctica siguen poco exploradas" pero el paper responde tres preguntas de investigación (la transferencia dialectal no está motivada en la Introducción). Cambiar "dos" por "tres" y añadir, tras el "Segundo, ...", este párrafo (ajustar transiciones):

> Tercero, aunque el español presenta una notable variación dialectal entre países, no se ha caracterizado empíricamente si los adaptadores LoRA entrenados sobre una variedad nacional transfieren sin penalización a otras—una cuestión con implicaciones directas para decidir entre desplegar un único adaptador panhispánico o adaptadores específicos por variedad.

**C3. Frase confusa en la Discusión.** "GPU de consumo de 8 GB (RTX 3070, RTX 4060 Ti) que no están disponibles en la gama H200/A100" no se entiende. Sustituir por: "GPU de consumo de 8 GB (RTX 3070, RTX 4060 Ti), a diferencia de la gama empresarial H200/A100".

**C4. Reconocer explícitamente que LoRA-4B excede el objetivo T4.** La Tabla III reporta 16.7 GB para Qwen3-4B LoRA frente al objetivo T4 de 16 GB del título, y el texto nunca lo dice. Insertar en la Sección IV-C (tras el párrafo de QLoRA) o en la Discusión:

> Cabe señalar que la configuración LoRA de Qwen3-4B alcanza una VRAM pico de 16.7 GB, marginalmente por encima del objetivo de despliegue T4 de 16 GB. En ese hardware, la vía recomendada para el modelo de 4B es QLoRA (6.79 GB); la variante LoRA sin cuantizar requiere una GPU con algo más de margen o técnicas adicionales de reducción de memoria. El modelo de 1.7B con LoRA (8.72 GB) sí cabe holgadamente en el objetivo T4.

**C5. Blindar la pregunta obvia sobre RoBERTuito.** RoBERTuito (0.758) supera al mejor resultado del estudio (0.707) y la Discusión no lo aborda. Insertar en la Sección V (p. ej. tras las implicaciones de la frontera de Pareto):

> Merece comentario que RoBERTuito (0.758, Tabla I) supere a todas las configuraciones controladas de este estudio. Este resultado no contradice nuestras conclusiones: RoBERTuito fue ajustado sobre TASS 2020, un corpus de tuits en español con una tarea idéntica, por lo que su evaluación "lista para usar" sobre Cardiff ES constituye una transferencia desde un dominio muy cercano y no una comparación controlada del coste de adaptación—de hecho, ilustra precisamente el valor del fine-tuning específico de tarea que este trabajo cuantifica. Nuestra contribución no es superar a todo sistema existente, sino medir, bajo un protocolo controlado con el mismo presupuesto de datos y cómputo, la frontera calidad-coste de adaptar modelos generativos pequeños frente a codificadores.

**C6. Consolidar las limitaciones duplicadas.** "Amenazas a la validez" (Sección V) y la lista "Limitaciones" (Sección VI) repiten casi el mismo contenido (dataset único, dominio Twitter, varianza de semilla, hardware, salvedades del Corte C). Dejar la lista detallada en la Sección VI y reducir "Amenazas a la validez" a un párrafo breve de 2–3 frases que remita a VI ("se enumeran en detalle en la Sección VI"). Elegir la opción que menos altere la maquetación; el paper debe seguir en ≥10 páginas (holgado: partimos de 14).

**C7. Dos notas al pie en la Tabla III (con verificación previa de ml-dev).**
- (a) La VRAM de prompting Qwen3-4B figura como 2.87 GB, pero los pesos bf16 de un 4B ocupan ~8 GB. **ml-dev debe comprobar en los logs/configs reales** qué se midió (¿carga cuantizada?, ¿medición solo del pico de generación?, ¿`max_memory_allocated` reseteado?). Según lo que se encuentre: corregir el número o añadir nota al pie explicando la medición. No inventar la explicación.
- (b) Full-FT 1.7B entrena en 124 s frente a 169 s de LoRA, lo cual es contraintuitivo. La explicación probable es el mínimo de pasos distinto (60 en full-FT frente a 80 en LoRA); **verificar en logs** y añadir nota al pie: "El full-FT completa menos pasos de optimización (mínimo 60 frente a 80 de LoRA), lo que explica su menor tiempo de entrenamiento pese al mayor número de parámetros."

---

## BLOQUE D — Añadidos (implementar todos; reportar en el resumen final)

**D1. Correo institucional.** Sustituir el Gmail del autor por el correo institucional de Comillas. (202216017@alu.comillas.edu)

**D2. Metadatos del PDF.** Title y Author están vacíos. Añadir `\hypersetup{pdftitle={...título del paper...}, pdfauthor={Beltrán Sánchez Careaga}}`.

**D3. Mención de robustez en Trabajo futuro** (una línea, sin sección nueva): añadir al final de "Trabajo futuro" una frase indicando que un análisis complementario de robustez frente a perturbaciones típicas de redes sociales (mayúsculas, elongaciones, emojis, etc.) apunta a una fragilidad diferencial de los codificadores *cased* frente a los modelos generativos, y que su caracterización completa se deja como extensión.

---

## Criterios de aceptación (qa-validator, obligatorios antes de cerrar)

1. `grep -ri "VERIFICAR"` sobre `.tex`/`.bib` y sobre el texto extraído del PDF → 0 resultados.
2. Ninguna anotación personal en las referencias del PDF compilado (revisar refs [3]–[16] una a una).
3. "bETO", "xLM" y "¿=" no aparecen en el PDF.
4. Sin doble numeración de figuras: ningún gráfico contiene "Figura A/B/C/D/E" incrustada; los pies van de Figura 1 a Figura 5 en orden.
5. Los assets originales de `defense/assets/` (A–E) permanecen intactos (diff limpio) y el explorer sigue funcionando.
6. Todas las menciones de InterTASS 2018 citan la nueva referencia de TASS 2018; [12] queda solo para TASS 2020/serie.
7. Diff numérico contra la versión congelada: ningún valor de resultados cambia (Tablas I–V byte a byte salvo el pie nuevo de la Tabla III y el pie corregido de la Figura 2).
8. El documento compila sin errores ni referencias indefinidas; ≥10 páginas.
9. Terminología conforme al glosario (muestreo de los párrafos nuevos: C2, C4, C5, D1, D5).
10. Las notas al pie de C7 están respaldadas por evidencia de logs (adjuntar la evidencia en el informe de QA); si (a) no pudo verificarse, el número queda marcado para revisión manual, no "explicado" sin evidencia.

## Entregable final

Resumen de cierre del sprint con: lista de cambios por bloque, evidencia de C7, diff de páginas (antes/después), y confirmación de los 10 criterios de QA.
