# Sprint 5 — Pulido editorial, figuras nuevas y traducción al español (fase 6)

**Presupuesto:** ~14 h.

## Objetivo
Dejar el paper listo para entrega: unificar la institución, eliminar las anotaciones de procedencia que desbordan, corregir todos los defectos de maquetación (tablas y figuras que invaden la columna contigua o se salen del margen, figuras que no renderizan), añadir 1–2 figuras que hagan el trabajo más visual, y traducir el documento completo al español. Al cerrar, el PDF compila sin desbordes que afecten a la lectura, todas las cifras siguen trazadas a sus tablas, y el idioma (texto + figuras) es coherente de punta a punta.

## Orden de ejecución (importante: NO es el orden de la lista)
La traducción reflota toda la maquetación a dos columnas (el español ocupa ~15–20 % más que el inglés). Por eso **los arreglos de maquetación van DESPUÉS de traducir y de añadir las figuras nuevas**, no antes — arreglar columnas primero y traducir después es trabajo tirado. Y como el paper acaba en español, las figuras A/B/C (ya en español) quedan correctas: no se reetiquetan, solo se generan las nuevas en español.

## Tareas

| # | Tarea | Agente | Est. |
|---|-------|--------|------|
| T1 | **Institución.** En el bloque `\author` del template IEEE, fijar la afiliación como *Universidad Pontificia de Comillas – ICAI*. Recompilar. | memoir-writer | 0.5 h |
| T2 | **Eliminar anotaciones de procedencia.** Quitar TODAS las `(source:configs/...)` del cuerpo, tablas y pies de figura. Comprobar que no quedan paréntesis huérfanos ni puntuación colgando. | memoir-writer | 1 h |
| T3 | **Traducción EN→ES.** Traducir TODO el texto (abstract, keywords, secciones, pies de figura, encabezados de tabla) al español, SOLO texto. Fijar un glosario de terminología y aplicarlo de forma consistente. **Gate de registro:** traducir primero una sección de muestra y parar a validar registro/terminología antes de seguir. | memoir-writer | 6 h |
| T4 | **Figuras nuevas (1–2), en español, desde datos que YA existen.** Generar 1–2 figuras de alto valor a partir de tablas/CSVs ya en el repo (cero datos inventados). Candidatas: (D) F1 por clase — la clase neutral como cuello de botella; (E) compromiso calidad/coste — parámetros entrenables y VRAM pico de LoRA/QLoRA/Full-FT/encoders. Exportar PNG+PDF a ruta versionada, integrar con `\includegraphics`. | ml-dev | 3 h |
| T5 | **Maquetación (AL FINAL).** Tras T3+T4, arreglar TODOS los desbordes: tablas/figuras que invaden la columna contigua o se salen del margen; figuras que no renderizan (path roto); figuras más anchas que `\columnwidth`. Usar `\columnwidth`/`\linewidth`, `figure*`/`table*` (doble columna), `\footnotesize`/`\resizebox` donde haga falta. | memoir-writer | 2 h |
| T6 | **QA final.** Checklist de cierre: compila exit 0; 0 refs/citas indefinidas; 0 `(source:...)` en el PDF; institución correcta; texto 100 % español + figuras coherentes; cada cifra trazada a su tabla; figuras nuevas trackeadas en origin. Diario del sprint. | qa-validator | 1.5 h |

**Total: ~14 h.**

## Criterios de aceptación
- PDF compila exit 0; 0 referencias/citas indefinidas; 0 cajas overfull/underfull que afecten a la lectura (las cosméticas residuales, documentadas).
- Ninguna tabla o figura invade la columna contigua ni se sale del margen; todas las figuras renderizan.
- Afiliación = *Universidad Pontificia de Comillas – ICAI*.
- 0 ocurrencias de `(source:...)` en el PDF.
- Texto 100 % en español; figuras (A, B, C + nuevas) con etiquetas en español, coherentes con el texto.
- Toda cifra del paper sigue trazada a su tabla/sección; ningún número, `\cite`, `\ref` o `\label` alterado por la traducción.
- 1–2 figuras nuevas integradas, generadas desde datos existentes (cero datos inventados), trackeadas en git/origin.

## Riesgos y notas
- **Idioma.** Confirma que la asignatura admite/prefiere el español antes de traducir 10 páginas. Si el inglés vale, la alternativa barata es reetiquetar solo las 3 figuras a inglés y dejar el texto como está; la traducción completa solo se justifica si el entregable debe ir en español.
- **La traducción NO toca** números, datos de tablas, fórmulas, `\cite`, `\ref`, `\label` ni la bibliografía (los títulos de las referencias se quedan en su idioma original).
- **Terminología consistente.** Fija un glosario al traducir la primera sección (p. ej. *parámetros entrenables*, *ajuste fino*, *codificador*, *frontera de Pareto*, *muestreo estratificado*) y mantén como anglicismos establecidos *LoRA*, *QLoRA*, *prompting*, *fine-tuning*. Valida ese registro antes de traducir el resto.
- **Figuras versionadas.** Las figuras originales estaban gitignoradas; las nuevas deben quedar en una ruta trackeada y subida a origin (relevante también para Claude Design después).
- **Alcance.** El paper está congelado salvo por estas tareas; cualquier otro cambio queda fuera.

## Dependencias
- Paper en su estado actual (post-corrección del factor de parámetros).
- CSVs de resultados (Cut A/B/C) y la tabla de F1 por clase, ya en origin, como única fuente de las figuras nuevas.

