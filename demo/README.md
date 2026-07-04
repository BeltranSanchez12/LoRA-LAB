# Demo de inferencia (T5) — sentimiento ES, 5 modelos en vivo

Demo interactiva (Gradio) para la defensa. **Dos modos**, mismo `app.py`:

- **GPU (principal)** — en la DGX (H200): compara **en vivo los 5 modelos del paper** sobre una
  frase (LoRA-1.7B, LoRA-4B, prompting k=4, BETO, XLM-R), con predicción + probabilidad por clase,
  y un botón de **colapso cased** (perturbación MAYÚSCULAS del Sprint 6) que enseña cómo los
  encoders cased se desploman y los LoRA aguantan. Se sirve por Gradio y se accede por **túnel SSH**
  desde el portátil.
- **CPU (fallback)** — `--cpu`: la versión ligera que corre en el portátil **sin GPU** (solo
  LoRA-1.7B + prompting k=4). Red de seguridad por si el día de la defensa falla el acceso a la DGX.

Todo es **inferencia sobre pesos congelados**: cero reentrenamiento, cero datos inventados. Mapeo
de etiquetas = el del paper (`{0:negative, 1:neutral, 2:positive}`). El paper NO se toca.

---

## 1. Modo GPU (principal, en la DGX)

### 1.1 Pesos que usa (y de dónde salen)

Todos existen ya en la DGX (reconstruidos, bit-exactos, semilla 42). **No van en git** (grandes):

| Modelo | Ruta en la DGX | Tamaño | Origen |
|--------|----------------|--------|--------|
| LoRA-1.7B | `adapters/lora_qwen1.7b_nfull_s42/` | 25 MB | `scripts/sprint6_reconstruct.py` |
| LoRA-4B | `adapters/lora_qwen4b_nfull_s42/` | 46 MB | `scripts/sprint6_reconstruct.py` |
| BETO | `results/checkpoints/encoder_beto_cardiff_es/` | 420 MB | `scripts/sprint6_reconstruct.py` |
| XLM-R | `results/checkpoints/encoder_xlmr_base_cardiff_es/` | 1.1 GB | `scripts/sprint6_reconstruct.py` |
| Prompting k=4 | Qwen3-4B base (HF) en 4-bit NF4 + `demo/fewshot_k4.json` | — | few-shot semilla 42 = el del paper |

> `adapters/` y los checkpoints de encoder están **gitignorados**; regeneran bit-exactos con
> `python scripts/sprint6_reconstruct.py` (documentado en `.gitignore`). En la DGX ya están.

### 1.2 Arranque en la DGX

```bash
conda activate ailab
# elige una GPU libre (nvidia-smi); la carga de los 5 modelos tarda ~15 s
CUDA_VISIBLE_DEVICES=0 python demo/app.py          # sirve en http://127.0.0.1:7860
```

### 1.3 Túnel SSH desde el portátil

```bash
# En el PORTÁTIL, abre el túnel (deja esta terminal abierta):
ssh -N -L 7860:localhost:7860 USUARIO@HOST_DGX
# luego abre en el navegador del portátil:
http://localhost:7860
```

- Si tu acceso a la DGX pasa por el mismo *port-forward* que usas para JupyterLab, reutiliza esa
  cadena (mismo esquema, puerto 7860).
- Si `localhost` del extremo SSH no alcanza al contenedor, arranca con `--host 0.0.0.0`
  (`CUDA_VISIBLE_DEVICES=0 python demo/app.py --host 0.0.0.0`) y tunela contra la IP del contenedor.
- Puerto configurable: `--port 7860` o env `DEMO_PORT`.

### 1.4 Tiempo por frase en GPU (medido, H200)

Carga (una vez): **~15 s** los 5 modelos. Después, por frase:

| Modelo | Tiempo/frase |
|--------|--------------|
| LoRA Qwen3-1.7B | **~23 ms** |
| LoRA Qwen3-4B | **~30 ms** |
| Prompting k=4 (Qwen3-4B, 4-bit) | **~48 ms** |
| BETO | **~3 ms** |
| XLM-R | **~3 ms** |
| **Los 5 juntos, por frase** | **~113 ms** |

### 1.5 Funciones (cada una = un hallazgo del paper)

- **A · Comparación 5 modelos.** Una frase → predicción + P(neg/neu/pos) de los 5 a la vez.
- **B · Colapso cased (MAYÚSCULAS).** Aplica la perturbación `mayusculas` del Sprint 6
  (`src.data.perturbations`, reutilizada, no reimplementada) y muestra antes/después. Enseña el
  **hallazgo titular**: MAYÚSCULAS hunde a los encoders cased
  (**BETO −0.184 / −27.8 %**, **XLM-R −0.110 / −17.0 %** de Macro-F1 en el test) mientras los LoRA
  apenas se mueven (−0.011 / −0.012) y prompting aguanta (+0.004). *(Cifras trazadas a
  `results/robustness_results.csv`.)*
- **C · Transferencia dialectal (Corte C).** Nota informativa: los adaptadores son de español
  general (Cardiff ES); el detalle 3×3 de transferencia entre variedades está en el explorador
  (`defense/deck_assets/explorer.html`, Vista 3) y en la Figura C. No se cargan adaptadores por
  variedad (solo están reconstruidos los de ES).

### 1.6 Probabilidades y fidelidad

- **Generativos** (LoRA 1.7B/4B, prompting): probabilidad = **verbalizador** (softmax sobre la
  verosimilitud de cada palabra-etiqueta tras el prompt de entrenamiento). El argmax coincide con
  la predicción greedy del paper: **LoRA-4B 12/12, prompting 12/12, LoRA-1.7B 11/12** (un caso
  frontera) sobre 12 frases del test.
- **Encoders** (BETO, XLM-R): softmax sobre los 3 logits del clasificador.

---

## 2. Modo CPU (fallback, en el portátil)

### 2.1 Requisitos e instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # rueda CPU (sin CUDA)
pip install -r demo/requirements.txt
```

Requisito real: `transformers>=4.51` (Qwen3). Probado con transformers 5.12.1 / peft 0.19.1 /
torch 2.12.1. No usa `datasets` (few-shot embebido en `demo/fewshot_k4.json`).

### 2.2 Adaptador (el portátil no puede regenerarlo sin GPU → cópialo una vez)

La demo busca el adaptador LoRA-1.7B en, por orden: `DEMO_ADAPTER_DIR`,
`adapters/lora_qwen1.7b_nfull_s42/`, `demo/adapter/`.

```bash
# opción A — dejarlo donde la demo lo encuentra sin configurar nada:
scp -r USUARIO@HOST_DGX:/home/jovyan/jupyterlab_container/AI_LAB/adapters/lora_qwen1.7b_nfull_s42 demo/adapter
# opción B — apuntar con la variable:
export DEMO_ADAPTER_DIR=/ruta/a/lora_qwen1.7b_nfull_s42
```

### 2.3 Arranque y tiempos

```bash
python demo/app.py --cpu          # abre http://127.0.0.1:7860
```

| Vía | Hilos CPU | Tiempo/frase |
|-----|-----------|--------------|
| LoRA-1.7B (verbalizador) | 4 | ~1.65 s |
| LoRA-1.7B (verbalizador) | 8 | ~1.01 s |
| Prompting k=4 (base 1.7B, sin LoRA) | 8 | ~0.94 s |

Variables opcionales: `DEMO_THREADS`, `DEMO_PORT`, `DEMO_HOST`.

---

## 3. Frases de ejemplo probadas (de punta a punta)

**Comparación 5 modelos (GPU):** los 5 coinciden en las tres:

| Frase | Predicción (5 modelos) |
|-------|------------------------|
| *Qué maravilla de película, me encantó de principio a fin.* | POSITIVO |
| *El paquete llegó tarde y encima roto, un auténtico desastre.* | NEGATIVO |
| *La reunión es el jueves a las seis en el auditorio principal.* | NEUTRAL |

**Colapso cased (GPU), frase recomendada:** `pues no está nada mal`
→ en MAYÚSCULAS, **BETO y XLM-R voltean positivo→negativo**, mientras **LoRA-1.7B y LoRA-4B
mantienen positivo**. (Frases con `@user` del test también lo disparan; ver la Figura F / el CSV.)

---

## 4. Ficheros

| Fichero | Qué es |
|---------|--------|
| `app.py` | Interfaz Gradio dual: GPU (5 modelos) por defecto, `--cpu` para el fallback. |
| `multimodel.py` | Motor GPU: carga los 5 modelos (loaders de `sprint6_robustness.py`), verbalizador + colapso cased. |
| `infer.py` | Núcleo CPU (`SentimentDemo`): LoRA-1.7B verbalizador + prompting k=4. |
| `fewshot_k4.json` | 4 ejemplos few-shot fijos (semilla 42) = `select_examples(k=4)` del paper (verificado). |
| `requirements.txt` | Dependencias mínimas del modo CPU. En la DGX se usa el env `ailab`. |
