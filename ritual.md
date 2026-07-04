# Ritual de arranque en la DGX

Los contenedores de la DGX son **efímeros**: al recrearlos se borra todo el sistema
**MENOS** `~/jupyterlab_container/`. Ahí persisten:

- el repositorio (`~/jupyterlab_container/AI_LAB`),
- el entorno conda (`~/jupyterlab_container/envs/ailab`),
- la caché de Hugging Face (`~/jupyterlab_container/hf_cache`).

Todo lo demás (paquetes del sistema como `curl`, binarios en `~/.local/bin` como
`claude`, variables de entorno, GPU seleccionada…) hay que reconstruirlo/reexportarlo
en cada sesión nueva.

> **Atajo:** en lugar de ejecutar los comandos a mano, usa el script idempotente:
> ```bash
> source setup.sh
> ```
> Hace todo lo de "POR SESIÓN" y se autocura si falta algo (entorno, requirements,
> torch CUDA, curl, claude). Tiene que **sourcearse**, no ejecutarse con `bash`
> (ver más abajo). Este documento explica qué hace por dentro y cómo arreglarlo a mano.

---

## POR SESIÓN — cada contenedor nuevo (repetir en la shell)

```bash
# 1. Activar el entorno conda persistente
conda activate ~/jupyterlab_container/envs/ailab

# 2. Apuntar la caché de Hugging Face al disco persistente
export HF_HOME=~/jupyterlab_container/hf_cache

# 3. Poner Claude Code en el PATH (su binario vive en ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# 4. Sincronizar el repo
git pull

# 5. Comprobar que PyTorch ve la GPU  ->  debe imprimir True
python -c "import torch; print(torch.cuda.is_available())"

# 6. Ver qué GPUs están libres
nvidia-smi

# 7. Elegir una GPU libre (el id VARÍA; míralo en nvidia-smi)
export CUDA_VISIBLE_DEVICES=<id de una GPU libre>
```

> El paso 7 es manual a propósito: en una DGX compartida hay que mirar `nvidia-smi`
> y elegir una GPU con memoria/utilización libres. `setup.sh` **no** lo fija por ti
> (no sabe cuál está libre), solo te imprime el estado para que decidas.

---

## SOLO SI FALTA — primera vez, o si el contenedor borró algo del home

```bash
# El entorno conda no existe (p. ej. se perdió envs/ailab)
conda create --prefix ~/jupyterlab_container/envs/ailab python=3.12 -y
conda activate ~/jupyterlab_container/envs/ailab
pip install -r requirements.txt

# torch.cuda.is_available() devuelve False (build de torch sin CUDA / CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# No está curl (sistema recién recreado)
apt-get update && apt-get install -y curl

# No está el binario claude (Claude Code)
curl -fsSL https://claude.ai/install.sh | bash
```

> **Por qué el torch CUDA va aparte y NO en `requirements.txt`:** el build de torch
> con CUDA (`cu126`) es específico de la DGX. `requirements.txt` se mantiene
> **agnóstico** (`torch>=2.3.0`, sin `--index-url`) para que en el portátil pip
> resuelva el build adecuado (CPU u otra CUDA) sin romperse. El build CUDA se instala
> como paso aparte aquí y en `setup.sh`.

---

## setup.sh — automatización idempotente

`setup.sh` hace todo lo anterior de forma segura de repetir:

- Crea el entorno **solo si no existe**.
- Instala `requirements.txt` **solo si falta** (detecta un paquete clave, p. ej. `transformers`).
- Instala el torch `cu126` **solo si `torch.cuda.is_available()` da False**.
- Instala `curl` / `claude` **solo si faltan**.
- Activa el entorno y exporta `HF_HOME` y el `PATH` de Claude Code.
- En sesiones repetidas es rápido: solo activa, exporta y comprueba.
- Al final imprime un resumen: entorno activo, `cuda True/False` y GPUs libres.

### Uso — IMPORTANTE: hay que SOURCEARLO

```bash
source setup.sh          # ✅  correcto
```

```bash
bash setup.sh            # ❌  NO sirve
./setup.sh               # ❌  NO sirve
```

`conda activate` y los `export` solo modifican la **shell actual**. Si ejecutas el
script en una subshell (`bash setup.sh` / `./setup.sh`), el entorno se activa y las
variables se exportan en esa subshell, que muere al terminar el script: tu shell se
queda **sin** entorno activo y **sin** las variables. Por eso hay que `source`-arlo.

---

## Para correr los baselines

Desde la raíz del repo, con el entorno activo y una GPU libre seleccionada:

```bash
# Baselines de encoders (BETO y XLM-R sobre Cardiff ES)
python scripts/run_encoder_baselines.py

# Baseline de prompting (few-shot / zero-shot) con un LLM generativo
python scripts/run_prompting_baseline.py --model_id Qwen/Qwen3-1.7B
```

Salidas en `results/baselines/`. Recuerda haber hecho `export CUDA_VISIBLE_DEVICES=<gpu>`
sobre una GPU libre antes de lanzarlos.
