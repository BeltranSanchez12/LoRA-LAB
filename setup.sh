#!/usr/bin/env bash
# setup.sh — Ritual de arranque idempotente para contenedores efímeros de la DGX.
#
#   IMPORTANTE: SOURCEAR, no ejecutar.
#       source setup.sh        ✅
#       bash setup.sh / ./setup.sh   ❌  (conda activate y export solo afectan
#                                          a la shell actual; en subshell se pierden)
#
# Seguro de correr siempre:
#   - crea el entorno conda solo si no existe,
#   - instala requirements solo si faltan,
#   - instala torch cu126 solo si cuda da False,
#   - instala curl / claude solo si faltan,
#   - activa el entorno y exporta HF_HOME y el PATH de Claude Code,
#   - al final imprime un resumen del estado.
# Ver ritual.md para la versión manual y las explicaciones.

# --- Detectar si nos están sourceando (si no, avisar y salir sin matar la shell) ---
_SOURCED=0
# bash
if [ -n "${BASH_VERSION:-}" ]; then
    [ "${BASH_SOURCE[0]}" != "${0}" ] && _SOURCED=1
# zsh
elif [ -n "${ZSH_VERSION:-}" ]; then
    [[ "${ZSH_EVAL_CONTEXT:-}" == *:file ]] && _SOURCED=1
fi
if [ "$_SOURCED" -eq 0 ]; then
    echo "❌ setup.sh debe SOURCEARSE, no ejecutarse."
    echo "   Usa:  source setup.sh   (no 'bash setup.sh' ni './setup.sh')"
    echo "   Motivo: conda activate y los export solo afectan a la shell actual."
    exit 1
fi

# A partir de aquí NO usamos 'set -e' ni 'exit': estamos en la shell del usuario
# y no queremos cerrarla si algo falla. Reportamos y seguimos.

# --- Rutas persistentes (sobreviven al borrado del contenedor) ---
PERSIST="$HOME/jupyterlab_container"
ENV_PREFIX="$PERSIST/envs/ailab"
HF_CACHE="$PERSIST/hf_cache"
# Raíz del repo = carpeta donde vive este script
if [ -n "${BASH_SOURCE:-}" ]; then
    _SELF="${BASH_SOURCE[0]}"
else
    _SELF="${(%):-%x}"   # zsh
fi
REPO_ROOT="$(cd "$(dirname "$_SELF")" && pwd)"

echo "=== Ritual de arranque DGX (idempotente) ==="
echo "Repo:        $REPO_ROOT"
echo "Entorno:     $ENV_PREFIX"
echo "HF cache:    $HF_CACHE"
echo

# --- 1. Caché de Hugging Face en disco persistente ---
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
echo "✔ HF_HOME=$HF_HOME"

# --- 2. PATH de Claude Code (binario en ~/.local/bin) ---
case ":$PATH:" in
    *":$HOME/.local/bin:"*) : ;;                       # ya está
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
echo "✔ PATH incluye ~/.local/bin"

# --- 3. Asegurar que conda está disponible ---
if ! command -v conda >/dev/null 2>&1; then
    # Intento de cargar el conda.sh estándar si conda no está en el PATH
    for _c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
              "$HOME/anaconda3/etc/profile.d/conda.sh" \
              "/opt/conda/etc/profile.d/conda.sh"; do
        [ -f "$_c" ] && . "$_c" && break
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "❌ conda no está disponible. Instálalo o ajusta el PATH y reintenta."
    return 1 2>/dev/null || exit 1
fi

# --- 4. Crear el entorno SOLO si no existe ---
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "… Entorno no encontrado. Creando con python=3.12 (primera vez)…"
    conda create --prefix "$ENV_PREFIX" python=3.12 -y
else
    echo "✔ Entorno ya existe"
fi

# --- 5. Activar el entorno (en la shell actual) ---
conda activate "$ENV_PREFIX"
if [ "${CONDA_PREFIX:-}" != "$ENV_PREFIX" ]; then
    echo "❌ No se pudo activar el entorno ($ENV_PREFIX)."
    return 1 2>/dev/null || exit 1
fi
echo "✔ Entorno activo: $CONDA_PREFIX"

# --- 6. Instalar requirements SOLO si faltan (sonda: transformers) ---
if ! python -c "import transformers" >/dev/null 2>&1; then
    echo "… Dependencias ausentes. Instalando requirements.txt…"
    pip install -r "$REPO_ROOT/requirements.txt"
else
    echo "✔ Requirements ya instalados"
fi

# --- 7. torch CUDA SOLO si cuda da False ---
#     (requirements.txt instala torch agnóstico; aquí forzamos el build cu126 de la DGX)
if python -c "import torch" >/dev/null 2>&1; then
    CUDA_OK="$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
else
    CUDA_OK="False"   # torch ni siquiera importa
fi
if [ "$CUDA_OK" != "True" ]; then
    echo "… torch.cuda.is_available()=$CUDA_OK. Instalando torch build cu126…"
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    CUDA_OK="$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo False)"
else
    echo "✔ CUDA disponible para torch"
fi

# --- 8. curl SOLO si falta ---
if ! command -v curl >/dev/null 2>&1; then
    echo "… curl ausente. Instalando…"
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y curl
    else
        apt-get update && apt-get install -y curl
    fi
else
    echo "✔ curl disponible"
fi

# --- 9. claude (Claude Code) SOLO si falta ---
if ! command -v claude >/dev/null 2>&1; then
    if command -v curl >/dev/null 2>&1; then
        echo "… claude ausente. Instalando Claude Code…"
        curl -fsSL https://claude.ai/install.sh | bash
        # refrescar PATH por si el instalador lo dejó en ~/.local/bin
        case ":$PATH:" in *":$HOME/.local/bin:"*) : ;; *) export PATH="$HOME/.local/bin:$PATH";; esac
    else
        echo "⚠ No se pudo instalar claude (falta curl)."
    fi
else
    echo "✔ claude disponible"
fi

# --- 10. Sincronizar el repo (no bloqueante) ---
if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "… git pull…"
    git -C "$REPO_ROOT" pull --ff-only || echo "⚠ git pull no limpio (revisa a mano)."
fi

# --- Resumen del estado ---
echo
echo "================== RESUMEN =================="
echo "Entorno activo : ${CONDA_PREFIX:-(ninguno)}"
echo "Python         : $(python --version 2>&1)"
echo "torch.cuda     : $CUDA_OK"
echo "HF_HOME        : $HF_HOME"
echo
echo "GPUs (nvidia-smi):"
if command -v nvidia-smi >/dev/null 2>&1; then
    # tabla compacta: id, nombre, memoria usada/total, utilización
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader 2>/dev/null \
        | awk -F',' '{printf "  GPU %s:%s  mem %s/%s  util %s\n",$1,$2,$3,$4,$5}'
    echo
    echo "  → Elige una GPU libre y fíjala:  export CUDA_VISIBLE_DEVICES=<id>"
else
    echo "  nvidia-smi no disponible en este contenedor."
fi
echo "============================================"

# limpieza de variables temporales
unset _SOURCED _SELF _c CUDA_OK PERSIST ENV_PREFIX HF_CACHE REPO_ROOT 2>/dev/null
