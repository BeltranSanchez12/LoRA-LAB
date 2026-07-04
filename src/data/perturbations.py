"""perturbations.py — Taxonomía de perturbaciones del español informal de redes
(Sprint 6, T1). Cada perturbación:

  * es una función pura  str -> str,
  * es DETERMINISTA: dado (texto, seed) siempre produce la misma salida
    (las que introducen ruido usan un RNG sembrado por-texto, independiente del orden),
  * es AISLADA: se aplica una sola perturbación por conjunto (no se componen).

Se usan para la prueba de robustez SOLO en inferencia (cero reentrenamiento): se perturba
el conjunto de test limpio de Cardiff ES y se re-evalúan los modelos ya entrenados.

Ejecutar como script para el informe de la taxonomía (cobertura + ejemplos antes/después)
sobre el test real, sin tocar ningún modelo:

    python -m src.data.perturbations
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Callable

PROJECT_SEED = 42

# ---------------------------------------------------------------------------
# RNG determinista por-texto (no depende del orden de iteración)
# ---------------------------------------------------------------------------

def _rng_for(text: str, seed: int, salt: str = "") -> "random.Random":
    import random
    h = hashlib.sha256(f"{seed}|{salt}|{text}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------------------
# 1. Sin tildes  (á→a, é→e, í→i, ó→o, ú→u, ü→u, ñ→n; y mayúsculas)
# ---------------------------------------------------------------------------

def strip_accents(text: str, seed: int = PROJECT_SEED) -> str:
    """Elimina diacríticos: descompone en NFD y quita las marcas combinantes.
    Cubre á/é/í/ó/ú/ü→vocal y ñ→n (Ñ→N), preservando el resto."""
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


# ---------------------------------------------------------------------------
# 2. Sin emojis
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # símbolos & pictogramas, suplemento, extendido-A
    "\U0001F1E6-\U0001F1FF"  # indicadores regionales (banderas)
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U00002B00-\U00002BFF"  # símbolos y flechas varias
    "\U00002190-\U000021FF"  # flechas
    "\U0000FE00-\U0000FE0F"  # selectores de variación
    "\U00002000-\U0000206F"  # puntuación general (incluye ZWJ 200D)  -> filtrado abajo
    "\U000020E3"             # combining enclosing keycap
    "\U0001F3FB-\U0001F3FF"  # modificadores de tono de piel
    "]+",
    flags=re.UNICODE,
)
# La puntuación general (2000-206F) contiene el ZWJ (200D) y el keycap, pero también
# espacios finos/comillas tipográficas; solo queremos quitar los invisibles de emoji.
_EMOJI_JOINERS = {"‍", "️", "⃣"}


def remove_emojis(text: str, seed: int = PROJECT_SEED) -> str:
    """Elimina emojis y sus uniones (ZWJ, selectores de variación, keycaps).
    No toca puntuación ni letras."""
    def _keep(ch: str) -> bool:
        if ch in _EMOJI_JOINERS:
            return False
        cp = ord(ch)
        if 0x1F300 <= cp <= 0x1FAFF: return False
        if 0x1F1E6 <= cp <= 0x1F1FF: return False
        if 0x2600 <= cp <= 0x27BF: return False
        if 0x2B00 <= cp <= 0x2BFF: return False
        if 0x1F3FB <= cp <= 0x1F3FF: return False
        return True
    kept = [ch for ch in text if _keep(ch)]
    if len(kept) == len(text):
        return text  # sin emojis -> texto intacto (evita contar cambios de whitespace)
    # solo limpiamos los huecos que deja el emoji eliminado (no tocamos saltos de línea)
    return re.sub(r"[ \t]{2,}", " ", "".join(kept)).strip()


# ---------------------------------------------------------------------------
# 3. Minúsculas / MAYÚSCULAS
# ---------------------------------------------------------------------------

def lowercase(text: str, seed: int = PROJECT_SEED) -> str:
    """Todo a minúsculas (pérdida de mayúsculas de énfasis/inicio de frase)."""
    return text.lower()


def uppercase(text: str, seed: int = PROJECT_SEED) -> str:
    """TODO A MAYÚSCULAS (estilo grito/énfasis en redes)."""
    return text.upper()


# ---------------------------------------------------------------------------
# 4. Alargamientos  ("holaaaa")
# ---------------------------------------------------------------------------

_VOWELS = "aeiouáéíóúAEIOUÁÉÍÓÚ"
_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)  # tokens alfabéticos


def elongate(text: str, seed: int = PROJECT_SEED, prob: float = 0.30) -> str:
    """Alarga la última vocal de algunos tokens (≥3 letras) repitiéndola 2-4 veces,
    simulando énfasis ("hola"→"holaaa"). Determinista: cada token se decide con un
    RNG sembrado por (seed, texto)."""
    rng = _rng_for(text, seed, salt="elongate")

    def _repl(m: re.Match) -> str:
        tok = m.group(0)
        if len(tok) < 3 or not any(c in _VOWELS for c in tok):
            return tok
        if rng.random() >= prob:
            return tok
        # índice de la última vocal
        idx = max(i for i, c in enumerate(tok) if c in _VOWELS)
        rep = rng.choice([2, 3, 4])
        return tok[: idx + 1] + tok[idx] * rep + tok[idx + 1:]

    return _WORD_RE.sub(_repl, text)


# ---------------------------------------------------------------------------
# 5. Abreviaturas de chat  (q / x / xq / tmb …)
# ---------------------------------------------------------------------------

# Multi-palabra primero (se aplican antes que las de una palabra).
_CHAT_MULTI = [
    ("por que", "xq"), ("por qué", "xq"), ("por favor", "xfa"),
    ("te quiero", "tq"), ("de nada", "dnd"),
]
_CHAT_SINGLE = {
    "que": "q", "qué": "q", "porque": "xq", "por": "x",
    "también": "tmb", "tambien": "tmb", "para": "pa",
    "bien": "bn", "mensaje": "msj", "gente": "gnt",
    "mucho": "mxo", "muchos": "mxos", "hombre": "wn",
}


def chat_abbrev(text: str, seed: int = PROJECT_SEED) -> str:
    """Sustituye palabras completas por abreviaturas de chat frecuentes en español.
    Insensible a mayúsculas en la coincidencia; la abreviatura va en minúscula."""
    out = text
    for phrase, repl in _CHAT_MULTI:
        out = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", repl, out, flags=re.IGNORECASE)
    def _repl(m: re.Match) -> str:
        return _CHAT_SINGLE[m.group(0).lower()]
    pattern = r"(?<!\w)(" + "|".join(re.escape(w) for w in _CHAT_SINGLE) + r")(?!\w)"
    return re.sub(pattern, _repl, out, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# 6. Sin puntuación
# ---------------------------------------------------------------------------

# @ y # se PRESERVAN: son placeholders vistos en entrenamiento (@user anonimizado,
# #hashtags). Quitarlos metería un cambio de distribución ajeno a la puntuación.
_KEEP_PUNCT = {"@", "#"}


def remove_punctuation(text: str, seed: int = PROJECT_SEED) -> str:
    """Elimina puntuación real (categoría Unicode P*: , . ! ? ¡ ¿ ; : … « » etc.),
    conservando letras, dígitos, emojis, espacios y los marcadores @user / #hashtag."""
    out = "".join(
        ch if (ch in _KEEP_PUNCT or not unicodedata.category(ch).startswith("P")) else " "
        for ch in text
    )
    return re.sub(r"\s{2,}", " ", out).strip()


# ---------------------------------------------------------------------------
# 7. (OPCIONAL) Code-switching ES→EN parcial
# ---------------------------------------------------------------------------

_CS_MAP = {
    "y": "and", "pero": "but", "muy": "very", "porque": "because",
    "gracias": "thanks", "hoy": "today", "mañana": "tomorrow",
    "siempre": "always", "nunca": "never", "gente": "people",
    "cuando": "when", "también": "also", "ahora": "now",
}


def code_switch(text: str, seed: int = PROJECT_SEED, prob: float = 0.50) -> str:
    """(OPCIONAL) Sustituye algunas palabras funcionales frecuentes por su equivalente
    en inglés, simulando code-switching parcial. Determinista por-texto. Se marca como
    opcional/reserva: toca palabras de contenido y podría confundir la señal de sentimiento."""
    rng = _rng_for(text, seed, salt="code_switch")
    def _repl(m: re.Match) -> str:
        w = m.group(0)
        low = w.lower()
        if low in _CS_MAP and rng.random() < prob:
            return _CS_MAP[low]
        return w
    pattern = r"(?<!\w)(" + "|".join(re.escape(w) for w in _CS_MAP) + r")(?!\w)"
    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

PERTURBATIONS: dict[str, Callable[..., str]] = {
    "sin_tildes": strip_accents,
    "sin_emojis": remove_emojis,
    "minusculas": lowercase,
    "mayusculas": uppercase,
    "alargamientos": elongate,
    "abrev_chat": chat_abbrev,
    "sin_puntuacion": remove_punctuation,
}
OPTIONAL_PERTURBATIONS: dict[str, Callable[..., str]] = {
    "code_switching": code_switch,
}


def apply_perturbation(name: str, texts: list[str], seed: int = PROJECT_SEED) -> list[str]:
    fn = {**PERTURBATIONS, **OPTIONAL_PERTURBATIONS}[name]
    return [fn(t, seed=seed) for t in texts]


# ---------------------------------------------------------------------------
# Informe de la taxonomía (cobertura + ejemplos) — sin modelos
# ---------------------------------------------------------------------------

def _load_test_texts() -> list[str]:
    from pathlib import Path
    from datasets import load_from_disk
    root = Path(__file__).resolve().parents[2]
    ds = load_from_disk(str(root / "data" / "processed" / "cardiff_es" / "test"))
    return list(ds["text"])


def _report() -> None:
    texts = _load_test_texts()
    n = len(texts)
    all_perts = {**PERTURBATIONS, **OPTIONAL_PERTURBATIONS}
    print(f"Test Cardiff ES: {n} ejemplos\n")
    print(f"{'perturbación':<16} {'% afectados':>12}   ejemplo antes → después")
    print("-" * 100)
    for name, fn in all_perts.items():
        changed_idx = []
        for i, t in enumerate(texts):
            if fn(t, seed=PROJECT_SEED) != t:
                changed_idx.append(i)
        pct = 100.0 * len(changed_idx) / n
        tag = " (opcional)" if name in OPTIONAL_PERTURBATIONS else ""
        # dos ejemplos afectados
        egs = changed_idx[:2]
        print(f"{name + tag:<16} {pct:>10.1f} %")
        for i in egs:
            before = texts[i].replace("\n", " ")
            after = fn(texts[i], seed=PROJECT_SEED).replace("\n", " ")
            print(f"    ANTES : {before[:88]}")
            print(f"    DESPUÉS: {after[:88]}")
        print()


if __name__ == "__main__":
    _report()
