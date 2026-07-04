"""
prompting.py — Baseline de prompting zero-shot y few-shot (Sprint 2, T4).

Clasificacion de sentimiento en tweets en espanol usando un LLM generativo
con prompting. Soporta zero-shot (k=0) y few-shot (k=4, 8, 16).

El protocolo completo esta congelado en configs/prompting_protocol.yaml.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

LABEL_NAMES = ("negative", "neutral", "positive")
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Orden de prioridad para el parsing (positivo primero para desambiguar
# si el modelo escribe "not positive" o similares — busqueda izquierda-derecha
# en la respuesta por defecto; la prioridad se aplica solo si hay empate posicional)
_PARSE_PRIORITY = ("positive", "negative", "neutral")

# Variantes aceptadas por el parser (inglés y español), mapeadas a la etiqueta
# canónica. Qwen3 a veces responde en español ("negativo"/"positiva"/…).
_LABEL_VARIANTS: dict[str, tuple[str, ...]] = {
    "positive": ("positive", "positivo", "positiva"),
    "negative": ("negative", "negativo", "negativa"),
    "neutral": ("neutral", "neutro", "neutra"),
}


# ---------------------------------------------------------------------------
# PromptingBaseline
# ---------------------------------------------------------------------------

class PromptingBaseline:
    """Clasificacion de sentimiento por prompting: zero-shot y few-shot.

    Parametros configurables (desde configs/prompting_protocol.yaml):
    - template: string con {text} y {examples} como placeholders.
    - k: numero de ejemplos para few-shot (0 para zero-shot).
    - example_selection: "stratified_random" (seed fija).
    - max_new_tokens: maximo de tokens a generar (default: 10).
    - fallback_label: etiqueta cuando el parsing falla (default: "neutral").

    Metodos publicos
    ----------------
    format_prompt(text, examples=None) -> str
    parse_response(response: str) -> str
        Devuelve "positive" | "negative" | "neutral".
    predict_batch(texts, model, tokenizer) -> tuple[list[str], float]
        Devuelve (predicciones_str, fallback_rate).
    select_examples(train_dataset, k, seed=42) -> list[dict]
        Selecciona k//3 ejemplos por clase de forma estratificada.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        k: int = 0,
    ) -> None:
        """
        Parametros
        ----------
        config_path :
            Ruta al YAML del protocolo.  Si es None se busca en la ubicacion
            estandar del proyecto (configs/prompting_protocol.yaml).
        k :
            Numero de ejemplos few-shot (0 = zero-shot).
        """
        self.k = k

        # Cargar protocolo
        if config_path is None:
            # Buscar relativo a la raiz del proyecto (2 niveles arriba de src/models/)
            default_path = Path(__file__).resolve().parents[2] / "configs" / "prompting_protocol.yaml"
            config_path = default_path

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Protocolo no encontrado: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        proto = raw["prompting_protocol"]
        self.template: str = proto["template"]
        self.fallback_label: str = proto.get("fallback_label", "neutral")
        self.max_new_tokens: int = proto.get("max_new_tokens", 16)
        self.generation_kwargs: dict = proto.get("generation_kwargs", {
            "do_sample": False,
        })

        # Protocolo de chat (modelos instruct tipo Qwen3)
        self.use_chat_template: bool = proto.get("use_chat_template", True)
        self.enable_thinking: bool = proto.get("enable_thinking", False)
        self.system_prompt: str = (proto.get("system_prompt") or "").strip()
        self.answer_cue: str = (proto.get("answer_cue") or "").strip()

        logger.info(
            "PromptingBaseline cargado. k=%d, fallback='%s', max_new_tokens=%d, "
            "chat_template=%s, thinking=%s",
            self.k, self.fallback_label, self.max_new_tokens,
            self.use_chat_template, self.enable_thinking,
        )

    # ------------------------------------------------------------------
    # Seleccion de ejemplos few-shot
    # ------------------------------------------------------------------

    def select_examples(
        self,
        train_dataset,
        k: int,
        seed: int = 42,
    ) -> list[dict]:
        """Selecciona k ejemplos de forma estratificada (k//3 por clase).

        Parametros
        ----------
        train_dataset :
            Dataset de HuggingFace con columnas "text" y "label" (int).
        k :
            Total de ejemplos few-shot.  Debe ser multiplo de 3 para
            distribucion exacta; el sobrante se asigna a clases con mayor
            indice.
        seed :
            Semilla para reproducibilidad (default: 42).

        Retorna
        -------
        Lista de dicts [{"text": ..., "label": "positive"}, ...] con exactamente
        k elementos, ordenados por clase.
        """
        if k == 0:
            return []

        rng = np.random.default_rng(seed)
        labels_arr = np.array(train_dataset["label"])
        texts_arr = np.array(train_dataset["text"])

        base_per_class = k // 3
        remainder = k - base_per_class * 3

        # Distribuir sobrante: clases 2, 1, 0 (positive, neutral, negative)
        n_per_class = {0: base_per_class, 1: base_per_class, 2: base_per_class}
        for i in range(remainder):
            n_per_class[2 - i] += 1

        examples: list[dict] = []
        for class_id in sorted(n_per_class.keys()):
            n = n_per_class[class_id]
            indices = np.where(labels_arr == class_id)[0]
            if len(indices) < n:
                logger.warning(
                    "Clase %d tiene solo %d ejemplos; se piden %d.",
                    class_id, len(indices), n,
                )
                n = len(indices)
            selected = rng.choice(indices, size=n, replace=False)
            for idx in selected:
                examples.append({
                    "text": str(texts_arr[idx]),
                    "label": ID2LABEL[class_id],
                })

        logger.info(
            "Seleccionados %d ejemplos few-shot (k=%d, seed=%d).",
            len(examples), k, seed,
        )
        return examples

    # ------------------------------------------------------------------
    # Formateo del prompt
    # ------------------------------------------------------------------

    def format_prompt(
        self,
        text: str,
        examples: list[dict] | None = None,
    ) -> str:
        """Formatea el prompt para un texto dado.

        Parametros
        ----------
        text :
            Texto del tweet a clasificar.
        examples :
            Lista de dicts [{"text": ..., "label": ...}].  Si es None o
            lista vacia, el bloque {examples} queda vacio (zero-shot).

        Retorna
        -------
        Prompt listo para enviar al LLM.
        """
        if examples:
            examples_block = ""
            for ex in examples:
                examples_block += f"Tweet: {ex['text']}\nSentiment: {ex['label']}\n\n"
        else:
            examples_block = ""

        prompt = self.template.format(examples=examples_block, text=text)
        return prompt

    # ------------------------------------------------------------------
    # Construccion del prompt de chat (modelos instruct)
    # ------------------------------------------------------------------

    def build_messages(
        self,
        text: str,
        examples: list[dict] | None = None,
    ) -> list[dict]:
        """Construye los mensajes de chat para clasificar *text*.

        Estructura: ``system`` (instrucción) + turnos few-shot
        (``user``: tweet / ``assistant``: etiqueta) + turno final ``user`` con
        el tweet y la señal de respuesta (``answer_cue``).
        """
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for ex in (examples or []):
            messages.append({"role": "user", "content": f"Tweet: {ex['text']}"})
            messages.append({"role": "assistant", "content": ex["label"]})
        user = f"Tweet: {text}"
        if self.answer_cue:
            user += f"\n\n{self.answer_cue}"
        messages.append({"role": "user", "content": user})
        return messages

    def render_prompt(self, text: str, examples: list[dict] | None, tokenizer) -> str:
        """Renderiza el prompt final que se envía al tokenizador.

        Usa el chat template del modelo (con ``enable_thinking`` si el
        tokenizador lo soporta); si no, recurre a la plantilla de texto plano
        (``format_prompt``).
        """
        if self.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = self.build_messages(text, examples)
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                # Tokenizador sin soporte de enable_thinking: se ignora ese kwarg
                # (el bloque <think> se limpia luego en el parser).
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return self.format_prompt(text, examples)

    # ------------------------------------------------------------------
    # Parsing de la respuesta
    # ------------------------------------------------------------------

    def _match_label(self, response: str) -> Optional[str]:
        """Detecta la etiqueta canónica en la respuesta, o None si no hay.

        - Ignora un bloque ``<think>...</think>`` (toma lo posterior al último
          ``</think>``), por si el modelo razona pese a enable_thinking=False.
        - Acepta variantes en inglés y español (case-insensitive).
        - Si hay varias, gana la que aparece primero en el texto; en empate
          posicional se usa el orden de prioridad (positive, negative, neutral).
        """
        text = response.lower()
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]

        best_label: Optional[str] = None
        best_pos = len(text) + 1
        for canon in _PARSE_PRIORITY:
            for variant in _LABEL_VARIANTS[canon]:
                # \b + variante: evita falsos positivos tipo "positively"
                match = re.search(r"\b" + variant, text)
                if match and match.start() < best_pos:
                    best_pos = match.start()
                    best_label = canon
        return best_label

    def parse_response(self, response: str) -> str:
        """Extrae la etiqueta de la respuesta del LLM.

        Busca "positive"/"negative"/"neutral" (y sus variantes en español) en
        la respuesta, case-insensitive.  Si hay varias coincidencias, devuelve
        la que aparece primero en el texto.  Si no encuentra ninguna, devuelve
        ``fallback_label``.

        Parametros
        ----------
        response :
            Texto generado por el LLM.

        Retorna
        -------
        "positive" | "negative" | "neutral"
        """
        label = self._match_label(response)
        if label is None:
            logger.debug("Fallback: no se encontro etiqueta en respuesta: %r", response[:80])
            return self.fallback_label
        return label

    # ------------------------------------------------------------------
    # Prediccion en batch
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        texts: Sequence[str],
        model,
        tokenizer,
        examples: list[dict] | None = None,
        device: str = "cpu",
    ) -> tuple[list[str], float]:
        """Genera predicciones para una lista de textos.

        Parametros
        ----------
        texts :
            Lista de textos a clasificar.
        model :
            Modelo causal (AutoModelForCausalLM) ya cargado.
        tokenizer :
            Tokenizador correspondiente.
        examples :
            Ejemplos few-shot pre-seleccionados (None = zero-shot).
        device :
            Dispositivo de inferencia ("cpu" o "cuda").

        Retorna
        -------
        (predictions, fallback_rate)
            predictions : lista de strings ("positive"|"negative"|"neutral").
            fallback_rate : fraccion de predicciones que cayeron a fallback.
        """
        import torch

        predictions: list[str] = []
        n_fallback = 0

        model.eval()
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            for i, text in enumerate(texts):
                prompt = self.render_prompt(text, examples, tokenizer)

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                gen_kwargs = dict(self.generation_kwargs)
                gen_kwargs["max_new_tokens"] = self.max_new_tokens

                # Eliminar temperature si do_sample=False (evita warning de HF)
                if not gen_kwargs.get("do_sample", True):
                    gen_kwargs.pop("temperature", None)

                output_ids = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.pad_token_id,
                    **gen_kwargs,
                )

                # Decodificar solo los tokens nuevos
                input_len = inputs["input_ids"].shape[1]
                new_tokens = output_ids[0][input_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                # Conteo de fallback preciso: solo cuenta si NO se detectó etiqueta.
                matched = self._match_label(response)
                if matched is None:
                    n_fallback += 1
                    label = self.fallback_label
                else:
                    label = matched

                predictions.append(label)

                if (i + 1) % 50 == 0:
                    logger.info("Procesados %d/%d textos", i + 1, len(texts))

        fallback_rate = n_fallback / len(texts) if texts else 0.0
        logger.info(
            "predict_batch completado. N=%d, fallback_rate=%.3f",
            len(texts), fallback_rate,
        )
        return predictions, fallback_rate


# ---------------------------------------------------------------------------
# Helpers para seleccion y dumping de ejemplos fijos
# ---------------------------------------------------------------------------

def select_and_print_fixed_examples(
    train_dataset,
    k_values: list[int] = (4, 8, 16),
    seed: int = 42,
) -> dict[int, list[dict]]:
    """Selecciona y devuelve los ejemplos few-shot fijos para cada k.

    Util para rellenar la seccion fixed_examples del YAML una sola vez.
    """
    baseline = PromptingBaseline(k=0)
    result: dict[int, list[dict]] = {}
    for k in k_values:
        examples = baseline.select_examples(train_dataset, k=k, seed=seed)
        result[k] = examples
        print(f"\n--- k={k} ---")
        for ex in examples:
            print(f"  [{ex['label']}] {ex['text'][:80]}")
    return result
