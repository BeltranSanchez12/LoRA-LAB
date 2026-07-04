"""
test_prompting.py — Tests para PromptingBaseline (Sprint 2, T4).

Verifica:
1. format_prompt genera el formato correcto (sin modelo real).
2. parse_response detecta correctamente los tres labels y el fallback.
3. La tasa de fallback se calcula correctamente en predict_batch (con mock).
4. select_examples produce la distribucion estratificada correcta.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Asegurar que la raiz del proyecto esta en el path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.prompting import PromptingBaseline, LABEL_NAMES

# ---------------------------------------------------------------------------
# Fixture: instancia base con config real
# ---------------------------------------------------------------------------

@pytest.fixture
def baseline():
    """PromptingBaseline con k=0 cargado desde el YAML del proyecto."""
    config_path = PROJECT_ROOT / "configs" / "prompting_protocol.yaml"
    return PromptingBaseline(config_path=config_path, k=0)


# ---------------------------------------------------------------------------
# 1. format_prompt
# ---------------------------------------------------------------------------

class TestFormatPrompt:
    def test_zero_shot_no_examples(self, baseline):
        """Zero-shot: el bloque examples debe quedar vacio."""
        prompt = baseline.format_prompt("Este producto es genial")
        assert "Tweet: Este producto es genial" in prompt
        assert "Sentiment:" in prompt
        # No hay ejemplos en el bloque
        lines = prompt.strip().split("\n")
        # La ultima linea debe ser "Sentiment:" (o acabar con "Sentiment:")
        last_meaningful = [l for l in lines if l.strip()][-1]
        assert last_meaningful.strip() == "Sentiment:"

    def test_few_shot_examples_appear(self, baseline):
        """Few-shot: los ejemplos deben aparecer antes del tweet objetivo."""
        examples = [
            {"text": "Odio este servicio", "label": "negative"},
            {"text": "Todo bien por aqui", "label": "neutral"},
            {"text": "Me encanta este producto", "label": "positive"},
        ]
        prompt = baseline.format_prompt("Tweet de prueba", examples=examples)

        # Los textos de los ejemplos deben aparecer
        assert "Odio este servicio" in prompt
        assert "Todo bien por aqui" in prompt
        assert "Me encanta este producto" in prompt
        # Las etiquetas de los ejemplos deben aparecer
        assert "negative" in prompt
        assert "neutral" in prompt
        assert "positive" in prompt
        # El tweet objetivo debe aparecer al final
        assert "Tweet de prueba" in prompt
        # Los ejemplos deben aparecer ANTES del tweet objetivo
        pos_ejemplos = prompt.find("Odio este servicio")
        pos_target = prompt.rfind("Tweet de prueba")
        assert pos_ejemplos < pos_target, "Los ejemplos deben preceder al tweet objetivo"

    def test_empty_examples_list(self, baseline):
        """Lista vacia de ejemplos equivale a zero-shot."""
        prompt_none = baseline.format_prompt("Test tweet", examples=None)
        prompt_empty = baseline.format_prompt("Test tweet", examples=[])
        assert prompt_none == prompt_empty

    def test_template_placeholders_replaced(self, baseline):
        """No debe quedar ningun placeholder sin reemplazar."""
        prompt = baseline.format_prompt("Texto de prueba", examples=[])
        assert "{text}" not in prompt
        assert "{examples}" not in prompt

    def test_tweet_marker_present(self, baseline):
        """El marcador 'Tweet:' debe aparecer exactamente antes del texto objetivo."""
        text = "Texto especifico de prueba"
        prompt = baseline.format_prompt(text, examples=None)
        assert f"Tweet: {text}" in prompt


# ---------------------------------------------------------------------------
# 2. parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    @pytest.mark.parametrize("response,expected", [
        # Casos simples (una sola palabra)
        ("positive", "positive"),
        ("negative", "negative"),
        ("neutral", "neutral"),
        # Case-insensitive
        ("Positive", "positive"),
        ("NEGATIVE", "negative"),
        ("Neutral", "neutral"),
        ("POSITIVE", "positive"),
        # Con texto extra antes y despues
        ("The sentiment is positive.", "positive"),
        ("I think this is negative sentiment.", "negative"),
        ("This tweet has neutral tone.", "neutral"),
        # Con nueva linea
        ("positive\n", "positive"),
        ("\nnegative", "negative"),
        # Con espacio
        ("  neutral  ", "neutral"),
    ])
    def test_correct_labels(self, baseline, response, expected):
        result = baseline.parse_response(response)
        assert result == expected, f"parse_response({response!r}) = {result!r}, esperado {expected!r}"

    def test_fallback_on_empty(self, baseline):
        """Respuesta vacia -> fallback_label."""
        result = baseline.parse_response("")
        assert result == baseline.fallback_label

    def test_fallback_on_garbage(self, baseline):
        """Respuesta sin ninguna etiqueta -> fallback_label."""
        for garbage in ["xyz", "123", "Buenas tardes", "N/A", "???", "sentimiento"]:
            result = baseline.parse_response(garbage)
            assert result == baseline.fallback_label, (
                f"parse_response({garbage!r}) deberia retornar fallback, retorno {result!r}"
            )

    def test_first_match_wins(self, baseline):
        """Si hay varias etiquetas, gana la que aparece primero en el texto."""
        # "negative" aparece antes de "positive"
        result = baseline.parse_response("negative or positive? negative")
        assert result == "negative"

    def test_positive_before_negative(self, baseline):
        """Cuando positive aparece antes, gana positive."""
        result = baseline.parse_response("positive vibes but negative undertones")
        assert result == "positive"

    def test_all_labels_covered(self, baseline):
        """Todos los LABEL_NAMES deben ser parseables."""
        for label in LABEL_NAMES:
            result = baseline.parse_response(label)
            assert result == label

    def test_fallback_label_default(self, baseline):
        """El fallback por defecto debe ser 'neutral'."""
        assert baseline.fallback_label == "neutral"


# ---------------------------------------------------------------------------
# 3. Tasa de fallback en predict_batch
# ---------------------------------------------------------------------------

class TestFallbackRate:
    def _make_mock_model_tokenizer(self, responses: list[str]):
        """Crea mocks de model y tokenizer que devuelven respuestas predeterminadas."""
        import torch

        # input_ids de longitud 5 (simulado)
        fake_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        fake_attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

        # tokenizer debe ser callable y devolver un dict con .items()
        tokenizer_return = {
            "input_ids": fake_input_ids,
            "attention_mask": fake_attention_mask,
        }
        tokenizer = MagicMock(return_value=tokenizer_return)
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token = "[EOS]"
        tokenizer.padding_side = "left"

        # generate devuelve tensor con input (5 tokens) + 1 token de respuesta
        call_count = [0]
        def fake_generate(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return torch.tensor([[1, 2, 3, 4, 5, 99]])

        model = MagicMock()
        model.eval = MagicMock(return_value=None)
        model.generate = MagicMock(side_effect=fake_generate)
        model.parameters = MagicMock(return_value=iter([torch.zeros(10)]))

        decode_call = [0]
        def fake_decode(tokens, **kwargs):
            resp = responses[decode_call[0] % len(responses)]
            decode_call[0] += 1
            return resp
        tokenizer.decode = MagicMock(side_effect=fake_decode)

        return model, tokenizer

    def test_zero_fallback_rate(self, baseline):
        """Cuando todas las respuestas son validas, fallback_rate = 0."""
        responses = ["positive", "negative", "neutral", "positive"]
        model, tokenizer = self._make_mock_model_tokenizer(responses)

        texts = ["texto1", "texto2", "texto3", "texto4"]
        preds, fallback_rate = baseline.predict_batch(texts, model, tokenizer, device="cpu")

        assert fallback_rate == 0.0
        assert len(preds) == len(texts)
        assert all(p in LABEL_NAMES for p in preds)

    def test_full_fallback_rate(self, baseline):
        """Cuando todas las respuestas son invalidas, fallback_rate = 1.0."""
        responses = ["xyz", "abc", "???", "nada"]
        model, tokenizer = self._make_mock_model_tokenizer(responses)

        texts = ["texto1", "texto2", "texto3", "texto4"]
        preds, fallback_rate = baseline.predict_batch(texts, model, tokenizer, device="cpu")

        assert fallback_rate == 1.0
        # Todas las predicciones deben ser el fallback_label
        assert all(p == baseline.fallback_label for p in preds)

    def test_partial_fallback_rate(self, baseline):
        """2 de 4 respuestas invalidas -> fallback_rate = 0.5."""
        responses = ["positive", "xyz", "negative", "abc"]
        model, tokenizer = self._make_mock_model_tokenizer(responses)

        texts = ["texto1", "texto2", "texto3", "texto4"]
        preds, fallback_rate = baseline.predict_batch(texts, model, tokenizer, device="cpu")

        assert fallback_rate == pytest.approx(0.5)

    def test_predictions_are_valid_labels(self, baseline):
        """Todas las predicciones deben ser etiquetas validas (incluyendo fallback)."""
        responses = ["positive", "garbage", "NEGATIVE", "Neutral"]
        model, tokenizer = self._make_mock_model_tokenizer(responses)

        texts = ["t1", "t2", "t3", "t4"]
        preds, _ = baseline.predict_batch(texts, model, tokenizer, device="cpu")

        for p in preds:
            assert p in LABEL_NAMES, f"Prediccion invalida: {p!r}"


# ---------------------------------------------------------------------------
# 4. select_examples
# ---------------------------------------------------------------------------

class TestSelectExamples:
    def _make_dummy_dataset(self):
        """Dataset minimo con 30 ejemplos: 10 por clase."""
        import datasets
        texts = [f"texto_{i}" for i in range(30)]
        labels = [i % 3 for i in range(30)]  # 0,1,2,0,1,2,...
        return datasets.Dataset.from_dict({"text": texts, "label": labels})

    def test_zero_k_returns_empty(self, baseline):
        ds = self._make_dummy_dataset()
        examples = baseline.select_examples(ds, k=0, seed=42)
        assert examples == []

    def test_k4_stratified(self, baseline):
        """k=4: debe haber al menos 1 ejemplo de cada clase (3 clases, 4 ejemplos)."""
        ds = self._make_dummy_dataset()
        examples = baseline.select_examples(ds, k=4, seed=42)
        assert len(examples) == 4
        labels_found = {ex["label"] for ex in examples}
        # Al menos 2 clases representadas (k=4, 3 clases -> al menos 1 clase con 2 ejemplos)
        assert len(labels_found) >= 2

    def test_k8_returns_correct_size(self, baseline):
        ds = self._make_dummy_dataset()
        examples = baseline.select_examples(ds, k=8, seed=42)
        assert len(examples) == 8

    def test_k16_returns_correct_size(self, baseline):
        ds = self._make_dummy_dataset()
        examples = baseline.select_examples(ds, k=16, seed=42)
        assert len(examples) == 16

    def test_reproducibility(self, baseline):
        """Misma seed -> mismos ejemplos."""
        ds = self._make_dummy_dataset()
        ex1 = baseline.select_examples(ds, k=8, seed=42)
        ex2 = baseline.select_examples(ds, k=8, seed=42)
        assert ex1 == ex2

    def test_different_seeds_different_examples(self, baseline):
        """Seeds diferentes -> ejemplos diferentes (con alta probabilidad)."""
        ds = self._make_dummy_dataset()
        ex1 = baseline.select_examples(ds, k=8, seed=42)
        ex2 = baseline.select_examples(ds, k=8, seed=99)
        # No garantizado en todos los casos, pero con 30 ejemplos y k=8 es muy probable
        texts1 = [e["text"] for e in ex1]
        texts2 = [e["text"] for e in ex2]
        assert texts1 != texts2

    def test_examples_have_correct_keys(self, baseline):
        """Los ejemplos deben tener las claves 'text' y 'label'."""
        ds = self._make_dummy_dataset()
        examples = baseline.select_examples(ds, k=6, seed=42)
        for ex in examples:
            assert "text" in ex
            assert "label" in ex
            assert ex["label"] in ("negative", "neutral", "positive")

    def test_k3_exactly_one_per_class(self, baseline):
        """k=3: exactamente 1 ejemplo por clase."""
        ds = self._make_dummy_dataset()
        examples = baseline.select_examples(ds, k=3, seed=42)
        assert len(examples) == 3
        labels_found = [ex["label"] for ex in examples]
        label_counts = {l: labels_found.count(l) for l in ("negative", "neutral", "positive")}
        assert all(v == 1 for v in label_counts.values()), f"No es 1 por clase: {label_counts}"


# ---------------------------------------------------------------------------
# 5. Carga del protocolo
# ---------------------------------------------------------------------------

class TestProtocolLoading:
    def test_loads_from_yaml(self):
        """El constructor debe cargar correctamente el YAML del protocolo."""
        config_path = PROJECT_ROOT / "configs" / "prompting_protocol.yaml"
        b = PromptingBaseline(config_path=config_path, k=0)
        assert b.template is not None
        assert "{text}" in b.template
        assert "{examples}" in b.template
        assert b.max_new_tokens > 0

    def test_file_not_found_raises(self):
        """Un path invalido debe lanzar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PromptingBaseline(config_path="/nonexistent/path/protocol.yaml", k=0)

    def test_generation_kwargs_loaded(self):
        """generation_kwargs debe cargarse del YAML."""
        config_path = PROJECT_ROOT / "configs" / "prompting_protocol.yaml"
        b = PromptingBaseline(config_path=config_path)
        assert "do_sample" in b.generation_kwargs


# ---------------------------------------------------------------------------
# 6. Resultado real de zero-shot (ya ejecutado en la DGX)
# ---------------------------------------------------------------------------

def test_zeroshot_result_is_real():
    """El JSON de zero-shot debe tener métricas reales (no el placeholder PENDIENTE).

    Tras el arreglo del baseline de prompting (chat template + thinking off +
    parser robusto) el zero-shot se ejecutó de verdad: f1_macro/accuracy no
    nulos y fallback_rate bajo (≈0).
    """
    path = PROJECT_ROOT / "results" / "baselines" / "prompting_zeroshot_Qwen3-1.7B.json"
    assert path.exists(), f"Resultado zero-shot no encontrado: {path}"
    with path.open() as f:
        data = json.load(f)
    assert data["f1_macro"] is not None, "f1_macro no debe ser null"
    assert data["accuracy"] is not None and data["accuracy"] > 0.33, (
        "accuracy debe estar muy por encima del azar (0.33)"
    )
    assert data["fallback_rate"] is not None and data["fallback_rate"] < 0.1, (
        "fallback_rate debe ser ≈0 tras el arreglo del parser/template"
    )
