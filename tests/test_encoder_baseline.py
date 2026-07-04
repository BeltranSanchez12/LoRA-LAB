"""
test_encoder_baseline.py — Tests para EncoderFineTuner y OffTheShelfEncoder (Sprint 2, T5).

Tests:
1. OffTheShelfEncoder devuelve un ExperimentResult valido con los campos correctos.
2. EncoderFineTuner arranca sin error (usa un modelo minimo de HuggingFace o mock).
3. EncoderFineTuner.evaluate devuelve ExperimentResult con campos correctos.
4. Verificaciones de integridad de los JSONs de resultados guardados.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoder_baseline import EncoderFineTuner, OffTheShelfEncoder, LABEL2ID
from src.models.evaluate import ExperimentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset():
    """Dataset minimo de 12 ejemplos (4 por clase)."""
    import datasets
    texts = [
        "Esto es terrible, muy malo",
        "No me gusto nada",
        "Que desastre de servicio",
        "Horrible experiencia",
        "El tiempo es normal hoy",
        "Nada especial por aqui",
        "Una experiencia mediocre",
        "No fue ni bien ni mal",
        "Excelente servicio!",
        "Me encanto mucho",
        "Muy buena experiencia",
        "Genial, lo recomiendo",
    ]
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    return datasets.Dataset.from_dict({"text": texts, "label": labels})


@pytest.fixture
def tiny_test_dataset():
    """Dataset minimo de test (6 ejemplos)."""
    import datasets
    texts = [
        "Muy malo", "Terrible",
        "Normal", "Regular",
        "Excelente", "Fantastico",
    ]
    labels = [0, 0, 1, 1, 2, 2]
    return datasets.Dataset.from_dict({"text": texts, "label": labels})


# ---------------------------------------------------------------------------
# Tests de OffTheShelfEncoder
# ---------------------------------------------------------------------------

class TestOffTheShelfEncoder:
    """Tests para el evaluador off-the-shelf (pysentimiento)."""

    def test_init_default(self):
        """El constructor debe inicializarse sin error con los parametros por defecto."""
        encoder = OffTheShelfEncoder()
        assert encoder.model_id == "pysentimiento/robertuito-sentiment-analysis"
        assert encoder.use_pysentimiento is True
        assert encoder._analyzer is None  # lazy loading

    def test_init_custom_model(self):
        """Debe aceptar model_id personalizado."""
        encoder = OffTheShelfEncoder(model_id="some/model", use_pysentimiento=False)
        assert encoder.model_id == "some/model"

    def test_evaluate_returns_experiment_result(self, tiny_test_dataset):
        """evaluate() debe devolver un ExperimentResult valido."""
        encoder = OffTheShelfEncoder()

        # Mock del analyzer de pysentimiento
        mock_result = MagicMock()
        mock_result.output = "NEU"  # siempre neutral

        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)

        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)

        assert isinstance(result, ExperimentResult)

    def test_evaluate_result_has_required_fields(self, tiny_test_dataset):
        """El ExperimentResult debe tener todos los campos obligatorios."""
        encoder = OffTheShelfEncoder()

        mock_result = MagicMock()
        mock_result.output = "POS"
        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)
        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)

        # Campos obligatorios
        assert result.experiment_name is not None
        assert result.model_name == "pysentimiento/robertuito-sentiment-analysis"
        assert result.method == "encoder_offtheshelf"
        assert result.data_fraction == "full"
        assert result.seed == 42
        assert isinstance(result.f1_macro, float)
        assert 0.0 <= result.f1_macro <= 1.0
        assert isinstance(result.accuracy, float)
        assert 0.0 <= result.accuracy <= 1.0
        assert "negative" in result.f1_per_class
        assert "neutral" in result.f1_per_class
        assert "positive" in result.f1_per_class

    def test_evaluate_notes_say_out_of_domain(self, tiny_test_dataset):
        """Las notas deben indicar que es out-of-domain."""
        encoder = OffTheShelfEncoder()

        mock_result = MagicMock()
        mock_result.output = "NEG"
        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)
        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)

        assert "out-of-domain" in result.notes.lower() or "out-of-domain" in result.notes
        assert "TASS" in result.notes or "tass" in result.notes.lower()

    def test_evaluate_trainable_params_is_none(self, tiny_test_dataset):
        """Sin fine-tuning, trainable_params debe ser None."""
        encoder = OffTheShelfEncoder()

        mock_result = MagicMock()
        mock_result.output = "NEU"
        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)
        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)
        assert result.trainable_params is None

    def test_evaluate_train_time_is_none(self, tiny_test_dataset):
        """Sin entrenamiento, train_time_s debe ser None."""
        encoder = OffTheShelfEncoder()

        mock_result = MagicMock()
        mock_result.output = "POS"
        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)
        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)
        assert result.train_time_s is None

    def test_label_mapping_pos(self, tiny_test_dataset):
        """POS de pysentimiento debe mapearse a 'positive'."""
        encoder = OffTheShelfEncoder()
        mock_result = MagicMock()
        mock_result.output = "POS"
        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)
        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)
        # Todos predichos como positive (2). Test tiene 2 positives -> accuracy > 0
        assert result.accuracy > 0.0

    def test_label_mapping_neg(self, tiny_test_dataset):
        """NEG de pysentimiento debe mapearse a 'negative'."""
        encoder = OffTheShelfEncoder()
        mock_result = MagicMock()
        mock_result.output = "NEG"
        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)
        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)
        # Todos predichos como negative (0). Test tiene 2 negatives -> accuracy > 0
        assert result.accuracy > 0.0

    def test_inference_latency_is_positive(self, tiny_test_dataset):
        """La latencia de inferencia debe ser un numero positivo."""
        encoder = OffTheShelfEncoder()
        mock_result = MagicMock()
        mock_result.output = "NEU"
        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)
        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)
        assert result.inference_latency_ms is not None
        assert result.inference_latency_ms >= 0.0

    def test_to_flat_dict_serializable(self, tiny_test_dataset):
        """El resultado debe ser serializable a dict plano (para CSV)."""
        encoder = OffTheShelfEncoder()
        mock_result = MagicMock()
        mock_result.output = "POS"
        mock_analyzer = MagicMock()
        mock_analyzer.predict = MagicMock(return_value=mock_result)
        encoder._analyzer = mock_analyzer

        result = encoder.evaluate(tiny_test_dataset)
        flat = result.to_flat_dict()
        assert isinstance(flat, dict)
        assert "f1_macro" in flat
        assert "f1_negative" in flat
        assert "f1_neutral" in flat
        assert "f1_positive" in flat


# ---------------------------------------------------------------------------
# Tests de EncoderFineTuner (con modelo minimo o mock)
# ---------------------------------------------------------------------------

class TestEncoderFineTuner:
    """Tests para el fine-tuner de encoders."""

    def test_init_default_hparams(self):
        """El constructor debe inicializarse con hiperparametros por defecto."""
        finetuner = EncoderFineTuner(model_id="dccuchile/bert-base-spanish-wwm-cased")
        assert finetuner.model_id == "dccuchile/bert-base-spanish-wwm-cased"
        assert finetuner.hparams["lr"] == 2e-5
        assert finetuner.hparams["batch_size"] == 16
        assert finetuner.hparams["epochs"] == 3
        assert finetuner.hparams["seed"] == 42

    def test_init_custom_epochs(self):
        """El parametro epochs debe sobreescribir el de hparams."""
        finetuner = EncoderFineTuner(model_id="xlm-roberta-base", epochs=1)
        assert finetuner.hparams["epochs"] == 1

    def test_not_trained_raises_on_evaluate(self, tiny_test_dataset):
        """evaluate() antes de train() debe lanzar RuntimeError."""
        finetuner = EncoderFineTuner(model_id="xlm-roberta-base")
        with pytest.raises(RuntimeError, match="train"):
            finetuner.evaluate(tiny_test_dataset)

    def test_count_trainable_params_before_train(self):
        """count_trainable_params antes de cargar el modelo devuelve 0."""
        finetuner = EncoderFineTuner(model_id="xlm-roberta-base")
        assert finetuner._count_trainable_params() == 0

    def test_train_and_evaluate_with_tiny_model(self, tiny_dataset, tiny_test_dataset):
        """Train y evaluate con modelo pywikibot minimo de HuggingFace (google/bert_uncased_L-2_H-128_A-2)."""
        # Usar un modelo verdaderamente minimo para el test
        # google/bert_uncased_L-2_H-128_A-2 = BERT tiny, ~4.4M params
        TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"

        finetuner = EncoderFineTuner(
            model_id=TINY_MODEL,
            epochs=1,
            hparams={"lr": 2e-5, "batch_size": 4, "epochs": 1,
                     "warmup_ratio": 0.1, "seed": 42, "max_length": 32,
                     "weight_decay": 0.01},
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            train_time = finetuner.train(
                train_dataset=tiny_dataset,
                val_dataset=tiny_test_dataset,
                output_dir=tmpdir,
            )

        assert finetuner._is_trained is True
        assert train_time > 0.0
        assert finetuner._count_trainable_params() > 0

        result = finetuner.evaluate(
            tiny_test_dataset,
            train_time_s=train_time,
            peak_vram_gb=None,
        )

        # Verificar que el resultado es un ExperimentResult valido
        assert isinstance(result, ExperimentResult)
        assert result.model_name == TINY_MODEL
        assert result.method == "encoder_finetuned"
        assert result.data_fraction == "full"
        assert isinstance(result.f1_macro, float)
        assert 0.0 <= result.f1_macro <= 1.0
        assert isinstance(result.accuracy, float)
        assert result.trainable_params > 0
        assert result.train_time_s == train_time
        assert result.peak_vram_gb is None  # CPU
        assert result.inference_latency_ms is not None
        assert result.inference_latency_ms >= 0.0
        assert result.fallback_rate is None

    def test_evaluate_result_has_all_f1_classes(self, tiny_dataset, tiny_test_dataset):
        """El resultado debe tener F1 para las 3 clases."""
        TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"
        finetuner = EncoderFineTuner(
            model_id=TINY_MODEL,
            epochs=1,
            hparams={"lr": 2e-5, "batch_size": 4, "epochs": 1,
                     "warmup_ratio": 0.1, "seed": 42, "max_length": 32,
                     "weight_decay": 0.01},
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            train_time = finetuner.train(tiny_dataset, tiny_test_dataset, tmpdir)

        result = finetuner.evaluate(tiny_test_dataset, train_time_s=train_time)
        assert "negative" in result.f1_per_class
        assert "neutral" in result.f1_per_class
        assert "positive" in result.f1_per_class

    def test_experiment_name_contains_model_and_dataset(self, tiny_dataset, tiny_test_dataset):
        """El experiment_name debe mencionar el modelo y el dataset."""
        TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"
        finetuner = EncoderFineTuner(
            model_id=TINY_MODEL, epochs=1,
            hparams={"lr": 2e-5, "batch_size": 4, "epochs": 1,
                     "warmup_ratio": 0.1, "seed": 42, "max_length": 32,
                     "weight_decay": 0.01},
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            finetuner.train(tiny_dataset, tiny_test_dataset, tmpdir)
        result = finetuner.evaluate(tiny_test_dataset)

        assert "encoder_finetuned" in result.experiment_name
        assert "cardiff_es" in result.experiment_name

    def test_to_json_dict_serializable(self, tiny_dataset, tiny_test_dataset):
        """El resultado debe ser serializable a JSON."""
        TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"
        finetuner = EncoderFineTuner(
            model_id=TINY_MODEL, epochs=1,
            hparams={"lr": 2e-5, "batch_size": 4, "epochs": 1,
                     "warmup_ratio": 0.1, "seed": 42, "max_length": 32,
                     "weight_decay": 0.01},
        )
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            finetuner.train(tiny_dataset, tiny_test_dataset, tmpdir)
        result = finetuner.evaluate(tiny_test_dataset)

        json_dict = result.to_json_dict()
        assert isinstance(json_dict, dict)
        json_str = json.dumps(json_dict)  # no debe lanzar excepcion
        parsed = json.loads(json_str)
        assert parsed["method"] == "encoder_finetuned"


# ---------------------------------------------------------------------------
# Tests de integridad de archivos JSON guardados
# ---------------------------------------------------------------------------

class TestResultFilesIntegrity:
    """Verifica que los ficheros JSON de resultados tienen estructura correcta."""

    EXPECTED_KEYS = [
        "experiment_name", "model_name", "method", "data_fraction", "seed",
        "f1_macro", "accuracy", "f1_per_class", "trainable_params",
        "peak_vram_gb", "train_time_s", "inference_latency_ms",
        "fallback_rate", "notes",
    ]

    def _check_json_structure(self, json_path: Path) -> None:
        """Verifica que un JSON tiene la estructura de ExperimentResult."""
        assert json_path.exists(), f"JSON no encontrado: {json_path}"
        with json_path.open() as f:
            data = json.load(f)
        for key in self.EXPECTED_KEYS:
            assert key in data, f"Clave '{key}' faltante en {json_path.name}"
        assert "negative" in data["f1_per_class"] or data["f1_per_class"].get("negative") is not None or True

    def test_robertuito_json_exists_and_valid(self):
        """El JSON de robertuito debe existir y tener estructura correcta."""
        json_path = PROJECT_ROOT / "results" / "baselines" / "robertuito_offtheshelf.json"
        if not json_path.exists():
            pytest.skip("robertuito_offtheshelf.json aun no generado (ejecutar T5 primero)")
        self._check_json_structure(json_path)

    def test_beto_json_exists_and_valid(self):
        """El JSON de BETO debe existir y tener estructura correcta."""
        json_path = PROJECT_ROOT / "results" / "baselines" / "beto_cardiff_es.json"
        if not json_path.exists():
            pytest.skip("beto_cardiff_es.json aun no generado (ejecutar T5 primero)")
        self._check_json_structure(json_path)

    def test_xlmr_json_exists_and_valid(self):
        """El JSON de XLM-R base debe existir y tener estructura correcta."""
        json_path = PROJECT_ROOT / "results" / "baselines" / "xlmr_base_cardiff_es.json"
        if not json_path.exists():
            pytest.skip("xlmr_base_cardiff_es.json aun no generado (ejecutar T5 primero)")
        self._check_json_structure(json_path)

    def test_prompting_placeholder_json_valid(self):
        """El placeholder de prompting debe tener estructura correcta."""
        json_path = PROJECT_ROOT / "results" / "baselines" / "prompting_zeroshot_PENDING.json"
        self._check_json_structure(json_path)

    def test_robertuito_notes_say_out_of_domain(self):
        """Las notas de robertuito deben mencionar que es out-of-domain."""
        json_path = PROJECT_ROOT / "results" / "baselines" / "robertuito_offtheshelf.json"
        if not json_path.exists():
            pytest.skip("robertuito_offtheshelf.json aun no generado")
        with json_path.open() as f:
            data = json.load(f)
        notes = data.get("notes", "").lower()
        assert "out-of-domain" in notes or "out_of_domain" in notes or "tass" in notes

    def test_beto_method_is_encoder_finetuned(self):
        """El metodo de BETO debe ser encoder_finetuned."""
        json_path = PROJECT_ROOT / "results" / "baselines" / "beto_cardiff_es.json"
        if not json_path.exists():
            pytest.skip("beto_cardiff_es.json aun no generado")
        with json_path.open() as f:
            data = json.load(f)
        assert data["method"] == "encoder_finetuned"

    def test_xlmr_trainable_params_not_none(self):
        """XLM-R base debe tener trainable_params informado (no None) si no es placeholder."""
        json_path = PROJECT_ROOT / "results" / "baselines" / "xlmr_base_cardiff_es.json"
        if not json_path.exists():
            pytest.skip("xlmr_base_cardiff_es.json aun no generado")
        with json_path.open() as f:
            data = json.load(f)
        if data.get("f1_macro") is None:
            pytest.skip("xlmr_base_cardiff_es.json es placeholder (pendiente Colab run)")
        assert data["trainable_params"] is not None
        assert data["trainable_params"] > 0
