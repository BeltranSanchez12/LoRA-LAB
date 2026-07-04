"""
encoder_baseline.py — Baselines encoder para clasificacion de sentimiento (Sprint 2, T5).

Dos tipos de baseline bien diferenciados:

A) Baselines CONTROLADOS (fine-tuning propio sobre Cardiff ES, mismos splits):
   - dccuchile/bert-base-spanish-wwm-cased (BETO)
   - xlm-roberta-base (XLM-R base)
   Estos son los puntos "encoder" de la frontera de Pareto.

B) Referencia OFF-THE-SHELF (evaluacion directa sin adaptar, out-of-domain):
   - pysentimiento/robertuito-sentiment-analysis (TASS 2020)
   Etiquetado explicitamente como "out-of-domain reference". NO en la Pareto.

NO usar: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
(distribution leakage confirmado, ver docs/research/xlmt_leakage_check.md).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES = ("negative", "neutral", "positive")

# Hiperparametros fijos (no optimizados, para comparacion justa)
DEFAULT_HPARAMS = {
    "lr": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "seed": 42,
    "max_length": 128,
    "weight_decay": 0.01,
}


# ---------------------------------------------------------------------------
# EncoderFineTuner
# ---------------------------------------------------------------------------

class EncoderFineTuner:
    """Fine-tuning de encoder (BERT/RoBERTa) para clasificacion de 3 clases.

    - Usa AutoModelForSequenceClassification con num_labels=3.
    - Entrenamiento con transformers Trainer.
    - Hiperparametros fijos (no optimizados, para comparacion justa):
        lr=2e-5, batch_size=16, epochs=3, warmup_ratio=0.1, seed=42.
    - Cuenta params entrenables (todos, en full fine-tuning).
    - Mide train_time_s con CostTracker.
    - Mide peak_vram_gb (None en CPU).
    - Mide inference_latency_ms sobre el test set.

    Parametros
    ----------
    model_id : str
        Identificador HuggingFace del encoder base.
    hparams : dict, opcional
        Hiperparametros. Si es None se usan los DEFAULT_HPARAMS.
    epochs : int, opcional
        Sobreescribe hparams["epochs"]. Util para reducir a 1 en CPU.
    """

    def __init__(
        self,
        model_id: str,
        hparams: dict | None = None,
        epochs: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.hparams = {**DEFAULT_HPARAMS, **(hparams or {})}
        if epochs is not None:
            self.hparams["epochs"] = epochs

        self._model = None
        self._tokenizer = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _count_trainable_params(self) -> int:
        """Cuenta parametros entrenables del modelo."""
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def _tokenize_dataset(self, dataset, max_length: int = 128):
        """Tokeniza un dataset de HuggingFace."""
        def tokenize_fn(examples):
            return self._tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        return dataset.map(tokenize_fn, batched=True, desc="Tokenizando")

    # ------------------------------------------------------------------
    # Conversion a PyTorch Dataset puro (workaround numpy 2.x / datasets bug)
    # ------------------------------------------------------------------

    def _hf_to_torch_dataset(self, hf_dataset, max_length: int):
        """Convierte un HuggingFace Dataset a un PyTorch Dataset puro.

        Workaround para el bug de incompatibilidad entre numpy>=2.0 y
        datasets<=2.20: set_format('torch') falla con ValueError en
        np.array(array, copy=False). Solucion: tokenizar al vuelo con
        list comprehension y almacenar como tensores nativos.
        """
        import torch
        from torch.utils.data import Dataset as TorchDataset

        texts = list(hf_dataset["text"])
        labels = list(hf_dataset["label"])

        # Tokenizar en batch (rapido)
        encoded = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        class _TorchDataset(TorchDataset):
            def __init__(self, enc, lbl):
                self.input_ids = enc["input_ids"]
                self.attention_mask = enc["attention_mask"]
                # token_type_ids es opcional (RoBERTa no lo usa)
                self.token_type_ids = enc.get("token_type_ids", None)
                self.labels = torch.tensor(lbl, dtype=torch.long)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {
                    "input_ids": self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels": self.labels[idx],
                }
                if self.token_type_ids is not None:
                    item["token_type_ids"] = self.token_type_ids[idx]
                return item

        return _TorchDataset(encoded, labels)

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset,
        val_dataset,
        output_dir: str | Path = "models/encoder_finetuned",
    ) -> float:
        """Entrena el encoder sobre el dataset de Cardiff ES.

        Parametros
        ----------
        train_dataset :
            Dataset HuggingFace con columnas "text" y "label" (int).
        val_dataset :
            Dataset de validacion para early-stopping / eval durante entrenamiento.
        output_dir :
            Directorio donde se guardan los checkpoints.

        Retorna
        -------
        Tiempo de entrenamiento en segundos.
        """
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed as hf_set_seed,
        )

        hf_set_seed(self.hparams["seed"])

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Cargando tokenizador y modelo: %s (epochs=%d, lr=%.0e, bs=%d)",
            self.model_id,
            self.hparams["epochs"],
            self.hparams["lr"],
            self.hparams["batch_size"],
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        # Convertir a PyTorch Dataset puro para evitar bug numpy 2.x / datasets
        max_length = self.hparams["max_length"]
        logger.info("Tokenizando train (%d ejemplos)...", len(train_dataset))
        train_torch = self._hf_to_torch_dataset(train_dataset, max_length)
        logger.info("Tokenizando val (%d ejemplos)...", len(val_dataset))
        val_torch = self._hf_to_torch_dataset(val_dataset, max_length)

        # Definir funcion de compute_metrics para el Trainer
        from sklearn.metrics import f1_score, accuracy_score

        def compute_metrics_trainer(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            acc = accuracy_score(labels, preds)
            return {"f1_macro": f1, "accuracy": acc}

        # Detectar dispositivo
        has_gpu = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.hparams["epochs"],
            per_device_train_batch_size=self.hparams["batch_size"],
            per_device_eval_batch_size=self.hparams["batch_size"],
            learning_rate=self.hparams["lr"],
            warmup_ratio=self.hparams["warmup_ratio"],
            weight_decay=self.hparams["weight_decay"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=self.hparams["seed"],
            report_to="none",  # Sin wandb/tensorboard
            logging_steps=50,
            fp16=has_gpu,  # Solo en GPU
            use_cpu=not has_gpu,
            dataloader_num_workers=0,  # Evitar problemas en WSL/CPU
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_torch,
            eval_dataset=val_torch,
            compute_metrics=compute_metrics_trainer,
        )

        logger.info("Iniciando entrenamiento de %s...", self.model_id)
        t0 = time.perf_counter()
        trainer.train()
        elapsed = time.perf_counter() - t0

        self._is_trained = True
        logger.info(
            "Entrenamiento completado en %.1fs. Params entrenables: %d",
            elapsed,
            self._count_trainable_params(),
        )
        return elapsed

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(self, save_dir: str | Path) -> Path:
        """Persiste el modelo entrenado (mejor época, ya cargado por
        load_best_model_at_end) y su tokenizador en ``save_dir``.

        Necesario para la re-inferencia de robustez (Sprint 6) sin reentrenar.
        """
        if not self._is_trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a train() primero.")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(save_dir))
        self._tokenizer.save_pretrained(str(save_dir))
        logger.info("Encoder guardado en %s", save_dir)
        return save_dir

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_dataset,
        train_time_s: float | None = None,
        peak_vram_gb: float | None = None,
    ):
        """Evalua el modelo entrenado sobre el test set.

        Parametros
        ----------
        test_dataset :
            Dataset HuggingFace con columnas "text" y "label" (int).
        train_time_s :
            Tiempo de entrenamiento (se pasa desde afuera si se midio con CostTracker).
        peak_vram_gb :
            VRAM pico durante entrenamiento (None en CPU).

        Retorna
        -------
        ExperimentResult con todas las metricas.
        """
        if not self._is_trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a train() primero.")

        import torch
        from src.models.evaluate import (
            ExperimentResult,
            compute_metrics,
            LABEL_NAMES as EVAL_LABEL_NAMES,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()

        test_texts = list(test_dataset["text"])
        y_true = list(test_dataset["label"])

        # Predecir en batches
        batch_size = self.hparams["batch_size"]
        y_pred: list[int] = []
        latencies: list[float] = []

        with torch.no_grad():
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i: i + batch_size]
                t0 = time.perf_counter()
                inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.hparams["max_length"],
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = self._model(**inputs).logits
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                elapsed_batch = time.perf_counter() - t0

                y_pred.extend(preds)
                # Latencia por ejemplo en ms
                latencies.extend(
                    [(elapsed_batch / len(batch_texts)) * 1000] * len(batch_texts)
                )

        metrics = compute_metrics(y_true, y_pred, label_names=EVAL_LABEL_NAMES)
        latency_ms = float(np.mean(latencies))

        # Nombre limpio del modelo para el experimento
        model_short = self.model_id.replace("/", "_")
        experiment_name = f"encoder_finetuned_{model_short}_cardiff_es"

        result = ExperimentResult(
            experiment_name=experiment_name,
            model_name=self.model_id,
            method="encoder_finetuned",
            data_fraction="full",
            seed=self.hparams["seed"],
            f1_macro=metrics["f1_macro"],
            accuracy=metrics["accuracy"],
            f1_per_class=metrics["f1_per_class"],
            trainable_params=self._count_trainable_params(),
            peak_vram_gb=peak_vram_gb,
            train_time_s=train_time_s,
            inference_latency_ms=latency_ms,
            fallback_rate=None,
            notes=(
                f"Full fine-tuning de {self.model_id} sobre Cardiff ES (train completo). "
                f"epochs={self.hparams['epochs']}, lr={self.hparams['lr']:.0e}, "
                f"bs={self.hparams['batch_size']}. "
                f"Punto A de la frontera de Pareto calidad-coste."
            ),
        )

        logger.info(
            "Evaluacion completada. F1-macro=%.4f, Accuracy=%.4f, Latencia=%.2fms",
            result.f1_macro, result.accuracy, result.inference_latency_ms,
        )
        return result


# ---------------------------------------------------------------------------
# OffTheShelfEncoder
# ---------------------------------------------------------------------------

class OffTheShelfEncoder:
    """Evaluacion directa de un clasificador pre-entrenado sin fine-tuning.

    Para pysentimiento: usa la pipeline de la libreria.
    Registra method="encoder_offtheshelf" y anota que es out-of-domain.

    IMPORTANTE: Este resultado se reporta por separado, NO forma parte de
    la frontera de Pareto (entrenado en TASS 2020, no en Cardiff ES).

    Parametros
    ----------
    model_id : str
        Identificador del modelo. Por defecto: "pysentimiento/robertuito-sentiment-analysis".
    use_pysentimiento : bool
        Si True, usa la API de pysentimiento. Si False, usa pipeline de transformers
        directamente (para otros modelos HuggingFace).
    """

    # Mapa de etiquetas de pysentimiento a las nuestras
    _PYSENTIMIENTO_LABEL_MAP = {
        "POS": "positive",
        "NEG": "negative",
        "NEU": "neutral",
        # Variantes por si acaso
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }

    def __init__(
        self,
        model_id: str = "pysentimiento/robertuito-sentiment-analysis",
        use_pysentimiento: bool = True,
    ) -> None:
        self.model_id = model_id
        self.use_pysentimiento = use_pysentimiento
        self._analyzer = None

    def _load_analyzer(self):
        """Carga el analizador (lazy loading)."""
        if self._analyzer is not None:
            return

        if self.use_pysentimiento:
            try:
                from pysentimiento import create_analyzer
                logger.info("Cargando pysentimiento analyzer: %s (lang='es')", self.model_id)
                self._analyzer = create_analyzer(task="sentiment", lang="es")
                logger.info("Analizador pysentimiento cargado.")
            except ImportError as e:
                raise ImportError(
                    "pysentimiento no instalado. Ejecuta: pip install pysentimiento>=0.7.0"
                ) from e
        else:
            from transformers import pipeline
            logger.info("Cargando pipeline HuggingFace: %s", self.model_id)
            self._analyzer = pipeline(
                "text-classification",
                model=self.model_id,
                device=-1,  # CPU
            )

    def _predict_pysentimiento(self, texts: list[str]) -> tuple[list[int], float]:
        """Predice usando pysentimiento. Retorna (y_pred_ids, latency_ms)."""
        y_pred: list[int] = []
        latencies: list[float] = []

        for text in texts:
            t0 = time.perf_counter()
            result = self._analyzer.predict(text)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # El output de pysentimiento es un objeto con .output
            raw_label = result.output
            label = self._PYSENTIMIENTO_LABEL_MAP.get(raw_label, "neutral")
            y_pred.append(LABEL2ID[label])
            latencies.append(elapsed_ms)

        return y_pred, float(np.mean(latencies))

    def _predict_hf_pipeline(self, texts: list[str]) -> tuple[list[int], float]:
        """Predice usando pipeline de HuggingFace."""
        y_pred: list[int] = []
        latencies: list[float] = []

        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            t0 = time.perf_counter()
            results = self._analyzer(batch, truncation=True, max_length=512)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            for res in results:
                raw_label = res["label"].upper()
                label = self._PYSENTIMIENTO_LABEL_MAP.get(raw_label, "neutral")
                y_pred.append(LABEL2ID[label])
            latencies.extend([elapsed_ms / len(batch)] * len(batch))

        return y_pred, float(np.mean(latencies))

    def evaluate(self, test_dataset):
        """Evalua el clasificador pre-entrenado sobre el test set.

        Parametros
        ----------
        test_dataset :
            Dataset HuggingFace con columnas "text" y "label" (int).

        Retorna
        -------
        ExperimentResult con todas las metricas.
        """
        from src.models.evaluate import (
            ExperimentResult,
            compute_metrics,
            LABEL_NAMES as EVAL_LABEL_NAMES,
        )

        self._load_analyzer()

        test_texts = list(test_dataset["text"])
        y_true = list(test_dataset["label"])

        logger.info(
            "Evaluando %s (off-the-shelf) sobre %d ejemplos...",
            self.model_id, len(test_texts),
        )

        if self.use_pysentimiento:
            y_pred, latency_ms = self._predict_pysentimiento(test_texts)
        else:
            y_pred, latency_ms = self._predict_hf_pipeline(test_texts)

        metrics = compute_metrics(y_true, y_pred, label_names=EVAL_LABEL_NAMES)

        model_short = self.model_id.replace("/", "_")
        experiment_name = f"encoder_offtheshelf_{model_short}"

        result = ExperimentResult(
            experiment_name=experiment_name,
            model_name=self.model_id,
            method="encoder_offtheshelf",
            data_fraction="full",
            seed=42,
            f1_macro=metrics["f1_macro"],
            accuracy=metrics["accuracy"],
            f1_per_class=metrics["f1_per_class"],
            trainable_params=None,  # no fine-tuning
            peak_vram_gb=None,
            train_time_s=None,
            inference_latency_ms=latency_ms,
            fallback_rate=None,
            notes=(
                f"out-of-domain reference: {self.model_id} entrenado en TASS 2020, "
                f"evaluado sobre Cardiff ES test sin fine-tuning. "
                f"NO incluir en la frontera de Pareto calidad-coste. "
                f"Solo referencia de comparacion."
            ),
        )

        logger.info(
            "Evaluacion off-the-shelf completada. F1-macro=%.4f, Accuracy=%.4f",
            result.f1_macro, result.accuracy,
        )
        return result


# ---------------------------------------------------------------------------
# Script de ejecucion directa (para uso en CPU o Colab)
# ---------------------------------------------------------------------------

def run_encoder_baselines(
    project_root: Path,
    seed: int = 42,
    epochs: int = 3,
    run_beto: bool = True,
    run_xlmr: bool = True,
    run_robertuito: bool = True,
    max_train_examples: int | None = None,
) -> dict:
    """Ejecuta todos los baselines encoder y guarda los resultados.

    Parametros
    ----------
    project_root :
        Raiz del proyecto.
    seed :
        Semilla global.
    epochs :
        Numero de epocas de fine-tuning (reducir a 1 en CPU si hay timeout).
    run_beto, run_xlmr, run_robertuito :
        Flags para elegir que modelos ejecutar.
    max_train_examples :
        Limitar train a N ejemplos (para pruebas rapidas).

    Retorna
    -------
    dict con los ExperimentResult de cada modelo ejecutado.
    """
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from datasets import load_from_disk
    from src.models.evaluate import (
        save_result,
        get_peak_vram_gb,
        reset_peak_vram,
    )
    import json

    data_dir = project_root / "data" / "processed" / "cardiff_es"
    train_ds = load_from_disk(str(data_dir / "train"))
    val_ds = load_from_disk(str(data_dir / "validation"))
    test_ds = load_from_disk(str(data_dir / "test"))

    if max_train_examples is not None:
        train_ds = train_ds.select(range(min(max_train_examples, len(train_ds))))
        logger.info("Train limitado a %d ejemplos.", len(train_ds))

    results_dir = project_root / "results"
    baselines_dir = results_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}

    # ---- A) Baselines controlados -------------------------------------------

    for model_id, flag, json_name in [
        ("dccuchile/bert-base-spanish-wwm-cased", run_beto, "beto_cardiff_es.json"),
        ("xlm-roberta-base", run_xlmr, "xlmr_base_cardiff_es.json"),
    ]:
        if not flag:
            continue

        model_short = model_id.split("/")[-1]
        logger.info("=" * 60)
        logger.info("Fine-tuning: %s", model_id)

        finetuner = EncoderFineTuner(model_id=model_id, epochs=epochs)

        reset_peak_vram()
        try:
            train_time_s = finetuner.train(
                train_dataset=train_ds,
                val_dataset=val_ds,
                output_dir=str(project_root / "models" / f"finetuned_{model_short}"),
            )
            peak_vram = get_peak_vram_gb()

            result = finetuner.evaluate(
                test_ds,
                train_time_s=train_time_s,
                peak_vram_gb=peak_vram,
            )

            save_result(result, results_dir=str(results_dir))
            json_path = baselines_dir / json_name
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result.to_json_dict(), f, ensure_ascii=False, indent=2)
            logger.info("Guardado: %s (F1-macro=%.4f)", json_path, result.f1_macro)
            all_results[model_id] = result

        except Exception as exc:
            logger.error("Error en %s: %s", model_id, exc, exc_info=True)
            # Guardar placeholder de error
            placeholder = {
                "experiment_name": f"encoder_finetuned_{model_id.replace('/', '_')}_cardiff_es",
                "model_name": model_id,
                "method": "encoder_finetuned",
                "data_fraction": "full",
                "seed": seed,
                "f1_macro": None,
                "accuracy": None,
                "f1_per_class": {"negative": None, "neutral": None, "positive": None},
                "trainable_params": None,
                "peak_vram_gb": None,
                "train_time_s": None,
                "inference_latency_ms": None,
                "fallback_rate": None,
                "notes": f"ERROR en ejecucion: {str(exc)[:200]}. Re-ejecutar en Colab con GPU.",
            }
            json_path = baselines_dir / json_name
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(placeholder, f, ensure_ascii=False, indent=2)
            logger.warning("Placeholder de error guardado: %s", json_path)

    # ---- B) Referencia off-the-shelf ----------------------------------------

    if run_robertuito:
        logger.info("=" * 60)
        logger.info("Off-the-shelf: pysentimiento/robertuito-sentiment-analysis")

        try:
            encoder = OffTheShelfEncoder()
            result = encoder.evaluate(test_ds)

            save_result(result, results_dir=str(results_dir))
            json_path = baselines_dir / "robertuito_offtheshelf.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result.to_json_dict(), f, ensure_ascii=False, indent=2)
            logger.info("Guardado: %s (F1-macro=%.4f)", json_path, result.f1_macro)
            all_results["robertuito"] = result

        except Exception as exc:
            logger.error("Error en robertuito: %s", exc, exc_info=True)

    return all_results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    import sys
    project_root = Path(__file__).resolve().parents[2]

    print("=" * 60)
    print("T5 — Baselines encoder")
    print("=" * 60)

    # En CPU reducir epochs a 1 para evitar timeout
    import torch
    has_gpu = torch.cuda.is_available()
    epochs = 3 if has_gpu else 1
    if not has_gpu:
        print(f"ADVERTENCIA: Sin GPU. Usando epochs={epochs}. Re-ejecutar con 3 epocas en Colab.")

    results = run_encoder_baselines(
        project_root=project_root,
        seed=42,
        epochs=epochs,
    )

    print("\nResultados:")
    for model_id, res in results.items():
        print(f"  {model_id}: F1-macro={res.f1_macro:.4f}, Acc={res.accuracy:.4f}")

    print("\nT5 completado.")


if __name__ == "__main__":
    main()
