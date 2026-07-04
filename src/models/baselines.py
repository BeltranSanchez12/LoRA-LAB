"""
baselines.py — Baseline clásico TF-IDF + Logistic Regression (Sprint 2, T3).

Sirve como punto de partida de la curva calidad-coste. No requiere GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TfidfLRBaseline
# ---------------------------------------------------------------------------

class TfidfLRBaseline:
    """Baseline TF-IDF + Regresión Logística para clasificación de sentimiento.

    Configuración fija (reproducible)
    ----------------------------------
    - TF-IDF:  char n-grams 2-4 + word unigrams, sublinear_tf=True, max_features=200_000.
    - LR:      C=1.0, max_iter=1000, solver='lbfgs', multi_class='multinomial', seed=42.

    Métodos
    -------
    fit(train_texts, train_labels)
        Entrena el pipeline.
    predict(texts) -> list[int]
        Devuelve predicciones de clase (ints).
    count_trainable_params() -> int
        Nº de parámetros entrenables: nº features * nº clases.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._pipeline: Pipeline | None = None
        self._n_features: int = 0
        self._n_classes: int = 0

    def _build_pipeline(self) -> Pipeline:
        # Vectorizador que combina char n-grams (2-4) y word unigrams
        char_ngram = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            max_features=100_000,
            strip_accents=None,  # Mantener acentos del español
        )
        word_unigram = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            sublinear_tf=True,
            max_features=100_000,
            strip_accents=None,
        )

        # Pipeline con FeatureUnion manual via sparse hstack
        # Usamos un Vectorizer combinado directamente (más simple y eficiente)
        combined_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            max_features=200_000,
            strip_accents=None,
        )

        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=self.seed,
            n_jobs=-1,
        )

        pipeline = Pipeline(
            [
                ("tfidf", combined_vectorizer),
                ("lr", lr),
            ]
        )
        return pipeline

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
    ) -> "TfidfLRBaseline":
        """Entrena el pipeline TF-IDF + LR.

        Parámetros
        ----------
        train_texts :
            Lista de textos de entrenamiento.
        train_labels :
            Lista de etiquetas (ints).

        Retorna
        -------
        self (para encadenamiento).
        """
        logger.info(
            "Entrenando TF-IDF+LR con %d ejemplos…", len(train_texts)
        )
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(train_texts, train_labels)

        # Registrar dimensiones para count_trainable_params
        tfidf = self._pipeline.named_steps["tfidf"]
        lr = self._pipeline.named_steps["lr"]
        self._n_features = len(tfidf.vocabulary_)
        self._n_classes = len(lr.classes_)

        logger.info(
            "Entrenamiento completado. Features: %d, clases: %d, params: %d",
            self._n_features,
            self._n_classes,
            self.count_trainable_params(),
        )
        return self

    def predict(self, texts: Sequence[str]) -> list[int]:
        """Predice etiquetas (ints) para una lista de textos."""
        if self._pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        preds = self._pipeline.predict(texts)
        return [int(p) for p in preds]

    def count_trainable_params(self) -> int:
        """Número de parámetros entrenables: n_features * n_classes.

        La LR multinomial tiene una matriz de coeficientes de forma
        (n_classes, n_features) más un vector de sesgos (n_classes),
        pero por convención del proyecto contamos features * clases.
        """
        return self._n_features * self._n_classes


# ---------------------------------------------------------------------------
# Script de ejecución directa
# ---------------------------------------------------------------------------

def run_tfidf_lr_baseline(
    project_root: Path,
    seed: int = 42,
) -> float:
    """Ejecuta el baseline TF-IDF+LR sobre Cardiff ES y guarda el resultado.

    Retorna el F1-macro en test.
    """
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.utils.seed import set_seed
    from src.data.load_data import (
        DataConfig,
        load_cardiff_es,
        LABEL2ID,
        ID2LABEL,
    )
    from src.models.evaluate import (
        ExperimentResult,
        CostTracker,
        compute_metrics,
        measure_inference_latency,
        save_result,
        LABEL_NAMES,
    )

    set_seed(seed)

    config = DataConfig()

    # ---- Cargar datos -------------------------------------------------------
    splits = load_cardiff_es(
        cache_dir=str(project_root / "data" / "raw" / "cardiff"),
        config=config,
    )
    train_ds = splits["train"]
    test_ds = splits["test"]

    train_texts = train_ds["text"]
    train_labels = train_ds["label"]
    test_texts = test_ds["text"]
    test_labels = test_ds["label"]

    # ---- Entrenar -----------------------------------------------------------
    model = TfidfLRBaseline(seed=seed)

    from src.models.evaluate import reset_peak_vram, get_peak_vram_gb
    reset_peak_vram()

    with CostTracker() as ct:
        model.fit(train_texts, train_labels)

    train_time_s = ct.elapsed
    peak_vram_gb = get_peak_vram_gb()  # None en CPU

    # ---- Predecir en test ---------------------------------------------------
    preds = model.predict(test_texts)

    # ---- Métricas de calidad ------------------------------------------------
    metrics = compute_metrics(test_labels, preds, label_names=LABEL_NAMES)

    # ---- Latencia de inferencia (muestra de test) ---------------------------
    sample_texts = list(test_texts[:200])  # suficiente para estimar latencia

    def predict_fn(texts: list[str]) -> list[int]:
        return model.predict(texts)

    latency_ms = measure_inference_latency(predict_fn, sample_texts, n_warmup=5)

    # ---- Construir resultado ------------------------------------------------
    result = ExperimentResult(
        experiment_name="tfidf_lr_cardiff_es_full",
        model_name="tfidf+lr",
        method="tfidf_lr",
        data_fraction="full",
        seed=seed,
        f1_macro=metrics["f1_macro"],
        accuracy=metrics["accuracy"],
        f1_per_class=metrics["f1_per_class"],
        trainable_params=model.count_trainable_params(),
        peak_vram_gb=peak_vram_gb,
        train_time_s=train_time_s,
        inference_latency_ms=latency_ms,
        fallback_rate=None,
        notes=(
            f"TF-IDF char_wb 2-4grams + LR C=1.0 multinomial. "
            f"Features: {model._n_features}, clases: {model._n_classes}."
        ),
    )

    # ---- Guardar ------------------------------------------------------------
    baselines_dir = project_root / "results" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    # JSON individual en results/baselines/
    import json
    json_path = baselines_dir / "tfidf_lr.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_json_dict(), f, ensure_ascii=False, indent=2)
    logger.info("JSON del baseline guardado en: %s", json_path)

    # Añadir también al CSV maestro results/all_results.csv
    save_result(result, results_dir=str(project_root / "results"))

    return result.f1_macro


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    import sys
    project_root = Path(__file__).resolve().parents[2]

    print("=" * 60)
    print("T3 — Baseline TF-IDF + Logistic Regression")
    print("=" * 60)

    f1_macro = run_tfidf_lr_baseline(project_root, seed=42)

    print(f"\nF1-macro en test (Cardiff ES): {f1_macro:.4f}")
    print(f"Resultado guardado en: {project_root / 'results' / 'baselines' / 'tfidf_lr.json'}")
    print("T3 completado correctamente.")


if __name__ == "__main__":
    main()
