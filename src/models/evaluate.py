"""
evaluate.py — Harness de evaluación unificado para AI Lab (Sprint 2, T2).

Recibe predicciones + etiquetas (y opcionalmente métricas de coste) y
devuelve un dict estandarizado. Funciona tanto en CPU como en GPU.

Todos los experimentos del proyecto (TF-IDF, encoders, LLMs con LoRA/QLoRA,
prompting) usan este mismo harness para que los resultados sean comparables.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Nombres de clase canónicos (en orden de ID)
LABEL_NAMES = ("negative", "neutral", "positive")

# Columnas del CSV maestro de resultados
_CSV_COLUMNS = [
    "experiment_name",
    "model_name",
    "method",
    "data_fraction",
    "seed",
    "f1_macro",
    "accuracy",
    "f1_negative",
    "f1_neutral",
    "f1_positive",
    "trainable_params",
    "peak_vram_gb",
    "train_time_s",
    "inference_latency_ms",
    "fallback_rate",
    "notes",
]


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Contenedor de resultados de un experimento.

    Campos de calidad
    -----------------
    f1_macro : float
        F1-macro sobre el conjunto de test.
    accuracy : float
        Accuracy (exactitud) sobre test.
    f1_per_class : dict[str, float]
        F1 por clase: {"negative": …, "neutral": …, "positive": …}.

    Campos de coste (None si no aplica)
    ------------------------------------
    trainable_params : int | None
        Número de parámetros entrenables del modelo.
    peak_vram_gb : float | None
        VRAM pico en GB durante el entrenamiento (solo GPU).
    train_time_s : float | None
        Tiempo de entrenamiento en segundos.
    inference_latency_ms : float | None
        Latencia media por ejemplo en ms sobre el conjunto de test.

    Extras
    ------
    fallback_rate : float | None
        Porcentaje de salidas que caen a etiqueta fallback (solo prompting).
    notes : str
        Notas libres para qa-validator o memoir-writer.
    """

    experiment_name: str
    model_name: str
    method: str  # "tfidf_lr" | "prompting_zeroshot" | "lora" | "qlora" | "full_ft" | …
    data_fraction: int | str  # 50, 100, 250, 500, 1000, "full"
    seed: int
    # Métricas de calidad
    f1_macro: float = 0.0
    accuracy: float = 0.0
    f1_per_class: dict[str, float] = field(
        default_factory=lambda: {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    )
    # Métricas de coste
    trainable_params: Optional[int] = None
    peak_vram_gb: Optional[float] = None
    train_time_s: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    # Extras
    fallback_rate: Optional[float] = None
    notes: str = ""

    def to_flat_dict(self) -> dict:
        """Serializa el resultado como diccionario plano (una fila de CSV)."""
        d = asdict(self)
        # Aplanar f1_per_class
        fpc = d.pop("f1_per_class", {})
        for label in LABEL_NAMES:
            d[f"f1_{label}"] = fpc.get(label)
        return d

    def to_json_dict(self) -> dict:
        """Serializa el resultado completo como dict (para JSON individual)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str] = LABEL_NAMES,
) -> dict:
    """Calcula métricas de calidad estándar.

    Parámetros
    ----------
    y_true :
        Etiquetas reales (ints).
    y_pred :
        Predicciones del modelo (ints).
    label_names :
        Nombres de clase en orden de ID (índice 0 = primera clase).

    Retorna
    -------
    dict con claves: "f1_macro", "accuracy", "f1_per_class".
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    # F1 por clase
    f1_each = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    f1_per_class = {
        label: float(f1_each[i])
        for i, label in enumerate(label_names)
        if i < len(f1_each)
    }

    return {
        "f1_macro": f1_macro,
        "accuracy": acc,
        "f1_per_class": f1_per_class,
    }


# ---------------------------------------------------------------------------
# measure_inference_latency
# ---------------------------------------------------------------------------

def measure_inference_latency(
    predict_fn: Callable[[list[str]], list],
    examples: Sequence[str],
    n_warmup: int = 5,
) -> float:
    """Mide la latencia media de inferencia en milisegundos.

    Parámetros
    ----------
    predict_fn :
        Función que recibe una lista de textos y devuelve predicciones.
        Se llama ejemplo a ejemplo para medir latencia individual.
    examples :
        Lista de textos de entrada (se descartarán los primeros n_warmup).
    n_warmup :
        Número de ejemplos de calentamiento (no se miden).

    Retorna
    -------
    Latencia media en ms por ejemplo.  0.0 si no hay ejemplos tras el warmup.
    """
    # Warmup
    for text in examples[:n_warmup]:
        predict_fn([text])

    eval_examples = list(examples[n_warmup:])
    if not eval_examples:
        logger.warning("No quedan ejemplos tras el warmup; latencia = 0.0 ms.")
        return 0.0

    times: list[float] = []
    for text in eval_examples:
        t0 = time.perf_counter()
        predict_fn([text])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1_000)  # ms

    latency_ms = float(np.mean(times))
    logger.debug(
        "Latencia: %.2f ms/ejemplo (N=%d, warmup=%d)", latency_ms, len(times), n_warmup
    )
    return latency_ms


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def CostTracker():
    """Context manager que mide tiempo de entrenamiento.

    Uso::

        with CostTracker() as ct:
            train(...)
        result.train_time_s = ct.elapsed

    El atributo ``elapsed`` (float, segundos) está disponible tras salir
    del bloque ``with``.
    """

    class _Tracker:
        elapsed: float = 0.0

    tracker = _Tracker()
    t0 = time.perf_counter()
    try:
        yield tracker
    finally:
        tracker.elapsed = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Medición de VRAM (solo GPU)
# ---------------------------------------------------------------------------

def reset_peak_vram() -> None:
    """Reinicia el contador de VRAM pico si hay GPU disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def get_peak_vram_gb() -> Optional[float]:
    """Retorna VRAM pico en GB, o None si no hay GPU disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 1e9
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# save_result
# ---------------------------------------------------------------------------

def save_result(
    result: ExperimentResult,
    results_dir: str | Path = "results/",
) -> Path:
    """Persiste un resultado en el CSV maestro y en JSON individual.

    Parámetros
    ----------
    result :
        Resultado del experimento.
    results_dir :
        Directorio raíz de resultados.

    Retorna
    -------
    Path al fichero JSON individual guardado.
    """
    results_path = Path(results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    # ---- CSV maestro --------------------------------------------------------
    csv_path = results_path / "all_results.csv"
    flat = result.to_flat_dict()

    # Asegurar que todas las columnas están presentes
    row = {col: flat.get(col) for col in _CSV_COLUMNS}

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Resultado añadido al CSV maestro: %s", csv_path)

    # ---- JSON individual ----------------------------------------------------
    json_path = results_path / f"{result.experiment_name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_json_dict(), f, ensure_ascii=False, indent=2)

    logger.info("JSON individual guardado: %s", json_path)
    return json_path


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------

def load_results(results_dir: str | Path = "results/") -> pd.DataFrame:
    """Carga el CSV maestro de resultados como DataFrame.

    Retorna un DataFrame vacío con las columnas correctas si el fichero
    no existe todavía.
    """
    csv_path = Path(results_dir).resolve() / "all_results.csv"
    if not csv_path.exists():
        logger.warning("No se encontró %s. Retornando DataFrame vacío.", csv_path)
        return pd.DataFrame(columns=_CSV_COLUMNS)
    df = pd.read_csv(csv_path)
    logger.info("Cargados %d resultados desde %s", len(df), csv_path)
    return df
