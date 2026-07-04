"""
src/models — Módulo de modelos, evaluación y baselines.
"""

from src.models.evaluate import (
    ExperimentResult,
    compute_metrics,
    measure_inference_latency,
    CostTracker,
    reset_peak_vram,
    get_peak_vram_gb,
    save_result,
    load_results,
    LABEL_NAMES,
)

from src.models.baselines import TfidfLRBaseline

__all__ = [
    "ExperimentResult",
    "compute_metrics",
    "measure_inference_latency",
    "CostTracker",
    "reset_peak_vram",
    "get_peak_vram_gb",
    "save_result",
    "load_results",
    "LABEL_NAMES",
    "TfidfLRBaseline",
]
