"""
src/data — Módulo de carga y preprocesado de datos.
"""

from src.data.load_data import (
    DataConfig,
    LABEL2ID,
    ID2LABEL,
    SPLIT_NAMES,
    DEFAULT_FRACTIONS,
    load_cardiff_es,
    make_fraction_subsets,
    save_processed,
)

__all__ = [
    "DataConfig",
    "LABEL2ID",
    "ID2LABEL",
    "SPLIT_NAMES",
    "DEFAULT_FRACTIONS",
    "load_cardiff_es",
    "make_fraction_subsets",
    "save_processed",
]
