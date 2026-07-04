"""
load_data.py — Pipeline de datos para AI Lab (Sprint 2, T1).

Descarga, valida y procesa el dataset cardiffnlp/tweet_sentiment_multilingual
(config 'spanish') y genera subconjuntos estratificados para el corte A.

Diseñado para enchufar otros datasets sin reescribir: usa DataConfig.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de etiquetas
# ---------------------------------------------------------------------------
LABEL2ID: dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# Nombres canónicos de splits
SPLIT_NAMES = ("train", "validation", "test")

# Fracciones por defecto para el corte A
DEFAULT_FRACTIONS = (50, 100, 250, 500, 1000)


# ---------------------------------------------------------------------------
# DataConfig — interfaz extensible para nuevos datasets
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Configuración de un dataset de sentimiento.

    Pensada para poder enchufar TASS/InterTASS u otros conjuntos después
    sin modificar las funciones de pipeline.

    Campos
    ------
    dataset_name : str
        Identificador HuggingFace Hub (p.ej. "cardiffnlp/tweet_sentiment_multilingual").
    language : str
        Código ISO-639 del idioma (p.ej. "es").
    subset : str
        Config/subset del dataset (p.ej. "spanish").
    variety : str, opcional
        Variedad regional o dominio (p.ej. "mx", "es-twitter").  None si no aplica.
    text_column : str
        Nombre de la columna de texto.
    label_column : str
        Nombre de la columna de etiqueta.
    label2id : dict
        Mapa etiqueta string -> int.
    revision : str
        Revisión del repo del Hub de la que leer los ficheros Parquet.  Por
        defecto ``refs/convert/parquet``, la rama de export automático a Parquet
        que el Hub mantiene para todos los datasets.  Leer de ahí evita el script
        de carga original (eliminado en ``datasets`` 4.x junto con
        ``trust_remote_code``) manteniendo exactamente los mismos splits,
        etiquetas y ejemplos.
    """

    dataset_name: str = "cardiffnlp/tweet_sentiment_multilingual"
    language: str = "es"
    subset: str = "spanish"
    variety: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    label2id: dict = field(default_factory=lambda: dict(LABEL2ID))
    revision: str = "refs/convert/parquet"

    @property
    def id2label(self) -> dict[int, str]:
        return {v: k for k, v in self.label2id.items()}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _retry(fn, max_attempts: int = 3, wait_s: float = 5.0):
    """Ejecuta *fn* con reintentos ante fallos de red."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Intento %d/%d fallido: %s. Reintentando en %.0fs…",
                attempt,
                max_attempts,
                exc,
                wait_s,
            )
            if attempt < max_attempts:
                time.sleep(wait_s)
    raise RuntimeError(
        f"Todos los intentos de descarga fallaron. Último error: {last_exc}"
    ) from last_exc


def _label_to_int(example: dict, label_column: str, label2id: dict[str, int]) -> dict:
    """Convierte etiqueta string a int (si no lo es ya)."""
    val = example[label_column]
    if isinstance(val, str):
        example[label_column] = label2id[val]
    return example


def _class_distribution(dataset: Dataset, label_column: str) -> dict[str, int]:
    """Cuenta ejemplos por etiqueta (int) y los devuelve como {str(id): count}."""
    from collections import Counter
    counts = Counter(dataset[label_column])
    return {str(k): int(v) for k, v in sorted(counts.items())}


def _verify_no_overlap(splits: dict[str, Dataset], text_column: str) -> None:
    """Comprueba que no hay textos duplicados entre splits distintos."""
    sets: dict[str, set] = {
        name: set(ds[text_column]) for name, ds in splits.items()
    }
    for i, (n1, s1) in enumerate(sets.items()):
        for n2, s2 in list(sets.items())[i + 1 :]:
            overlap = s1 & s2
            if overlap:
                raise ValueError(
                    f"Solapamiento de {len(overlap)} textos entre '{n1}' y '{n2}'."
                )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_cardiff_es(
    cache_dir: str = "data/raw/cardiff",
    config: DataConfig | None = None,
) -> dict[str, Dataset]:
    """Descarga y cachea el dataset Cardiff ES.

    Parámetros
    ----------
    cache_dir :
        Directorio local de caché de HuggingFace Datasets.
    config :
        DataConfig con los parámetros del dataset.  Si es None se usa
        la configuración por defecto para Cardiff ES.

    Retorna
    -------
    dict con claves "train", "validation" y "test".

    Verifica
    --------
    - Las 3 clases (negative, neutral, positive) están presentes en cada split.
    - No hay solapamiento de textos entre splits.
    """
    if config is None:
        config = DataConfig()

    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    # Leemos los ficheros Parquet del export automático del Hub
    # (rama ``refs/convert/parquet``) en lugar del script de carga original.
    # `datasets` 4.x eliminó los scripts de carga y `trust_remote_code`, por lo
    # que `load_dataset(name, subset, trust_remote_code=True)` ya no funciona.
    # El export Parquet contiene EXACTAMENTE los mismos splits, el mismo esquema
    # de etiquetas (ClassLabel negative/neutral/positive -> 0/1/2) y los mismos
    # ejemplos, así que los baselines ya calculados siguen siendo válidos.
    base_uri = (
        f"hf://datasets/{config.dataset_name}@{config.revision}/{config.subset}"
    )
    data_files = {
        split_name: f"{base_uri}/{split_name}/*.parquet"
        for split_name in SPLIT_NAMES
    }

    logger.info(
        "Descargando %s / %s (Parquet @ %s) desde HuggingFace Hub (caché: %s)…",
        config.dataset_name,
        config.subset,
        config.revision,
        cache_path,
    )

    raw: DatasetDict = _retry(
        lambda: load_dataset(
            "parquet",
            data_files=data_files,
            cache_dir=str(cache_path),
        )
    )

    # Mapear a los splits canónicos (Cardiff usa "train", "validation", "test")
    splits: dict[str, Dataset] = {}
    for split_name in SPLIT_NAMES:
        if split_name not in raw:
            raise ValueError(
                f"Split '{split_name}' no encontrado. Disponibles: {list(raw.keys())}"
            )
        ds = raw[split_name]

        # Normalizar etiquetas a int si son strings
        if isinstance(ds[config.label_column][0], str):
            ds = ds.map(
                lambda ex: _label_to_int(ex, config.label_column, config.label2id),
                desc=f"Normalizar etiquetas [{split_name}]",
            )

        splits[split_name] = ds

    # Verificaciones
    _verify_labels(splits, config)
    _verify_no_overlap(splits, config.text_column)

    logger.info(
        "Dataset cargado correctamente. Tamaños: %s",
        {k: len(v) for k, v in splits.items()},
    )
    return splits


def _verify_labels(splits: dict[str, Dataset], config: DataConfig) -> None:
    """Comprueba que los 3 IDs de clase aparecen en cada split."""
    required_ids = set(config.label2id.values())
    for split_name, ds in splits.items():
        found_ids = set(ds[config.label_column])
        missing = required_ids - found_ids
        if missing:
            missing_names = {config.id2label[i] for i in missing}
            logger.warning(
                "Split '%s' no contiene clases: %s", split_name, missing_names
            )


def make_fraction_subsets(
    train_dataset: Dataset,
    fractions: tuple[int, ...] = DEFAULT_FRACTIONS,
    seed: int = 42,
    label_column: str = "label",
) -> dict[int | str, Dataset]:
    """Crea subconjuntos estratificados del split de entrenamiento.

    Parámetros
    ----------
    train_dataset :
        Dataset completo de entrenamiento.
    fractions :
        Tamaños absolutos a muestrear (p.ej. 50, 100, 250, …).
    seed :
        Semilla fija para reproducibilidad.
    label_column :
        Nombre de la columna de etiqueta (int).

    Retorna
    -------
    dict con claves int (tamaño muestral) y "full" (todo el train).
    """
    result: dict[int | str, Dataset] = {}

    # Subset "full"
    result["full"] = train_dataset

    n_total = len(train_dataset)
    labels = np.array(train_dataset[label_column])
    unique_classes = np.unique(labels)

    for n in fractions:
        if n >= n_total:
            logger.warning(
                "Fracción n=%d >= tamaño train (%d). Se usa el train completo.",
                n,
                n_total,
            )
            result[n] = train_dataset
            continue

        # Muestreo estratificado manualmente (proporcional a cada clase)
        rng = np.random.default_rng(seed)
        selected_indices: list[int] = []

        # Calcular cuántos ejemplos de cada clase
        class_counts = {c: int(np.sum(labels == c)) for c in unique_classes}
        n_per_class: dict[int, int] = {}
        remainder: dict[int, float] = {}
        assigned = 0

        for c in unique_classes:
            exact = n * class_counts[c] / n_total
            n_per_class[int(c)] = int(exact)
            remainder[int(c)] = exact - int(exact)
            assigned += int(exact)

        # Distribuir los ejemplos restantes a las clases con mayor fracción decimal
        leftover = n - assigned
        sorted_classes = sorted(
            remainder.keys(), key=lambda c: remainder[c], reverse=True
        )
        for i in range(leftover):
            n_per_class[sorted_classes[i % len(sorted_classes)]] += 1

        # Muestrear por clase
        for c in unique_classes:
            class_indices = np.where(labels == c)[0]
            k = min(n_per_class[int(c)], len(class_indices))
            sampled = rng.choice(class_indices, size=k, replace=False)
            selected_indices.extend(sampled.tolist())

        # Mezclar los índices seleccionados
        rng.shuffle(selected_indices)
        result[n] = train_dataset.select(selected_indices)

        logger.info(
            "Fracción n=%d: %d ejemplos seleccionados, distribución: %s",
            n,
            len(result[n]),
            _class_distribution(result[n], label_column),
        )

    return result


def save_processed(
    splits: dict[str, Dataset],
    fractions: dict[int | str, Dataset],
    out_dir: str = "data/processed/cardiff_es",
    config: DataConfig | None = None,
    seed: int = 42,
) -> Path:
    """Guarda splits y fracciones en disco (formato Arrow/Parquet).

    Genera también ``metadata.json`` con tamaños, distribución de clases,
    semilla y fecha.

    Retorna
    -------
    Path al directorio de salida.
    """
    if config is None:
        config = DataConfig()

    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "dataset_name": config.dataset_name,
        "language": config.language,
        "subset": config.subset,
        "variety": config.variety,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label2id": config.label2id,
        "splits": {},
        "fractions": {},
    }

    # Guardar splits principales
    for split_name, ds in splits.items():
        split_dir = out_path / split_name
        ds.save_to_disk(str(split_dir))
        metadata["splits"][split_name] = {
            "n_examples": len(ds),
            "class_distribution": _class_distribution(ds, config.label_column),
        }
        logger.info("Guardado split '%s' en %s (%d ejemplos)", split_name, split_dir, len(ds))

    # Guardar fracciones del train
    frac_dir = out_path / "train_fractions"
    frac_dir.mkdir(exist_ok=True)
    for key, ds in fractions.items():
        fname = f"n_{key}"
        ds.save_to_disk(str(frac_dir / fname))
        metadata["fractions"][str(key)] = {
            "n_examples": len(ds),
            "class_distribution": _class_distribution(ds, config.label_column),
        }
        logger.info("Guardada fracción '%s' (%d ejemplos)", key, len(ds))

    # Guardar metadata.json
    meta_path = out_path / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("metadata.json guardado en %s", meta_path)
    return out_path


# ---------------------------------------------------------------------------
# Script de ejecución directa
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    import sys
    # Añadir raíz del proyecto al path si se ejecuta directamente
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.utils.seed import set_seed
    set_seed(42)

    config = DataConfig()

    print("=" * 60)
    print("T1 — Pipeline de datos: Cardiff ES")
    print("=" * 60)

    # T1.1 Descargar
    splits = load_cardiff_es(
        cache_dir=str(project_root / "data" / "raw" / "cardiff"),
        config=config,
    )
    print("\nSplits cargados:")
    for name, ds in splits.items():
        dist = _class_distribution(ds, config.label_column)
        print(f"  {name:12s}: {len(ds):5d} ejemplos | distribución: {dist}")

    # T1.2 Subconjuntos estratificados
    fractions = make_fraction_subsets(
        splits["train"],
        fractions=DEFAULT_FRACTIONS,
        seed=42,
        label_column=config.label_column,
    )
    print("\nFracciones generadas:")
    for key, ds in fractions.items():
        dist = _class_distribution(ds, config.label_column)
        print(f"  n={str(key):6s}: {len(ds):5d} ejemplos | distribución: {dist}")

    # T1.3 Guardar
    out_path = save_processed(
        splits,
        fractions,
        out_dir=str(project_root / "data" / "processed" / "cardiff_es"),
        config=config,
        seed=42,
    )
    print(f"\nDatos guardados en: {out_path}")
    print(f"metadata.json: {out_path / 'metadata.json'}")
    print("T1 completado correctamente.")


if __name__ == "__main__":
    main()
