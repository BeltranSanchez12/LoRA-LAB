"""
load_tass.py — Pipeline de datos InterTASS 2018 para el Corte C (Sprint 4).

Transferencia dialectal: variedades del español (ES, CR, PE) en formato XML de
InterTASS 2018. Cada país trae dos ficheros ETIQUETADOS:
    intertass-{PAIS}-train-tagged.xml         -> split de ENTRENAMIENTO
    intertass-{PAIS}-development-tagged.xml    -> split de EVALUACIÓN (dev)

Por qué dev y no test
---------------------
El gold del split de test de InterTASS 2018 se distribuye en un fichero .qrel
que hoy devuelve 404 (inaccesible). Por eso entrenamos sobre TRAIN y evaluamos
sobre DEVELOPMENT de cada país. Queda documentado en el diario y en el paper.

Mapeo a 3 clases (idéntico al esquema del estudio principal, Cardiff ES)
------------------------------------------------------------------------
    P    -> positive
    N    -> negative
    NEU  -> neutral
    NONE -> DESCARTADO

NONE = ausencia de sentimiento, fuera del esquema de polaridad. Fundirlo con NEU
contaminaría la clase neutral (NEU = sentimiento mixto/neutro presente, distinto
de "sin sentimiento"), así que se descarta. El resto queda en el mismo espacio
de etiquetas {negative:0, neutral:1, positive:2} que usa todo el proyecto.

El resultado se guarda en el MISMO layout que Cardiff ES para reutilizar sin
cambios el harness de fine-tuning/eval del Sprint 3:
    data/processed/tass_{cc}/
        train                      (train del país, menos el hold-out de val)
        validation                 (hold-out estratificado del train; selección de checkpoint)
        test                       (= development del país; objetivo de evaluación)
        train_fractions/n_full     (= train reducido; lo que el harness carga como "full")
        metadata.json
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from sklearn.model_selection import train_test_split

from src.data.load_data import LABEL2ID, _class_distribution

logger = logging.getLogger(__name__)

# InterTASS polaridad -> etiqueta canónica del proyecto. NONE se descarta (no está aquí).
POLARITY_MAP: dict[str, str] = {
    "P": "positive",
    "N": "negative",
    "NEU": "neutral",
}

COUNTRIES = ("ES", "CR", "PE")


# ---------------------------------------------------------------------------
# Parser XML
# ---------------------------------------------------------------------------

def parse_tass_xml(path: str | Path) -> dict:
    """Parsea un fichero InterTASS *-tagged.xml a ejemplos (texto, etiqueta).

    Deduplica por texto exacto DENTRO del fichero (mantiene la primera aparición).
    InterTASS-PE-train trae 1 tweet nativo repetido 3 veces; sin deduplicar, el
    hold-out de validación podría recibir una copia idéntica a una de train (micro-
    fuga en la selección de checkpoint). Se registra cuántas copias se eliminan.

    Retorna
    -------
    dict con:
        "examples": list[{"text": str, "label": int}]  (NONE descartado, sin duplicados)
        "raw_counts": {"P","N","NEU","NONE", ...}  conteo bruto de polaridades
        "n_discarded_none": int
        "n_skipped_empty": int       (tweets sin contenido textual)
        "n_duplicates_dropped": int  (copias exactas eliminadas tras el primer visto)
    """
    path = Path(path)
    tree = ET.parse(str(path))
    root = tree.getroot()

    examples: list[dict] = []
    raw_counts: dict[str, int] = {}
    n_none = 0
    n_empty = 0
    n_dup = 0
    seen_texts: set[str] = set()

    for tweet in root.findall(".//tweet"):
        content_el = tweet.find("content")
        text = (content_el.text or "").strip() if content_el is not None else ""

        value_el = tweet.find("./sentiment/polarity/value")
        polarity = (value_el.text or "").strip() if value_el is not None else ""

        raw_counts[polarity] = raw_counts.get(polarity, 0) + 1

        if not text:
            n_empty += 1
            continue
        if polarity == "NONE":
            n_none += 1
            continue
        if polarity not in POLARITY_MAP:
            # Polaridad inesperada: la registramos y la saltamos (no debería ocurrir).
            logger.warning("Polaridad no reconocida '%s' en %s; tweet omitido.", polarity, path.name)
            continue
        if text in seen_texts:
            n_dup += 1
            continue
        seen_texts.add(text)

        examples.append({"text": text, "label": LABEL2ID[POLARITY_MAP[polarity]]})

    return {
        "examples": examples,
        "raw_counts": raw_counts,
        "n_discarded_none": n_none,
        "n_skipped_empty": n_empty,
        "n_duplicates_dropped": n_dup,
    }


def _examples_to_dataset(examples: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "text": [e["text"] for e in examples],
        "label": [e["label"] for e in examples],
    })


# ---------------------------------------------------------------------------
# Construcción de splits por país
# ---------------------------------------------------------------------------

def build_country(
    country: str,
    raw_dir: str | Path = "data/raw/tass",
    out_root: str | Path = "data/processed",
    val_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """Construye y guarda los splits procesados de un país InterTASS.

    - train (tagged)      -> se divide ESTRATIFICADAMENTE en train' + validation
                             (la validation se usa SOLO para selección de checkpoint,
                             igual que Cardiff; nunca se evalúa sobre ella).
    - development (tagged) -> test (objetivo de evaluación; el gold de test real es 404).

    Retorna el dict de metadata (también se persiste como metadata.json).
    """
    cc = country.upper()
    raw_dir = Path(raw_dir)
    train_xml = raw_dir / f"intertass-{cc}-train-tagged.xml"
    dev_xml = raw_dir / f"intertass-{cc}-development-tagged.xml"
    for p in (train_xml, dev_xml):
        if not p.exists():
            raise FileNotFoundError(f"No existe el fichero InterTASS esperado: {p}")

    train_parsed = parse_tass_xml(train_xml)
    dev_parsed = parse_tass_xml(dev_xml)

    train_ex = train_parsed["examples"]
    dev_ex = dev_parsed["examples"]

    # Hold-out estratificado de validation desde el train (selección de checkpoint).
    labels = [e["label"] for e in train_ex]
    tr_idx, val_idx = train_test_split(
        list(range(len(train_ex))),
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )
    train_sub = [train_ex[i] for i in tr_idx]
    val_sub = [train_ex[i] for i in val_idx]

    ds_train = _examples_to_dataset(train_sub)
    ds_val = _examples_to_dataset(val_sub)
    ds_test = _examples_to_dataset(dev_ex)  # development = objetivo de evaluación

    out_dir = Path(out_root) / f"tass_{cc.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_train.save_to_disk(str(out_dir / "train"))
    ds_val.save_to_disk(str(out_dir / "validation"))
    ds_test.save_to_disk(str(out_dir / "test"))
    # El harness carga la fracción "full" desde train_fractions/n_full.
    (out_dir / "train_fractions").mkdir(parents=True, exist_ok=True)
    ds_train.save_to_disk(str(out_dir / "train_fractions" / "n_full"))

    metadata = {
        "country": cc,
        "source": "InterTASS 2018",
        "eval_split_note": (
            "Se evalúa sobre DEVELOPMENT (guardado como 'test'): el gold del split "
            "de test de InterTASS 2018 se distribuye en un .qrel inaccesible (404)."
        ),
        "label2id": LABEL2ID,
        "polarity_map": POLARITY_MAP,
        "none_discarded": True,
        "val_size": val_size,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw": {
            "train_raw_counts": train_parsed["raw_counts"],
            "train_none_discarded": train_parsed["n_discarded_none"],
            "train_empty_skipped": train_parsed["n_skipped_empty"],
            "train_duplicates_dropped": train_parsed["n_duplicates_dropped"],
            "dev_raw_counts": dev_parsed["raw_counts"],
            "dev_none_discarded": dev_parsed["n_discarded_none"],
            "dev_empty_skipped": dev_parsed["n_skipped_empty"],
            "dev_duplicates_dropped": dev_parsed["n_duplicates_dropped"],
        },
        "splits": {
            "train": {
                "n_examples": len(ds_train),
                "class_distribution": _class_distribution(ds_train, "label"),
            },
            "validation": {
                "n_examples": len(ds_val),
                "class_distribution": _class_distribution(ds_val, "label"),
            },
            "test": {
                "n_examples": len(ds_test),
                "class_distribution": _class_distribution(ds_test, "label"),
            },
        },
        "train_plus_val_after_none": len(train_ex),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("[%s] train=%d val=%d test(dev)=%d (guardado en %s)",
                cc, len(ds_train), len(ds_val), len(ds_test), out_dir)
    return metadata
