"""
build_fraction_subsets.py — Construye subconjuntos estratificados POR SEMILLA
para el corte A del grid de 3 semillas (Sprint 3).

Motivación
----------
El estudio de F1 vs. tamaño de datos a fracciones pequeñas (n=4,7,10,16,25,…)
necesita barras de error que capturen la VARIANZA DE MUESTREO de los datos: a
n pequeño, *qué* ejemplos te tocan domina el ruido. Por eso cada semilla
(42/43/44) muestrea un subconjunto estratificado DISTINTO.

Layout en disco (reemplaza el layout plano `n_50/` del piloto de 1 semilla):
    data/processed/cardiff_es/train_fractions/n_{frac}/s{seed}
    data/processed/cardiff_es/train_fractions/n_full          (semilla-agnóstico)

`n_full` es idéntico para todas las semillas (es todo el train), así que no se
duplica por semilla. El loader (src/models/lora_finetune.py) resuelve la ruta
con la semilla del experimento.

Uso:
    python scripts/build_fraction_subsets.py
    python scripts/build_fraction_subsets.py --seeds 42 43 44 --fractions 4 7 10 16 25 50 100 250 500 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.load_data import make_fraction_subsets, _class_distribution  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_fraction_subsets")

DEFAULT_SEEDS = (42, 43, 44)
# Corte A: foco en fracciones pequeñas; añadidos n=4 y n=7 para capturar el cruce (<10).
DEFAULT_FRACTIONS = (4, 7, 10, 16, 25, 50, 100, 250, 500, 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/cardiff_es")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--fractions", type=int, nargs="+", default=list(DEFAULT_FRACTIONS))
    ap.add_argument("--label_column", default="label")
    args = ap.parse_args()

    processed = (PROJECT_ROOT / args.processed_dir).resolve()
    train = load_from_disk(str(processed / "train"))
    logger.info("Train cargado: %d ejemplos, dist=%s",
                len(train), _class_distribution(train, args.label_column))

    frac_root = processed / "train_fractions"
    frac_root.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": "stratified, resampled per seed (data-sampling variance for error bars)",
        "seeds": list(args.seeds),
        "fractions": list(args.fractions),
        "train_n": len(train),
        "per_seed": {},
    }

    for seed in args.seeds:
        subsets = make_fraction_subsets(
            train,
            fractions=tuple(args.fractions),
            seed=seed,
            label_column=args.label_column,
        )
        meta["per_seed"][str(seed)] = {}
        for frac in args.fractions:
            ds = subsets[frac]
            out = frac_root / f"n_{frac}" / f"s{seed}"
            out.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(out))
            dist = _class_distribution(ds, args.label_column)
            meta["per_seed"][str(seed)][str(frac)] = {"n": len(ds), "dist": dist}
            logger.info("seed=%d n=%d -> %d ej, dist=%s", seed, frac, len(ds), dist)

    # n_full (semilla-agnóstico): garantiza que existe a partir del train completo.
    full_dir = frac_root / "n_full"
    if not full_dir.exists():
        train.save_to_disk(str(full_dir))
        logger.info("n_full creado (%d ej).", len(train))
    meta["full"] = {"n": len(train), "dist": _class_distribution(train, args.label_column)}

    meta_path = frac_root / "SUBSETS_META.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info("Metadatos escritos en %s", meta_path)

    # Sanidad: distintas semillas -> distintos índices a n pequeño.
    s_a = load_from_disk(str(frac_root / "n_10" / f"s{args.seeds[0]}"))["text"]
    s_b = load_from_disk(str(frac_root / "n_10" / f"s{args.seeds[1]}"))["text"]
    overlap = len(set(s_a) & set(s_b))
    logger.info("Chequeo varianza de muestreo n=10: solapamiento s%d∩s%d = %d/%d textos",
                args.seeds[0], args.seeds[1], overlap, len(s_a))


if __name__ == "__main__":
    main()
