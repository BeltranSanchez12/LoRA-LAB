"""
build_tass_subsets.py — Construye los splits procesados InterTASS por país (Corte C).

Parsea ES/CR/PE (train + development), descarta NONE, mapea P/N/NEU a las 3 clases
del proyecto y guarda cada país en data/processed/tass_{cc}/ con el mismo layout que
Cardiff ES (train / validation / test / train_fractions/n_full / metadata.json).

Imprime, POR PAÍS Y SPLIT:
  - conteo bruto de polaridades (incluido NONE) y cuántos NONE/empty se descartan,
  - nº de ejemplos tras el descarte y balance de clases (negative/neutral/positive).

Uso:
    python scripts/build_tass_subsets.py
    python scripts/build_tass_subsets.py --countries ES CR PE --val_size 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_tass import COUNTRIES, build_country  # noqa: E402
from src.data.load_data import ID2LABEL  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_tass_subsets")


def _fmt_dist(dist: dict[str, int]) -> str:
    """Distribución {id->count} a 'negative=.. neutral=.. positive=..'."""
    parts = []
    for cid in (0, 1, 2):
        parts.append(f"{ID2LABEL[cid]}={dist.get(str(cid), 0)}")
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=list(COUNTRIES))
    ap.add_argument("--raw_dir", default="data/raw/tass")
    ap.add_argument("--out_root", default="data/processed")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    print("=" * 78)
    print("Corte C — InterTASS 2018: conteos por país y split (NONE descartado)")
    print("Evaluación sobre DEVELOPMENT (gold de test = 404). val = hold-out del train.")
    print("=" * 78)

    grand = {}
    for cc in args.countries:
        meta = build_country(
            cc,
            raw_dir=str(PROJECT_ROOT / args.raw_dir),
            out_root=str(PROJECT_ROOT / args.out_root),
            val_size=args.val_size,
            seed=args.seed,
        )
        grand[cc] = meta
        raw = meta["raw"]
        sp = meta["splits"]
        print(f"\n### {cc}")
        print(f"  TRAIN-tagged  brutos: {raw['train_raw_counts']}  "
              f"(NONE descartados={raw['train_none_discarded']}, vacíos={raw['train_empty_skipped']}, "
              f"duplicados eliminados={raw['train_duplicates_dropped']})")
        print(f"  DEV-tagged    brutos: {raw['dev_raw_counts']}  "
              f"(NONE descartados={raw['dev_none_discarded']}, vacíos={raw['dev_empty_skipped']}, "
              f"duplicados eliminados={raw['dev_duplicates_dropped']})")
        print(f"  train (post-NONE, menos val) : n={sp['train']['n_examples']:4d}  | {_fmt_dist(sp['train']['class_distribution'])}")
        print(f"  validation (hold-out train)  : n={sp['validation']['n_examples']:4d}  | {_fmt_dist(sp['validation']['class_distribution'])}")
        print(f"  test (= development)         : n={sp['test']['n_examples']:4d}  | {_fmt_dist(sp['test']['class_distribution'])}")

    # Tabla-resumen compacta
    print("\n" + "=" * 78)
    print(f"{'país':5s} {'train':>7s} {'val':>5s} {'test/dev':>9s}   balance test/dev (neg/neu/pos)")
    print("-" * 78)
    for cc, meta in grand.items():
        sp = meta["splits"]
        td = sp["test"]["class_distribution"]
        print(f"{cc:5s} {sp['train']['n_examples']:7d} {sp['validation']['n_examples']:5d} "
              f"{sp['test']['n_examples']:9d}   {td.get('0',0)}/{td.get('1',0)}/{td.get('2',0)}")
    print("=" * 78)
    print("\nSplits guardados en data/processed/tass_{es,cr,pe}/  (layout Cardiff-compatible).")


if __name__ == "__main__":
    main()
