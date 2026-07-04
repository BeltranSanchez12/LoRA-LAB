"""
aggregate_cutc.py — Agrega la matriz de transferencia dialectal (Cut C, Sprint 4)
y dibuja el heatmap de publicación.

Lee results/sprint4/*.json (una celda = cutc_{model}_{src}2{tgt}_s{seed}), agrega
F1-macro como MEDIA±STD sobre las semillas {42,43,44}, y produce, POR MODELO:
  - results/sprint4/transfer_matrix_{model}.csv   (mean, std, n_seeds por celda)
  - una matriz 3×3 (fila=fuente/train, col=objetivo/eval=dev del país).

Heatmap (results/figures/figC_transfer.{png,pdf}): un panel por modelo, color = F1
media, anotación = "mean\n±std", diagonal (techo en-dominio) recuadrada. Fuera de
la diagonal = transferencia dialectal. Lectura POR COLUMNA: comparar cada celda con
la diagonal de su columna (mismo objetivo) da la penalización por transferencia.

Además (Sprint 7) se genera la variante para el paper SIN suptitle incrustado
(la numeración la pone el pie IEEE): figC_transfer_paper.{png,pdf}. Idéntica a la
original salvo el título.

Uso:
    python scripts/aggregate_cutc.py                # CSVs + figC + figC_paper
    python scripts/aggregate_cutc.py --paper-only   # SOLO figC_paper (no toca CSVs
                                                    # ni la figura original)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESDIR = ROOT / "results" / "sprint4"
FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["es", "cr", "pe"]
CC_LABEL = {"es": "ES", "cr": "CR", "pe": "PE"}
MODEL_TAGS = ["qwen1.7b", "qwen4b"]
MODEL_TITLE = {"qwen1.7b": "Qwen3-1.7B", "qwen4b": "Qwen3-4B"}

NAME_RE = re.compile(r"cutc_(?P<tag>qwen[\d.]+b)_(?P<src>es|cr|pe)2(?P<tgt>es|cr|pe)_s(?P<seed>\d+)")

plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight"})


def load_cells() -> pd.DataFrame:
    """Carga todas las celdas del grid desde los JSON individuales."""
    rows = []
    for jf in sorted(RESDIR.glob("cutc_*.json")):
        m = NAME_RE.match(jf.stem)
        if not m:
            continue
        d = json.loads(jf.read_text())
        rows.append({
            "tag": m["tag"], "src": m["src"], "tgt": m["tgt"], "seed": int(m["seed"]),
            "f1_macro": d["f1_macro"], "accuracy": d["accuracy"],
            "f1_neutral": d["f1_per_class"].get("neutral"),
        })
    if not rows:
        sys.exit(f"No hay celdas cutc_*.json en {RESDIR}. ¿Ha corrido el grid?")
    return pd.DataFrame(rows)


def matrices(df: pd.DataFrame, tag: str):
    """Devuelve (mean, std, n) como DataFrames 3×3 (fila=src, col=tgt)."""
    sub = df[df.tag == tag]
    mean = pd.DataFrame(index=COUNTRIES, columns=COUNTRIES, dtype=float)
    std = pd.DataFrame(index=COUNTRIES, columns=COUNTRIES, dtype=float)
    n = pd.DataFrame(index=COUNTRIES, columns=COUNTRIES, dtype=int)
    for src in COUNTRIES:
        for tgt in COUNTRIES:
            cell = sub[(sub.src == src) & (sub.tgt == tgt)]["f1_macro"]
            mean.loc[src, tgt] = cell.mean() if len(cell) else np.nan
            std.loc[src, tgt] = cell.std(ddof=0) if len(cell) else np.nan
            n.loc[src, tgt] = len(cell)
    return mean, std, n


def transfer_gap(mean: pd.DataFrame) -> pd.DataFrame:
    """Penalización por columna: diagonal(objetivo) - celda. >0 = peor que en-dominio."""
    gap = mean.copy()
    for tgt in COUNTRIES:
        diag = mean.loc[tgt, tgt]
        for src in COUNTRIES:
            gap.loc[src, tgt] = diag - mean.loc[src, tgt]
    return gap


def save_csv(tag, mean, std, n):
    out = RESDIR / f"transfer_matrix_{tag}.csv"
    rows = []
    for src in COUNTRIES:
        for tgt in COUNTRIES:
            rows.append({
                "model": MODEL_TITLE[tag], "train": CC_LABEL[src], "eval": CC_LABEL[tgt],
                "f1_macro_mean": round(float(mean.loc[src, tgt]), 4),
                "f1_macro_std": round(float(std.loc[src, tgt]), 4),
                "n_seeds": int(n.loc[src, tgt]), "diagonal": src == tgt,
            })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  matriz -> {out}")


def heatmap(ax, tag, mean, std, vmin, vmax):
    M = mean.values.astype(float)
    im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([CC_LABEL[c] for c in COUNTRIES])
    ax.set_yticklabels([CC_LABEL[c] for c in COUNTRIES])
    ax.set_xlabel("Evaluación (dev del país objetivo)")
    ax.set_ylabel("Entrenamiento (país fuente)")
    ax.set_title(MODEL_TITLE[tag])
    for i in range(3):
        for j in range(3):
            v = M[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.3f}\n±{std.values[i, j]:.3f}"
            # contraste de texto según luminancia de la celda (escala compartida)
            lum = (v - vmin) / (vmax - vmin + 1e-9)
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if lum < 0.55 else "black", fontsize=9)
            if i == j:  # diagonal = techo en-dominio
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="red", lw=2.5))
    return im


def build_figure(means, tags, vmin, vmax, with_title: bool = True):
    """Construye el heatmap multi-panel. with_title=False -> variante del paper
    sin suptitle incrustado (la numeración la pone el pie IEEE)."""
    fig, axes = plt.subplots(1, len(tags), figsize=(5.4 * len(tags), 4.8))
    if len(tags) == 1:
        axes = [axes]
    last_im = None
    for ax, tag in zip(axes, tags):
        mean, std, _n = means[tag]
        last_im = heatmap(ax, tag, mean, std, vmin, vmax)
    fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.04, label="F1-macro")
    if with_title:
        fig.suptitle("Cut C — Transferencia dialectal de adapters LoRA (InterTASS 2018)\n"
                     "diagonal (rojo) = techo en-dominio · fuera = transferencia", y=1.02)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-only", action="store_true",
                        help="Genera SOLO figC_transfer_paper.{png,pdf} (sin título "
                             "incrustado), sin reescribir los CSVs agregados ni la "
                             "figura original figC_transfer.")
    args = parser.parse_args()

    df = load_cells()
    print(f"Celdas cargadas: {len(df)}  (modelos={sorted(df.tag.unique())}, "
          f"semillas={sorted(df.seed.unique())})")
    tags = [t for t in MODEL_TAGS if t in df.tag.unique()]

    # Escala de color COMPARTIDA entre paneles para que los colores sean comparables
    # entre modelos y casen con la única barra de color.
    means = {t: matrices(df, t) for t in tags}
    all_vals = np.concatenate([m[0].values.astype(float).ravel() for m in means.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    if not args.paper_only:
        for tag in tags:
            mean, std, n = means[tag]
            save_csv(tag, mean, std, n)
            print(f"\n### {MODEL_TITLE[tag]} — F1-macro media (fila=train, col=eval)")
            print(mean.round(3).to_string())
            print("Penalización por transferencia (diag_objetivo - celda):")
            print(transfer_gap(mean).round(3).to_string())
            miss = int((n.values == 0).sum())
            if miss:
                print(f"  [aviso] {miss} celdas sin datos todavía (grid incompleto).")

        fig = build_figure(means, tags, vmin, vmax, with_title=True)
        for ext in ("png", "pdf"):
            out = FIGDIR / f"figC_transfer.{ext}"
            fig.savefig(out)
            print(f"figura -> {out}")
        plt.close(fig)

    # Variante del paper (Sprint 7): sin suptitle, misma figura en lo demás.
    fig = build_figure(means, tags, vmin, vmax, with_title=False)
    for ext in ("png", "pdf"):
        out = FIGDIR / f"figC_transfer_paper.{ext}"
        fig.savefig(out)
        print(f"figura (paper) -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
