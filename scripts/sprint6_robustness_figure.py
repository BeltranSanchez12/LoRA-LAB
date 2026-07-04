#!/usr/bin/env python
"""sprint6_robustness_figure.py — Figura de robustez del Sprint 6 (T3).

Lee results/robustness_results.csv (T2) y produce results/figures/figF_robustness.{png,pdf}:
  * Panel A: heatmap Δ Macro-F1 (perturbado − limpio) por perturbación (7 nucleares) × método
             (5 modelos), con una fila 'media (7 pert.)' para el ranking de robustez.
  * Panel B: heatmap del ángulo POR CLASE — Δ F1 medio (sobre las 7 perturbaciones) de
             negative/neutral/positive × método, para ver si el ruido hunde la neutral.
code_switching (opcional) NO entra en la figura; se reporta aparte en el análisis.

Cada celda sale directamente del CSV (cero cifras inventadas).

Uso: python scripts/sprint6_robustness_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "robustness_results.csv"
OUT = ROOT / "results" / "figures" / "figF_robustness"

CORE_ORDER = ["sin_tildes", "sin_emojis", "minusculas", "mayusculas",
              "alargamientos", "abrev_chat", "sin_puntuacion"]
PERT_LABEL = {
    "sin_tildes": "sin tildes", "sin_emojis": "sin emojis", "minusculas": "minúsculas",
    "mayusculas": "MAYÚSCULAS", "alargamientos": "alargamientos",
    "abrev_chat": "abrev. chat", "sin_puntuacion": "sin puntuación",
}
MODEL_ORDER = ["lora_1.7b", "lora_4b", "prompt_k4_4b", "beto", "xlmr"]
MODEL_LABEL = {"lora_1.7b": "LoRA 1.7B", "lora_4b": "LoRA 4B",
               "prompt_k4_4b": "Prompt k=4", "beto": "BETO", "xlmr": "XLM-R"}


def _annot(ax, M, fmt="{:+.3f}", size=8):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=size,
                    color="black")


def main():
    if not CSV.exists():
        sys.exit(f"No existe {CSV}. Corre antes scripts/sprint6_robustness.py (T2).")
    df = pd.read_csv(CSV)
    core = df[(~df["optional"].astype(bool)) & (df["perturbation"] != "limpio")]

    # Panel A: Δ macro-F1 por perturbación × modelo
    A = np.full((len(CORE_ORDER) + 1, len(MODEL_ORDER)), np.nan)
    for i, p in enumerate(CORE_ORDER):
        for j, m in enumerate(MODEL_ORDER):
            r = core[(core["perturbation"] == p) & (core["model"] == m)]
            if len(r):
                A[i, j] = float(r["delta_abs"].iloc[0])
    A[-1, :] = np.nanmean(A[:-1, :], axis=0)  # fila media

    # Panel B: Δ F1 medio por clase × modelo (sobre las 7 perturbaciones)
    classes = ["delta_f1_negative", "delta_f1_neutral", "delta_f1_positive"]
    class_lbl = ["negative", "neutral", "positive"]
    B = np.full((3, len(MODEL_ORDER)), np.nan)
    for ci, c in enumerate(classes):
        for j, m in enumerate(MODEL_ORDER):
            vals = core[core["model"] == m][c].astype(float)
            if len(vals):
                B[ci, j] = float(vals.mean())

    vmaxA = np.nanmax(np.abs(A))
    normA = TwoSlopeNorm(vmin=-vmaxA, vcenter=0.0, vmax=max(vmaxA, 1e-6))
    cmap = plt.get_cmap("RdYlGn")

    # Un solo panel limpio: heatmap Δ Macro-F1 (perturbación × modelo) + fila media.
    # El ángulo por-clase se reporta en results/sprint6/robustness_analysis.md (donde el
    # artefacto de la neutral bajo mayúsculas puede etiquetarse sin ambigüedad visual).
    fig, axA = plt.subplots(figsize=(8.6, 6.6))
    imA = axA.imshow(A, cmap=cmap, norm=normA, aspect="auto")
    axA.set_xticks(range(len(MODEL_ORDER)))
    axA.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    axA.set_yticks(range(len(CORE_ORDER) + 1))
    axA.set_yticklabels([PERT_LABEL[p] for p in CORE_ORDER] + ["media (7 pert.)"])
    axA.axhline(len(CORE_ORDER) - 0.5, color="black", lw=1.5)
    _annot(axA, A)
    fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.02, label="Δ Macro-F1")
    axA.set_title(
        "Robustez al ruido del español de redes (re-inferencia, pesos congelados)\n"
        "Δ Macro-F1 vs limpio (perturbado − limpio) — rojo = se degrada\n"
        "Test Cardiff ES (n=870, semilla 42). Fuente: results/robustness_results.csv",
        fontsize=10, pad=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    print(f"Figura guardada: {OUT}.png / {OUT}.pdf")

    # Ángulo por-clase (no se dibuja; va al análisis): Δ F1 medio por clase × modelo
    print("\nΔ F1 medio por clase (sobre 7 pert.), por modelo:")
    for ci, lbl in enumerate(class_lbl):
        print("  " + f"{lbl:<9} " + "  ".join(
            f"{MODEL_LABEL[MODEL_ORDER[j]]}={B[ci, j]:+.4f}" for j in range(len(MODEL_ORDER))))

    # Ranking de robustez (media Δ, menos negativo = más robusto) para el análisis
    print("\nRanking de robustez (media Δ Macro-F1 sobre 7 perturbaciones, ↑ mejor):")
    order = np.argsort(-A[-1, :])
    for j in order:
        print(f"  {MODEL_LABEL[MODEL_ORDER[j]]:<12} media Δ = {A[-1, j]:+.4f}")


if __name__ == "__main__":
    main()
