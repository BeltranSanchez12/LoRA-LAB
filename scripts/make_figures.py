"""
make_figures.py — Genera las Figuras A y B del paper a partir de los resúmenes
del grid del Sprint 3.

Figura A — F1-macro vs. tamaño de datos (corte A):
    Curvas LoRA {Qwen3-1.7B, Qwen3-4B} con barras de error (media±std sobre 3
    semillas), eje x logarítmico (n=4…full). Líneas de referencia N-independientes:
    prompting zero-shot y mejor few-shot por modelo, y encoders fine-tuned a datos
    completos. El interés está en el CRUCE: a qué n el fine-tuning iguala/supera al
    prompting (por debajo de n≈10).

Figura B — Calidad vs. coste a datos completos (corte B):
    F1-macro frente a parámetros entrenables (escala log) para lora/qlora/full_ft
    y encoders; prompting (0 params entrenables) como línea de referencia.
    Frontera de Pareto resaltada.

Salidas: results/figures/figA_f1_vs_n.{png,pdf} y figB_quality_cost.{png,pdf}
Además (Sprint 7) se generan variantes para el paper SIN título incrustado
(la numeración la pone el pie IEEE): figA_f1_vs_n_paper.{png,pdf} y
figB_quality_cost_paper.{png,pdf}. Idénticas a las originales salvo el título.

Uso:
    python scripts/aggregate_results.py && python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.aggregate_results import frac_to_n, short_model, TRAIN_FULL_N  # noqa: E402

FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MODEL_COLOR = {"Qwen3-1.7B": "#1f77b4", "Qwen3-4B": "#d62728"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def enc_friendly(model_name: str) -> str:
    """Nombre corto y legible de los encoders fine-tuned para las etiquetas."""
    n = str(model_name).lower()
    if "bert-base-spanish" in n or "beto" in n:
        return "BETO"
    if "xlm-roberta" in n:
        return "XLM-R"
    return str(model_name).split("/")[-1]


def _load_csv() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "all_results.csv")
    df = df.drop_duplicates(subset=["experiment_name", "seed"], keep="last").reset_index(drop=True)
    df["model_short"] = df["model_name"].map(short_model)
    df["n_train"] = df["data_fraction"].map(frac_to_n)
    return df


# ---------------------------------------------------------------------------
# Figura A
# ---------------------------------------------------------------------------

def figure_a(df: pd.DataFrame, paper: bool = False) -> None:
    """Con paper=True genera la variante *_paper sin título incrustado."""
    summ = pd.read_csv(ROOT / "results" / "summary_corteA.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for model in ["Qwen3-1.7B", "Qwen3-4B"]:
        sub = summ[summ["model_short"] == model].sort_values("n_train")
        if sub.empty:
            continue
        c = MODEL_COLOR[model]
        ax.errorbar(sub["n_train"], sub["f1_mean"], yerr=sub["f1_std"],
                    marker="o", ms=5, lw=1.8, capsize=3, color=c,
                    label=f"LoRA {model} (media±std, 3 semillas)")
        ax.fill_between(sub["n_train"], sub["f1_mean"] - sub["f1_std"],
                        sub["f1_mean"] + sub["f1_std"], color=c, alpha=0.12)

    # Referencias de prompting (N-independientes), por modelo
    prompt = df[df["method"].str.startswith("prompting", na=False)]
    for model in ["Qwen3-1.7B", "Qwen3-4B"]:
        pm = prompt[prompt["model_short"] == model]
        if pm.empty:
            continue
        c = MODEL_COLOR[model]
        zs = pm[pm["method"] == "prompting_zeroshot"]["f1_macro"]
        fs = pm[pm["method"].str.startswith("prompting_fewshot")]["f1_macro"]
        if not zs.empty:
            ax.axhline(zs.iloc[0], ls=":", lw=1.3, color=c, alpha=0.8,
                       label=f"{model} 0-shot ({zs.iloc[0]:.3f})")
        if not fs.empty:
            ax.axhline(fs.max(), ls="--", lw=1.3, color=c, alpha=0.8,
                       label=f"{model} mejor few-shot ({fs.max():.3f})")

    # Encoders fine-tuned (topline supervisado a datos completos).
    # Etiqueta INLINE en el extremo derecho de cada línea, por encima de la
    # leyenda (lower right) y de las curvas -> BETO y XLM-R legibles.
    enc = df[df["method"] == "encoder_finetuned"]
    x_max = max(summ["n_train"]) if not summ.empty else TRAIN_FULL_N
    for _, r in enc.iterrows():
        f1 = r["f1_macro"]
        ax.axhline(f1, ls="-.", lw=1.0, color="gray", alpha=0.6)
        ax.text(x_max, f1 + 0.0015,
                f"{enc_friendly(r['model_name'])} (FT completo): {f1:.3f}",
                fontsize=8, color="dimgray", va="bottom", ha="right")

    ax.set_xscale("log")
    ax.set_xlabel("Ejemplos de entrenamiento (n, escala log)")
    ax.set_ylabel("F1-macro (test completo, n=870)")
    if not paper:  # la variante _paper va sin título (numeración en el pie IEEE)
        ax.set_title("Figura A — Fine-tuning LoRA vs. prompting según tamaño de datos")
    # Ticks legibles en las fracciones reales
    xticks = sorted(summ["n_train"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels(["full" if x == TRAIN_FULL_N else str(int(x)) for x in xticks],
                       rotation=45, fontsize=8)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9, ncol=1)
    fig.tight_layout()
    stem = "figA_f1_vs_n_paper" if paper else "figA_f1_vs_n"
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"Figura A{' (paper)' if paper else ''} -> {FIGDIR / (stem + '.png')}")


# ---------------------------------------------------------------------------
# Figura B
# ---------------------------------------------------------------------------

def _pareto_front(points):
    """Devuelve índices Pareto-óptimos: menor coste (x) y mayor F1 (y)."""
    idx = sorted(range(len(points)), key=lambda i: points[i][0])
    front, best_y = [], -np.inf
    for i in idx:
        if points[i][1] > best_y:
            front.append(i)
            best_y = points[i][1]
    return front


def figure_b(df: pd.DataFrame, paper: bool = False) -> None:
    """Con paper=True genera la variante *_paper sin título incrustado."""
    full = df[df["n_train"] == TRAIN_FULL_N].copy()
    trained = full[full["method"].isin(["lora", "qlora", "full_ft", "encoder_finetuned"])].copy()
    trained = trained[trained["trainable_params"] > 0]

    marker = {"lora": "o", "qlora": "s", "full_ft": "^", "encoder_finetuned": "D"}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # LoRA a datos completos: colapsa las 3 semillas a UN marcador media±std por
    # modelo (en vez de 3 puntos solapados). El resto (qlora/full_ft/encoders)
    # es 1 semilla -> punto único.
    pts = []
    plotted = []  # (method, model) ya dibujados para no duplicar
    lora = trained[trained["method"] == "lora"]
    for model, g in lora.groupby("model_short"):
        x = float(g["trainable_params"].median())
        ymean = float(g["f1_macro"].mean())
        ystd = float(g["f1_macro"].std(ddof=1)) if len(g) > 1 else 0.0
        color = MODEL_COLOR.get(model, "#555555")
        ax.errorbar(x, ymean, yerr=ystd, marker="o", ms=10, color=color,
                    mec="black", mew=0.6, capsize=4, lw=0, elinewidth=1.3, zorder=3)
        ax.annotate(f"LoRA {model}\n{ymean:.3f}±{ystd:.3f} (n=3)",
                    (x, ymean), fontsize=7, xytext=(8, 5), textcoords="offset points")
        pts.append((x, ymean))
        plotted.append(("lora", model))

    for _, r in trained[trained["method"] != "lora"].iterrows():
        m, model = r["method"], r["model_short"]
        color = MODEL_COLOR.get(model, "#555555")
        ax.scatter(r["trainable_params"], r["f1_macro"], s=95,
                   marker=marker.get(m, "o"), color=color, edgecolor="black", lw=0.6, zorder=3)
        if m == "encoder_finetuned":
            txt, ha, off = enc_friendly(r["model_name"]), "right", (-9, 4)
        elif m == "qlora":
            # comparte x con LoRA del mismo modelo -> etiqueta DEBAJO para no solapar
            txt, ha, off = f"QLoRA {model}", "left", (8, -14)
        else:
            txt, ha, off = f"full-FT {model}", "left", (8, 4)
        ax.annotate(txt, (r["trainable_params"], r["f1_macro"]),
                    fontsize=7, ha=ha, xytext=off, textcoords="offset points")
        pts.append((r["trainable_params"], r["f1_macro"]))

    # Frontera de Pareto (menor coste, mayor calidad)
    if pts:
        front = _pareto_front(pts)
        fx = [pts[i][0] for i in front]
        fy = [pts[i][1] for i in front]
        ax.plot(fx, fy, ls="--", lw=1.3, color="green", alpha=0.7, zorder=2, label="Frontera de Pareto")

    # Prompting: 0 params entrenables -> línea de referencia
    prompt = df[df["method"].str.startswith("prompting", na=False)]
    if not prompt.empty:
        best = prompt.loc[prompt["f1_macro"].idxmax()]
        ax.axhline(best["f1_macro"], ls=":", color="purple", alpha=0.8,
                   label=f"Mejor prompting ({best['model_short']}, {best['f1_macro']:.3f}, 0 params)")

    # Marcadores fantasma para la leyenda de métodos.
    for m, lab in [("lora", "LoRA (media±std, 3 semillas)"), ("qlora", "QLoRA"),
                   ("full_ft", "Full FT"), ("encoder_finetuned", "Encoder FT")]:
        ax.scatter([], [], marker=marker[m], color="#888888", edgecolor="black",
                   lw=0.6, s=80, label=lab)

    ax.set_xscale("log")
    ax.set_xlabel("Parámetros entrenables (escala log)")
    ax.set_ylabel("F1-macro (test completo, n=870)")
    if not paper:  # la variante _paper va sin título (numeración en el pie IEEE)
        ax.set_title("Figura B — Calidad vs. coste de adaptación (datos completos)")
    ax.margins(x=0.18)  # aire para las etiquetas de los encoders a la derecha
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9, ncol=1)
    fig.tight_layout()
    stem = "figB_quality_cost_paper" if paper else "figB_quality_cost"
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"Figura B{' (paper)' if paper else ''} -> {FIGDIR / (stem + '.png')}")


def main() -> None:
    df = _load_csv()
    figure_a(df)
    figure_a(df, paper=True)
    figure_b(df)
    figure_b(df, paper=True)


if __name__ == "__main__":
    main()
