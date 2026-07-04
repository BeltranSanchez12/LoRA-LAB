"""
Figuras D y E (Sprint 5) — versiones definitivas para integrar.
D: F1 por clase (neg/neu/pos) — fuente f1_per_class de los JSON; LoRA = media 3 semillas.
E: Dispersión 4D — F1(y) vs VRAM(x), tamaño = parámetros (M), latencia anotada.
   Valores exactos de Tabla III (Corte B).
Salida: results/figures/figD_per_class.{png,pdf}, figE_quality_cost.{png,pdf}
Además (Sprint 7) variantes para el paper SIN título incrustado (la numeración
la pone el pie IEEE): figD_per_class_paper.{png,pdf}, figE_quality_cost_paper.{png,pdf}.
Idénticas a las originales salvo el título.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGDIR = RES / "figures"

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
C17, C4 = "#1f77b4", "#d62728"
C_BETO, C_XLMR, C_FFT = "#2ca02c", "#9467bd", "#ff7f0e"

def pc(path):
    return json.load(open(path))["f1_per_class"]

def mean_pc(paths):
    ds = [pc(p) for p in paths]
    return {k: float(np.mean([d[k] for d in ds])) for k in ("negative", "neutral", "positive")}

# ---------- Figura D ----------
lora4 = mean_pc([RES / f"lora_qwen4b_nfull_s{s}.json" for s in (42, 43, 44)])
lora17 = mean_pc([RES / f"lora_qwen1.7b_nfull_s{s}.json" for s in (42, 43, 44)])
beto = pc(RES / "encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es.json")
xlmr = pc(RES / "encoder_finetuned_xlm-roberta-base_cardiff_es.json")
systems = [("LoRA Qwen3-4B", lora4, C4), ("LoRA Qwen3-1.7B", lora17, C17),
           ("BETO (FT completo)", beto, C_BETO), ("XLM-R (FT completo)", xlmr, C_XLMR)]
classes = ["negative", "neutral", "positive"]
labels_es = ["Negativo", "Neutral", "Positivo"]

def figure_d(paper: bool = False) -> None:
    """Con paper=True genera la variante *_paper sin título incrustado."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(classes)); w = 0.2
    for i, (name, d, col) in enumerate(systems):
        vals = [d[c] for c in classes]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=col,
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.3)
    ax.axvspan(0.5, 1.5, color="#cccccc", alpha=0.20, zorder=0)
    ax.annotate("cuello de botella\n(clase neutral)", xy=(1, 0.500), xytext=(1.62, 0.50),
                fontsize=8.5, style="italic", color="#555555", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
    ax.set_xticks(x); ax.set_xticklabels(labels_es)
    ax.set_ylabel("F1 por clase (test Cardiff ES, n=870)")
    ax.set_ylim(0.45, 0.82); ax.set_xlabel("Clase de sentimiento")
    if not paper:  # la variante _paper va sin título (numeración en el pie IEEE)
        ax.set_title("Figura D — F1 por clase: la clase neutral es el cuello de botella")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              framealpha=0.9)
    fig.tight_layout()
    stem = "figD_per_class_paper" if paper else "figD_per_class"
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"Figura D{' (paper)' if paper else ''} -> {FIGDIR / (stem + '.png')}")

# ---------- Figura E: dispersión 4D ----------
#   (etiqueta, F1, params(M), VRAM(GB), lat(ms), color, marcador)  -- Tabla III exacta
pts = [
    ("LoRA 4B",    0.707, 11.80, 16.7,  68,  C4,    "o"),
    ("LoRA 1.7B",  0.698,  6.42,  8.72, 54,  C17,   "o"),
    ("QLoRA 4B",   0.701, 11.80,  6.79, 132, C4,    "s"),
    ("QLoRA 1.7B", 0.681,  6.42,  4.93, 102, C17,   "s"),
    ("Full-FT 1.7B",0.696,1720.0, 15.84,33,  C_FFT, "^"),
    ("BETO",       0.661, 110.0,  2.47, 0.6, C_BETO,"D"),
    ("XLM-R",      0.646, 278.0,  4.59, 0.5, C_XLMR,"D"),
]
def bub(p):  # tamaño del marcador proporcional a log(parámetros)
    return 70 + 90 * (np.log10(p) - np.log10(6.42))

def figure_e(paper: bool = False) -> None:
    """Con paper=True genera la variante *_paper sin título incrustado."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for lab, f1, par, vram, lat, col, mk in pts:
        ax.scatter(vram, f1, s=bub(par), marker=mk, color=col, edgecolor="black",
                   linewidth=0.7, alpha=0.85, zorder=3)
        dy = 0.004 if lab not in ("QLoRA 1.7B",) else -0.010
        ax.annotate(f"{lab}\n{lat:g} ms", (vram, f1), fontsize=7.2,
                    xytext=(7, 6 if dy > 0 else -16), textcoords="offset points",
                    ha="left", va="bottom" if dy > 0 else "top")
    # Esquina deseable: alta calidad, baja VRAM (arriba-izquierda)
    ax.annotate("mejor compromiso\n(alta F1, baja VRAM)", xy=(4.93, 0.706),
                xytext=(8.5, 0.668), fontsize=8, color="green", ha="left",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.1, alpha=0.7))
    ax.set_xlabel("VRAM pico (GB)")
    ax.set_ylabel("Calidad — macro-F1 (test Cardiff ES)")
    ax.set_xlim(0, 18.5); ax.set_ylim(0.635, 0.72)
    if not paper:  # la variante _paper va sin título (numeración en el pie IEEE)
        ax.set_title("Figura E — Compromiso calidad/coste (4D)")
    # leyenda de métodos (marcadores) y nota de tamaño
    mk_leg = [Line2D([], [], marker=m, color="w", markerfacecolor="#888", markeredgecolor="black",
                     markersize=9, label=l) for m, l in
              [("o", "LoRA"), ("s", "QLoRA"), ("^", "Full-FT"), ("D", "Encoder FT")]]
    leg1 = ax.legend(handles=mk_leg, fontsize=8, loc="lower right", title="Método",
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.text(0.015, 0.04, "tamaño del marcador $\\propto$ parámetros entrenables (M)\n"
            "anotación = latencia (ms/ej.)", transform=ax.transAxes, fontsize=7.3,
            va="bottom", ha="left", color="#444444",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
    fig.tight_layout()
    stem = "figE_quality_cost_paper" if paper else "figE_quality_cost"
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"Figura E{' (paper)' if paper else ''} -> {FIGDIR / (stem + '.png')}")

if __name__ == "__main__":
    figure_d()
    figure_d(paper=True)
    figure_e()
    figure_e(paper=True)
