"""
aggregate_results.py — Agrega los resultados del grid del Sprint 3 en media ± std
por celda y escribe resúmenes para las Figuras A y B.

Fuente única: results/all_results.csv (lo escribe save_result en cada experimento).

Salidas:
  results/summary_corteA.csv  — LoRA × {1.7B,4B} × fracción: F1 media±std sobre semillas.
  results/summary_corteB.csv  — datos completos: lora/qlora/full_ft + encoders + prompting (coste-calidad).

Uso:
    python scripts/aggregate_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "all_results.csv"

TRAIN_FULL_N = 1839  # tamaño del train completo de Cardiff ES


def frac_to_n(x) -> int:
    s = str(x).lower()
    if s in ("full", "all", "1.0", "1", "nan", ""):
        return TRAIN_FULL_N
    return int(float(x))


def short_model(name: str) -> str:
    n = str(name)
    if "Qwen3-1.7B" in n:
        return "Qwen3-1.7B"
    if "Qwen3-4B" in n:
        return "Qwen3-4B"
    return n.split("/")[-1]


def main() -> None:
    if not CSV.exists():
        sys.exit(f"No existe {CSV}")
    df = pd.read_csv(CSV)
    # Dedup defensivo: re-runs (prompting) o overwrite dejan filas repetidas por
    # (experiment_name, seed). Nos quedamos con la última (la más reciente).
    before = len(df)
    df = df.drop_duplicates(subset=["experiment_name", "seed"], keep="last").reset_index(drop=True)
    if len(df) < before:
        print(f"[dedup] {before - len(df)} filas duplicadas descartadas (re-runs).")
    df["model_short"] = df["model_name"].map(short_model)
    df["n_train"] = df["data_fraction"].map(frac_to_n)

    # ---------------- Corte A: LoRA × fracción, media±std sobre semillas ----------------
    lora = df[df["method"] == "lora"].copy()
    if lora.empty:
        print("[aviso] aún no hay filas LoRA en el CSV.")
    gcols = ["model_short", "n_train"]
    agg = (
        lora.groupby(gcols)
        .agg(
            n_seeds=("seed", "nunique"),
            seeds=("seed", lambda s: ",".join(map(str, sorted(set(s))))),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            f1_min=("f1_macro", "min"),
            f1_max=("f1_macro", "max"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            trainable_params=("trainable_params", "median"),
            vram_gb=("peak_vram_gb", "mean"),
            train_time_s=("train_time_s", "mean"),
            latency_ms=("inference_latency_ms", "mean"),
            fallback=("fallback_rate", "mean"),
        )
        .reset_index()
        .sort_values(["model_short", "n_train"])
    )
    agg["f1_std"] = agg["f1_std"].fillna(0.0)  # std de 1 muestra -> 0
    out_a = ROOT / "results" / "summary_corteA.csv"
    agg.to_csv(out_a, index=False)

    print("=" * 88)
    print("CORTE A — LoRA F1-macro media±std por (modelo, n_train)")
    print("=" * 88)
    for model in sorted(agg["model_short"].unique()):
        sub = agg[agg["model_short"] == model]
        print(f"\n  {model}:")
        print(f"    {'n':>6} {'seeds':>3}  {'F1 mean±std':>16}  {'[min,max]':>16}")
        for _, r in sub.iterrows():
            ntxt = "full" if r["n_train"] == TRAIN_FULL_N else str(int(r["n_train"]))
            print(f"    {ntxt:>6} {int(r['n_seeds']):>3}  "
                  f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}  "
                  f"[{r['f1_min']:.4f},{r['f1_max']:.4f}]")

    # ---------------- Corte B: datos completos, coste-calidad ----------------
    full = df[df["n_train"] == TRAIN_FULL_N].copy()
    keep_methods = ["lora", "qlora", "full_ft", "encoder_finetuned",
                    "prompting_zeroshot", "prompting_fewshot_k4",
                    "prompting_fewshot_k8", "prompting_fewshot_k16"]
    full = full[full["method"].isin(keep_methods)]
    cols_b = ["experiment_name", "model_short", "method", "seed", "f1_macro",
              "accuracy", "trainable_params", "peak_vram_gb", "train_time_s",
              "inference_latency_ms", "fallback_rate"]
    full_b = full[cols_b].sort_values(["method", "model_short"]).reset_index(drop=True)
    out_b = ROOT / "results" / "summary_corteB.csv"
    full_b.to_csv(out_b, index=False)

    print("\n" + "=" * 88)
    print("CORTE B — datos completos (coste-calidad, 1 semilla)")
    print("=" * 88)
    print(full_b.to_string(index=False, max_colwidth=28))

    print(f"\nEscritos:\n  {out_a}\n  {out_b}")


if __name__ == "__main__":
    main()
