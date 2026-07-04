"""
gen_sprint3_grid.py — Genera los configs del GRID DEFINITIVO del Sprint 3.

Diseño (acordado con el director del proyecto):

  Corte A (curva F1 vs. tamaño de datos, Figura A) — 3 SEMILLAS {42,43,44}:
    LoRA × {Qwen3-1.7B, Qwen3-4B}
         × fracciones {4,7,10,16,25,50,100,250,500,1000,full}
    Se añaden n=4 y n=7 para capturar el CRUCE (por debajo de n=10), con las 3
    semillas desde el inicio. A fracciones pequeñas cada semilla re-muestrea su
    subconjunto (varianza de muestreo -> barras de error reales). Eval SIEMPRE
    sobre el test completo (870).
    => 2 modelos × 11 fracciones × 3 semillas = 66 celdas.

  Corte B (calidad-coste a datos completos, Figura B) — 1 SEMILLA {42}:
    A datos completos el ruido entre semillas es bajo, así que reservamos las 3
    semillas para el corte A. Spot-checks:
        qlora_qwen1.7b_full, full_ft_qwen1.7b_full, qlora_qwen4b_full
    => 3 celdas.

Hiperparámetros fijos del estudio controlado: lr=2e-4, r=16, epochs=5,
min_steps=80, mejor-checkpoint por F1-macro de validación.

GPU lanes: 1.7B -> GPU 6 ; 4B -> GPU 7 (gpu_group A/B es solo etiqueta de lane).
Escribe un YAML por celda en configs/sprint3/grid/.
"""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "sprint3" / "grid"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS_A = [42, 43, 44]
SEED_B = 42
FRACTIONS = [4, 7, 10, 16, 25, 50, 100, 250, 500, 1000, "full"]
MODELS = {"qwen1.7b": "Qwen/Qwen3-1.7B", "qwen4b": "Qwen/Qwen3-4B"}
# Lane de GPU por modelo (etiqueta; el runner fija CUDA_VISIBLE_DEVICES).
LANE = {"qwen1.7b": "A", "qwen4b": "B"}

BASE_TRAIN = dict(
    epochs=5, min_steps=80, batch_size=8, grad_accum=1,
    learning_rate=2.0e-4, warmup_ratio=0.1, max_length=256,
    lr_scheduler_type="cosine", eval_points=4,
)
PEFT = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])


def cfg(name, base_model, method, fraction, seed, gpu_group):
    c = {
        "experiment": {"name": name, "seed": seed, "overwrite": False, "gpu_group": gpu_group},
        "model": {"base_model": base_model, "method": method},
        "data": {"processed_dir": "data/processed/cardiff_es", "train_fraction": fraction},
        "training": dict(BASE_TRAIN),
        "eval": {"max_examples": None},   # test completo (870)
        "output": {"results_dir": "results/"},
    }
    if method in ("lora", "qlora"):
        c["model"]["peft"] = dict(PEFT)
    if method == "full_ft":
        c["training"] = {**BASE_TRAIN, "learning_rate": 1.0e-5, "min_steps": 60}
    return c


def main():
    written = []
    # --- Corte A: LoRA × 2 modelos × fracciones × 3 semillas ---
    for tag, base in MODELS.items():
        group = LANE[tag]
        for frac in FRACTIONS:
            fr = "full" if frac == "full" else int(frac)
            for seed in SEEDS_A:
                name = f"lora_{tag}_n{fr}_s{seed}"
                path = OUT / f"{name}.yaml"
                path.write_text(yaml.safe_dump(
                    cfg(name, base, "lora", fr, seed, group),
                    sort_keys=False, allow_unicode=True))
                written.append((group, "A", path.name))

    # --- Corte B: 1 semilla, spot-checks a datos completos ---
    spot = [
        ("qlora", "Qwen/Qwen3-1.7B", "qwen1.7b", "A"),
        ("full_ft", "Qwen/Qwen3-1.7B", "qwen1.7b", "A"),
        ("qlora", "Qwen/Qwen3-4B", "qwen4b", "B"),
    ]
    for method, base, tag, group in spot:
        name = f"{method}_{tag}_nfull_s{SEED_B}"
        path = OUT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(
            cfg(name, base, method, "full", SEED_B, group),
            sort_keys=False, allow_unicode=True))
        written.append((group, "B", path.name))

    n_a = sum(1 for _, corte, _ in written if corte == "A")
    n_b = sum(1 for _, corte, _ in written if corte == "B")
    print(f"Escritos {len(written)} configs en {OUT}")
    print(f"  Corte A (3 semillas): {n_a} celdas")
    print(f"  Corte B (1 semilla):  {n_b} celdas")
    for lane in ("A", "B"):
        cells = sorted(n for grp, _, n in written if grp == lane)
        gpu = "GPU6/1.7B" if lane == "A" else "GPU7/4B"
        print(f"  lane {lane} ({gpu}): {len(cells)} celdas")


if __name__ == "__main__":
    main()
