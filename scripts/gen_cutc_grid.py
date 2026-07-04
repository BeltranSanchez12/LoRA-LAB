"""
gen_cutc_grid.py — Configs del Corte C (Sprint 4): transferencia dialectal LoRA.

Matriz de transferencia 3×3 (train país A -> eval país B-dev) para Qwen3-1.7B y 4B.
MISMOS hiperparámetros que el Corte B: LoRA r=16, lr=2e-4, epochs=5, min_steps=80,
mejor-checkpoint por F1-macro de validación. MISMO harness F1-macro.

  - processed_dir = dominio FUENTE  (train + validation, país A)  -> data/processed/tass_{A}
  - eval_dir      = dominio OBJETIVO (test = development, país B)  -> data/processed/tass_{B}
  - Diagonal (A==B) = techo dentro del dominio; fuera de diagonal = penalización dialectal.

Genera además el SMOKE TEST: 1.7B entrenado en ES, evaluado en ES (diagonal) y PE (fuera).

Salida: configs/sprint4/grid/*.yaml y configs/sprint4/smoke/*.yaml
"""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "configs" / "sprint4" / "grid"
SMOKE = ROOT / "configs" / "sprint4" / "smoke"
GRID.mkdir(parents=True, exist_ok=True)
SMOKE.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["es", "cr", "pe"]
MODELS = {"qwen1.7b": "Qwen/Qwen3-1.7B", "qwen4b": "Qwen/Qwen3-4B"}
LANE = {"qwen1.7b": "A", "qwen4b": "B"}
# 3 semillas para media±desv y barras de error reales (subsets pequeños, 539-738/país).
# El split train/val es fijo (construido con seed 42); la semilla solo varía la
# aleatoriedad de ENTRENAMIENTO (init LoRA, orden de datos), igual que Cut B a datos full.
SEEDS = [42, 43, 44]
SMOKE_SEED = 42

# Idénticos al Corte B (gen_sprint3_grid.py).
BASE_TRAIN = dict(
    epochs=5, min_steps=80, batch_size=8, grad_accum=1,
    learning_rate=2.0e-4, warmup_ratio=0.1, max_length=256,
    lr_scheduler_type="cosine", eval_points=4,
)
PEFT = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])


def cfg(name, base_model, src, tgt, gpu_group, seed):
    return {
        "experiment": {"name": name, "seed": seed, "overwrite": False, "gpu_group": gpu_group},
        "model": {"base_model": base_model, "method": "lora", "peft": dict(PEFT)},
        "data": {
            "processed_dir": f"data/processed/tass_{src}",   # fuente: train + val
            "eval_dir": f"data/processed/tass_{tgt}",         # objetivo: test = dev país B
            "train_fraction": "full",
        },
        "training": dict(BASE_TRAIN),
        "eval": {"max_examples": None},
        "output": {"results_dir": "results/sprint4/"},
    }


def main():
    written = []
    # --- Matriz 3×3 por modelo × semilla ---
    for tag, base in MODELS.items():
        group = LANE[tag]
        for src in COUNTRIES:
            for tgt in COUNTRIES:
                for seed in SEEDS:
                    name = f"cutc_{tag}_{src}2{tgt}_s{seed}"
                    (GRID / f"{name}.yaml").write_text(
                        yaml.safe_dump(cfg(name, base, src, tgt, group, seed),
                                       sort_keys=False, allow_unicode=True))
                    written.append(name)

    # --- Smoke test: 1.7B train ES -> eval ES (diagonal) y PE (fuera de diagonal) ---
    smoke = []
    for tgt in ("es", "pe"):
        name = f"smoke_cutc_qwen1.7b_es2{tgt}_s{SMOKE_SEED}"
        (SMOKE / f"{name}.yaml").write_text(
            yaml.safe_dump(cfg(name, MODELS["qwen1.7b"], "es", tgt, "A", SMOKE_SEED),
                           sort_keys=False, allow_unicode=True))
        smoke.append(name)

    print(f"Grid 3×3 × {len(MODELS)} modelos × {len(SEEDS)} semillas: {len(written)} configs en {GRID}")
    for n in written:
        print(f"  {n}")
    print(f"\nSmoke test: {len(smoke)} configs en {SMOKE}")
    for n in smoke:
        print(f"  {n}")


if __name__ == "__main__":
    main()
