"""
gen_sprint3_pilot.py — Genera los configs del PILOTO de 1 seed del Sprint 3.

Piloto (seed 42):
  - Núcleo del corte A: LoRA × {Qwen3-1.7B, Qwen3-4B} × fracciones
    {10,16,25,50,100,250,500,1000,full}.
  - Spot-check del corte B (full data): QLoRA en 1.7B y 4B, y full-FT en 1.7B.

Hiperparámetros fijos del estudio controlado: lr=2e-4, r=16. Política de pasos:
epochs=5 con min_steps=80 (fracciones pequeñas), mejor-checkpoint por F1 de validación.
Escribe un YAML por celda en configs/sprint3/pilot/.
"""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "sprint3" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
FRACTIONS = [10, 16, 25, 50, 100, 250, 500, 1000, "full"]
MODELS = {"qwen1.7b": "Qwen/Qwen3-1.7B", "qwen4b": "Qwen/Qwen3-4B"}

BASE_TRAIN = dict(
    epochs=5, min_steps=80, batch_size=8, grad_accum=1,
    learning_rate=2.0e-4, warmup_ratio=0.1, max_length=256,
    lr_scheduler_type="cosine", eval_points=4,
)
PEFT = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])


def cfg(name, base_model, method, fraction, gpu_group):
    c = {
        "experiment": {"name": name, "seed": SEED, "overwrite": False, "gpu_group": gpu_group},
        "model": {"base_model": base_model, "method": method},
        "data": {"processed_dir": "data/processed/cardiff_es", "train_fraction": fraction},
        "training": dict(BASE_TRAIN),
        "eval": {"max_examples": None},
        "output": {"results_dir": "results/"},
    }
    if method in ("lora", "qlora"):
        c["model"]["peft"] = dict(PEFT)
    if method == "full_ft":
        # full-FT: lr más bajo y sin min_steps inflado para no destrozar pesos
        c["training"] = {**BASE_TRAIN, "learning_rate": 1.0e-5, "min_steps": 60}
    return c


def main():
    written = []
    # Corte A: LoRA en ambos modelos × fracciones. 1.7B -> grupo gpuA, 4B -> grupo gpuB.
    for tag, base in MODELS.items():
        group = "A" if tag == "qwen1.7b" else "B"
        for frac in FRACTIONS:
            fr = "full" if frac == "full" else int(frac)
            name = f"lora_{tag}_n{fr}_s{SEED}"
            path = OUT / f"{name}.yaml"
            path.write_text(yaml.safe_dump(cfg(name, base, "lora", fr, group), sort_keys=False, allow_unicode=True))
            written.append((group, path.name))
    # Corte B spot-check (full data)
    spot = [
        ("qlora", "Qwen/Qwen3-1.7B", "qwen1.7b", "A"),
        ("full_ft", "Qwen/Qwen3-1.7B", "qwen1.7b", "A"),
        ("qlora", "Qwen/Qwen3-4B", "qwen4b", "B"),
    ]
    for method, base, tag, group in spot:
        name = f"{method}_{tag}_nfull_s{SEED}"
        path = OUT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg(name, base, method, "full", group), sort_keys=False, allow_unicode=True))
        written.append((group, path.name))

    print(f"Escritos {len(written)} configs en {OUT}")
    for g in ("A", "B"):
        cells = [n for grp, n in written if grp == g]
        print(f"  grupo {g} ({'GPU0/1.7B' if g=='A' else 'GPU6/4B'}): {len(cells)} celdas")
        for n in cells:
            print(f"    {n}")


if __name__ == "__main__":
    main()
