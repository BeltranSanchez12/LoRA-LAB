# GPU Runs Pending — Sprint 2 Closure (Colab/Kaggle T4)

Three results still need to be produced on a GPU (no local GPU available) before
Sprint 2 can close:

| # | Run | Replaces / produces | Output file |
|---|-----|---|---|
| 1 | BETO fine-tuned, **3 epochs** (definitive) | provisional 1-epoch CPU result | `results/baselines/beto_cardiff_es.json` |
| 2 | xlm-roberta-base fine-tuned, **3 epochs** | placeholder | `results/baselines/xlmr_base_cardiff_es.json` |
| 3 | Qwen3-1.7B prompting, **k=0/4/8/16** (records `fallback_rate`) | placeholder | `results/baselines/prompting_*` |

Sprint 2 closes when all three have real numbers **and**
`pytest tests/` reports `0 skipped` (i.e. `test_xlmr_trainable_params_not_none`
goes from SKIP to PASS).

---

## Option A — Run the notebook (recommended: one session, all three runs)

1. Open `notebooks/sprint2_gpu_baselines.ipynb` in Google Colab.
   `Runtime → Change runtime type → T4 GPU`.
2. Clone/upload the repo to `/content/AI_LAB` (adjust `REPO_ROOT` in cell 3 if different).
3. Run the cells in order:
   - **Cells 1–4** — install deps, detect GPU, load Cardiff ES splits.
   - **Cells 5–6** — T5-A: BETO, 3 epochs (overwrites the provisional 1-epoch CPU result).
   - **Cells 7–8** — T5-B: xlm-roberta-base, 3 epochs (controlled encoder for the Pareto B).
   - **Cells 9–11** — T4: Qwen3-1.7B prompting, k=0, 4, 8, 16 (records `fallback_rate`;
     expect ~30–60 min for 870 test examples × 4 variants).
   - **Cell 12** — prints the results summary table for a quick sanity check.
   - **Cell 14** — zips `results/baselines/` + `results/all_results.csv` and downloads.
4. Copy the downloaded JSONs into the local `results/baselines/`, overwriting the placeholders.

## Option B — Run the standalone scripts (more control, can be split across sessions)

Both scripts auto-detect the GPU, write the JSON to `results/baselines/`, and append
to `results/all_results.csv` via `save_result`.

```bash
# T5-A + T5-B: BETO and xlm-roberta-base, 3 epochs each (~5-8 min/model on a T4)
python scripts/run_encoder_baselines.py

# T4: Qwen3-1.7B prompting, all k from configs/prompting_protocol.yaml (4-bit on GPU)
python scripts/run_prompting_baseline.py --model_id Qwen/Qwen3-1.7B
```

Useful `run_prompting_baseline.py` flags:
- `--k_values 0 4 8` — restrict to a subset of k (default: all four from the YAML)
- `--max_examples 50` — quick smoke-test on part of the test set before the full run
- `--dump-examples` — print the selected few-shot examples without running inference
  (useful to freeze `fixed_examples` in `configs/prompting_protocol.yaml` beforehand)

---

## After the runs — closing checklist

1. Confirm each new/updated JSON in `results/baselines/` has non-null `f1_macro`,
   `trainable_params` (encoders) and `fallback_rate` (prompting).
2. Replace the placeholders: `xlmr_base_cardiff_es.json` and
   `prompting_zeroshot_PENDING.json` (the script/notebook write the real files directly).
3. Re-run `pytest tests/ -v` — the suite should report **0 skipped**.
4. Review `results/all_results.csv` and update the comparison tables/figures in
   `paper/main.tex` (Section IV) with the definitive numbers.
5. Only then: commit + push to close Sprint 2.
