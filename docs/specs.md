# Project Specifications

**Project title:** Efficient Adaptation of Small Open LLMs for Spanish Sentiment Analysis — an Empirical Study of Quality, Cost, and Robustness under a Fixed Compute Budget.

## Research Questions

### Cut A — Data-efficiency crossover  (Priority: CORE)

> From how many labelled examples does a small LLM with LoRA outperform the same
> model under few-shot prompting, and how does this crossover point depend on
> model size?

This cut varies the training-set fraction (≈5 %, 10 %, 25 %, 50 %, 100 %) and
compares LoRA fine-tuning against zero/few-shot prompting on the same model
family.  The expected deliverable is a **data-efficiency curve with a visible
crossover point** (or the honest finding that no crossover exists in this range).

**Gap:** No systematic study characterises this crossover for Spanish sentiment
under free-tier compute constraints with current small open LLMs.

### Cut B — Generative LLMs vs. Spanish encoders  (Priority: CORE)

> Do small generative LLMs with LoRA/QLoRA beat established Spanish encoders
> (BETO / RoBERTuito) on the quality-cost Pareto frontier, or do they only match
> them at higher cost?

This cut operates at full data and plots every method (LoRA, QLoRA, full-FT on
small models; BETO/RoBERTuito from Sprint 2) in a single **Pareto quality-cost
plane**.  Cost axis: trainable parameters, peak VRAM, training time, inference
latency.

**Gap:** Direct comparison of generative PEFT vs. encoder fine-tuning in a
single reproducible pipeline for Spanish is absent from the recent literature.

### Cut C — Dialectal robustness  (Priority: BUFFER — trimmed first)

> How much does a LoRA adapter trained on one Spanish variety transfer to other
> varieties (Mexican, Uruguayan, Costa Rican, Peruvian, Peninsular), and from
> English?

This cut requires labelled data per variety (TASS / InterTASS corpus, pending
licence).  If the licence does not arrive in time, Cut C becomes future work —
this is planned and acknowledged.

**Gap:** PEFT transfer across Spanish varieties has not been systematically
benchmarked on open small LLMs.

## Priority

| Cut | Status | Rationale |
|-----|--------|-----------|
| A + B | Committed core | Share the same pipeline and experiment grid |
| C | Planned buffer | Reuses the same machinery; data availability is the gating dependency |

## Quality Metrics

Primary: **F1 macro** over three sentiment classes (positive / neutral / negative).

Secondary (reported per run):
- Accuracy
- F1 per class (positive, neutral, negative)

## Cost Metrics

Reported for every experiment run:

| Metric | Unit | How measured |
|--------|------|--------------|
| `trainable_params` | count | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| `peak_vram_gb` | GB | `torch.cuda.max_memory_allocated()` |
| `train_time_s` | seconds | wall-clock from first step to last step |
| `inference_latency_ms` | ms | median over test set, one sample at a time |

## Dataset

- **Cuts A and B:** `cardiffnlp/tweet_sentiment_multilingual`, Spanish subset.
  Three classes, pre-defined train/validation/test splits, downloaded via
  HuggingFace `datasets`.
- **Cut C (pending):** TASS / InterTASS (SEPLN), labelled by country variety.
  Requires signing an academic licence.

## Compute Budget

- **Hardware:** free-tier GPU (NVIDIA T4, 16 GB VRAM) on Google Colab or
  Kaggle.
- **Total budget:** 180 GPU-hours.
- **Model sizes:** workhorse 1–4 B parameters (many configurations); optional
  point at 7–8 B via QLoRA if budget permits.
- **Seeds:** 3 per configuration cell; reduced to 2 if the clock runs short.
  Never below 2.

## Experiment Configuration

Each experiment is fully described by a YAML file in `configs/`.  See
`configs/example_experiment.yaml` for the canonical schema.  Results from every
run are persisted in `results/` as structured CSV/JSON rows alongside the config
hash that produced them, ensuring full traceability.
