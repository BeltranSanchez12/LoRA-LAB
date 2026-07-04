# AI Lab — Efficient Adaptation of Small Open LLMs for Spanish Sentiment Analysis

An empirical study of LoRA and QLoRA fine-tuning on small open-weight LLMs
for Spanish sentiment analysis.  The project characterises the quality-cost
trade-off of parameter-efficient fine-tuning (PEFT) under a fixed free-tier
GPU budget (T4 16 GB, ~180 h total), producing reproducible results from a
single shared pipeline.

## Research Questions

| Cut | Question |
|-----|----------|
| **A — Data-efficiency crossover** | From how many labelled examples does a small LLM with LoRA outperform the same model under few-shot prompting, and how does this depend on model size? |
| **B — Generative vs. encoder models** | Do small generative LLMs with LoRA beat established Spanish encoders (BETO / RoBERTuito) on the quality-cost frontier, or do they only match them at higher cost? |
| **C — Dialectal robustness** *(buffer, may be reduced)* | How well does a LoRA adapter transfer between Spanish varieties (and from English)? |

Cuts A and B are the committed core.  Cut C is a planned extension that is the
first to be trimmed if the compute budget runs short.

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd AI_LAB

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run the test suite
pytest tests/
```

## Project Structure

```
AI_LAB/
├── configs/                   # One YAML file per experiment
│   └── example_experiment.yaml
├── data/
│   ├── raw/                   # Original downloaded data  [git-ignored]
│   └── processed/             # Tokenised / split data    [git-ignored]
├── docs/
│   ├── planning/              # Sprint specs (read-only)
│   └── specs.md               # Research questions, metrics, constraints
├── results/                   # Metrics, figures, CSVs    [git-ignored except .gitkeep]
├── src/
│   ├── data/                  # Data loading and preprocessing modules
│   ├── models/                # Training and evaluation modules
│   └── utils/
│       └── seed.py            # Global seed fixture (seed = 42)
├── tests/                     # pytest test suite
├── paper/                     # LaTeX source
├── diary/                     # Sprint diary entries
├── defense/                   # Oral defense material
├── .gitignore
├── requirements.txt
└── README.md
```

## Reproducibility

- **Global seed:** 42 for all runs.  Call `from src.utils.seed import set_seed; set_seed()` at the top of every script.
- **One config per experiment:** every run is fully described by a YAML file in `configs/`.  Results are stored in `results/` as structured CSV/JSON alongside the config that produced them.
- **Pinned dependencies:** `requirements.txt` lists minimum-version constraints tested on Python 3.11.
- **Three seeds per cell** in the main experiment grid to allow reporting of mean ± std.
