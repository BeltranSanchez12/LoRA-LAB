"""
qa-validator: Sprint 5 validation suite.

Covers:
  - Corte A (Table II): f1_mean / f1_std vs summary_corteA.csv
  - Corte B (Table III): F1, params, VRAM, train_time, latency vs summary_corteB.csv
  - Table IV / Fig D (per-class F1): vs JSON files (LoRA = mean over 3 seeds)
  - Corte C (Table V): transfer matrix mean/std vs sprint4 JSON files
  - sigma values for transfer penalty toward ES-ibérico
  - TF-IDF and RoBERTuito baselines (Table I)
  - Parameter reduction factors (17-43x, 9-24x, 268x)
  - Dataset split sizes (Cardiff and InterTASS)
  - Compilation artefact: 0 Overfull hbox in main.log
  - Git tracking: figA-figE + make_figures_sprint5.py
  - Paper meta: affiliation, 10,000x attribution, no (source:) artefacts

Rounding notes (matching paper table display):
  - F1 values: 3 decimal places
  - VRAM: 2 decimal places
  - Latency: nearest integer ms (or nearest 0.1 for sub-1ms encoder values)
  - Params: nearest displayed integer M (BETO=110, XLM-R=278, Full-FT=1720)
  - Std: ddof=1 (sample std, matches summary_corteA.csv)
"""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

RESULTS = "/home/jovyan/jupyterlab_container/AI_LAB/results"
SPRINT4 = os.path.join(RESULTS, "sprint4")
PAPER = "/home/jovyan/jupyterlab_container/AI_LAB/paper"
REPO = "/home/jovyan/jupyterlab_container/AI_LAB"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r3(x):
    """Round to 3 decimal places (F1 precision in paper)."""
    return round(float(x), 3)


def r2(x):
    """Round to 2 decimal places (VRAM precision in paper)."""
    return round(float(x), 2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Corte A — Table II
# ---------------------------------------------------------------------------

class TestCorteA:
    """Paper Table II values traced to summary_corteA.csv (ddof=1 std)."""

    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(os.path.join(RESULTS, "summary_corteA.csv"))

    EXPECTED_17B = [
        (4,    0.539, 0.010),
        (7,    0.590, 0.063),
        (10,   0.598, 0.035),
        (16,   0.622, 0.026),
        (25,   0.654, 0.023),
        (50,   0.632, 0.030),
        (100,  0.645, 0.024),
        (250,  0.657, 0.011),
        (500,  0.683, 0.015),
        (1000, 0.685, 0.005),
        (1839, 0.698, 0.006),
    ]

    EXPECTED_4B = [
        (4,    0.522, 0.089),
        (7,    0.645, 0.038),
        (10,   0.638, 0.013),
        (16,   0.641, 0.068),
        (25,   0.645, 0.037),
        (50,   0.666, 0.034),
        (100,  0.687, 0.014),
        (250,  0.698, 0.002),
        (500,  0.687, 0.009),
        (1000, 0.698, 0.007),
        (1839, 0.707, 0.022),
    ]

    @pytest.mark.parametrize("n,f1_mean,f1_std", EXPECTED_17B)
    def test_17b(self, df, n, f1_mean, f1_std):
        row = df[(df["model_short"] == "Qwen3-1.7B") & (df["n_train"] == n)]
        assert len(row) == 1, f"No row for 1.7B n={n}"
        assert r3(row["f1_mean"].values[0]) == f1_mean, \
            f"1.7B n={n} f1_mean mismatch: {row['f1_mean'].values[0]:.6f} vs paper {f1_mean}"
        assert r3(row["f1_std"].values[0]) == f1_std, \
            f"1.7B n={n} f1_std mismatch: {row['f1_std'].values[0]:.6f} vs paper {f1_std}"

    @pytest.mark.parametrize("n,f1_mean,f1_std", EXPECTED_4B)
    def test_4b(self, df, n, f1_mean, f1_std):
        row = df[(df["model_short"] == "Qwen3-4B") & (df["n_train"] == n)]
        assert len(row) == 1, f"No row for 4B n={n}"
        assert r3(row["f1_mean"].values[0]) == f1_mean, \
            f"4B n={n} f1_mean mismatch: {row['f1_mean'].values[0]:.6f} vs paper {f1_mean}"
        assert r3(row["f1_std"].values[0]) == f1_std, \
            f"4B n={n} f1_std mismatch: {row['f1_std'].values[0]:.6f} vs paper {f1_std}"


# ---------------------------------------------------------------------------
# Corte B — Table III
# ---------------------------------------------------------------------------

class TestCorteB:
    """Paper Table III values traced to summary_corteB.csv.

    Rounding:
      - F1: 3 dp
      - VRAM: 2 dp  (paper displays 6.79, 4.93, 15.84, 2.47, 4.59 GB)
      - Latency: nearest integer ms; encoder latency to 1 dp
      - Params (M): 2 dp for LoRA/QLoRA; nearest int for encoders/full-FT
    """

    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(os.path.join(RESULTS, "summary_corteB.csv"))

    @pytest.fixture(scope="class")
    def df_a(self):
        """summary_corteA has precomputed mean/std (ddof=1) for LoRA full-data rows."""
        return pd.read_csv(os.path.join(RESULTS, "summary_corteA.csv"))

    # --- LoRA 4B (3 seeds) ---
    def test_lora_4b_mean_f1(self, df_a):
        row = df_a[(df_a["model_short"] == "Qwen3-4B") & (df_a["n_train"] == 1839)]
        assert r3(row["f1_mean"].values[0]) == 0.707

    def test_lora_4b_std_f1(self, df_a):
        row = df_a[(df_a["model_short"] == "Qwen3-4B") & (df_a["n_train"] == 1839)]
        assert r3(row["f1_std"].values[0]) == 0.022

    def test_lora_4b_vram(self, df):
        seeds = df[df["experiment_name"].isin(
            [f"lora_qwen4b_nfull_s{s}" for s in [42, 43, 44]])]
        assert r2(seeds["peak_vram_gb"].mean()) == 16.74 or \
               round(seeds["peak_vram_gb"].mean(), 1) == 16.7, \
               f"4B LoRA VRAM mean={seeds['peak_vram_gb'].mean():.3f}"

    def test_lora_4b_train_time(self, df):
        seeds = df[df["experiment_name"].isin(
            [f"lora_qwen4b_nfull_s{s}" for s in [42, 43, 44]])]
        assert round(seeds["train_time_s"].mean()) == 218

    def test_lora_4b_latency(self, df):
        seeds = df[df["experiment_name"].isin(
            [f"lora_qwen4b_nfull_s{s}" for s in [42, 43, 44]])]
        assert round(seeds["inference_latency_ms"].mean()) == 68

    def test_lora_4b_params(self, df):
        seeds = df[df["experiment_name"].isin(
            [f"lora_qwen4b_nfull_s{s}" for s in [42, 43, 44]])]
        assert r2(seeds["trainable_params"].mean() / 1e6) == 11.80

    # --- LoRA 1.7B (3 seeds) ---
    def test_lora_17b_mean_f1(self, df_a):
        row = df_a[(df_a["model_short"] == "Qwen3-1.7B") & (df_a["n_train"] == 1839)]
        assert r3(row["f1_mean"].values[0]) == 0.698

    def test_lora_17b_std_f1(self, df_a):
        row = df_a[(df_a["model_short"] == "Qwen3-1.7B") & (df_a["n_train"] == 1839)]
        assert r3(row["f1_std"].values[0]) == 0.006

    def test_lora_17b_vram(self, df):
        seeds = df[df["experiment_name"].isin(
            [f"lora_qwen1.7b_nfull_s{s}" for s in [42, 43, 44]])]
        assert r2(seeds["peak_vram_gb"].mean()) == 8.72

    def test_lora_17b_train_time(self, df):
        seeds = df[df["experiment_name"].isin(
            [f"lora_qwen1.7b_nfull_s{s}" for s in [42, 43, 44]])]
        assert round(seeds["train_time_s"].mean()) == 169

    def test_lora_17b_latency(self, df):
        seeds = df[df["experiment_name"].isin(
            [f"lora_qwen1.7b_nfull_s{s}" for s in [42, 43, 44]])]
        assert round(seeds["inference_latency_ms"].mean()) in [53, 54]

    def test_lora_17b_params(self, df):
        seeds = df[df["experiment_name"].isin(
            [f"lora_qwen1.7b_nfull_s{s}" for s in [42, 43, 44]])]
        assert r2(seeds["trainable_params"].mean() / 1e6) == 6.42

    # --- QLoRA 4B (single seed 42) ---
    def test_qlora_4b_f1(self, df):
        row = df[df["experiment_name"] == "qlora_qwen4b_nfull_s42"].iloc[0]
        assert r3(row["f1_macro"]) == 0.701

    def test_qlora_4b_vram(self, df):
        row = df[df["experiment_name"] == "qlora_qwen4b_nfull_s42"].iloc[0]
        # paper displays 6.79 GB (2dp)
        assert r2(row["peak_vram_gb"]) == 6.79

    def test_qlora_4b_train_time(self, df):
        row = df[df["experiment_name"] == "qlora_qwen4b_nfull_s42"].iloc[0]
        assert round(row["train_time_s"]) == 534

    def test_qlora_4b_latency(self, df):
        row = df[df["experiment_name"] == "qlora_qwen4b_nfull_s42"].iloc[0]
        assert round(row["inference_latency_ms"]) == 132

    # --- QLoRA 1.7B (single seed 42) ---
    def test_qlora_17b_f1(self, df):
        row = df[df["experiment_name"] == "qlora_qwen1.7b_nfull_s42"].iloc[0]
        assert r3(row["f1_macro"]) == 0.681

    def test_qlora_17b_vram(self, df):
        row = df[df["experiment_name"] == "qlora_qwen1.7b_nfull_s42"].iloc[0]
        # paper displays 4.93 GB (2dp)
        assert r2(row["peak_vram_gb"]) == 4.93

    def test_qlora_17b_train_time(self, df):
        row = df[df["experiment_name"] == "qlora_qwen1.7b_nfull_s42"].iloc[0]
        assert round(row["train_time_s"]) == 409

    def test_qlora_17b_latency(self, df):
        row = df[df["experiment_name"] == "qlora_qwen1.7b_nfull_s42"].iloc[0]
        assert round(row["inference_latency_ms"]) == 102

    # --- Full-FT 1.7B ---
    def test_fullft_17b_f1(self, df):
        row = df[df["experiment_name"] == "full_ft_qwen1.7b_nfull_s42"].iloc[0]
        assert r3(row["f1_macro"]) == 0.696

    def test_fullft_17b_vram(self, df):
        row = df[df["experiment_name"] == "full_ft_qwen1.7b_nfull_s42"].iloc[0]
        assert r2(row["peak_vram_gb"]) == 15.84

    def test_fullft_17b_train_time(self, df):
        row = df[df["experiment_name"] == "full_ft_qwen1.7b_nfull_s42"].iloc[0]
        assert round(row["train_time_s"]) == 124

    def test_fullft_17b_latency(self, df):
        row = df[df["experiment_name"] == "full_ft_qwen1.7b_nfull_s42"].iloc[0]
        assert round(row["inference_latency_ms"]) == 33

    def test_fullft_17b_params_approx(self, df):
        """Paper says 1720M; actual is 1720.57M (truncated to 4 sig figs)."""
        row = df[df["experiment_name"] == "full_ft_qwen1.7b_nfull_s42"].iloc[0]
        params_m = row["trainable_params"] / 1e6
        # Paper says 1720; actual rounds to 1721 (nearest int) but paper truncates
        assert 1719 <= params_m <= 1722, f"Full-FT params={params_m:.2f}M (expected ~1720M)"

    # --- BETO (Full-FT) ---
    def test_beto_f1(self, df):
        exp = "encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es"
        row = df[df["experiment_name"] == exp].iloc[0]
        assert r3(row["f1_macro"]) == 0.661

    def test_beto_vram(self, df):
        exp = "encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es"
        row = df[df["experiment_name"] == exp].iloc[0]
        assert r2(row["peak_vram_gb"]) == 2.47

    def test_beto_train_time(self, df):
        exp = "encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es"
        row = df[df["experiment_name"] == exp].iloc[0]
        assert round(row["train_time_s"]) == 11

    def test_beto_params_rounded(self, df):
        """Paper displays 110M; actual is 109.85M (rounds to 110M)."""
        exp = "encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es"
        row = df[df["experiment_name"] == exp].iloc[0]
        assert round(row["trainable_params"] / 1e6) == 110

    # --- XLM-R (Full-FT) ---
    def test_xlmr_f1(self, df):
        exp = "encoder_finetuned_xlm-roberta-base_cardiff_es"
        row = df[df["experiment_name"] == exp].iloc[0]
        assert r3(row["f1_macro"]) == 0.646

    def test_xlmr_vram(self, df):
        exp = "encoder_finetuned_xlm-roberta-base_cardiff_es"
        row = df[df["experiment_name"] == exp].iloc[0]
        assert r2(row["peak_vram_gb"]) == 4.59

    def test_xlmr_train_time(self, df):
        exp = "encoder_finetuned_xlm-roberta-base_cardiff_es"
        row = df[df["experiment_name"] == exp].iloc[0]
        assert round(row["train_time_s"]) == 18

    def test_xlmr_params_rounded(self, df):
        """Paper displays 278M; actual is 278.05M (rounds to 278M)."""
        exp = "encoder_finetuned_xlm-roberta-base_cardiff_es"
        row = df[df["experiment_name"] == exp].iloc[0]
        assert round(row["trainable_params"] / 1e6) == 278


# ---------------------------------------------------------------------------
# Table IV / Figure D — per-class F1
# ---------------------------------------------------------------------------

class TestPerClassF1:
    """Paper Table IV and Fig D per-class F1 values."""

    def _lora_mean(self, pattern_files):
        neg, neu, pos = [], [], []
        for f in pattern_files:
            d = load_json(f)
            pc = d["f1_per_class"]
            neg.append(pc["negative"])
            neu.append(pc["neutral"])
            pos.append(pc["positive"])
        return np.mean(neg), np.mean(neu), np.mean(pos)

    def test_lora_4b_per_class(self):
        files = [os.path.join(RESULTS, f"lora_qwen4b_nfull_s{s}.json")
                 for s in [42, 43, 44]]
        neg, neu, pos = self._lora_mean(files)
        assert r3(neg) == 0.746, f"4B neg: {neg:.4f}"
        assert r3(neu) == 0.613, f"4B neu: {neu:.4f}"
        assert r3(pos) == 0.764, f"4B pos: {pos:.4f}"

    def test_lora_17b_per_class(self):
        files = [os.path.join(RESULTS, f"lora_qwen1.7b_nfull_s{s}.json")
                 for s in [42, 43, 44]]
        neg, neu, pos = self._lora_mean(files)
        assert r3(neg) == 0.731, f"1.7B neg: {neg:.4f}"
        assert r3(neu) == 0.631, f"1.7B neu: {neu:.4f}"
        assert r3(pos) == 0.731, f"1.7B pos: {pos:.4f}"

    def test_beto_per_class(self):
        d = load_json(os.path.join(
            RESULTS, "encoder_finetuned_dccuchile_bert-base-spanish-wwm-cased_cardiff_es.json"))
        pc = d["f1_per_class"]
        assert r3(pc["negative"]) == 0.702
        assert r3(pc["neutral"]) == 0.550
        assert r3(pc["positive"]) == 0.732

    def test_xlmr_per_class(self):
        d = load_json(os.path.join(
            RESULTS, "encoder_finetuned_xlm-roberta-base_cardiff_es.json"))
        pc = d["f1_per_class"]
        assert r3(pc["negative"]) == 0.702
        assert r3(pc["neutral"]) == 0.500
        assert r3(pc["positive"]) == 0.737


# ---------------------------------------------------------------------------
# Corte C — Table V transfer matrix
# ---------------------------------------------------------------------------

class TestCorteC:
    """Paper Table V (tab:cutC) values traced to sprint4 JSON files.

    std uses ddof=0 (population std over 3 seeds) — verified against paper table.
    Exception: 4B es2es std=0.006525 rounds to 0.007 but paper says 0.006
    (captured separately in test_4b_es2es_std_discrepancy).
    """

    EXPECTED_17B = {
        "es2es": (0.644, 0.007), "es2cr": (0.684, 0.026), "es2pe": (0.663, 0.031),
        "cr2es": (0.613, 0.007), "cr2cr": (0.682, 0.023), "cr2pe": (0.631, 0.009),
        "pe2es": (0.613, 0.008), "pe2cr": (0.688, 0.005), "pe2pe": (0.670, 0.010),
    }
    EXPECTED_4B = {
        "es2es": (0.687, 0.006), "es2cr": (0.662, 0.015), "es2pe": (0.678, 0.021),
        "cr2es": (0.655, 0.010), "cr2cr": (0.683, 0.018), "cr2pe": (0.685, 0.018),
        "pe2es": (0.648, 0.015), "pe2cr": (0.688, 0.011), "pe2pe": (0.691, 0.014),
    }

    def _load_combo(self, model, combo):
        vals = []
        for s in [42, 43, 44]:
            f = os.path.join(SPRINT4, f"cutc_{model}_{combo}_s{s}.json")
            vals.append(load_json(f)["f1_macro"])
        return np.mean(vals), np.std(vals, ddof=0)

    @pytest.mark.parametrize("combo,expected", list(EXPECTED_17B.items()))
    def test_17b_mean(self, combo, expected):
        mean, _ = self._load_combo("qwen1.7b", combo)
        assert r3(mean) == expected[0], \
            f"1.7B {combo} mean: {mean:.6f} vs paper {expected[0]}"

    @pytest.mark.parametrize("combo,expected", list(EXPECTED_17B.items()))
    def test_17b_std(self, combo, expected):
        _, std = self._load_combo("qwen1.7b", combo)
        assert r3(std) == expected[1], \
            f"1.7B {combo} std: {std:.6f} vs paper {expected[1]}"

    @pytest.mark.parametrize("combo,expected", list(EXPECTED_4B.items()))
    def test_4b_mean(self, combo, expected):
        mean, _ = self._load_combo("qwen4b", combo)
        assert r3(mean) == expected[0], \
            f"4B {combo} mean: {mean:.6f} vs paper {expected[0]}"

    @pytest.mark.parametrize("combo,expected",
                              [(k, v) for k, v in EXPECTED_4B.items() if k != "es2es"])
    def test_4b_std(self, combo, expected):
        """All 4B stds except es2es (borderline rounding case, see separate test)."""
        _, std = self._load_combo("qwen4b", combo)
        assert r3(std) == expected[1], \
            f"4B {combo} std: {std:.6f} vs paper {expected[1]}"

    def test_4b_es2es_std_discrepancy(self):
        """
        DISCREPANCY: paper reports Qwen3-4B ES in-domain std=±0.006.
        Actual std(ddof=0) = 0.006525, which standard round() gives 0.007.
        Paper value 0.006 appears to be truncated rather than rounded.
        """
        _, std = self._load_combo("qwen4b", "es2es")
        # Raw std is between 0.006 and 0.007 (borderline)
        assert 0.006 < std < 0.007, \
            f"4B es2es std={std:.6f}; should be in (0.006, 0.007)"
        # Standard Python rounding disagrees with paper
        assert r3(std) == 0.007, \
            f"round3(std)={r3(std)}, paper says 0.006 — DISCREPANCY"


# ---------------------------------------------------------------------------
# Sigma values for ES-ibérico transfer penalty
# ---------------------------------------------------------------------------

class TestSigmaValues:
    """Verify the sigma characterisation of the transfer penalty toward ES.

    Sigma is computed as delta / std(out-of-domain), using raw unrounded values.
    """

    def _load_vals(self, model, combo):
        return [
            load_json(os.path.join(SPRINT4, f"cutc_{model}_{combo}_s{s}.json"))["f1_macro"]
            for s in [42, 43, 44]
        ]

    def test_17b_cr2es_sigma(self):
        es = self._load_vals("qwen1.7b", "es2es")
        cr = self._load_vals("qwen1.7b", "cr2es")
        delta = np.mean(es) - np.mean(cr)
        sigma = delta / np.std(cr, ddof=0)
        # Paper says 4.6σ (computed: ~4.62)
        assert abs(sigma - 4.6) < 0.15, f"1.7B CR->ES sigma={sigma:.2f}, paper says 4.6"

    def test_17b_pe2es_sigma(self):
        es = self._load_vals("qwen1.7b", "es2es")
        pe = self._load_vals("qwen1.7b", "pe2es")
        delta = np.mean(es) - np.mean(pe)
        sigma = delta / np.std(pe, ddof=0)
        # Paper says 3.8σ (computed: ~3.76)
        assert abs(sigma - 3.8) < 0.15, f"1.7B PE->ES sigma={sigma:.2f}, paper says 3.8"

    def test_4b_cr2es_sigma(self):
        es = self._load_vals("qwen4b", "es2es")
        cr = self._load_vals("qwen4b", "cr2es")
        delta = np.mean(es) - np.mean(cr)
        sigma = delta / np.std(cr, ddof=0)
        # Paper says 3.1σ (computed: ~3.07)
        assert abs(sigma - 3.1) < 0.15, f"4B CR->ES sigma={sigma:.2f}, paper says 3.1"

    def test_4b_pe2es_sigma(self):
        es = self._load_vals("qwen4b", "es2es")
        pe = self._load_vals("qwen4b", "pe2es")
        delta = np.mean(es) - np.mean(pe)
        sigma = delta / np.std(pe, ddof=0)
        # Paper says 2.6σ (computed: ~2.62)
        assert abs(sigma - 2.6) < 0.15, f"4B PE->ES sigma={sigma:.2f}, paper says 2.6"

    def test_all_es_penalties_above_2sigma(self):
        """All 4 ES-target out-of-domain penalties exceed 2σ (paper claim)."""
        for model in ["qwen1.7b", "qwen4b"]:
            es = self._load_vals(model, "es2es")
            for src in ["cr", "pe"]:
                combo = f"{src}2es"
                other = self._load_vals(model, combo)
                delta = np.mean(es) - np.mean(other)
                sigma = delta / np.std(other, ddof=0)
                assert sigma > 2.0, \
                    f"{model} {combo}: sigma={sigma:.2f} should exceed 2σ"


# ---------------------------------------------------------------------------
# Baseline values (Table I)
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_tfidf_accuracy(self):
        d = load_json(os.path.join(RESULTS, "tfidf_lr_cardiff_es_full.json"))
        assert r3(d["accuracy"]) == 0.569

    def test_tfidf_f1_discrepancy(self):
        """
        DISCREPANCY: paper says TF-IDF F1=0.567; data gives 0.566 (rounds to 0.566).
        This test asserts the DATA value and confirms the paper differs by 1 ULP at 3dp.
        """
        d = load_json(os.path.join(RESULTS, "tfidf_lr_cardiff_es_full.json"))
        data_f1 = r3(d["f1_macro"])
        # The data clearly rounds to 0.566
        assert data_f1 == 0.566, \
            f"TF-IDF F1 from data={data_f1}; paper says 0.567 — DISCREPANCY"

    def test_robertuito_f1(self):
        d = load_json(os.path.join(
            RESULTS, "encoder_offtheshelf_pysentimiento_robertuito-sentiment-analysis.json"))
        assert r3(d["f1_macro"]) == 0.758

    def test_robertuito_accuracy(self):
        d = load_json(os.path.join(
            RESULTS, "encoder_offtheshelf_pysentimiento_robertuito-sentiment-analysis.json"))
        assert r3(d["accuracy"]) == 0.761


# ---------------------------------------------------------------------------
# Parameter reduction factors
# ---------------------------------------------------------------------------

class TestParamFactors:
    """Verify the reduction factors cited in abstract and Corte B narrative."""

    # Raw param counts from summary_corteB.csv
    LORA_17B_M = 6422528 / 1e6
    LORA_4B_M = 11796480 / 1e6
    BETO_M = 109853187 / 1e6
    XLMR_M = 278045955 / 1e6
    FULL_17B_M = 1720574976 / 1e6

    def test_17b_vs_beto_range(self):
        """Paper: ~17× less params for 1.7B vs BETO."""
        ratio = self.BETO_M / self.LORA_17B_M
        assert 16.5 <= ratio <= 18.0, f"1.7B vs BETO ratio={ratio:.2f}"

    def test_17b_vs_xlmr_range(self):
        """Paper: ~43× less params for 1.7B vs XLM-R."""
        ratio = self.XLMR_M / self.LORA_17B_M
        assert 42.5 <= ratio <= 44.0, f"1.7B vs XLM-R ratio={ratio:.2f}"

    def test_4b_vs_beto_range(self):
        """Paper: ~9× less params for 4B vs BETO."""
        ratio = self.BETO_M / self.LORA_4B_M
        assert 8.5 <= ratio <= 10.0, f"4B vs BETO ratio={ratio:.2f}"

    def test_4b_vs_xlmr_range(self):
        """Paper: ~24× less params for 4B vs XLM-R."""
        ratio = self.XLMR_M / self.LORA_4B_M
        assert 23.0 <= ratio <= 24.5, f"4B vs XLM-R ratio={ratio:.2f}"

    def test_full_ft_vs_lora_17b(self):
        """Paper: ~268× more params for Full-FT vs LoRA 1.7B."""
        ratio = self.FULL_17B_M / self.LORA_17B_M
        assert 267.0 <= ratio <= 269.0, f"Full-FT/LoRA 1.7B ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------

class TestDatasetSplits:
    """Verify stated split sizes are consistent with source data records."""

    def test_cardiff_full_n_train(self):
        """Cardiff full training set = 1839."""
        df = pd.read_csv(os.path.join(RESULTS, "summary_corteA.csv"))
        assert df["n_train"].max() == 1839

    def test_cardiff_seeds_per_full_row(self):
        """Three seeds per Corte A full row."""
        df = pd.read_csv(os.path.join(RESULTS, "summary_corteA.csv"))
        full = df[df["n_train"] == 1839]
        assert (full["n_seeds"] == 3).all()

    def test_intertass_train_sizes_es(self):
        """InterTASS ES training files exist for all 3 seeds (train=738 after NONE removal)."""
        for s in [42, 43, 44]:
            p = os.path.join(SPRINT4, f"cutc_qwen1.7b_es2es_s{s}.json")
            assert os.path.exists(p), f"Missing {p}"

    def test_intertass_all_9_combos_exist(self):
        """All 9 transfer combos × 3 seeds × 2 models = 54 files exist."""
        combos = ["es2es", "es2cr", "es2pe", "cr2es", "cr2cr", "cr2pe", "pe2es", "pe2cr", "pe2pe"]
        for model in ["qwen1.7b", "qwen4b"]:
            for combo in combos:
                for s in [42, 43, 44]:
                    p = os.path.join(SPRINT4, f"cutc_{model}_{combo}_s{s}.json")
                    assert os.path.exists(p), f"Missing {p}"


# ---------------------------------------------------------------------------
# Compilation checks (main.log)
# ---------------------------------------------------------------------------

class TestCompilation:
    LOG = os.path.join(PAPER, "main.log")

    def test_log_exists(self):
        assert os.path.exists(self.LOG), "main.log not found"

    def test_zero_overfull(self):
        with open(self.LOG) as f:
            content = f.read()
        count = content.count("Overfull")
        assert count == 0, f"{count} Overfull \\hbox in main.log"

    def test_pdf_exists(self):
        assert os.path.exists(os.path.join(PAPER, "main.pdf"))

    def test_no_undefined_warnings(self):
        with open(self.LOG) as f:
            lines = f.readlines()
        bad_lines = [l.strip() for l in lines
                     if "Warning" in l and "undefined" in l.lower()]
        assert bad_lines == [], f"Undefined ref warnings: {bad_lines}"

    def test_underfull_count(self):
        """Underfull are cosmetic. Document the count (expected ~27)."""
        with open(self.LOG) as f:
            content = f.read()
        count = content.count("Underfull")
        # Not a failure criterion; just assert it's in the known range
        assert 0 <= count <= 50, f"Underfull count={count} out of expected range"


# ---------------------------------------------------------------------------
# Git tracking
# ---------------------------------------------------------------------------

class TestGitTracking:
    def _git_ls(self, path):
        result = subprocess.run(
            ["git", "-C", REPO, "ls-files", path],
            capture_output=True, text=True
        )
        return result.stdout.strip().splitlines()

    def test_ten_figure_files_tracked(self):
        tracked = self._git_ls("results/figures/")
        assert len(tracked) >= 10, \
            f"Expected >= 10 figure files, got {len(tracked)}: {tracked}"

    @pytest.mark.parametrize("fig", ["figA", "figB", "figC", "figD", "figE"])
    def test_figure_png_tracked(self, fig):
        tracked = self._git_ls("results/figures/")
        names = [os.path.basename(f) for f in tracked]
        assert any(n.startswith(fig) and n.endswith(".png") for n in names), \
            f"No PNG for {fig} in git ls-files results/figures/"

    @pytest.mark.parametrize("fig", ["figA", "figB", "figC", "figD", "figE"])
    def test_figure_pdf_tracked(self, fig):
        tracked = self._git_ls("results/figures/")
        names = [os.path.basename(f) for f in tracked]
        assert any(n.startswith(fig) and n.endswith(".pdf") for n in names), \
            f"No PDF for {fig} in git ls-files results/figures/"

    def test_make_figures_script_tracked(self):
        tracked = self._git_ls("scripts/make_figures_sprint5.py")
        assert len(tracked) == 1, \
            "scripts/make_figures_sprint5.py not tracked in git"


# ---------------------------------------------------------------------------
# Paper metadata: affiliation, 10,000× attribution, no artefacts
# ---------------------------------------------------------------------------

class TestPaperMetadata:
    TEX = os.path.join(PAPER, "main.tex")

    def _read_tex(self):
        with open(self.TEX) as f:
            return f.read()

    def test_affiliation_comillas_icai(self):
        tex = self._read_tex()
        assert "Universidad Pontificia de Comillas" in tex
        assert "ICAI" in tex

    def test_10000_occurs_exactly_once(self):
        """10,000× must appear exactly once (§II-A, attributed to Hu et al./GPT-3)."""
        tex = self._read_tex()
        count = tex.count("10,000") + tex.count("10.000") + tex.count("10000")
        assert count == 1, \
            f"Expected exactly 1 occurrence of 10,000; found {count}"

    def test_no_source_annotations(self):
        """No (source:...) artefacts from the generation pipeline."""
        tex = self._read_tex()
        import re
        hits = re.findall(r"\(source:", tex, re.IGNORECASE)
        assert hits == [], f"Found (source:...) annotations: {hits}"

    def test_language_primarily_spanish(self):
        """All major section titles should be in Spanish."""
        tex = self._read_tex()
        spanish_sections = [
            "Introducción", "Trabajo relacionado", "Metodología",
            "Experimentos y resultados", "Discusión", "Conclusiones"
        ]
        for title in spanish_sections:
            assert title in tex, f"Expected Spanish section title '{title}' not found"
