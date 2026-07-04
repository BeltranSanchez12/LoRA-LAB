"""
test_cutc_grid_audit.py
QA-validator — Auditoría completa de la matriz 3x3 de transferencia dialectal LoRA
Sprint 4 / Corte C.

Cubre 5 puntos:
  P1 – Completitud: 54 celdas, sin huecos ni duplicados, 3 semillas exactas por celda.
  P2 – Trazabilidad: notes.train_dir / notes.eval_dir concuerdan con el nombre del fichero;
       n_test coincide con el split test del país objetivo.
  P3 – Agregación: recomputa media y std (ddof=0) desde los JSON y compara con los CSV.
  P4 – Sanidad: fallback_rate == 0.0 en todas las celdas.
  P5 – Varianza/significancia: para cada celda fuera de la diagonal, compara el gap de
       transferencia (diag_objetivo − celda_media) contra 2·max(σ_celda, σ_diag).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESDIR = ROOT / "results" / "sprint4"

TAGS   = ["qwen1.7b", "qwen4b"]
SRCS   = ["es", "cr", "pe"]
TGTS   = ["es", "cr", "pe"]
SEEDS  = [42, 43, 44]

# Tamaños de test esperados por país objetivo (dev de InterTASS 2018)
EXPECTED_N_TEST = {"es": 444, "cr": 242, "pe": 262}

FNAME_RE = re.compile(
    r"cutc_(?P<tag>qwen[\d.]+b)_(?P<src>es|cr|pe)2(?P<tgt>es|cr|pe)_s(?P<seed>\d+)"
)
NOTES_RE_TRAIN = re.compile(r"train_dir=tass_(\w+)")
NOTES_RE_EVAL  = re.compile(r"eval_dir=tass_(\w+)")
NOTES_RE_NTEST = re.compile(r"n_test=(\d+)")

TOL_F1  = 1e-4   # tolerancia para comparar medias y stds con CSV
TOL_GAP = 2      # umbral en múltiplos de sigma para considerar gap "significativo"


# ---------------------------------------------------------------------------
# Fixture: carga todos los JSON una vez
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def all_cells() -> dict:
    """
    Devuelve un dict keyed por (tag, src, tgt, seed) con el contenido del JSON.
    """
    cells = {}
    for jf in sorted(RESDIR.glob("cutc_*.json")):
        m = FNAME_RE.match(jf.stem)
        if not m:
            continue
        key = (m["tag"], m["src"], m["tgt"], int(m["seed"]))
        cells[key] = json.loads(jf.read_text())
    return cells


# ---------------------------------------------------------------------------
# P1 – COMPLETITUD
# ---------------------------------------------------------------------------
class TestP1Completeness:

    def test_total_count(self, all_cells):
        """Deben existir exactamente 54 celdas (2×3×3×3)."""
        assert len(all_cells) == 54, (
            f"Se esperaban 54 celdas, se encontraron {len(all_cells)}. "
            f"Claves faltantes o duplicadas."
        )

    def test_no_gaps(self, all_cells):
        """Cada combinación (tag, src, tgt, seed) debe existir."""
        missing = []
        for tag in TAGS:
            for src in SRCS:
                for tgt in TGTS:
                    for seed in SEEDS:
                        if (tag, src, tgt, seed) not in all_cells:
                            missing.append((tag, src, tgt, seed))
        assert not missing, f"Celdas faltantes: {missing}"

    def test_exactly_3_seeds_per_cell(self, all_cells):
        """Cada (tag, src, tgt) debe tener exactamente 3 semillas."""
        counts = defaultdict(int)
        for (tag, src, tgt, seed) in all_cells:
            counts[(tag, src, tgt)] += 1
        wrong = {k: v for k, v in counts.items() if v != 3}
        assert not wrong, f"Celdas con != 3 semillas: {wrong}"

    def test_no_duplicates(self, all_cells):
        """No debe haber duplicados de (tag, src, tgt, seed)."""
        # Si el fixture no los duplica al cargar, este test confirma que
        # tampoco había dos ficheros para la misma clave (sobrescritura silenciosa).
        # Contamos ficheros directamente.
        keys_from_files = []
        for jf in RESDIR.glob("cutc_*.json"):
            m = FNAME_RE.match(jf.stem)
            if m:
                keys_from_files.append((m["tag"], m["src"], m["tgt"], int(m["seed"])))
        assert len(keys_from_files) == len(set(keys_from_files)), (
            "Ficheros duplicados detectados (misma clave, distinto fichero)."
        )


# ---------------------------------------------------------------------------
# P2 – TRAZABILIDAD
# ---------------------------------------------------------------------------
class TestP2Traceability:

    def test_train_dir_matches_filename_src(self, all_cells):
        """notes.train_dir debe coincidir con src del nombre del fichero."""
        mismatches = []
        for (tag, src, tgt, seed), d in all_cells.items():
            notes = d.get("notes", "")
            m = NOTES_RE_TRAIN.search(notes)
            if not m:
                mismatches.append(((tag, src, tgt, seed), "train_dir_not_found"))
                continue
            if m.group(1) != src:
                mismatches.append(((tag, src, tgt, seed), f"train_dir={m.group(1)} vs src={src}"))
        assert not mismatches, f"Desajustes train_dir vs src: {mismatches}"

    def test_eval_dir_matches_filename_tgt(self, all_cells):
        """notes.eval_dir debe coincidir con tgt del nombre del fichero."""
        mismatches = []
        for (tag, src, tgt, seed), d in all_cells.items():
            notes = d.get("notes", "")
            m = NOTES_RE_EVAL.search(notes)
            if not m:
                mismatches.append(((tag, src, tgt, seed), "eval_dir_not_found"))
                continue
            if m.group(1) != tgt:
                mismatches.append(((tag, src, tgt, seed), f"eval_dir={m.group(1)} vs tgt={tgt}"))
        assert not mismatches, f"Desajustes eval_dir vs tgt: {mismatches}"

    def test_n_test_matches_expected_target_size(self, all_cells):
        """n_test en notes debe coincidir con el tamaño del split test del país objetivo."""
        mismatches = []
        for (tag, src, tgt, seed), d in all_cells.items():
            notes = d.get("notes", "")
            m = NOTES_RE_NTEST.search(notes)
            if not m:
                mismatches.append(((tag, src, tgt, seed), "n_test_not_found_in_notes"))
                continue
            actual = int(m.group(1))
            expected = EXPECTED_N_TEST[tgt]
            if actual != expected:
                mismatches.append(
                    ((tag, src, tgt, seed), f"n_test={actual} vs expected={expected} for tgt={tgt}")
                )
        assert not mismatches, f"Desajustes n_test vs tamaño de split objetivo: {mismatches}"

    def test_f1_varies_across_eval_targets_same_training(self, all_cells):
        """
        Para cada (tag, src, seed), los f1_macro evaluados en los 3 países distintos
        no deben ser todos idénticos. Detecta si se reutilizó la evaluación del mismo
        país para todos los targets.
        """
        groups = defaultdict(dict)
        for (tag, src, tgt, seed), d in all_cells.items():
            groups[(tag, src, seed)][tgt] = d["f1_macro"]

        identical_groups = []
        for (tag, src, seed), tgt_f1 in groups.items():
            vals = list(tgt_f1.values())
            if len(set(round(v, 8) for v in vals)) == 1:
                identical_groups.append(((tag, src, seed), vals))

        assert not identical_groups, (
            f"FUGA DE EVALUACIÓN: los siguientes grupos tienen F1 idéntico "
            f"en los 3 targets (misma evaluación reusada): {identical_groups}"
        )


# ---------------------------------------------------------------------------
# P3 – AGREGACIÓN
# ---------------------------------------------------------------------------
class TestP3Aggregation:

    @pytest.fixture(scope="class")
    def recomputed(self, all_cells):
        """Recomputa media y std (ddof=0) desde los JSON."""
        result = {}
        for tag in TAGS:
            means = {}
            stds = {}
            for src in SRCS:
                for tgt in TGTS:
                    vals = [
                        all_cells[(tag, src, tgt, s)]["f1_macro"]
                        for s in SEEDS
                        if (tag, src, tgt, s) in all_cells
                    ]
                    means[(src, tgt)] = float(np.mean(vals))
                    stds[(src, tgt)]  = float(np.std(vals, ddof=0))
            result[tag] = {"mean": means, "std": stds}
        return result

    def _load_csv(self, tag: str) -> pd.DataFrame:
        fname = "qwen1.7b" if "1.7" in tag else tag
        path = RESDIR / f"transfer_matrix_{fname}.csv"
        assert path.exists(), f"CSV no encontrado: {path}"
        return pd.read_csv(path)

    def test_means_match_csv_qwen17b(self, recomputed):
        """Media recomputada desde JSON vs CSV para Qwen3-1.7B."""
        df = self._load_csv("qwen1.7b")
        cc = {"ES": "es", "CR": "cr", "PE": "pe"}
        mismatches = []
        for _, row in df.iterrows():
            src = cc[row["train"]]
            tgt = cc[row["eval"]]
            computed = recomputed["qwen1.7b"]["mean"][(src, tgt)]
            csv_val  = float(row["f1_macro_mean"])
            if abs(computed - csv_val) > TOL_F1:
                mismatches.append(
                    f"qwen1.7b {src}->{tgt}: recomputed={computed:.6f} csv={csv_val:.6f} "
                    f"diff={abs(computed-csv_val):.6f}"
                )
        assert not mismatches, f"Medias no coinciden:\n" + "\n".join(mismatches)

    def test_stds_match_csv_qwen17b(self, recomputed):
        """Std (ddof=0) recomputado desde JSON vs CSV para Qwen3-1.7B."""
        df = self._load_csv("qwen1.7b")
        cc = {"ES": "es", "CR": "cr", "PE": "pe"}
        mismatches = []
        for _, row in df.iterrows():
            src = cc[row["train"]]
            tgt = cc[row["eval"]]
            computed = recomputed["qwen1.7b"]["std"][(src, tgt)]
            csv_val  = float(row["f1_macro_std"])
            if abs(computed - csv_val) > TOL_F1:
                mismatches.append(
                    f"qwen1.7b {src}->{tgt}: recomputed_std={computed:.6f} csv_std={csv_val:.6f} "
                    f"diff={abs(computed-csv_val):.6f}"
                )
        assert not mismatches, f"Stds no coinciden (ddof=0):\n" + "\n".join(mismatches)

    def test_means_match_csv_qwen4b(self, recomputed):
        """Media recomputada desde JSON vs CSV para Qwen3-4B."""
        df = self._load_csv("qwen4b")
        cc = {"ES": "es", "CR": "cr", "PE": "pe"}
        mismatches = []
        for _, row in df.iterrows():
            src = cc[row["train"]]
            tgt = cc[row["eval"]]
            computed = recomputed["qwen4b"]["mean"][(src, tgt)]
            csv_val  = float(row["f1_macro_mean"])
            if abs(computed - csv_val) > TOL_F1:
                mismatches.append(
                    f"qwen4b {src}->{tgt}: recomputed={computed:.6f} csv={csv_val:.6f} "
                    f"diff={abs(computed-csv_val):.6f}"
                )
        assert not mismatches, f"Medias no coinciden:\n" + "\n".join(mismatches)

    def test_stds_match_csv_qwen4b(self, recomputed):
        """Std (ddof=0) recomputado desde JSON vs CSV para Qwen3-4B."""
        df = self._load_csv("qwen4b")
        cc = {"ES": "es", "CR": "cr", "PE": "pe"}
        mismatches = []
        for _, row in df.iterrows():
            src = cc[row["train"]]
            tgt = cc[row["eval"]]
            computed = recomputed["qwen4b"]["std"][(src, tgt)]
            csv_val  = float(row["f1_macro_std"])
            if abs(computed - csv_val) > TOL_F1:
                mismatches.append(
                    f"qwen4b {src}->{tgt}: recomputed_std={computed:.6f} csv_std={csv_val:.6f} "
                    f"diff={abs(computed-csv_val):.6f}"
                )
        assert not mismatches, f"Stds no coinciden (ddof=0):\n" + "\n".join(mismatches)

    def test_n_seeds_is_3_in_csv(self):
        """n_seeds debe ser 3 en todas las filas del CSV."""
        for tag in ["qwen1.7b", "qwen4b"]:
            df = self._load_csv(tag)
            wrong = df[df["n_seeds"] != 3]
            assert wrong.empty, f"{tag}: filas con n_seeds != 3:\n{wrong}"

    def test_diagonal_flag_correct(self):
        """El flag diagonal debe ser True iff train==eval."""
        for tag in ["qwen1.7b", "qwen4b"]:
            df = self._load_csv(tag)
            for _, row in df.iterrows():
                expected_diag = (row["train"] == row["eval"])
                assert bool(row["diagonal"]) == expected_diag, (
                    f"{tag}: diagonal={row['diagonal']} pero train={row['train']} eval={row['eval']}"
                )


# ---------------------------------------------------------------------------
# P4 – SANIDAD (fallback_rate)
# ---------------------------------------------------------------------------
class TestP4Sanity:

    def test_fallback_rate_zero_all_cells(self, all_cells):
        """fallback_rate debe ser exactamente 0.0 en todas las 54 celdas."""
        nonzero = []
        for (tag, src, tgt, seed), d in all_cells.items():
            fr = d.get("fallback_rate")
            if fr is None:
                nonzero.append(((tag, src, tgt, seed), "MISSING"))
            elif fr != 0.0:
                nonzero.append(((tag, src, tgt, seed), fr))
        assert not nonzero, f"Celdas con fallback_rate != 0: {nonzero}"

    def test_fallback_rate_max_reported(self, all_cells):
        """Informe del máximo fallback_rate observado (siempre pasa; solo reporta)."""
        rates = [d.get("fallback_rate", 0.0) for d in all_cells.values()]
        max_rate = max(rates)
        # Este test siempre pasa; el valor queda en el output de pytest -s
        print(f"\n[P4] max fallback_rate = {max_rate}")
        assert True


# ---------------------------------------------------------------------------
# P5 – VARIANZA / SIGNIFICANCIA ENTRE SEMILLAS
# ---------------------------------------------------------------------------
class TestP5Significance:
    """
    Para cada celda fuera de la diagonal, calcula:
        gap   = mean(diag_objetivo) − mean(celda)
        sigma = max(std(diag_objetivo), std(celda))   [ddof=0, sobre 3 semillas]

    Si gap > TOL_GAP * sigma → la penalización por transferencia supera el ruido.
    """

    @pytest.fixture(scope="class")
    def stats(self, all_cells):
        result = {}
        for tag in TAGS:
            means = {}
            stds  = {}
            for src in SRCS:
                for tgt in TGTS:
                    vals = [all_cells[(tag, src, tgt, s)]["f1_macro"] for s in SEEDS]
                    means[(src, tgt)] = float(np.mean(vals))
                    stds[(src, tgt)]  = float(np.std(vals, ddof=0))
            result[tag] = {"mean": means, "std": stds}
        return result

    def _off_diagonal_analysis(self, tag: str, stats: dict) -> list[dict]:
        """Devuelve lista de dicts con el análisis gap/sigma para celdas off-diagonal."""
        rows = []
        means = stats[tag]["mean"]
        stds  = stats[tag]["std"]
        for tgt in TGTS:
            diag_mean = means[(tgt, tgt)]
            diag_std  = stds[(tgt, tgt)]
            for src in SRCS:
                if src == tgt:
                    continue
                cell_mean = means[(src, tgt)]
                cell_std  = stds[(src, tgt)]
                gap   = diag_mean - cell_mean
                sigma = max(diag_std, cell_std)
                rows.append({
                    "tag": tag, "src": src, "tgt": tgt,
                    "gap": round(gap, 4),
                    "sigma": round(sigma, 4),
                    "ratio": round(gap / sigma, 2) if sigma > 0 else float("inf"),
                    "exceeds_2sigma": (gap > TOL_GAP * sigma),
                })
        return rows

    def test_print_gap_analysis_qwen17b(self, stats, capsys):
        """Imprime tabla de gaps vs 2σ para Qwen3-1.7B (siempre pasa)."""
        rows = self._off_diagonal_analysis("qwen1.7b", stats)
        print(f"\n[P5] Qwen3-1.7B — gap de transferencia (diag_tgt − celda) vs 2σ:")
        print(f"{'src':>3} → {'tgt':>3}  gap     σ      gap/σ  >2σ?")
        for r in rows:
            print(
                f"{r['src']:>3} → {r['tgt']:>3}  {r['gap']:+.4f}  {r['sigma']:.4f}  "
                f"{r['ratio']:>5.2f}   {'YES' if r['exceeds_2sigma'] else 'no'}"
            )
        exceeds = [r for r in rows if r["exceeds_2sigma"]]
        print(f"  Celdas que superan 2σ: {len(exceeds)}/6")
        assert True  # Este test siempre pasa; el veredicto queda en el output

    def test_print_gap_analysis_qwen4b(self, stats, capsys):
        """Imprime tabla de gaps vs 2σ para Qwen3-4B (siempre pasa)."""
        rows = self._off_diagonal_analysis("qwen4b", stats)
        print(f"\n[P5] Qwen3-4B — gap de transferencia (diag_tgt − celda) vs 2σ:")
        print(f"{'src':>3} → {'tgt':>3}  gap     σ      gap/σ  >2σ?")
        for r in rows:
            print(
                f"{r['src']:>3} → {r['tgt']:>3}  {r['gap']:+.4f}  {r['sigma']:.4f}  "
                f"{r['ratio']:>5.2f}   {'YES' if r['exceeds_2sigma'] else 'no'}"
            )
        exceeds = [r for r in rows if r["exceeds_2sigma"]]
        print(f"  Celdas que superan 2σ: {len(exceeds)}/6")
        assert True

    def test_gap_stats_collected(self, stats):
        """
        Verifica que el análisis pueda ejecutarse sobre todas las celdas sin errores.
        (test de smoke para la función de análisis en sí)
        """
        for tag in TAGS:
            rows = self._off_diagonal_analysis(tag, stats)
            assert len(rows) == 6, f"{tag}: se esperaban 6 celdas off-diagonal, se obtuvieron {len(rows)}"
            for r in rows:
                assert "gap" in r and "sigma" in r and "ratio" in r
