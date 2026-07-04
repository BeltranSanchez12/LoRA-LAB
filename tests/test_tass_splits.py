"""
tests/test_tass_splits.py — QA-Validator: auditoría de splits InterTASS 2018 (Sprint 4).

Cubre los 4 puntos de aceptación originales más deduplicación (Sprint 4, rev.2):
  1. Mapeo de polaridades y alineación con LABEL2ID del proyecto.
  2. Descarte de NONE (ningún ejemplo NONE en los splits).
  3. 'test' en disco == fichero -development-tagged.xml mapeado.
  4. Validación de hold-out: solapamientos, fuentes, estratificación.
  5. Deduplicación por texto exacto (primera aparición): metadata registra
     train_duplicates_dropped / dev_duplicates_dropped; ningún texto aparece
     más de una vez dentro de cada split en disco.

El reparser independiente aplica la misma política de deduplicación que el pipeline
(primera aparición de cada texto exacto), de modo que los conteos coincidan con el disco.

Usa parsers propios (sin importar src/) para contrastar con lo guardado en disco.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Rutas (absolutas, sin depender de cwd)
# ---------------------------------------------------------------------------
PROJECT = Path("/home/jovyan/jupyterlab_container/AI_LAB")
RAW_DIR = PROJECT / "data" / "raw" / "tass"
PROC_DIR = PROJECT / "data" / "processed"

COUNTRIES = ("ES", "CR", "PE")

# ---------------------------------------------------------------------------
# LABEL2ID de referencia independiente (debe coincidir con load_data.py)
# ---------------------------------------------------------------------------
LABEL2ID_REF = {"negative": 0, "neutral": 1, "positive": 2}
POLARITY_MAP_REF = {"P": "positive", "N": "negative", "NEU": "neutral"}
# Mapeo directo polaridad XML -> label int (para aserciones precisas)
POL2INT = {pol: LABEL2ID_REF[canon] for pol, canon in POLARITY_MAP_REF.items()}


# ---------------------------------------------------------------------------
# Parser independiente (sin reutilizar src/)
# Aplica la misma política que el pipeline:
#   - descarta NONE y vacíos
#   - deduplica por texto exacto, conservando la primera aparición
# ---------------------------------------------------------------------------

def _parse_xml(path: Path) -> dict:
    """Parser independiente con deduplicación por texto exacto (primera aparición).

    Retorna
    -------
    dict con:
        "examples"   : list[{"text", "label", "polarity"}]  (sin NONE, sin duplicados)
        "raw_counts" : Counter  (conteo bruto de polaridades, incluido NONE)
        "n_none"     : int  (NONE descartados)
        "n_empty"    : int  (tweets sin texto)
        "n_dup"      : int  (copias exactas eliminadas tras la primera aparición)
    """
    tree = ET.parse(str(path))
    root = tree.getroot()
    examples, raw_counts = [], Counter()
    seen_texts: set[str] = set()
    n_none = n_empty = n_dup = 0

    for tweet in root.findall(".//tweet"):
        content_el = tweet.find("content")
        text = (content_el.text or "").strip() if content_el is not None else ""
        value_el = tweet.find("./sentiment/polarity/value")
        polarity = (value_el.text or "").strip() if value_el is not None else ""

        raw_counts[polarity] += 1

        if not text:
            n_empty += 1
            continue
        if polarity == "NONE":
            n_none += 1
            continue
        if polarity not in POLARITY_MAP_REF:
            continue
        if text in seen_texts:
            n_dup += 1
            continue

        seen_texts.add(text)
        examples.append({"text": text, "label": POL2INT[polarity], "polarity": polarity})

    return {
        "examples": examples,
        "raw_counts": dict(raw_counts),
        "n_none": n_none,
        "n_empty": n_empty,
        "n_dup": n_dup,
    }


# ---------------------------------------------------------------------------
# Fixtures por país
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=COUNTRIES)
def country_data(request):
    cc = request.param
    train_path = RAW_DIR / f"intertass-{cc}-train-tagged.xml"
    dev_path   = RAW_DIR / f"intertass-{cc}-development-tagged.xml"
    assert train_path.exists(), f"Fichero no encontrado: {train_path}"
    assert dev_path.exists(),   f"Fichero no encontrado: {dev_path}"

    # Parsers independientes CON deduplicación, alineados con el pipeline
    train_r = _parse_xml(train_path)
    dev_r   = _parse_xml(dev_path)
    train_ex = train_r["examples"]       # ya deduplicados
    labels   = [e["label"] for e in train_ex]
    tr_idx, val_idx = train_test_split(
        list(range(len(train_ex))),
        test_size=0.15, random_state=42, stratify=labels,
    )
    train_sub = [train_ex[i] for i in tr_idx]
    val_sub   = [train_ex[i] for i in val_idx]

    meta_path = PROC_DIR / f"tass_{cc.lower()}" / "metadata.json"
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "cc": cc,
        "train_r": train_r,
        "dev_r": dev_r,
        "train_ex": train_ex,
        "train_sub": train_sub,
        "val_sub": val_sub,
        "dev_ex": dev_r["examples"],
        "meta": meta,
        "proc_dir": PROC_DIR / f"tass_{cc.lower()}",
    }


# ---------------------------------------------------------------------------
# PUNTO 1 — Mapeo de polaridades y alineación con LABEL2ID del proyecto
# ---------------------------------------------------------------------------

class TestLabelMapping:
    """P1: mapeo 3 clases correcto y alineado con el proyecto."""

    def test_polarity_map_values(self):
        """P->2, N->0, NEU->1 según LABEL2ID del proyecto."""
        assert POL2INT["P"]   == 2, "P debería ser positive=2"
        assert POL2INT["N"]   == 0, "N debería ser negative=0"
        assert POL2INT["NEU"] == 1, "NEU debería ser neutral=1"

    def test_label2id_from_source_matches_ref(self):
        """El LABEL2ID importado de load_data.py coincide con la referencia."""
        import sys
        sys.path.insert(0, str(PROJECT))
        from src.data.load_data import LABEL2ID as L2ID_SRC
        assert L2ID_SRC == LABEL2ID_REF, (
            f"LABEL2ID del proyecto: {L2ID_SRC}; referencia esperada: {LABEL2ID_REF}"
        )

    def test_polarity_map_in_source_matches_ref(self):
        """El POLARITY_MAP de load_tass.py coincide con la referencia de auditoría."""
        import sys
        sys.path.insert(0, str(PROJECT))
        from src.data.load_tass import POLARITY_MAP as PM_SRC
        assert PM_SRC == POLARITY_MAP_REF, (
            f"POLARITY_MAP de load_tass.py: {PM_SRC}; referencia: {POLARITY_MAP_REF}"
        )

    def test_all_disk_labels_in_valid_set(self, country_data):
        """Todos los labels guardados en disco están en {0, 1, 2}."""
        cc   = country_data["cc"]
        pdir = country_data["proc_dir"]
        for split_name in ("train", "validation", "test"):
            ds = load_from_disk(str(pdir / split_name))
            bad = [l for l in ds["label"] if l not in (0, 1, 2)]
            assert not bad, (
                f"[{cc}] split '{split_name}' contiene labels fuera de {{0,1,2}}: {set(bad)}"
            )


# ---------------------------------------------------------------------------
# PUNTO 2 — NONE descartado por completo
# ---------------------------------------------------------------------------

class TestNoneDiscarded:
    """P2: ningún ejemplo NONE acaba como ejemplo etiquetado."""

    def test_no_none_in_parsed_train(self, country_data):
        """El parser independiente no incluye NONE en train_sub ni val_sub."""
        cc = country_data["cc"]
        for split_name, split in (("train_sub", country_data["train_sub"]),
                                   ("val_sub",   country_data["val_sub"])):
            for ex in split:
                assert ex["polarity"] != "NONE", (
                    f"[{cc}] {split_name}: ejemplo con polarity NONE encontrado"
                )

    def test_no_none_in_parsed_dev(self, country_data):
        """El parser independiente no incluye NONE en dev_ex (test en disco)."""
        cc = country_data["cc"]
        for ex in country_data["dev_ex"]:
            assert ex["polarity"] != "NONE", (
                f"[{cc}] dev_ex: ejemplo con polarity NONE encontrado"
            )

    def test_none_count_matches_metadata(self, country_data):
        """Los NONE descartados por el parser independiente coinciden con metadata.json."""
        cc   = country_data["cc"]
        meta = country_data["meta"]
        raw  = meta["raw"]
        assert country_data["train_r"]["n_none"] == raw["train_none_discarded"], (
            f"[{cc}] NONE train: auditor={country_data['train_r']['n_none']} "
            f"vs meta={raw['train_none_discarded']}"
        )
        assert country_data["dev_r"]["n_none"] == raw["dev_none_discarded"], (
            f"[{cc}] NONE dev: auditor={country_data['dev_r']['n_none']} "
            f"vs meta={raw['dev_none_discarded']}"
        )

    def test_metadata_flags_none_discarded(self, country_data):
        """metadata.json documenta none_discarded=True."""
        assert country_data["meta"]["none_discarded"] is True, (
            f"[{country_data['cc']}] metadata.json.none_discarded no es True"
        )


# ---------------------------------------------------------------------------
# PUNTO 3 — 'test' en disco == fichero -development-tagged.xml mapeado
# ---------------------------------------------------------------------------

class TestTestEqualsDevFile:
    """P3: el split 'test' guardado en disco es el development mapeado."""

    def test_test_size_matches_dev(self, country_data):
        """Número de ejemplos en disco/test == dev parseado independientemente."""
        cc   = country_data["cc"]
        ds   = load_from_disk(str(country_data["proc_dir"] / "test"))
        assert len(ds) == len(country_data["dev_ex"]), (
            f"[{cc}] test en disco: {len(ds)}; dev parseado: {len(country_data['dev_ex'])}"
        )

    def test_test_texts_match_dev(self, country_data):
        """Los textos del split 'test' en disco coinciden (multiset) con el dev XML."""
        cc  = country_data["cc"]
        ds  = load_from_disk(str(country_data["proc_dir"] / "test"))
        disk_texts = Counter(ds["text"])
        dev_texts  = Counter(e["text"] for e in country_data["dev_ex"])
        assert disk_texts == dev_texts, (
            f"[{cc}] Los textos de 'test' en disco no coinciden con -development-tagged.xml"
        )

    def test_test_labels_match_dev(self, country_data):
        """Los labels del split 'test' en disco coinciden (multiset) con el dev XML."""
        cc  = country_data["cc"]
        ds  = load_from_disk(str(country_data["proc_dir"] / "test"))
        disk_labels = Counter(ds["label"])
        dev_labels  = Counter(e["label"] for e in country_data["dev_ex"])
        assert disk_labels == dev_labels, (
            f"[{cc}] Labels de 'test' en disco: {disk_labels}; "
            f"dev parseado: {dev_labels}"
        )

    def test_test_source_is_dev_not_train(self, country_data):
        """Ningún texto del 'test' procede del XML de train (fuentes separadas)."""
        cc = country_data["cc"]
        train_texts = {e["text"] for e in country_data["train_ex"]}
        test_texts  = {e["text"] for e in country_data["dev_ex"]}
        cross = train_texts & test_texts
        assert not cross, (
            f"[{cc}] {len(cross)} textos aparecen tanto en train XML como en dev XML"
        )


# ---------------------------------------------------------------------------
# PUNTO 4 — Validación del hold-out: solapamientos, fuentes, estratificación
# ---------------------------------------------------------------------------

class TestValidationHoldout:
    """P4: hold-out estratificado del 15% sin filtraciones."""

    def test_no_index_overlap_train_val(self, country_data):
        """(a) Los índices de train y val son disjuntos (no hay ejemplar compartido)."""
        cc = country_data["cc"]
        labels_all = [e["label"] for e in country_data["train_ex"]]
        tr_idx, val_idx = train_test_split(
            list(range(len(country_data["train_ex"]))),
            test_size=0.15, random_state=42, stratify=labels_all,
        )
        assert set(tr_idx).isdisjoint(set(val_idx)), (
            f"[{cc}] Los índices de train y val se solapan (fuga de datos)"
        )

    def test_no_text_overlap_train_val_after_dedup(self, country_data):
        """(a2) Tras deduplicación, ningún texto aparece a la vez en train_sub y val_sub.
        Con el parser deduplicado, cada texto es único en train_ex, por lo que train_test_split
        nunca puede asignarlo a ambos lados."""
        cc = country_data["cc"]
        train_texts = {e["text"] for e in country_data["train_sub"]}
        val_texts   = {e["text"] for e in country_data["val_sub"]}
        overlap = train_texts & val_texts
        assert not overlap, (
            f"[{cc}] {len(overlap)} textos aparecen tanto en train_sub como en val_sub "
            f"tras deduplicación: {list(overlap)[:3]}"
        )

    def test_no_index_overlap_train_test(self, country_data):
        """(b) train y test provienen de ficheros XML distintos (sin cruces de texto)."""
        cc = country_data["cc"]
        train_texts = {e["text"] for e in country_data["train_ex"]}
        test_texts  = {e["text"] for e in country_data["dev_ex"]}
        cross = train_texts & test_texts
        assert not cross, (
            f"[{cc}] {len(cross)} textos en train XML también en dev XML"
        )

    def test_val_size_is_15_percent(self, country_data):
        """El tamaño de val es aproximadamente el 15% del train+val combinado."""
        cc      = country_data["cc"]
        n_total = len(country_data["train_sub"]) + len(country_data["val_sub"])
        ratio   = len(country_data["val_sub"]) / n_total
        assert 0.13 <= ratio <= 0.17, (
            f"[{cc}] ratio val/total={ratio:.3f} fuera del rango esperado [0.13, 0.17]"
        )

    def test_stratification_preserves_class_proportions(self, country_data):
        """(c) La estratificación mantiene proporciones de clase dentro de ±3 p.p."""
        cc       = country_data["cc"]
        all_ex   = country_data["train_ex"]
        n_total  = len(all_ex)
        orig_dist = Counter(e["label"] for e in all_ex)
        val_dist  = Counter(e["label"] for e in country_data["val_sub"])
        n_val = len(country_data["val_sub"])

        for label in (0, 1, 2):
            p_orig = orig_dist[label] / n_total
            p_val  = val_dist[label] / n_val if n_val else 0
            diff   = abs(p_orig - p_val)
            assert diff <= 0.03, (
                f"[{cc}] label={label}: proporción original={p_orig:.3f} vs val={p_val:.3f} "
                f"(diferencia={diff:.3f} > 0.03 → estratificación rota)"
            )

    def test_disk_train_val_counts_match_parsed(self, country_data):
        """Los conteos en disco (metadata.json) coinciden con el parser independiente
        (que aplica la misma deduplicación que el pipeline)."""
        cc   = country_data["cc"]
        meta = country_data["meta"]
        sp   = meta["splits"]

        assert sp["train"]["n_examples"] == len(country_data["train_sub"]), (
            f"[{cc}] train en disco: {sp['train']['n_examples']} "
            f"vs auditado (dedup): {len(country_data['train_sub'])}"
        )
        assert sp["validation"]["n_examples"] == len(country_data["val_sub"]), (
            f"[{cc}] val en disco: {sp['validation']['n_examples']} "
            f"vs auditado (dedup): {len(country_data['val_sub'])}"
        )
        assert sp["test"]["n_examples"] == len(country_data["dev_ex"]), (
            f"[{cc}] test en disco: {sp['test']['n_examples']} "
            f"vs auditado (dedup): {len(country_data['dev_ex'])}"
        )


# ---------------------------------------------------------------------------
# PUNTO 5 — Deduplicación: unicidad en disco y conteos en metadata
# ---------------------------------------------------------------------------

class TestDeduplication:
    """P5: deduplicación por texto exacto aplicada y documentada correctamente."""

    # Conteos esperados de duplicados eliminados (auditados de los XML fuente)
    EXPECTED_TRAIN_DUPS = {"ES": 0, "CR": 0, "PE": 2}
    EXPECTED_DEV_DUPS   = {"ES": 0, "CR": 0, "PE": 0}

    def test_metadata_has_duplicates_dropped_fields(self, country_data):
        """metadata.json contiene los campos train_duplicates_dropped y dev_duplicates_dropped."""
        cc  = country_data["cc"]
        raw = country_data["meta"]["raw"]
        assert "train_duplicates_dropped" in raw, (
            f"[{cc}] metadata.raw no contiene 'train_duplicates_dropped'"
        )
        assert "dev_duplicates_dropped" in raw, (
            f"[{cc}] metadata.raw no contiene 'dev_duplicates_dropped'"
        )

    def test_metadata_train_duplicates_dropped_value(self, country_data):
        """train_duplicates_dropped en metadata coincide con el valor auditado."""
        cc       = country_data["cc"]
        raw      = country_data["meta"]["raw"]
        expected = self.EXPECTED_TRAIN_DUPS[cc]
        actual   = raw["train_duplicates_dropped"]
        assert actual == expected, (
            f"[{cc}] train_duplicates_dropped: metadata={actual}, esperado={expected}"
        )

    def test_metadata_dev_duplicates_dropped_value(self, country_data):
        """dev_duplicates_dropped en metadata coincide con el valor auditado (0 para todos)."""
        cc       = country_data["cc"]
        raw      = country_data["meta"]["raw"]
        expected = self.EXPECTED_DEV_DUPS[cc]
        actual   = raw["dev_duplicates_dropped"]
        assert actual == expected, (
            f"[{cc}] dev_duplicates_dropped: metadata={actual}, esperado={expected}"
        )

    def test_independent_parser_dup_count_matches_metadata(self, country_data):
        """El conteo de duplicados del parser independiente coincide con metadata."""
        cc  = country_data["cc"]
        raw = country_data["meta"]["raw"]
        assert country_data["train_r"]["n_dup"] == raw["train_duplicates_dropped"], (
            f"[{cc}] train n_dup: auditor={country_data['train_r']['n_dup']} "
            f"vs meta={raw['train_duplicates_dropped']}"
        )
        assert country_data["dev_r"]["n_dup"] == raw["dev_duplicates_dropped"], (
            f"[{cc}] dev n_dup: auditor={country_data['dev_r']['n_dup']} "
            f"vs meta={raw['dev_duplicates_dropped']}"
        )

    def test_no_duplicate_texts_in_disk_train(self, country_data):
        """Ningún texto aparece más de una vez en el split 'train' guardado en disco."""
        cc = country_data["cc"]
        ds = load_from_disk(str(country_data["proc_dir"] / "train"))
        counts = Counter(ds["text"])
        dupes = {t: c for t, c in counts.items() if c > 1}
        assert not dupes, (
            f"[{cc}] {len(dupes)} textos duplicados en disk/train: "
            f"{list(dupes.items())[:3]}"
        )

    def test_no_duplicate_texts_in_disk_validation(self, country_data):
        """Ningún texto aparece más de una vez en el split 'validation' guardado en disco."""
        cc = country_data["cc"]
        ds = load_from_disk(str(country_data["proc_dir"] / "validation"))
        counts = Counter(ds["text"])
        dupes = {t: c for t, c in counts.items() if c > 1}
        assert not dupes, (
            f"[{cc}] {len(dupes)} textos duplicados en disk/validation: "
            f"{list(dupes.items())[:3]}"
        )

    def test_no_duplicate_texts_in_disk_test(self, country_data):
        """Ningún texto aparece más de una vez en el split 'test' guardado en disco."""
        cc = country_data["cc"]
        ds = load_from_disk(str(country_data["proc_dir"] / "test"))
        counts = Counter(ds["text"])
        dupes = {t: c for t, c in counts.items() if c > 1}
        assert not dupes, (
            f"[{cc}] {len(dupes)} textos duplicados en disk/test: "
            f"{list(dupes.items())[:3]}"
        )


# ---------------------------------------------------------------------------
# PUNTO EXTRA — Conteos brutos globales (referencia del enunciado)
# ---------------------------------------------------------------------------

class TestGlobalCounts:
    """Verifica el conteo bruto total declarado en el enunciado (todos los ficheros).
    Los raw_counts se toman ANTES de cualquier descarte, por lo que la deduplicación
    no afecta a estos totales (los duplicados sí tienen polaridad contabilizada en bruto)."""

    def test_global_raw_counts(self):
        """N=1406, NEU=562, NONE=1023, P=1123 sobre los 6 ficheros XML."""
        grand = Counter()
        for cc in COUNTRIES:
            for split_name in ("train", "development"):
                path = RAW_DIR / f"intertass-{cc}-{split_name}-tagged.xml"
                r = _parse_xml(path)
                for pol, cnt in r["raw_counts"].items():
                    grand[pol] += cnt

        assert grand["N"]    == 1406, f"N global: {grand['N']} != 1406"
        assert grand["NEU"]  == 562,  f"NEU global: {grand['NEU']} != 562"
        assert grand["NONE"] == 1023, f"NONE global: {grand['NONE']} != 1023"
        assert grand["P"]    == 1123, f"P global: {grand['P']} != 1123"
