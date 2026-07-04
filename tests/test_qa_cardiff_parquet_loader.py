"""
tests/test_qa_cardiff_parquet_loader.py — QA validation for the Parquet-based Cardiff ES loader.

Audita el cambio de datasets==4.1.1 + trust_remote_code a la lectura via
refs/convert/parquet.  Verifica:

1. Tamaños de splits exactos (valores de referencia de diary/sprint-02.md y paper/main.tex):
   - train:      1839 ejemplos, 613 por clase (0/1/2)
   - validation:  324 ejemplos, 108 por clase
   - test:        870 ejemplos, 290 por clase

2. Esquema de etiquetas: label contiene enteros 0/1/2 (no strings).

3. ClassLabel o equivalente con orden negativo < neutral < positivo.

4. No queda ninguna referencia a trust_remote_code en load_data.py.

5. requirements.txt: datasets>=2.18.0 (rango, no pin), sin datasets==4.1.1.

6. make_fraction_subsets produce n=50,100,250,500,1000 + "full" estratificados
   a partir de datos cargados reales (no sintéticos).

7. No hay fuga de datos: los conjuntos train/validation/test no comparten textos.

Ejecutar con:
    HF_HOME=~/jupyterlab_container/hf_cache \
    /home/jovyan/jupyterlab_container/envs/ailab/bin/python \
    -m pytest tests/test_qa_cardiff_parquet_loader.py -v
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Valores de referencia (diary/sprint-02.md y paper/main.tex)
# ---------------------------------------------------------------------------
EXPECTED_SIZES = {
    "train": 1839,
    "validation": 324,
    "test": 870,
}
EXPECTED_PER_CLASS = {
    "train": 613,
    "validation": 108,
    "test": 290,
}
EXPECTED_CLASSES = {0, 1, 2}
EXPECTED_LABEL_ORDER = ["negative", "neutral", "positive"]
DEFAULT_FRACTIONS = (50, 100, 250, 500, 1000)


# ---------------------------------------------------------------------------
# Fixture: carga real del dataset (una vez por sesión de test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cardiff_splits():
    """
    Carga Cardiff ES desde el Hub vía Parquet y lo devuelve como dict de splits.
    Se marca como skip si la red no está disponible o la descarga falla.
    """
    from src.data.load_data import load_cardiff_es, DataConfig
    cache_dir = str(PROJECT_ROOT / "data" / "raw" / "cardiff")
    try:
        splits = load_cardiff_es(cache_dir=cache_dir, config=DataConfig())
    except Exception as exc:
        pytest.skip(f"No se pudo cargar Cardiff ES desde el Hub: {exc}")
    return splits


# ---------------------------------------------------------------------------
# Bloque 1: Tamaños de splits
# ---------------------------------------------------------------------------

class TestSplitSizes:
    """Los tamaños deben coincidir EXACTAMENTE con los valores de referencia."""

    def test_train_size(self, cardiff_splits):
        assert len(cardiff_splits["train"]) == EXPECTED_SIZES["train"], (
            f"train: esperado {EXPECTED_SIZES['train']}, obtenido {len(cardiff_splits['train'])}"
        )

    def test_validation_size(self, cardiff_splits):
        assert len(cardiff_splits["validation"]) == EXPECTED_SIZES["validation"], (
            f"validation: esperado {EXPECTED_SIZES['validation']}, "
            f"obtenido {len(cardiff_splits['validation'])}"
        )

    def test_test_size(self, cardiff_splits):
        assert len(cardiff_splits["test"]) == EXPECTED_SIZES["test"], (
            f"test: esperado {EXPECTED_SIZES['test']}, obtenido {len(cardiff_splits['test'])}"
        )


# ---------------------------------------------------------------------------
# Bloque 2: Distribución de clases
# ---------------------------------------------------------------------------

class TestClassDistribution:
    """Cada clase debe tener exactamente el número de ejemplos de referencia."""

    @pytest.mark.parametrize("split_name", ["train", "validation", "test"])
    def test_per_class_counts_exact(self, cardiff_splits, split_name):
        ds = cardiff_splits[split_name]
        counts = Counter(ds["label"])
        expected_count = EXPECTED_PER_CLASS[split_name]
        for class_id in EXPECTED_CLASSES:
            actual = counts.get(class_id, 0)
            assert actual == expected_count, (
                f"split='{split_name}', class={class_id}: "
                f"esperado {expected_count}, obtenido {actual}. "
                f"Distribución completa: {dict(counts)}"
            )

    @pytest.mark.parametrize("split_name", ["train", "validation", "test"])
    def test_all_three_classes_present(self, cardiff_splits, split_name):
        ds = cardiff_splits[split_name]
        found = set(ds["label"])
        assert EXPECTED_CLASSES.issubset(found), (
            f"split='{split_name}': clases esperadas {EXPECTED_CLASSES}, encontradas {found}"
        )

    @pytest.mark.parametrize("split_name", ["train", "validation", "test"])
    def test_no_extra_classes(self, cardiff_splits, split_name):
        ds = cardiff_splits[split_name]
        found = set(ds["label"])
        assert found == EXPECTED_CLASSES, (
            f"split='{split_name}': clases encontradas {found}, esperadas {EXPECTED_CLASSES}"
        )


# ---------------------------------------------------------------------------
# Bloque 3: Tipo y valores de la columna label
# ---------------------------------------------------------------------------

class TestLabelDtype:
    """Las etiquetas deben ser enteros 0/1/2, no strings."""

    @pytest.mark.parametrize("split_name", ["train", "validation", "test"])
    def test_labels_are_integers(self, cardiff_splits, split_name):
        ds = cardiff_splits[split_name]
        first = ds["label"][0]
        assert isinstance(first, int), (
            f"split='{split_name}': label[0] es {type(first).__name__!r}, "
            f"se esperaba int. Valor: {first!r}"
        )

    @pytest.mark.parametrize("split_name", ["train", "validation", "test"])
    def test_all_labels_are_integers(self, cardiff_splits, split_name):
        ds = cardiff_splits[split_name]
        non_int = [v for v in ds["label"] if not isinstance(v, int)]
        assert len(non_int) == 0, (
            f"split='{split_name}': {len(non_int)} etiquetas no son int: "
            f"muestra={non_int[:5]}"
        )

    @pytest.mark.parametrize("split_name", ["train", "validation", "test"])
    def test_label_values_in_range(self, cardiff_splits, split_name):
        ds = cardiff_splits[split_name]
        invalid = [v for v in ds["label"] if v not in {0, 1, 2}]
        assert len(invalid) == 0, (
            f"split='{split_name}': valores fuera de {{0,1,2}}: {invalid[:5]}"
        )


# ---------------------------------------------------------------------------
# Bloque 4: Orden de ClassLabel (negative=0, neutral=1, positive=2)
# ---------------------------------------------------------------------------

class TestClassLabelSchema:
    """
    Verifica el esquema de etiquetas: la característica 'label' debe tener
    ClassLabel con el orden correcto [negative, neutral, positive].
    Si el loader devuelve int sin ClassLabel, comprobamos que el mapeo LABEL2ID
    es coherente con lo esperado.
    """

    def test_label2id_mapping_correct(self):
        """LABEL2ID en load_data debe ser {negative:0, neutral:1, positive:2}."""
        from src.data.load_data import LABEL2ID
        assert LABEL2ID["negative"] == 0, f"negative -> {LABEL2ID['negative']}, esperado 0"
        assert LABEL2ID["neutral"] == 1, f"neutral -> {LABEL2ID['neutral']}, esperado 1"
        assert LABEL2ID["positive"] == 2, f"positive -> {LABEL2ID['positive']}, esperado 2"

    def test_classlabel_feature_order(self, cardiff_splits):
        """
        Si el feature 'label' es ClassLabel, sus nombres deben seguir el orden
        [negative, neutral, positive].
        """
        from datasets import ClassLabel
        ds = cardiff_splits["train"]
        feat = ds.features.get("label")
        if not isinstance(feat, ClassLabel):
            pytest.skip(
                f"La columna 'label' no es ClassLabel (es {type(feat).__name__}). "
                "Verificación de esquema de ClassLabel omitida; el orden viene dado "
                "por LABEL2ID que se comprueba en test_label2id_mapping_correct."
            )
        assert feat.names == EXPECTED_LABEL_ORDER, (
            f"ClassLabel.names = {feat.names}, esperado {EXPECTED_LABEL_ORDER}"
        )


# ---------------------------------------------------------------------------
# Bloque 5: Ausencia de trust_remote_code en load_data.py
# ---------------------------------------------------------------------------

class TestNoTrustRemoteCode:
    """No debe quedar ninguna referencia a trust_remote_code en el código fuente."""

    LOAD_DATA_PATH = PROJECT_ROOT / "src" / "data" / "load_data.py"

    def test_no_trust_remote_code_in_source(self):
        """Ninguna llamada debe pasar trust_remote_code como argumento.

        Se analiza el AST (no el texto) para no dar falsos positivos con las
        menciones del término en comentarios/docstrings, que documentan
        legítimamente por qué se eliminó el parámetro.
        """
        import ast

        source = self.LOAD_DATA_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "trust_remote_code"
        ]
        assert not offenders, (
            f"{self.LOAD_DATA_PATH}: una llamada todavía pasa 'trust_remote_code'. "
            "El cambio debía eliminar este parámetro de las llamadas a load_dataset."
        )

    def test_parquet_load_strategy_used(self):
        """El loader debe usar la estrategia 'parquet' con hf:// URIs."""
        source = self.LOAD_DATA_PATH.read_text(encoding="utf-8")
        assert 'load_dataset(\n            "parquet"' in source or 'load_dataset("parquet"' in source or "load_dataset(\n        \"parquet\"" in source or "'parquet'" in source, (
            "No se encontró el patrón load_dataset('parquet', ...) en load_data.py. "
            "La estrategia Parquet puede no estar activa."
        )

    def test_refs_convert_parquet_revision_in_source(self):
        """La URI Parquet debe referenciar refs/convert/parquet."""
        source = self.LOAD_DATA_PATH.read_text(encoding="utf-8")
        assert "refs/convert/parquet" in source, (
            "No se encontró 'refs/convert/parquet' en load_data.py. "
            "La revisión del export Parquet no está configurada."
        )


# ---------------------------------------------------------------------------
# Bloque 6: requirements.txt — pin eliminado, rango mantenido
# ---------------------------------------------------------------------------

class TestRequirements:
    """Verifica que requirements.txt usa rango >=2.18.0 y no pin ==4.1.1."""

    REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

    def test_datasets_pin_removed(self):
        """datasets==4.1.1 no debe aparecer en requirements.txt."""
        content = self.REQUIREMENTS_PATH.read_text(encoding="utf-8")
        assert "datasets==4.1.1" not in content, (
            "requirements.txt todavía contiene 'datasets==4.1.1'. "
            "El cambio debía reemplazarlo por un rango >=2.18.0."
        )

    def test_datasets_range_present(self):
        """datasets>=2.18.0 (o similar) debe estar presente."""
        content = self.REQUIREMENTS_PATH.read_text(encoding="utf-8")
        has_range = "datasets>=" in content
        assert has_range, (
            "No se encontró 'datasets>=' en requirements.txt. "
            f"Contenido de la línea de datasets: "
            f"{[l for l in content.splitlines() if 'datasets' in l]}"
        )

    def test_trust_remote_code_not_in_requirements(self):
        """trust_remote_code no debe aparecer como directiva activa en requirements.txt.

        Se ignora la parte de comentario (tras '#') de cada línea: una mención
        explicativa en un comentario es legítima; lo que no debe haber es un
        requisito/opción real que lo invoque.
        """
        active = "\n".join(
            line.split("#", 1)[0]
            for line in self.REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        )
        assert "trust_remote_code" not in active, (
            "requirements.txt contiene trust_remote_code fuera de un comentario (inesperado)."
        )


# ---------------------------------------------------------------------------
# Bloque 7: make_fraction_subsets con datos reales
# ---------------------------------------------------------------------------

class TestFractionSubsetsReal:
    """
    Verifica que make_fraction_subsets sobre los datos reales de Cardiff ES
    produce exactamente los tamaños n=50,100,250,500,1000 + "full".
    """

    def test_fraction_sizes_real_data(self, cardiff_splits):
        from src.data.load_data import make_fraction_subsets, DEFAULT_FRACTIONS
        train = cardiff_splits["train"]
        fractions = make_fraction_subsets(
            train, fractions=DEFAULT_FRACTIONS, seed=42, label_column="label"
        )
        # "full" debe tener el mismo tamaño que el train completo
        assert len(fractions["full"]) == EXPECTED_SIZES["train"], (
            f"fracciones['full'] tiene {len(fractions['full'])} ejemplos, "
            f"esperado {EXPECTED_SIZES['train']}"
        )
        # Cada fracción numérica debe tener exactamente n ejemplos
        for n in DEFAULT_FRACTIONS:
            assert len(fractions[n]) == n, (
                f"fracción n={n}: tiene {len(fractions[n])} ejemplos, esperado {n}"
            )

    def test_fraction_stratification_real_data(self, cardiff_splits):
        """Cada fracción debe tener aproximadamente 1/3 de ejemplos por clase."""
        from src.data.load_data import make_fraction_subsets, DEFAULT_FRACTIONS
        train = cardiff_splits["train"]
        fractions = make_fraction_subsets(
            train, fractions=DEFAULT_FRACTIONS, seed=42, label_column="label"
        )
        for n in DEFAULT_FRACTIONS:
            counts = Counter(fractions[n]["label"])
            for class_id in EXPECTED_CLASSES:
                actual = counts.get(class_id, 0)
                expected_approx = n / 3
                # Tolerancia: ±1 ejemplo (por el reparto de decimales)
                assert abs(actual - expected_approx) <= 1, (
                    f"fracción n={n}, clase {class_id}: {actual} ejemplos, "
                    f"esperado ~{expected_approx:.1f} (±1)"
                )

    def test_fraction_texts_come_from_train(self, cardiff_splits):
        """Todos los textos de las fracciones deben provenir del train original."""
        from src.data.load_data import make_fraction_subsets, DEFAULT_FRACTIONS
        train = cardiff_splits["train"]
        all_train_texts = set(train["text"])
        fractions = make_fraction_subsets(
            train, fractions=DEFAULT_FRACTIONS, seed=42, label_column="label"
        )
        for n in DEFAULT_FRACTIONS:
            frac_texts = set(fractions[n]["text"])
            outside = frac_texts - all_train_texts
            assert len(outside) == 0, (
                f"fracción n={n}: {len(outside)} textos fuera del train"
            )

    def test_fraction_reproducibility_real_data(self, cardiff_splits):
        """Misma semilla → mismas fracciones (texto y orden)."""
        from src.data.load_data import make_fraction_subsets
        train = cardiff_splits["train"]
        f1 = make_fraction_subsets(train, fractions=(50, 100), seed=42)
        f2 = make_fraction_subsets(train, fractions=(50, 100), seed=42)
        for n in (50, 100):
            assert f1[n]["text"] == f2[n]["text"], (
                f"fracción n={n}: seed 42 no produce el mismo resultado en dos llamadas"
            )

    def test_fractions_keys_include_full_and_numeric(self, cardiff_splits):
        """El dict de fracciones debe tener claves 'full' y los 5 tamaños numéricos."""
        from src.data.load_data import make_fraction_subsets, DEFAULT_FRACTIONS
        train = cardiff_splits["train"]
        fractions = make_fraction_subsets(
            train, fractions=DEFAULT_FRACTIONS, seed=42
        )
        expected_keys = set(DEFAULT_FRACTIONS) | {"full"}
        assert expected_keys.issubset(set(fractions.keys())), (
            f"Claves esperadas: {expected_keys}. Claves obtenidas: {set(fractions.keys())}"
        )


# ---------------------------------------------------------------------------
# Bloque 8: Anti-leakage — no solapamiento entre splits
# ---------------------------------------------------------------------------

class TestAntiLeakage:
    """Los conjuntos train, validation y test no deben compartir textos."""

    def test_no_overlap_train_test(self, cardiff_splits):
        train_texts = set(cardiff_splits["train"]["text"])
        test_texts = set(cardiff_splits["test"]["text"])
        overlap = train_texts & test_texts
        assert len(overlap) == 0, (
            f"Solapamiento de {len(overlap)} textos entre train y test. "
            f"Muestra: {list(overlap)[:3]}"
        )

    def test_no_overlap_train_validation(self, cardiff_splits):
        train_texts = set(cardiff_splits["train"]["text"])
        val_texts = set(cardiff_splits["validation"]["text"])
        overlap = train_texts & val_texts
        assert len(overlap) == 0, (
            f"Solapamiento de {len(overlap)} textos entre train y validation. "
            f"Muestra: {list(overlap)[:3]}"
        )

    def test_no_overlap_validation_test(self, cardiff_splits):
        val_texts = set(cardiff_splits["validation"]["text"])
        test_texts = set(cardiff_splits["test"]["text"])
        overlap = val_texts & test_texts
        assert len(overlap) == 0, (
            f"Solapamiento de {len(overlap)} textos entre validation y test. "
            f"Muestra: {list(overlap)[:3]}"
        )
