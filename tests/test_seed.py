"""
tests/test_seed.py

Smoke tests for the seed fixture.  These run without GPU or heavy dependencies.
"""

import random

import numpy as np
import pytest


def test_set_seed_is_importable():
    from src.utils.seed import set_seed, PROJECT_SEED

    assert PROJECT_SEED == 42
    assert callable(set_seed)


def test_set_seed_fixes_python_random():
    from src.utils.seed import set_seed

    set_seed(42)
    a = random.random()
    set_seed(42)
    b = random.random()
    assert a == b, "Python random is not deterministic after set_seed"


def test_set_seed_fixes_numpy():
    from src.utils.seed import set_seed

    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b, err_msg="NumPy random is not deterministic after set_seed")


def test_set_seed_different_seeds_differ():
    from src.utils.seed import set_seed

    set_seed(0)
    a = random.random()
    set_seed(1)
    b = random.random()
    assert a != b, "Different seeds should (almost surely) produce different values"


def test_set_seed_torch_if_available():
    """If torch is present, verify it is seeded reproducibly."""
    pytest.importorskip("torch")
    import torch
    from src.utils.seed import set_seed

    set_seed(42)
    a = torch.rand(4)
    set_seed(42)
    b = torch.rand(4)
    assert torch.allclose(a, b), "torch.rand is not deterministic after set_seed"
