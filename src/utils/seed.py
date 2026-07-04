"""
seed.py — Global seed fixture for the AI Lab project.

Project seed: 42.

Usage
-----
    from src.utils.seed import set_seed
    set_seed()          # uses PROJECT_SEED (42)
    set_seed(123)       # override for a specific run
"""

import os
import random

import numpy as np

PROJECT_SEED: int = 42


def set_seed(seed: int = PROJECT_SEED) -> None:
    """Fix all relevant random-number generators to make runs reproducible.

    Covers: Python stdlib `random`, NumPy, and PyTorch (CPU + CUDA).
    Also sets PYTHONHASHSEED for deterministic dict ordering and
    enables PyTorch's deterministic-algorithm mode.

    Parameters
    ----------
    seed:
        Integer seed to use.  Defaults to PROJECT_SEED (42).
    """
    # Python stdlib
    random.seed(seed)

    # Environment (affects hash randomisation in Python 3.3+)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch — imported lazily so that the module can be imported even
    # in environments where torch is not yet installed.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Request deterministic algorithms where available.
        # WARN=True surfaces a warning instead of crashing when a
        # deterministic algorithm is not available for a given op.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Disable benchmark mode so cuDNN picks the same kernel every run.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass  # torch not installed — skip silently
