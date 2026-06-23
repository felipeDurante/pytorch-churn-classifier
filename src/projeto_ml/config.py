import random

import numpy as np

try:  # torch is optional during development
    import torch
except Exception:  # pragma: no cover - torch may not be installed
    torch = None


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set seeds for reproducibility across libraries used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
