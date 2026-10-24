from __future__ import annotations

import hashlib
import numpy as np
from typing import List

def structure_hash(symbols: List[str], coords: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update((",".join(symbols)).encode("utf-8"))
    rc = np.round(coords.astype(float), 4)
    h.update(rc.tobytes())
    return h.hexdigest()[:16]
