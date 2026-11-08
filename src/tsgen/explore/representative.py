from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min


def has_informative_uncertainty(sigma: np.ndarray) -> bool:
    """Check for a broad uncertainty spread, separate from model calibration."""
    if sigma.size < 2 or not np.isfinite(sigma).all() or (sigma < 0).any():
        return False
    low, high = np.percentile(sigma, [10, 90])
    return bool(high - low > max(1e-8, 1e-3 * high))


def select_representatives(
    df: pd.DataFrame,
    X: np.ndarray,
    k: int,
    labeled_X: np.ndarray,
    labeled_species: set[str],
    *,
    min_support_frames: int,
    new_species_slots: int,
    paired_species: set[str] | None = None,
    scores: np.ndarray | None = None,
    preselect_factor: int = 20,
) -> list[int]:
    """Select positional rows from trajectory medoids and preserve species discovery.

    Supported unseen species receive the exploration slots first. A weakly
    supported medoid can only enter through a reserved slot when no supported
    unseen species is available. Regular slots can be left empty.
    """
    if k <= 0 or df.empty:
        return []
    if len(df) != len(X) or (scores is not None and scores.shape != (len(df),)):
        raise ValueError("Representative features and scores must align with the candidate rows")

    species = df["species_id"].astype(str).to_numpy()
    available = ~df["is_seed"].to_numpy(dtype=bool)
    if not available.any():
        return []
    support = df["support_frames"].where(available, 0).to_numpy(dtype=int)
    supported = available & (support >= min_support_frames)
    support_score = np.log1p(support)
    support_score /= max(1.0, support_score[available].max())

    paired = paired_species or set()
    unseen = available & ~np.isin(species, sorted(labeled_species | paired))
    supported_unseen = supported & unseen
    discovery = supported_unseen if supported_unseen.any() else unseen
    discovery_slots = min(new_species_slots, len(set(species[discovery])))
    reserved = discovery_slots if supported_unseen.any() else 0

    distances = None
    if labeled_X.size:
        _, distances = pairwise_distances_argmin_min(X, labeled_X, metric="euclidean")
    selected: list[int] = []
    selected_species: set[str] = set()

    def pick(mask: np.ndarray, slots: int, *, unique_species: bool) -> None:
        nonlocal distances
        slots = min(slots, k - len(selected))
        for step in range(slots):
            eligible = available & mask
            distinct = eligible & ~np.isin(species, sorted(selected_species))
            if unique_species or distinct.any():
                eligible = distinct
            options = np.flatnonzero(eligible)
            if not len(options):
                break

            if scores is not None:
                order = np.argsort(-scores[options], kind="stable")
                options = options[order[: preselect_factor * (slots - step)]]
                best = options[0] if step == 0 else options[np.argmax(distances[options])]
            else:
                priority = support_score[options]
                if distances is not None:
                    separation = distances[options]
                    positive = separation[separation > 0]
                    scale = np.median(positive) if len(positive) else 1.0
                    priority = priority + separation / (separation + scale)
                best = options[np.argmax(priority)]

            selected.append(int(best))
            selected_species.add(species[best])
            available[best] = False
            separation = np.linalg.norm(X - X[best], axis=1)
            distances = separation if distances is None else np.minimum(distances, separation)

    if paired:
        pick(supported & np.isin(species, sorted(paired)), k - reserved, unique_species=True)
    pick(discovery, discovery_slots, unique_species=True)
    pick(supported, k - len(selected), unique_species=False)
    return selected
