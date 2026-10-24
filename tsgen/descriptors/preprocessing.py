"""Fixed, auditable pool-wide feature transform for transductive Explore AL."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def preprocess_features(X: np.ndarray, cfg, artifact_dir: Path) -> np.ndarray:
    """Fit once on pool descriptors (no labels); reuse exactly on resume.

    All training, holdout, acquisition and anchor rows use the returned matrix.
    This transductive transform is not a train-only held-out benchmark protocol.
    No whitening or per-column standardization is performed.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('Descriptor matrix must be finite and two-dimensional')
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = cfg.model_dump(mode='json')
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(str(X.shape).encode())
    digest.update(np.ascontiguousarray(X).tobytes())
    signature = digest.hexdigest()
    metadata_path = artifact_dir / 'feature_metadata.json'
    basis_path = artifact_dir / 'feature_transform.npz'
    Xn = X.copy()
    if cfg.normalization == 'l2':
        Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-15)

    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if meta['input_signature'] != signature:
            raise ValueError('Pool descriptors/preprocessing changed; start a new run instead of resuming')
        with np.load(basis_path, allow_pickle=False) as state:
            return ((Xn - state['mean']) @ state['components'].T
                    if meta['pca_applied'] else Xn)

    mean = np.zeros(X.shape[1])
    components = np.empty((0, X.shape[1]))
    ratios = np.empty(0)
    applied = bool(cfg.pca.enabled and X.shape[0] and X.shape[1])
    if applied:
        mean = Xn.mean(axis=0)
        if len(Xn) < 2 or not np.any(Xn != mean):
            # A constant pool has no variance; avoid sklearn's NaN ratios.
            components = np.zeros((1, X.shape[1]))
            ratios = np.empty(0)
        else:
            pca = PCA(n_components=cfg.pca.variance_ratio, svd_solver='full', whiten=False)
            pca.fit(Xn)
            mean, components = pca.mean_, pca.components_
            ratios = pca.explained_variance_ratio_
    Y = (Xn - mean) @ components.T if applied else Xn
    np.save(artifact_dir / 'features_raw.npy', X)
    np.savez_compressed(basis_path, mean=mean, components=components, explained_variance_ratio=ratios)
    meta = dict(input_signature=signature, descriptor=config,
                fit_scope='fixed candidate pool descriptors only; no labels (transductive AL)',
                n_samples=len(X), raw_dimension=X.shape[1], output_dimension=Y.shape[1],
                pca_applied=applied, whiten=False,
                retained_variance_ratio=float(ratios.sum()) if len(ratios) else None,
                degenerate_variance=bool(applied and not len(ratios)))
    metadata_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return Y
