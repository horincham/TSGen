"""Full descriptor features for computation; PCA is a separate visualization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tsgen.descriptors.pca import pca_2d


def _signature(X, config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(str(X.shape).encode())
    digest.update(np.ascontiguousarray(X).tobytes())
    return digest.hexdigest()


def preprocess_features(X: np.ndarray, cfg, artifact_dir: Path) -> np.ndarray:
    """Return every descriptor column, optionally row-normalized as configured.

    PCA coordinates are saved separately and NEVER returned to model/selection
    callers. Toggling visualization does not change the computational signature.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('Descriptor matrix must be finite and two-dimensional')
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = cfg.model_dump(mode='json')
    computational_config = {k: v for k, v in config.items() if k != 'pca'}
    signature = _signature(X, computational_config)
    metadata_path = artifact_dir / 'feature_metadata.json'
    raw_path = artifact_dir / 'features_raw.npy'
    basis_path = artifact_dir / 'feature_transform.npz'

    meta = {}
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if meta.get('pca_applied'):
            raise ValueError('Previously PCA-reduced run cannot resume in full descriptor space; start a new run')
        if meta['input_signature'] != signature:
            # Compatible migration of older runs that never used PCA.
            old_config = meta.get('descriptor', {})
            old_computational = {k: v for k, v in old_config.items() if k != 'pca'}
            changed = 'Pool descriptors/preprocessing changed; start a new run instead of resuming'
            if old_computational != computational_config:
                raise ValueError(changed)
            if meta['input_signature'] != _signature(X, old_config):
                # Reuse the original descriptors when another host reproduces them within roundoff.
                if not raw_path.is_file():
                    raise ValueError(changed)
                cached = np.load(raw_path, allow_pickle=False)
                cached_signature = _signature(cached, computational_config)
                if (
                    cached.shape != X.shape
                    or meta['input_signature'] not in {cached_signature, _signature(cached, old_config)}
                    or not np.allclose(X, cached, rtol=1e-9, atol=1e-9)
                ):
                    raise ValueError(changed)
                X = cached
                signature = cached_signature
    else:
        np.save(raw_path, X)

    Xn = X.copy()
    if cfg.normalization == 'l2':
        Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-15)
    if not basis_path.exists():
        np.savez_compressed(basis_path, mean=np.zeros(X.shape[1]), components=np.empty((0, X.shape[1])),
                            explained_variance_ratio=np.empty(0))
    visual_path = artifact_dir / 'features_pca_2d.npy'
    if cfg.pca.enabled and (not visual_path.exists() or meta.get('pca_visualization_file') != visual_path.name):
        np.save(visual_path, pca_2d(Xn))
    meta = dict(input_signature=signature, descriptor=config,
                feature_policy='full_descriptors_v2',
                fit_scope='row normalization only; PCA fit on candidate descriptors for visualization only',
                n_samples=len(X), raw_dimension=X.shape[1], output_dimension=Xn.shape[1],
                pca_applied=False, whiten=False, retained_variance_ratio=None,
                pca_visualization_enabled=cfg.pca.enabled,
                pca_visualization_file=visual_path.name if cfg.pca.enabled else None,
                legacy_variance_ratio_ignored=True)
    metadata_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return Xn
