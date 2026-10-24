from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from tsgen.data.xyz import Frame


@dataclass(frozen=True)
class SoapSettings:
    species: Sequence[str]
    r_cut: float = 5.0
    n_max: int = 8
    l_max: int = 6
    sigma: float = 0.5
    periodic: bool = False
    pooling: str = "mean_std"


def _settings_from_cfg(cfg) -> SoapSettings:
    return SoapSettings(
        species=list(getattr(cfg, "soap_species", ["H", "C", "N", "O"])),
        r_cut=float(getattr(cfg, "soap_r_cut", 5.0)),
        n_max=int(getattr(cfg, "soap_n_max", 8)),
        l_max=int(getattr(cfg, "soap_l_max", 6)),
        sigma=float(getattr(cfg, "soap_sigma", 0.5)),
        periodic=bool(getattr(cfg, "soap_periodic", False)),
        pooling=str(getattr(cfg, "soap_pooling", "mean_std")),
    )


def _make_atoms(frame: Frame):
    try:
        from ase import Atoms
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "SOAP descriptors require ASE and DScribe. Install them with: "
            "pip install ase dscribe"
        ) from exc

    atoms = Atoms(symbols=list(frame.symbols), positions=np.asarray(frame.coords, dtype=float))
    if not np.any(atoms.cell.lengths()):
        # Molecular default: a generous nonperiodic box keeps DScribe happy
        # without changing the nonperiodic SOAP definition.
        atoms.set_cell([80.0, 80.0, 80.0])
        atoms.center()
    atoms.set_pbc(False)
    return atoms


def _make_soap(settings: SoapSettings):
    try:
        from dscribe.descriptors import SOAP
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "SOAP descriptors require DScribe and ASE. Install them with: "
            "pip install ase dscribe"
        ) from exc

    kwargs = {
        "species": list(settings.species),
        "periodic": settings.periodic,
        "r_cut": settings.r_cut,
        "n_max": settings.n_max,
        "l_max": settings.l_max,
        "sigma": settings.sigma,
    }
    try:
        return SOAP(**kwargs)
    except TypeError:
        # Older DScribe releases used rcut/nmax/lmax keyword spellings.
        kwargs["rcut"] = kwargs.pop("r_cut")
        kwargs["nmax"] = kwargs.pop("n_max")
        kwargs["lmax"] = kwargs.pop("l_max")
        return SOAP(**kwargs)


def _pool_atom_features(atom_features: np.ndarray, pooling: str) -> np.ndarray:
    X = np.asarray(atom_features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    mean = X.mean(axis=0)
    if pooling == "mean":
        return mean
    if pooling == "mean_std":
        return np.concatenate([mean, X.std(axis=0)], axis=0)
    raise ValueError(f"Unsupported SOAP pooling mode: {pooling}")


def soap_descriptor(frame: Frame, cfg) -> np.ndarray:
    """Return a fixed-length, structure-level SOAP descriptor.

    DScribe SOAP is atom-centered.  For molecule-level active learning we pool
    per-atom descriptors into mean and standard-deviation statistics, a common
    descriptor-space visualization and FPS strategy for heterogeneous datasets.
    """
    settings = _settings_from_cfg(cfg)
    atoms = _make_atoms(frame)
    soap = _make_soap(settings)
    atom_features = soap.create(atoms)
    return _pool_atom_features(np.asarray(atom_features, dtype=float), settings.pooling)


def soap_feature_matrix(frames: List[Frame], cfg) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0), dtype=float)
    rows = [soap_descriptor(fr, cfg) for fr in frames]
    return np.vstack(rows).astype(float)
