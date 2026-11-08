from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re

import numpy as np
import pandas as pd

from tsgen.data.xyz import Frame, read_xyz_multiframe
from tsgen.utils.hashing import structure_hash
from tsgen.utils.geometry import composition_key, pair_distance_fingerprint, fingerprint_distance
from tsgen.selection.fps import fps_indices
from tsgen.descriptors.preprocessing import preprocess_features

from tsgen.explore.config import CandidatePoolCfg, DescriptorCfg, SpeciesCfg
from tsgen.explore.species import neutral_hydrocarbon_drop_reason, species_signature, split_frame_components
from tsgen.explore.surrogate.features import compute_features


def _write_xyz_multiframe(path: Path, frames: List[Frame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fr in frames:
            syms = fr.symbols
            coords = np.asarray(fr.coords, dtype=float)
            f.write(f"{len(syms)}\n")
            comment = getattr(fr, "comment", "") or ""
            f.write(f"{comment}\n")
            for s, (x, y, z) in zip(syms, coords):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def _write_xyz_single(path: Path, fr: Frame) -> None:
    """Write a single-frame XYZ file."""
    _write_xyz_multiframe(path, [fr])


def _read_xyz_multiframe_streaming(path: Path) -> List[Frame]:
    """Read XYZ frames and recover complete frames from a truncated trajectory."""
    try:
        return read_xyz_multiframe(path)
    except Exception:
        pass

    frames: List[Frame] = []

    def _parse_atom_line(ln: str):
        parts = ln.split()
        if len(parts) < 4:
            return None
        s = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            return None
        return s, x, y, z

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        while True:
            natoms_line = fh.readline()
            if not natoms_line:
                break
            natoms_line = natoms_line.strip()
            if natoms_line == "":
                continue
            try:
                natoms = int(natoms_line)
            except Exception:
                # desync, keep scanning
                continue

            comment = fh.readline()
            if comment is None or comment == "":
                break
            comment = comment.rstrip("\n")

            atom_lines: List[str] = []
            ok = True
            for _ in range(natoms):
                ln = fh.readline()
                if not ln:
                    ok = False
                    break
                ln = ln.strip()
                if ln == "":
                    ok = False
                    break
                atom_lines.append(ln)
            if not ok:
                break

            symbols: List[str] = []
            coords: List[List[float]] = []
            bad = False
            for ln in atom_lines:
                p = _parse_atom_line(ln)
                if p is None:
                    bad = True
                    break
                s, x, y, z = p
                symbols.append(s)
                coords.append([x, y, z])
            if bad:
                # skip malformed frame, keep going
                continue

            try:
                frames.append(Frame(symbols=symbols, coords=np.asarray(coords, dtype=float), comment=comment))
            except TypeError:
                # fallback constructor
                frames.append(Frame(symbols, np.asarray(coords, dtype=float), comment))

    return frames


_ATOMIC_NUMBERS_SIMPLE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "Br": 35, "I": 53,
}


def _formula_from_species_id(species_id: str) -> str:
    return str(species_id).split("_", 1)[0]


def _electron_count_from_counts(counts: Dict[str, int], charge: int = 0) -> Optional[int]:
    total = 0
    for sym, n in counts.items():
        z = _ATOMIC_NUMBERS_SIMPLE.get(sym)
        if z is None:
            return None
        total += z * int(n)
    return total - int(charge or 0)


def _pool_chemistry_drop_reason(frame: Frame, chem_cfg, bond_tol: float = 1.2) -> str:
    if not bool(getattr(chem_cfg, "enabled", False)):
        return ""
    counts = Counter(frame.symbols)
    allowed = getattr(chem_cfg, "allowed_elements", None)
    if allowed is not None:
        allowed_set = {str(x) for x in allowed}
        extra = sorted(set(counts) - allowed_set)
        if extra:
            return f"elements {extra} not in allowed_elements={sorted(allowed_set)}"
    required = {str(x) for x in list(getattr(chem_cfg, "required_elements", []) or [])}
    missing = sorted([el for el in required if int(counts.get(el, 0)) <= 0])
    if missing:
        return f"missing required_elements={missing}"
    excluded = {str(x) for x in list(getattr(chem_cfg, "exclude_elements", []) or [])}
    present_excluded = sorted([el for el in excluded if int(counts.get(el, 0)) > 0])
    if present_excluded:
        return f"contains exclude_elements={present_excluded}"
    for el, mn in dict(getattr(chem_cfg, "min_element_counts", {}) or {}).items():
        if int(counts.get(str(el), 0)) < int(mn):
            return f"{el} count < {mn}"
    for el, mx in dict(getattr(chem_cfg, "max_element_counts", {}) or {}).items():
        if int(counts.get(str(el), 0)) > int(mx):
            return f"{el} count > {mx}"
    if bool(getattr(chem_cfg, "exclude_odd_electron", False)):
        ne = _electron_count_from_counts(counts, charge=0)
        if ne is not None and ne % 2 == 1:
            return "odd-electron formula excluded"
    if chem_cfg.require_neutral_hydrocarbon_valence:
        return neutral_hydrocarbon_drop_reason(frame, bond_tol)
    return ""


def _evenly_spaced_indices(indices: List[int], k: int) -> List[int]:
    if k <= 0 or len(indices) <= k:
        return list(indices)
    # Use linspace in positional space to preserve trajectory coverage rather than
    # taking only early-time frames from a long-lived species.
    pos = np.linspace(0, len(indices) - 1, k)
    out = []
    seen = set()
    for x in pos:
        ii = indices[int(round(float(x)))]
        if ii not in seen:
            out.append(ii)
            seen.add(ii)
    # Fill any rounding gaps.
    for ii in indices:
        if len(out) >= k:
            break
        if ii not in seen:
            out.append(ii)
            seen.add(ii)
    return sorted(out)


@dataclass
class CandidatePool:
    """Construct candidate geometries, metadata, and aligned descriptors."""

    pool_cfg: CandidatePoolCfg
    desc_cfg: DescriptorCfg
    species_cfg: SpeciesCfg

    pool_dir: Optional[Path] = None

    # global pool (built once)
    frames: List[Frame] = None  # type: ignore[assignment]
    df: pd.DataFrame = None  # type: ignore[assignment]
    X: np.ndarray = None  # type: ignore[assignment]

    # bookkeeping
    _selected_ids: set[str] = None  # type: ignore[assignment]
    _seed_species: set[str] = None  # type: ignore[assignment]

    def initialize(self, pool_dir: Path | str, *, logger: Logger | None = None) -> None:
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._selected_ids = set()

        frames, df = self._build_frames_and_df(logger=logger)
        if logger:
            logger.info("[pool] Descriptor calculation | backend=%s candidates=%d", self.desc_cfg.backend, len(frames))
        frames, df, X_raw = self._balance_pool(frames, df)
        if logger and "support_frames" in df:
            logger.info("[pool] Representative clusters | trajectory_representatives=%d",
                        int(df["support_frames"].gt(0).sum()))
        if X_raw is None:
            X_raw = compute_features(frames, self.desc_cfg)
        index_path = self.pool_dir / 'pool_index.csv'
        if index_path.exists():
            previous = pd.read_csv(index_path, dtype={'id': str})
            if previous['id'].astype(str).tolist() != df['id'].astype(str).tolist():
                raise ValueError('Candidate pool changed; start a new run instead of resuming')
            if previous['species_id'].astype(str).tolist() != df['species_id'].astype(str).tolist():
                raise ValueError('Species fingerprints changed; use the original fingerprint_version or start a new run')
            if not (self.pool_dir / 'feature_metadata.json').exists() and (
                self.desc_cfg.normalization != 'none'
            ):
                raise ValueError('Legacy raw-feature run cannot enable preprocessing on resume; start a new run')
        X = preprocess_features(X_raw, self.desc_cfg, self.pool_dir)
        if logger:
            logger.info("[pool] Descriptor pool ready | candidates=%d dimensions=%d", *X.shape)

        self.frames = frames
        self.df = df
        self.X = X

        if "status" not in self.df.columns:
            self.df["status"] = "unlabeled"  # unlabeled|submitted|labeled|failed
        if "selected_iter" not in self.df.columns:
            self.df["selected_iter"] = -1
        if "label_E_hartree" not in self.df.columns:
            self.df["label_E_hartree"] = np.nan
        if "label_E_kcalmol" not in self.df.columns:
            self.df["label_E_kcalmol"] = np.nan
        if "label_E_kcalmol_rel" not in self.df.columns:
            self.df["label_E_kcalmol_rel"] = np.nan

        # seed species set (for "new_species" export)
        self._seed_species = set(df.loc[df["is_seed"], "species_id"].astype(str).tolist())

        try:
            df.to_csv(self.pool_dir / "pool_index.csv", index=False)
            np.save(self.pool_dir / "features.npy", X)
        except Exception:
            pass

    def _balance_pool(self, frames, df):
        """Cap each group after deduplication, preserving its seeds."""
        cfg = self.pool_cfg.balance
        if not cfg.enabled or df.empty:
            return frames, df, None
        groups = {}
        for i, sid in enumerate(df['species_id'].astype(str)):
            key = _formula_from_species_id(sid) if cfg.by == 'formula' else sid
            groups.setdefault(key, []).append(i)
        keep, summary, features = [], [], []
        seed_mask = df["is_seed"].to_numpy(dtype=bool)
        if cfg.method == "representative":
            origins = list(zip(df["trajectory"], df["trajectory_frame"]))
            support_frames = np.zeros(len(df), dtype=int)
        for key, indices in sorted(groups.items()):
            seeds = [j for j, i in enumerate(indices) if seed_mask[i]]
            candidates = [j for j, i in enumerate(indices) if not seed_mask[i]]
            count = len(candidates) if cfg.max_per_group is None else max(0, cfg.max_per_group - len(seeds))
            raw = None
            if cfg.method != 'evenly_spaced':
                raw = compute_features([frames[i] for i in indices], self.desc_cfg)
            if cfg.method == 'coverage':
                values = raw
                if self.desc_cfg.normalization == "l2":
                    values = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
                chosen = self._coverage_selection(values, seeds, candidates, count, cfg.min_novelty)
            elif cfg.method == 'fps':
                chosen = []
                if candidates and count:
                    reference = raw[seeds].mean(axis=0) if seeds else raw.mean(axis=0)
                    start = int(np.argmax(np.linalg.norm(raw[candidates] - reference, axis=1)))
                    chosen = [candidates[j] for j in fps_indices(raw[candidates], count, start=start)]
            elif cfg.method == "representative":
                source_frames = len({origins[indices[j]] for j in candidates})
                cluster_count = min(count, max(1, source_frames // 2))
                clusters = self._representative_selection(raw, candidates, cluster_count)
                chosen = list(clusters)
                for representative, members in clusters.items():
                    support_frames[indices[representative]] = len({origins[indices[j]] for j in members})
            else:
                chosen = _evenly_spaced_indices(candidates, count) if count else []
            selected = seeds + chosen
            keep.extend(indices[j] for j in selected)
            if raw is not None:
                features.append(raw[selected])
            summary.append(dict(group=key, n_before=len(indices), n_after=len(selected),
                                balance_by=cfg.by, method=cfg.method, n_seeds=len(seeds)))
        order = np.argsort(keep)
        rows = np.asarray(keep, dtype=int)[order]
        pd.DataFrame(summary).to_csv(self.pool_dir / 'pool_balance_summary.csv', index=False)
        result = df.iloc[rows].reset_index(drop=True)
        result['pool_idx'] = np.arange(len(result), dtype=int)
        if cfg.method == "representative":
            result["support_frames"] = support_frames[rows]
        return [frames[i] for i in rows], result, np.vstack(features)[order] if features else None

    def _representative_selection(self, raw, candidates, count):
        """Map the actual frame closest to each centroid to its cluster members."""
        if count == 0 or not candidates:
            return {}
        if len(candidates) <= count:
            return {i: [i] for i in candidates}

        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_limits

        values = raw[candidates].astype(np.float32)
        if self.desc_cfg.normalization == "l2":
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        if count == 1:
            labels = np.zeros(len(candidates), dtype=int)
        else:
            model = MiniBatchKMeans(
                n_clusters=count, random_state=0, n_init=1, batch_size=512,
                max_iter=30, reassignment_ratio=0.0,
            )
            with threadpool_limits(limits=1):
                labels = model.fit_predict(values)

        representatives = {}
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            vectors = values[members]
            centroid = vectors.mean(axis=0)
            nearest = members[np.argmin(np.linalg.norm(vectors - centroid, axis=1))]
            representatives[candidates[nearest]] = [candidates[j] for j in members]
        return representatives

    def _coverage_selection(self, X, seeds, candidates, count, min_novelty):
        if not candidates or count == 0:
            return []
        values = X[candidates]
        if seeds:
            distances = np.min([np.linalg.norm(values - X[i], axis=1) for i in seeds], axis=0)
        else:
            distances = np.linalg.norm(values - X.mean(axis=0), axis=1)
        chosen = []
        for _ in range(min(count, len(candidates))):
            next_index = int(distances.argmax())
            if (seeds or chosen) and distances[next_index] < min_novelty:
                break
            chosen.append(candidates[next_index])
            distances = np.minimum(distances, np.linalg.norm(values - values[next_index], axis=1))
            distances[next_index] = -np.inf
        return chosen

    def sample(self, it: int) -> Tuple[pd.DataFrame, np.ndarray, List[Frame]]:
        """Return current candidate set for this iteration.

        For now we return all not-yet-selected candidates.
        """
        if self.df is None or self.X is None or self.frames is None:
            raise RuntimeError("CandidatePool not initialized.")

        mask = ~self.df["id"].isin(list(self._selected_ids))
        df_c = self.df.loc[mask].reset_index(drop=True)
        idx = np.where(mask.to_numpy())[0]
        X_c = self.X[idx]
        fr_c = [self.frames[i] for i in idx.tolist()]
        return df_c, X_c, fr_c

    def mark_selected(self, selected_df: pd.DataFrame) -> None:
        for _id in selected_df["id"].astype(str).tolist():
            self._selected_ids.add(_id)

    def export_xyz(self, df: pd.DataFrame, frames: List[Frame], out_path: Path) -> None:
        # df is aligned with frames; export all frames as multi-frame xyz
        _write_xyz_multiframe(out_path, frames)

    def export_new_species_xyz(self, selected_df: pd.DataFrame, selected_frames: List[Frame], out_path: Path) -> None:
        if "species_id" not in selected_df.columns:
            _write_xyz_multiframe(out_path, [])
            return
        keep = [str(sp) not in self._seed_species for sp in selected_df["species_id"].astype(str).tolist()]
        fr2 = [fr for fr, k in zip(selected_frames, keep) if k]
        _write_xyz_multiframe(out_path, fr2)

    def export_validated_new_species_dir(self, out_dir: Path) -> Path:
        """Export labeled non-seed frames, favoring species absent from the seed set."""
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / "index.csv"

        if self.df is None or self.frames is None:
            # Write empty index for traceability
            index_path.write_text("id,species_id,selected_iter,xyz_file\n", encoding="utf-8")
            return out_dir

        df = self.df.copy()
        if "status" not in df.columns:
            df["status"] = "unlabeled"
        if "selected_iter" not in df.columns:
            df["selected_iter"] = -1
        if "species_id" not in df.columns:
            df["species_id"] = "unknown"

        # Candidate rows to export
        if "is_seed" not in df.columns:
            df["is_seed"] = False
        mask = (
            (df["status"].astype(str) == "labeled")
            & (df["selected_iter"].astype(int) >= 0)
            & (~df["is_seed"].astype(bool))
        )
        dfv = df.loc[mask].copy()

        # Prefer previously unseen species when any have labels.
        seed_species = getattr(self, "_seed_species", set()) or set()
        if seed_species and not dfv.empty:
            df_strict = dfv.loc[~dfv["species_id"].astype(str).isin(seed_species)].copy()
            if not df_strict.empty:
                dfv = df_strict

        # Always write an index (even if empty)
        import csv
        with index_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "species_id", "selected_iter", "label_E_hartree", "label_E_kcalmol_rel", "xyz_file"])

            if dfv.empty:
                try:
                    (out_dir / "README.txt").write_text(
                        "No validated non-seed structures were available to export.\n"
                        "This typically means: (i) no non-seed candidates were selected, or (ii) the oracle jobs \
did not finish successfully yet.\n"
                        "Check labels/labels.csv and iter_*/gaussian/jobs/* for job status.\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                return out_dir

            for ridx, row in dfv.iterrows():
                _id = str(row.get("id", ""))
                sp = str(row.get("species_id", "unknown"))
                it = int(row.get("selected_iter", -1))
                eh = row.get("label_E_hartree", "")
                er = row.get("label_E_kcalmol_rel", "")

                try:
                    fr = self.frames[int(ridx)]
                except Exception:
                    continue

                # Create a readable but stable file name
                safe_sp = re.sub(r"[^A-Za-z0-9_.-]+", "_", sp)[:80]
                xyz_name = f"{safe_sp}__{_id}.xyz" if _id else f"{safe_sp}__r{int(ridx):06d}.xyz"
                xyz_path = out_dir / xyz_name
                _write_xyz_single(xyz_path, fr)
                w.writerow([_id, sp, it, eh, er, str(xyz_name)])

        return out_dir

    # -------------------------
    # internal build helpers
    # -------------------------
    def _build_frames_and_df(self, *, logger: Logger | None = None) -> Tuple[List[Frame], pd.DataFrame]:
        chem_cfg = self.pool_cfg.chemistry_filter
        neighborhood = chem_cfg.reference_neighborhood
        references = {}
        if neighborhood is not None:
            reference_path = Path(neighborhood.xyz_file)
            for frame in read_xyz_multiframe(reference_path):
                sid = species_signature(
                    frame, self.species_cfg.bond_tol, fingerprint_version=self.species_cfg.fingerprint_version
                )
                references.setdefault(sid, []).append(pair_distance_fingerprint(frame))
            if not references:
                raise ValueError(f"Reference geometry library is empty: {reference_path}")
            if logger:
                logger.info("[pool] Reference neighborhood | library=%s species=%d max_pair_delta=%.3f A",
                            reference_path, len(references), neighborhood.max_pair_distance_delta)

        sources = []
        for source in self.pool_cfg.sources:
            path = Path(source.path)
            if not path.exists():
                raise FileNotFoundError(f"Candidate source does not exist: {path}")
            is_seed = source.type.startswith("seed_")
            if source.type.endswith("_dir"):
                files = sorted(path.rglob(source.pattern) if source.recursive else path.glob(source.pattern))
            else:
                files = [path]
            if not files:
                raise ValueError(f"No candidate XYZ files in {path}")
            if logger:
                logger.info("[pool] Reading %s | source=%s files=%d", path.name, source.type, len(files))
            start = len(sources)
            for file in files:
                stride = 1 if is_seed else source.stride
                if stride < 1:
                    raise ValueError("Trajectory stride must be positive")
                sampled = list(enumerate(_read_xyz_multiframe_streaming(file)))[::stride]
                if not is_seed and source.max_frames is not None:
                    sampled = sampled[:source.max_frames]
                sources.extend((frame, is_seed, str(file.resolve()), frame_idx) for frame_idx, frame in sampled)
            if logger:
                logger.info("[pool] Loaded %s | sampled_frames=%d", path.name, len(sources) - start)

        if logger:
            logger.info("[pool] Building molecular candidates | frames=%d split_fragments=%s",
                        len(sources), self.pool_cfg.split_fragments)

        # split fragments if requested (note: split_frame_components returns (Frame, species_id))
        frames: List[Frame] = []
        is_seed: List[bool] = []
        species_ids: List[str] = []
        provenance: List[Tuple[str, int]] = []

        do_split = bool(getattr(self.pool_cfg, "split_fragments", False))
        min_atoms = int(getattr(self.pool_cfg, "min_fragment_atoms", 1))
        bond_tol = float(getattr(self.species_cfg, "bond_tol", 1.2))
        fingerprint_version = self.species_cfg.fingerprint_version

        for fr, seed_flag, source_path, frame_idx in sources:
            if do_split:
                comps = split_frame_components(fr, bond_tol=bond_tol, fingerprint_version=fingerprint_version)
                for c_fr, sid in comps:
                    if len(c_fr.symbols) >= min_atoms:
                        frames.append(c_fr)
                        is_seed.append(seed_flag)
                        species_ids.append(str(sid))
                        provenance.append((source_path, frame_idx))
            else:
                frames.append(fr)
                is_seed.append(seed_flag)
                species_ids.append(species_signature(fr, bond_tol=bond_tol, fingerprint_version=fingerprint_version))
                provenance.append((source_path, frame_idx))

        # Screen composition and geometry before computing descriptors.
        if chem_cfg.enabled:
            f2: List[Frame] = []
            s2: List[bool] = []
            sp2: List[str] = []
            dropped_rows: List[Dict] = []
            kept_rows = []
            for idx, (fr, seed_flag, sid) in enumerate(zip(frames, is_seed, species_ids)):
                match_reference = neighborhood is not None and not seed_flag
                if match_reference and sid not in references:
                    reason = "species absent from reference library"
                else:
                    reason = _pool_chemistry_drop_reason(fr, chem_cfg, bond_tol)
                    if match_reference and not reason:
                        fingerprint = pair_distance_fingerprint(fr)
                        distance = min(fingerprint_distance(fingerprint, ref) for ref in references[sid])
                        if distance > neighborhood.max_pair_distance_delta:
                            reason = f"reference distance exceeded ({distance:.3f} A)"
                if reason:
                    dropped_rows.append({"pool_pre_idx": idx, "species_id": str(sid), "reason": reason})
                    continue
                f2.append(fr)
                s2.append(seed_flag)
                sp2.append(str(sid))
                kept_rows.append(idx)
            frames, is_seed, species_ids = f2, s2, sp2
            provenance = [provenance[i] for i in kept_rows]
            try:
                if self.pool_dir is not None:
                    pd.DataFrame(dropped_rows).to_csv(Path(self.pool_dir) / "pool_prefilter_dropped.csv", index=False)
            except Exception:
                pass

        if self.pool_dir is not None:
            (self.pool_dir / "chemically_admissible_count.txt").write_text(f"{len(frames)}\n", encoding="utf-8")

        # Remove invariant near-duplicates before balancing. Seeds win over
        # trajectory copies even when a seed source appears later in the YAML.
        dedup_cfg = self.pool_cfg.deduplication
        if dedup_cfg.enabled:
            kept_indices = []
            representatives = {}
            exact_representatives = {}
            tolerance = dedup_cfg.fingerprint_tol
            audit = []
            order = sorted(range(len(frames)), key=lambda i: (not is_seed[i], i))
            for i in order:
                fr = frames[i]
                key = (species_ids[i], composition_key(fr))
                fp = pair_distance_fingerprint(fr)
                duplicate = None
                distance = float('inf')
                if tolerance == 0:
                    duplicate = exact_representatives.setdefault((key, fp), i)
                    if duplicate == i:
                        duplicate = None
                    else:
                        distance = 0.0
                else:
                    for j, old_fp in representatives.get(key, []):
                        distance = fingerprint_distance(fp, old_fp)
                        if distance <= tolerance:
                            duplicate = j
                            break
                if duplicate is None:
                    if tolerance > 0:
                        representatives.setdefault(key, []).append((i, fp))
                    kept_indices.append(i)
                audit.append(dict(source_index=i, id=structure_hash(fr.symbols, fr.coords),
                                  species_id=species_ids[i], is_seed=is_seed[i],
                                  decision='keep' if duplicate is None else 'drop',
                                  duplicate_of='' if duplicate is None else structure_hash(frames[duplicate].symbols, frames[duplicate].coords),
                                  fingerprint_distance=None if duplicate is None else distance))
            if self.pool_dir is not None:
                pd.DataFrame(audit, columns=['source_index', 'id', 'species_id', 'is_seed', 'decision', 'duplicate_of', 'fingerprint_distance']).to_csv(
                    self.pool_dir / 'pool_deduplication.csv', index=False)
            keep = sorted(kept_indices)
            frames = [frames[i] for i in keep]
            is_seed = [is_seed[i] for i in keep]
            species_ids = [species_ids[i] for i in keep]
            provenance = [provenance[i] for i in keep]

        # metadata df aligned with frames
        rows: List[Dict] = []
        for fr, seed_flag, sid, (source_path, frame_idx) in zip(frames, is_seed, species_ids, provenance):
            _id = structure_hash(fr.symbols, fr.coords)
            rows.append(
                {
                    "id": str(_id),
                    "species_id": str(sid),
                    "natoms": int(len(fr.symbols)),
                    "is_seed": bool(seed_flag),
                    "comment": fr.comment,
                    "trajectory": source_path,
                    "trajectory_frame": frame_idx,
                }
            )
        df = pd.DataFrame(rows, columns=["id", "species_id", "natoms", "is_seed", "comment", "trajectory", "trajectory_frame"])

        # de-duplicate by id
        if len(df) > 0:
            # Merge seed provenance before retaining the first exact coordinate copy.
            seed_by_id = df.groupby('id')['is_seed'].any()
            df['is_seed'] = df['id'].map(seed_by_id).astype(bool)
            keep = ~df["id"].duplicated(keep="first")
            df = df.loc[keep].reset_index(drop=True)
            frames = [fr for fr, k in zip(frames, keep.to_numpy()) if bool(k)]

        # stable positional index for alignment across filtering/reset_index
        df.insert(0, "pool_idx", np.arange(len(df), dtype=int))

        return frames, df
