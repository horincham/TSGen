from __future__ import annotations

from pathlib import Path
from logging import Logger
from typing import List, Dict, Any
import json
import pandas as pd
import numpy as np

from tsgen.config import TSGenConfig, load_config
from tsgen.paths import RunPaths
from tsgen.state import RunState, save_state, load_state
from tsgen.data.xyz import read_xyz_multiframe, write_xyz_multiframe, Frame
from tsgen.data.manifest import ManifestRow, write_manifest
from tsgen.data.lineage import LineageRecord, write_lineage_json
from tsgen.utils.hashing import structure_hash
from tsgen.filters.physics import PhysicsFilter
from tsgen.generators.scan_bond import BondScan
from tsgen.generators.scan_angle_dihedral import AngleDihedralScan
from tsgen.generators.random_perturb import RandomPerturb
from tsgen.descriptors.features import compute_descriptor_matrix
from tsgen.descriptors.pca import pca_2d
from tsgen.selection.fps import fps_indices
from tsgen.selection.scoring import reaction_relevance_geometry, ScoreWeights, compute_total_score
from tsgen.selection.stratify import apply_quotas
from tsgen.review_bundle.contract import write_review_bundle, read_decision_jsonl, apply_decisions
from tsgen.labeling.mock_backend import MockLabelBackend
from tsgen.labeling.gaussian_stub import GaussianStubBackend
from tsgen.labeling.gaussian_backend import GaussianBackend
from tsgen.utils.io import write_features_csv
from tsgen.metrics.geom import compute_geom_metrics
from tsgen.figure3.dataset import append_run_dataset
from tsgen.benchmark.coverage import descriptor_coverage


def _iter_log(logger: Logger, it: int, msg: str, **kv: Any) -> None:
    """Iteration-scoped logs with a stable, paper-friendly format.

    We keep the iter tag in the message string so the output remains
    readable under any logging formatter.
    Example: "[iter=0] generate.done n=120".
    """
    parts = [f"[iter={it}] {msg}"]
    for k, v in kv.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    logger.info(" ".join(parts))

def _backend_from_cfg(cfg: TSGenConfig):
    if cfg.labeling.backend == "mock":
        return MockLabelBackend(seed=cfg.run.run_id)
    if cfg.labeling.backend == "gaussian_stub":
        return GaussianStubBackend()
    if cfg.labeling.backend == "gaussian":
        g = cfg.labeling.gaussian
        return GaussianBackend(
            mode=g.mode,
            route=g.route,
            charge=g.charge,
            multiplicity=g.multiplicity,
            nproc=g.nproc,
            mem=g.mem,
            g16_cmd=g.g16_cmd,
            slurm=g.slurm.model_dump(),
            slurm_extra_lines=g.slurm.extra_lines,
        )
    raise ValueError(f"Unknown backend: {cfg.labeling.backend}")

def _ensure_iter_dirs(rp: RunPaths, it: int) -> Dict[str, Path]:
    idir = rp.iter_dir(it)
    idir.mkdir(parents=True, exist_ok=True)
    out = {
        "iter_dir": idir,
        "candidates_xyz": idir / "candidates.xyz",
        "manifest_jsonl": idir / "manifest.jsonl",
        "features_csv": idir / "features.csv",
        "selection_csv": idir / "selection.csv",
        "geom_metrics_csv": idir / "geom_metrics.csv",
        "descriptors_npy": idir / "descriptors.npy",
        "descriptor_ids_json": idir / "descriptor_ids.json",
        "review_bundle": idir / "review_bundle",
        "selected_ids_json": idir / "selected_ids.json",
        "label_tasks": idir / "label_tasks",
        "lineage_json": idir / "lineage.json",
        "kept_xyz": idir / "kept.xyz",
        "selected_xyz": idir / "selected.xyz",
        "filter_report_csv": idir / "filter_report.csv",
    }
    return out

def _new_candidate_id(it: int, idx: int) -> str:
    return f"cand_{it:03d}_{idx:08d}"

def _update_summary(rp: RunPaths, row: Dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    if rp.summary_csv.exists():
        old = pd.read_csv(rp.summary_csv)
        df = pd.concat([old, df], axis=0)
    df.to_csv(rp.summary_csv, index=False)

def run_workflow(cfg: TSGenConfig, rp: RunPaths, logger: Logger) -> None:
    if cfg.workflow.resume and rp.state_file.exists():
        logger.info("workflow.resume.detected", extra={"extra": {"state_file": str(rp.state_file)}})
        resume_workflow(rp, logger)
        return

    state = RunState(stage="INIT", iteration=0, dft_used=0)
    save_state(rp.state_file, state)

    # Persist config for resume
    cfg_path = rp.run_dir / "config.yaml"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("workflow.init", extra={"extra": {"run_dir": str(rp.run_dir)}})
    _main_loop(cfg, rp, logger, state)

def resume_workflow(rp: RunPaths, logger: Logger) -> None:
    cfg_path = rp.run_dir / "config.yaml"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing {cfg_path}. For resumable runs, TSGen writes config.yaml on first run.")
    cfg = TSGenConfig.model_validate(json.loads(cfg_path.read_text(encoding="utf-8")))
    state = load_state(rp.state_file)
    _main_loop(cfg, rp, logger, state)

def _main_loop(cfg: TSGenConfig, rp: RunPaths, logger: Logger, state: RunState) -> None:
    seed_frames = read_xyz_multiframe(Path(cfg.run.seed_xyz))
    bench_frames = read_xyz_multiframe(Path(cfg.run.benchmark_xyz))
    _ = bench_frames  # placeholder (future: real benchmark eval)

    D_frames: List[Frame] = seed_frames[:]
    D_ids: List[str] = [fr.get_id() or f"seed_{i:04d}" for i, fr in enumerate(D_frames)]

    rng = np.random.default_rng(cfg.run.run_id)
    backend = _backend_from_cfg(cfg)

    while True:
        it = state.iteration
        if it >= cfg.workflow.max_iters:
            state.stage = "DONE"
            save_state(rp.state_file, state)
            logger.info("workflow.done.max_iters", extra={"extra": {"iter": it}})
            break
        if state.dft_used >= cfg.workflow.dft_budget:
            state.stage = "DONE"
            save_state(rp.state_file, state)
            logger.info("workflow.done.budget", extra={"extra": {"dft_used": state.dft_used}})
            break

        io = _ensure_iter_dirs(rp, it)

        state.stage = "GENERATE"
        save_state(rp.state_file, state)
        _iter_log(logger, it, "generate.start")

        parents = D_frames[-min(len(D_frames), 3):]
        parent_ids = D_ids[-min(len(D_ids), 3):]

        proposals = []
        if cfg.generators.bond_scan.enabled:
            g = BondScan(
                n_per_parent=cfg.generators.bond_scan.n_per_parent,
                i=cfg.generators.bond_scan.i,
                j=cfg.generators.bond_scan.j,
                r_min=cfg.generators.bond_scan.r_min,
                r_max=cfg.generators.bond_scan.r_max,
            )
            proposals += g.generate(parents, parent_ids, rng)
        if cfg.generators.angle_dihedral_scan.enabled:
            g2 = AngleDihedralScan(
                n_per_parent=cfg.generators.angle_dihedral_scan.n_per_parent,
                i=cfg.generators.angle_dihedral_scan.i,
                j=cfg.generators.angle_dihedral_scan.j,
                k=cfg.generators.angle_dihedral_scan.k,
                angle_min_deg=cfg.generators.angle_dihedral_scan.angle_min_deg,
                angle_max_deg=cfg.generators.angle_dihedral_scan.angle_max_deg,
                dihedral_enabled=cfg.generators.angle_dihedral_scan.dihedral.enabled,
                l=cfg.generators.angle_dihedral_scan.dihedral.l,
                phi_min_deg=cfg.generators.angle_dihedral_scan.dihedral.phi_min_deg,
                phi_max_deg=cfg.generators.angle_dihedral_scan.dihedral.phi_max_deg,
            )
            proposals += g2.generate(parents, parent_ids, rng)
        if cfg.generators.random_perturb.enabled:
            gr = RandomPerturb(
                n_per_parent=cfg.generators.random_perturb.n_per_parent,
                sigma_angstrom=cfg.generators.random_perturb.sigma_angstrom,
            )
            proposals += gr.generate(parents, parent_ids, rng)

        cand_frames: List[Frame] = []
        manifest_rows: List[ManifestRow] = []
        lineage: Dict[str, LineageRecord] = {}

        for idx, prop in enumerate(proposals):
            cid = _new_candidate_id(it, idx)
            fr = Frame(prop.frame.symbols, prop.frame.coords, comment=f"id={cid} source={prop.source} parent={prop.parent_id}")
            h = structure_hash(fr.symbols, fr.coords)
            cand_frames.append(fr)
            manifest_rows.append(
                ManifestRow(
                    id=cid,
                    iter=it,
                    source=prop.source,
                    propose_params=prop.propose_params,
                    hash=h,
                    status="proposed",
                    xyz_frame_index=idx,
                    parent_id=prop.parent_id,
                )
            )
            lineage[cid] = LineageRecord(
                id=cid, parent_id=prop.parent_id, source=prop.source, propose_params=prop.propose_params, structure_hash=h
            )

        write_xyz_multiframe(io["candidates_xyz"], cand_frames)
        write_manifest(io["manifest_jsonl"], manifest_rows)
        write_lineage_json(io["lineage_json"], lineage)
        _iter_log(logger, it, "generate.done", n=len(cand_frames))

        state.stage = "FILTER"
        save_state(rp.state_file, state)

        pf = PhysicsFilter(
            min_distance_angstrom=cfg.filters.min_distance_angstrom,
            angle_limits_deg=cfg.filters.angle_limits_deg,
        )
        kept, flags = pf.apply(cand_frames)
        keep_mask = np.array([f["keep"] for f in flags], dtype=bool)
        kept_ids = [manifest_rows[i].id for i in range(len(manifest_rows)) if keep_mask[i]]
        # --- P0.1/P0.2 observability outputs ---
        # P0.1: kept.xyz exports frames that pass physics filter
        write_xyz_multiframe(io["kept_xyz"], kept)

        # P0.2: filter_report.csv exports per-candidate keep/drop + reasons
        filter_rows: List[Dict[str, Any]] = []
        for i, f in enumerate(flags):
            mr = manifest_rows[i]
            filter_rows.append({
                "id": mr.id,
                "iter": it,
                "source": mr.source,
                "parent_id": mr.parent_id,
                "xyz_frame_index": mr.xyz_frame_index,
                "hash": mr.hash,
                "keep": bool(f.get("keep")),
                "min_dist": f.get("min_dist"),
                "extreme_angle": f.get("extreme_angle"),
                "reasons": f.get("reasons", ""),
            })
        pd.DataFrame(filter_rows).to_csv(io["filter_report_csv"], index=False)

        _iter_log(logger, it, "filter.done", kept=len(kept), dropped=int((~keep_mask).sum()))

        if len(kept) == 0:
            _update_summary(rp, {"iter": it, "kept": 0, "selected": 0, "labeled": 0, "bench_err": None})
            state.iteration += 1
            continue

        state.stage = "FEATURIZE"
        save_state(rp.state_file, state)

        X = compute_descriptor_matrix(kept, cfg.descriptor)
        # Persist raw descriptors for Panel A (PCA coverage) and future re-embeddings.
        np.save(io["descriptors_npy"], X)
        io["descriptor_ids_json"].write_text(json.dumps(kept_ids, indent=2), encoding="utf-8")
        Y = pca_2d(X, n_components=cfg.descriptor.pca.n_components)
        PCA1, PCA2 = Y[:, 0], Y[:, 1]

        from tsgen.committee.mock_predictor import MockCommittee
        committee = MockCommittee(size=cfg.committee.size, jitter_sigma=cfg.committee.jitter_sigma, seed=cfg.run.run_id + it)
        score_u = committee.uncertainty_uE(kept)

        score_r = reaction_relevance_geometry(kept)

        fps_k = min(cfg.workflow.select_k, len(kept))
        fps_sel = fps_indices(X, k=fps_k, start=0)
        chosen_X = X[fps_sel]
        dists = np.linalg.norm(X[:, None, :] - chosen_X[None, :, :], axis=2)
        score_d = np.min(dists, axis=1)

        df = pd.DataFrame({
            "id": kept_ids,
            "score_u": score_u,
            "score_r": score_r,
            "score_d": score_d,
            "PCA1": PCA1,
            "PCA2": PCA2,
            "filter_flags": ["OK" for _ in kept_ids],
        })

        state.stage = "SCORE_SELECT"
        save_state(rp.state_file, state)

        w = ScoreWeights(
            lambda_u=cfg.scoring.lambda_u,
            lambda_r=cfg.scoring.lambda_r,
            lambda_d=cfg.scoring.lambda_d,
            use_u=cfg.scoring.ablation.u,
            use_r=cfg.scoring.ablation.r,
            use_d=cfg.scoring.ablation.d,
        )
        df = compute_total_score(df, w)
        df["iter"] = it
        write_features_csv(io["features_csv"], df)

        if cfg.scoring.quotas.enabled:
            picked = apply_quotas(df, by_field=cfg.scoring.quotas.by_field, quotas=cfg.scoring.quotas.quotas)
            picked = picked.sort_values("S", ascending=False)
        else:
            picked = df.sort_values("S", ascending=False).head(cfg.workflow.select_k)

        auto_selected_ids = picked["id"].astype(str).tolist()
        # P0.1: In kit mode, we may pause for human decision; write the machine shortlist as selected.xyz
        # (This file may be overwritten after decision/resume.)
        id_to_frame = {fr.get_id(): fr for fr in cand_frames}
        auto_selected_frames = [id_to_frame[i] for i in auto_selected_ids if i in id_to_frame]
        write_xyz_multiframe(io["selected_xyz"], auto_selected_frames)


        if cfg.workflow.gate == "kit":
            state.stage = "GATE_WAIT"
            save_state(rp.state_file, state)

            write_review_bundle(io["review_bundle"], io["candidates_xyz"], io["manifest_jsonl"], io["features_csv"])
            decision_path = io["review_bundle"] / "decision.jsonl"

            if not decision_path.exists():
                _iter_log(logger, it, "gate.kit.waiting", decision=str(decision_path))
                _iter_log(logger, it, "gate.kit.hint", resume_cmd=f"tsgen resume --run-dir {rp.run_dir}")
                return

            decisions = read_decision_jsonl(decision_path)
            keep_ids, drop_ids = apply_decisions(df, decisions)
            selected_ids = keep_ids if keep_ids else auto_selected_ids
            _iter_log(logger, it, "select.done", k=len(selected_ids), keep=len(keep_ids), drop=len(drop_ids))
        else:
            selected_ids = auto_selected_ids
            # ✅ P0.1: write FINAL selected.xyz (after gate decision)
            id_to_frame = {fr.get_id(): fr for fr in cand_frames}
            selected_frames = [id_to_frame[i] for i in selected_ids if i in id_to_frame]
            write_xyz_multiframe(io["selected_xyz"], selected_frames)

            _iter_log(logger, it, "select.done", k=len(selected_ids))

        # Label submission: task materialization + submit/wait happens downstream.
        _iter_log(logger, it, "label.submit", n=len(selected_ids))

        # --- Figure 3 data plumbing (Panels A–C) ---
        # selection.csv is the authoritative table distinguishing selected vs rejected vs dropped.
        sel_base = pd.DataFrame(filter_rows)
        sel_base = sel_base.rename(columns={"keep": "is_kept", "reasons": "drop_reason"})
        sel_base["is_dropped"] = (~sel_base["is_kept"].astype(bool)).astype(bool)
        sel_base["is_selected"] = sel_base["id"].astype(str).isin([str(x) for x in selected_ids])

        sel_base.to_csv(io["selection_csv"], index=False)

        # geom_metrics.csv stores explicit reaction-coordinate values (Panel C)
        id_to_frame = {fr.get_id(): fr for fr in cand_frames}
        frames_aligned: List[Frame] = []
        ids_aligned: List[str] = []
        for cid in selected_ids:
            if cid in id_to_frame:
                frames_aligned.append(id_to_frame[cid])
                ids_aligned.append(cid)
        gm = compute_geom_metrics(frames_aligned, ids_aligned, it, cfg.reaction_coords)
        pd.DataFrame(gm.rows).to_csv(io["geom_metrics_csv"], index=False)

        io["selected_ids_json"].write_text(json.dumps(selected_ids, indent=2), encoding="utf-8")

        state.stage = "LABEL_TASKS"
        save_state(rp.state_file, state)

        # NOTE: the next step may spend most walltime in Slurm queue / Gaussian runtime.
        # GaussianBackend itself emits periodic progress logs; we also mark submission here.
        _iter_log(logger, it, "label.submit", n=len(selected_ids))

        id_to_frame = {fr.get_id(): fr for fr in cand_frames}
        selected_frames = [id_to_frame[i] for i in selected_ids if i in id_to_frame]
        backend.generate_tasks(io["label_tasks"], selected_frames, selected_ids, layer_cfg=cfg.labeling.model_dump())
        backend.run_and_wait(io["label_tasks"])
        state.dft_used += len(selected_ids)
        state.selected_ids_total += selected_ids
        save_state(rp.state_file, state)

        state.stage = "LABEL_PARSE"
        save_state(rp.state_file, state)

        results = backend.parse_results(io["label_tasks"])
        # Only accept labels that are explicitly parsed and flagged as OK.
        labeled: List[str] = []
        for cid in selected_ids:
            if cid not in results:
                continue
            r = results[cid]
            if isinstance(r, dict) and r.get("status") == "ok":
                labeled.append(cid)
            # legacy mock backend support
            elif isinstance(r, dict) and ("energy" in r):
                labeled.append(cid)

        # For real backends, if any labels are pending/unavailable, stop here and resume later.
        if cfg.labeling.backend != "mock" and len(labeled) < len(selected_ids):
            pending = [cid for cid in selected_ids if cid not in labeled]
            state.stage = "LABEL_WAIT"
            save_state(rp.state_file, state)
            _iter_log(
                logger,
                it,
                "label.waiting",
                pending=len(pending),
                hint=f"place Gaussian outputs under {io['label_tasks']} and re-run: tsgen resume --run-dir {rp.run_dir}",
            )
            return
        state.labeled_ids_total += labeled
        save_state(rp.state_file, state)

        for cid in labeled:
            D_frames.append(id_to_frame[cid])
            D_ids.append(cid)

        state.stage = "EVAL_BENCH"
        save_state(rp.state_file, state)

        cov = descriptor_coverage(D_frames, bench_frames, descriptor_cfg=cfg.descriptor)
        _iter_log(logger, it, "bench.eval.coverage", **cov.as_dict())

        _update_summary(rp, {
            "iter": it,
            "kept": int(len(kept)),
            "selected": int(len(selected_ids)),
            "labeled": int(len(labeled)),
            **cov.as_dict(),
            "dft_used": state.dft_used,
        })

        # Append a merged, cross-iteration dataset for Figure 3
        append_run_dataset(rp.run_dir, io["iter_dir"], out_name="figure3_dataset.csv")

        state.iteration += 1
        save_state(rp.state_file, state)

    (rp.run_dir / "final_ffield.placeholder").write_text("TODO: export final ffield\n", encoding="utf-8")
    (rp.run_dir / "final_trainset.placeholder").write_text("TODO: export standardized ReaxFF trainset/GEO\n", encoding="utf-8")
