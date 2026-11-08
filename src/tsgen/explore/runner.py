from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tsgen.explore.config import ExploreCfg, save_explore_config
from tsgen.explore.pool import CandidatePool
from tsgen.explore.gating import PhysicsGater
from tsgen.explore.acquisition import Acquisition
from tsgen.explore.anchors import AnchorManager
from tsgen.explore.io import ExplorePaths, ExploreState, load_state, save_state
from tsgen.explore.label_store import LabelStore, LabelRecord, HARTREE_TO_KCALMOL
from tsgen.explore.oracle_gaussian import GaussianOracle, GaussianJobSpec
from tsgen.explore.representative import has_informative_uncertainty, select_representatives
from tsgen.explore.surrogate import build_surrogate
from tsgen.explore.metrics import ALMetrics, evaluate_holdout, append_metrics_csv
from tsgen.logging import setup_logging
from tsgen.explore.trainset_export import export_trainset_products
from tsgen.selection.fps import fps_indices


class ExploreRunner:
    """Coordinate candidate selection and Gaussian labeling across resumable rounds."""

    def __init__(self, cfg: ExploreCfg, paths: ExplorePaths):
        self.cfg = cfg
        self.paths = paths
        self.paths.ensure()
        self.logger = setup_logging(self.paths.run_dir / "run.log.jsonl", console=True)
        self._selection_exhausted = False

        self.state = load_state(self.paths.run_dir / "state.json")

        # Older state files can contain null counters.
        if getattr(self.state, "iter", None) is None:
            self.state.iter = 0
        if getattr(self.state, "dft_used", None) is None:
            self.state.dft_used = 0
        if getattr(self.state, "selected", None) is None:
            self.state.selected = 0
        if getattr(self.state, "submitted", None) is None:
            self.state.submitted = 0
        if getattr(self.state, "labeled", None) is None:
            self.state.labeled = 0

        if self.state.labeled_ids is None:
            self.state.labeled_ids = []
        if self.state.holdout_ids is None:
            self.state.holdout_ids = []

        self.pool = CandidatePool(cfg.candidate_pool, cfg.descriptor, cfg.species, pool_dir=self.paths.pool_dir)
        self.gater = PhysicsGater(cfg.gating)
        self.acq = Acquisition(cfg.acquisition)
        self.anchors = AnchorManager(cfg.anchors)

        self.oracle = GaussianOracle(cfg.labeling.gaussian, cfg.loop.max_wait_seconds)
        self.labels = LabelStore(self.paths.labels_jsonl,
                                 reference_path=self.paths.run_dir/'energy_references.json',
                                 excluded_ids=self.state.holdout_ids,
                                 require_optimized=cfg.labeling.gaussian.reference_mode == "optimized")
        self.reference_oracle = None
        if self.labels.require_optimized:
            from tsgen.explore.references import OptimizedReferences
            self.reference_oracle = OptimizedReferences(cfg.labeling.gaussian, cfg.species.bond_tol,
                                                        paths.run_dir / "references" / "jobs", self.labels,
                                                        timeout_seconds=cfg.loop.max_wait_seconds,
                                                        fingerprint_version=cfg.species.fingerprint_version,
                                                        logger=self.logger)

    def _resolve_surrogate_target(self, df_labels: pd.DataFrame) -> Tuple[np.ndarray, str]:
        """Return training energies in kcal/mol and their target name."""
        target = str(self.cfg.surrogate.target).strip()

        # aliases for user-friendly config
        alias = {
            "dft_energy": "e_kcalmol_abs",
            "dft_energy_kcalmol": "e_kcalmol_abs",
            "dft_energy_rel": "e_kcalmol_rel",
            "hartree": "e_hartree",
            "energy_hartree": "e_hartree",
        }
        key = alias.get(target, target)

        if key not in df_labels.columns:
            raise KeyError(
                f"Surrogate target '{target}' resolved to '{key}', but '{key}' not found in labels dataframe. "
                f"Available columns: {list(df_labels.columns)}. "
                "Fix by: (i) updating config surrogate.target, or (ii) ensuring label collection writes the target field."
            )

        y = df_labels[key].astype(float).to_numpy()
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite target: every labeled species requires a fixed energy reference')

        # For e_hartree, convert to kcal/mol to keep downstream columns (mu_kcalmol/sigma_kcalmol) consistent
        if key == "e_hartree":
            y = y * float(HARTREE_TO_KCALMOL)
            return y, "e_hartree(kcal/mol)"

        return y, f"{key}(kcal/mol)"

    def run(self) -> None:
        """Collect completed labels and advance one manual step or an automatic loop."""
        self._selection_exhausted = False
        self.logger.info(
            "[run] Explore | directory=%s gaussian=%s loop=%s",
            self.paths.run_dir, self.cfg.labeling.gaussian.mode, self.cfg.loop.mode,
        )
        snapshot = self.paths.run_dir / "config.yaml"
        if not snapshot.exists():
            save_explore_config(self.cfg, snapshot)
        self._ensure_pool()
        self._collect_and_export()
        if self.cfg.loop.mode == "step":
            if not self._should_stop():
                self.step_once()
                if self.cfg.labeling.gaussian.mode == "local":
                    self._collect_and_export()
        else:
            while True:
                if self.reference_oracle is not None and self._pending_batch_jobs():
                    iteration = self.state.last_iter_submitted
                elif self._should_stop():
                    break
                else:
                    _, iteration = self.step_once()
                if iteration is None:
                    break
                finished = self.wait_iter_done(iteration)
                self._collect_and_export()
                if not finished:
                    break
        self._report_outcome()

    def _pending_batch_jobs(self) -> list[Path]:
        iteration = self.state.last_iter_submitted
        if iteration is None:
            return []
        metadata = sorted(self.paths.jobs_dir(iteration).glob("job_*/meta.json"))
        jobs = [meta.parent for meta in metadata]
        if self.reference_oracle is not None:
            refs = self.labels.energy_references()
            species = {str(json.loads(meta.read_text(encoding="utf-8"))["species_id"]) for meta in metadata}
            jobs.extend(self.reference_oracle.active_job_dir(sid) for sid in sorted(species - refs.keys()))
        return [job for job in jobs if self.oracle.job_status(job) == "pending"]

    def _report_outcome(self) -> None:
        pending = self._pending_batch_jobs()
        if pending:
            if self.cfg.labeling.gaussian.mode == "manual":
                self.logger.info("[run] Awaiting Gaussian logs | pending_jobs=%d labels=%d",
                                 len(pending), self.state.labeled)
                self.logger.info("[run] Run the SP and reference Opt inputs and save each log as input.log")
            else:
                self.logger.info("[run] Awaiting Gaussian jobs | pending_jobs=%d labels=%d",
                                 len(pending), self.state.labeled)
        elif self._selection_exhausted or self._should_stop() or not self._eligible_unlabeled_mask().any():
            labels = self.labels.load()
            training = labels.loc[~labels["id"].astype(str).isin(self.state.holdout_ids or [])]
            fitted = self._fit_surrogate(training, "model_final.pkl", "[model]")
            if fitted is None:
                self.logger.warning("[run] Stopped without training labels | rounds=%d failed_labels=%d",
                                    self.state.iter, self.state.failed)
            else:
                self.logger.info("[run] Finished | rounds=%d labels=%d failed_labels=%d",
                                 self.state.iter, self.state.labeled, self.state.failed)
            return
        else:
            self.logger.info("[run] Ready for the next round | labels=%d", self.state.labeled)
        self.logger.info('[run] Continue with tsgen explore resume --run-dir "%s"', self.paths.run_dir)

    def _round_tag(self, iteration: int) -> str:
        max_iters = int(self.cfg.stop.max_iters or self.cfg.run.max_iters)
        return f"[round {iteration + 1}/{max_iters}]"

    def _collect_and_export(self) -> None:
        previous_failures = self.state.failed
        collected_ok, collected_fail = self._collect_oracle_labels()
        self._write_exports()
        if collected_ok or collected_fail != previous_failures:
            self.logger.info("[labels] Collected | new=%d total=%d failed_labels=%d",
                             collected_ok, self.state.labeled, collected_fail)
            self._write_metrics(submitted_this_iter=0, collected_ok=collected_ok, collected_fail=collected_fail)
        save_state(self.paths.run_dir / "state.json", self.state)

    def step_once(self) -> tuple[bool, int | None]:
        """Execute a single recoverable AL step.

        Returns:
            (did_submit, iter_submitted_or_None)
        """
        self._ensure_pool()
        collect_ok, collect_fail = self._collect_oracle_labels()

        if self.reference_oracle is not None and self._pending_batch_jobs():
            return False, self.state.last_iter_submitted

        # Stage determination
        self._update_stage()

        # DFT budget
        budget = int(self.cfg.stop.dft_budget or self.cfg.run.dft_budget)
        if self.state.dft_used >= budget:
            self.logger.info(f"DFT budget exhausted: {self.state.dft_used}/{budget}.")
            self._write_metrics(submitted_this_iter=0, collected_ok=collect_ok, collected_fail=collect_fail)
            save_state(self.paths.run_dir / "state.json", self.state)
            # Keep exports up-to-date even when we stop early
            try:
                self._write_exports()
            except Exception:
                pass
            return False, None

        # Label legacy anchors up to their bootstrap target.
        submitted_this_iter = 0
        it = self.state.iter
        if self.cfg.anchors.enabled and not self.cfg.holdout.size:
            n_holdout_labeled = self._count_labeled_ids(set(self.state.holdout_ids or []))
            if n_holdout_labeled < int(self.cfg.anchors.min_labeled):
                k = min(int(self.cfg.acquisition.k_select), budget - self.state.dft_used)
                n_need = int(self.cfg.anchors.min_labeled) - n_holdout_labeled
                k = min(k, max(0, n_need))
                if k > 0:
                    idx = self._select_holdout_unlabeled(k=k)
                    if idx:
                        submitted_this_iter = self._submit_selected(idx)
                        # advance iteration for holdout bootstrap submissions as well
                        self.state.last_iter_submitted = it
                        self.state.iter += 1
                        self._write_metrics(submitted_this_iter=submitted_this_iter, collected_ok=collect_ok, collected_fail=collect_fail)
                        save_state(self.paths.run_dir / "state.json", self.state)
                        try:
                            self._write_exports()
                        except Exception:
                            pass
                        return submitted_this_iter > 0, it

        k = min(int(self.cfg.acquisition.k_select), budget - self.state.dft_used)
        if k <= 0:
            self._write_metrics(submitted_this_iter=0, collected_ok=collect_ok, collected_fail=collect_fail)
            save_state(self.paths.run_dir / "state.json", self.state)
            try:
                self._write_exports()
            except Exception:
                pass
            return False, None

        self.logger.info("%s Selecting candidates | eligible=%d batch_limit=%d",
                         self._round_tag(it), int(self._eligible_unlabeled_mask().sum()), k)
        selected = self._select_next_batch(k=k)
        if not selected:
            self._selection_exhausted = True
            self.logger.info("%s No candidates available under the selection policy", self._round_tag(it))
            self._write_metrics(submitted_this_iter=0, collected_ok=collect_ok, collected_fail=collect_fail)
            save_state(self.paths.run_dir / "state.json", self.state)
            try:
                self._write_exports()
            except Exception:
                pass
            return False, None

        submitted_this_iter = self._submit_selected(selected)

        # Record the new iteration after preparing its jobs.
        self.state.last_iter_submitted = it
        self.state.iter += 1

        self._write_metrics(submitted_this_iter=submitted_this_iter, collected_ok=collect_ok, collected_fail=collect_fail)
        save_state(self.paths.run_dir / "state.json", self.state)
        try:
            self._write_exports()
        except Exception:
            pass
        return submitted_this_iter > 0, it

    def wait_iter_done(self, it: int) -> bool:
        """Poll a fixed submission list and record snapshots in jobs_status.csv."""
        loop = self.cfg.loop
        it_dir = self.paths.iter_dir(it)

        submitted_csv = it_dir / "jobs_submitted.csv"
        status_csv = it_dir / "jobs_status.csv"
        # Backward compatible fallback
        src_csv = submitted_csv if submitted_csv.exists() else (it_dir / "jobs_manifest.csv")
        if not src_csv.exists():
            return True

        import time

        t0 = time.time()
        poll = max(5, int(loop.poll_seconds))
        max_wait = int(loop.max_wait_seconds)

        df0 = pd.read_csv(src_csv)
        if "job_dir" not in df0.columns or len(df0) == 0:
            return True
        job_dirs = [Path(p) for p in df0["job_dir"].astype(str).tolist()]

        ref_species = []
        if self.reference_oracle is not None:
            ref_species = sorted(set(df0["species_id"]) - self.labels.energy_references().keys())
        n_total = len(job_dirs) + len(ref_species)
        last_logged = None

        while True:
            statuses = [self.oracle.job_status(jd) for jd in job_dirs]
            df = df0.copy()
            df["status"] = statuses
            if ref_species:
                self.reference_oracle.collect()
                ref_jobs = []
                for sid in ref_species:
                    job = self.reference_oracle.active_job_dir(sid)
                    status = "failed" if sid in self.reference_oracle.failures else self.oracle.job_status(job)
                    ref_jobs.append({"species_id": sid, "job_dir": job, "status": status})
                df = pd.concat([df, pd.DataFrame(ref_jobs)], ignore_index=True)
            df.to_csv(status_csv, index=False)

            n_ok = int((df["status"] == "ok").sum())
            n_fail = int((df["status"] == "failed").sum())
            n_done = n_ok + n_fail
            n_pending = n_total - n_done
            ok_frac = (n_ok / n_total) if n_total > 0 else 0.0
            snapshot = (n_ok, n_fail, n_pending)

            if snapshot != last_logged:
                self.logger.info("%s Gaussian jobs | ok=%d failed=%d pending=%d",
                                 self._round_tag(it), n_ok, n_fail, n_pending)
                last_logged = snapshot

            # completion criteria
            if n_done == n_total:
                return True
            if self.reference_oracle is None and (not loop.require_all_ok) and (ok_frac >= float(loop.min_ok_frac)) and (n_done > 0):
                return True

            if (time.time() - t0) > max_wait:
                self.logger.info(f"Max wait exceeded ({max_wait}s). Resume after the jobs complete.")
                return False

            time.sleep(poll)

    def _ensure_pool(self, *, create_holdout: bool = True) -> None:
        # Build df/X/frames if not present
        if (
            getattr(self.pool, "df", None) is None
            or getattr(self.pool, "X", None) is None
            or getattr(self.pool, "frames", None) is None
        ):
            # CandidatePool.initialize persists pool_index/features; safe for resume.
            self.pool.initialize(self.paths.pool_dir, logger=self.logger)

        # Reapply after threshold or reference changes, including resumed pools.
        df = self.pool.df
        signature = self.gater.signature()
        if ("passed_gating" not in df.columns or 'gating_signature' not in df.columns
                or not df['gating_signature'].eq(signature).all()):
            passed = np.ones((len(df),), dtype=bool)
            try:
                kept_frames, df_gate = self.gater.filter(self.pool.frames, df)
                passed = df_gate.get('keep', pd.Series([True]*len(df))).to_numpy(dtype=bool)
                # keep df_gate columns for audit
                for c in df_gate.columns:
                    df[c] = df_gate[c].values
                # if filtering removed frames, keep the original pool.frames unchanged; passed_gating mask controls eligibility
            except Exception as exc:
                raise RuntimeError('Physics gating failed; refusing to accept an unchecked pool') from exc
            self.pool.df["passed_gating"] = passed
            audit_columns = [c for c in ('id', 'species_id', 'passed_gating', 'reasons', 'min_dist',
                             'rho_max', 'theta_min', 'rho_status', 'reference_bond_count',
                             'gating_reference_id', 'gating_signature') if c in df.columns]
            self.pool.df[audit_columns].to_csv(self.paths.pool_dir / 'geometry_screening.csv', index=False)
            self.logger.info("[pool] Geometry screening | retained=%d rejected=%d",
                             int(passed.sum()), int((~passed).sum()))

        if not create_holdout:
            self.pool.df.to_csv(self.paths.pool_dir / "pool_index.csv", index=False)
            return
        if self.cfg.holdout.size:
            from tsgen.data.xyz import Frame, write_xyz_multiframe
            from tsgen.explore.holdout import grouped_holdout

            split_path = self.paths.run_dir / "pool_split.csv"
            if split_path.exists():
                split = pd.read_csv(split_path, dtype={"id": str, "split_group": str})
                eligible_ids = set(df.loc[df["passed_gating"], "id"])
                if set(split["id"]) != eligible_ids:
                    raise ValueError("Screened candidates changed after holdout creation; start a new run")
            else:
                split = grouped_holdout(df, self.pool.X, self.cfg.holdout, self.cfg.run.seed)
                split.to_csv(split_path, index=False)
            split = split.set_index("id")
            for column in ("split_group", "soap_cluster", "is_holdout"):
                df[column] = df["id"].map(split[column])
            keep = df["is_holdout"].eq(True)
            holdout = df.loc[keep]
            if len(holdout) != self.cfg.holdout.size or holdout["is_seed"].any():
                raise ValueError("Persisted holdout does not match the configured protocol")
            ids = holdout["id"].tolist()
            if self.state.holdout_ids and ids != self.state.holdout_ids:
                raise ValueError("The reserved test set changed; start a new run")
            self.state.holdout_ids = ids
            holdout.to_csv(self.paths.run_dir / "holdout.csv", index=False)
            frames = (Frame(f.symbols, f.coords, f"id={iid} {f.comment}") for f, iid, flag in
                      zip(self.pool.frames, df["id"], keep) if flag)
            write_xyz_multiframe(self.paths.run_dir / "holdout.xyz", frames, precision=16)
            save_state(self.paths.run_dir / "state.json", self.state)
            self.labels.excluded_ids = set(ids)
            df.to_csv(self.paths.pool_dir / "pool_index.csv", index=False)
            return

        # Anchors selection (for holdout evaluation) is deterministic and persisted by AnchorManager
        try:
            anchors = self.anchors.init_anchors(df_pool=self.pool.df, X_pool=self.pool.X, run_dir=self.paths.run_dir)
            if self.cfg.anchors.enabled and len(anchors) > 0 and not (self.state.holdout_ids and len(self.state.holdout_ids) > 0):
                # select a fixed holdout subset from anchors
                n_hold = max(1, int(round(float(self.cfg.anchors.holdout_frac) * len(anchors))))
                hold = anchors.sample(n=min(n_hold, len(anchors)), random_state=int(self.cfg.run.seed)).copy()
                self.state.holdout_ids = hold['id'].astype(str).tolist()
                hold.to_csv(self.paths.run_dir / 'holdout.csv', index=False)
                save_state(self.paths.run_dir / 'state.json', self.state)
        except Exception:
            pass

        self.labels.excluded_ids = set(self.state.holdout_ids or [])

    def _eligible_unlabeled_mask(self) -> np.ndarray:
        df = self.pool.df
        passed = df.get("passed_gating", pd.Series([True] * len(df))).to_numpy(dtype=bool)
        status = df.get("status", pd.Series(["unlabeled"] * len(df))).astype(str).to_numpy()
        unlabeled = status == "unlabeled"
        protected=df['id'].astype(str).isin(self.state.holdout_ids or []).to_numpy()
        eligible = passed & unlabeled & ~protected
        if self.cfg.acquisition.trajectory_only_after_initialization and self.state.labeled:
            eligible &= ~df["is_seed"].to_numpy(dtype=bool)
        if self.reference_oracle is not None and self.reference_oracle.failures:
            eligible &= ~df["species_id"].isin(self.reference_oracle.failures).to_numpy()
        return eligible

    # Oracle collection
    def _collect_oracle_labels(self) -> tuple[int, int]:
        """Refresh labels and pool state, returning new-label and failure counts."""
        failed_references = {}
        if self.reference_oracle is not None:
            self.reference_oracle.collect()
            failed_references = self.reference_oracle.failures
        existing = self.labels.load()
        existing_ids = set(existing["id"].astype(str).tolist()) if not existing.empty else set()

        # A submission may reach disk before the iteration counter is saved.
        iter_nums: set[int] = set()
        for p in self.paths.run_dir.glob("iter_*"):
            if not p.is_dir():
                continue
            try:
                iter_nums.add(int(p.name.split("_")[-1]))
            except Exception:
                continue
        # Backward-compatible safety: also include range up to state.iter.
        try:
            for it in range(int(self.state.iter) + 1):
                iter_nums.add(it)
        except Exception:
            pass

        new_records: List[LabelRecord] = []
        n_failed = 0
        max_collected_iter = None

        for it in sorted(iter_nums):
            jobs_dir = self.paths.jobs_dir(it)
            if not jobs_dir.exists():
                continue

            for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
                if (job_dir / "meta.json").exists() and self.oracle.job_status(job_dir) == "pending":
                    self.oracle.submit(job_dir)
                st = self.oracle.job_status(job_dir)
                if st == "failed":
                    n_failed += 1
                elif st == "ok" and failed_references and (job_dir / "meta.json").is_file():
                    meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
                    n_failed += int(meta["species_id"] in failed_references)

            recs = self.oracle.collect_labels(jobs_dir, iter_idx=it)
            for r in recs:
                if r.id and r.id not in existing_ids:
                    new_records.append(r)
                    existing_ids.add(r.id)
                    max_collected_iter = it if max_collected_iter is None else max(max_collected_iter, it)

        if self.reference_oracle is not None:
            refs = self.labels.energy_references()
            new_records = [record for record in new_records if record.species_id in refs]
        if new_records:
            self.labels.append(new_records)

        # Always reload and rewrite the tabular label view. This matters for
        # resumed runs where jsonl already contains labels but pool_index.csv was
        # stale in an older version.
        df_labels = self.labels.load()
        if not df_labels.empty:
            try:
                df_labels.to_csv(self.paths.labels_jsonl.with_suffix(".csv"), index=False)
            except Exception:
                pass

        # Synchronize pool bookkeeping from label store and job manifests.
        try:
            self._sync_pool_with_jobs_and_labels(df_labels)
        except Exception as exc:
            self.logger.warning(f"Pool/label synchronization failed: {exc}")

        if max_collected_iter is not None:
            self.state.last_iter_collected = int(max_collected_iter)

        self.state.failed = n_failed
        return int(len(new_records)), int(n_failed)

    def _sync_pool_with_jobs_and_labels(self, df_labels: pd.DataFrame | None = None) -> None:
        """Restore pool status and provenance from job manifests and labels."""
        if getattr(self.pool, "df", None) is None:
            return
        df = self.pool.df
        if df_labels is None:
            df_labels = self.labels.load()

        # Ensure columns exist.
        defaults = {
            "status": "unlabeled",
            "selected_iter": -1,
            "job_dir": "",
            "slurm_jobid": "",
            "label_E_hartree": np.nan,
            "label_E_kcalmol": np.nan,
            "label_E_kcalmol_rel": np.nan,
        }
        for c, v in defaults.items():
            if c not in df.columns:
                df[c] = v

        id_to_row = {str(iid): j for j, iid in enumerate(df["id"].astype(str).tolist())}

        # Mark submitted/prepared jobs from submitted manifests.
        for it_dir in sorted(self.paths.run_dir.glob("iter_*")):
            if not it_dir.is_dir():
                continue
            try:
                it = int(it_dir.name.split("_")[-1])
            except Exception:
                continue
            for manifest_name in ("jobs_submitted.csv", "jobs_manifest.csv"):
                mf = it_dir / manifest_name
                if not mf.exists():
                    continue
                try:
                    jm = pd.read_csv(mf)
                except Exception:
                    continue
                if "id" not in jm.columns:
                    continue
                for _, r in jm.iterrows():
                    iid = str(r.get("id", ""))
                    if iid not in id_to_row:
                        continue
                    j = id_to_row[iid]
                    # Do not overwrite labeled with submitted.
                    if str(df.at[j, "status"]) != "labeled":
                        df.at[j, "status"] = str(r.get("status", "submitted")) if str(r.get("status", "")) else "submitted"
                    df.at[j, "selected_iter"] = int(it)
                    if "job_dir" in jm.columns:
                        df.at[j, "job_dir"] = str(r.get("job_dir", ""))
                    if "jobid" in jm.columns and pd.notna(r.get("jobid", None)):
                        df.at[j, "slurm_jobid"] = str(r.get("jobid", ""))

        prepared_ids = set()
        prepared_iterations = []
        for job_dir in self.paths.run_dir.glob("iter_*/gaussian/jobs/job_*"):
            metadata = job_dir / "meta.json"
            if not metadata.is_file() or not (job_dir / "input.gjf").is_file():
                continue
            iid = str(json.loads(metadata.read_text(encoding="utf-8"))["id"])
            if iid not in id_to_row:
                raise ValueError(f"Gaussian job lies outside the candidate pool: {job_dir}")
            prepared_ids.add(iid)
            iteration = int(job_dir.parents[2].name.split("_")[-1])
            prepared_iterations.append(iteration)
            row = id_to_row[iid]
            df.at[row, "selected_iter"] = iteration
            df.at[row, "job_dir"] = str(job_dir)
            if str(df.at[row, "status"]) != "labeled":
                status = self.oracle.job_status(job_dir)
                ref_failed = (
                    self.reference_oracle is not None
                    and df.at[row, "species_id"] in self.reference_oracle.failures
                )
                df.at[row, "status"] = "failed" if status == "failed" or (status == "ok" and ref_failed) else "prepared"
        if prepared_iterations:
            latest = max(prepared_iterations)
            self.state.iter = max(self.state.iter, latest + 1)
            self.state.last_iter_submitted = max(self.state.last_iter_submitted or 0, latest)
        self.state.selected = max(self.state.selected, len(prepared_ids))
        self.state.submitted = max(self.state.submitted, len(prepared_ids))

        # Mark successful labels and energies from label store.
        if df_labels is not None and not df_labels.empty:
            for _, r in df_labels.iterrows():
                iid = str(r.get("id", ""))
                if iid not in id_to_row:
                    continue
                j = id_to_row[iid]
                df.at[j, "status"] = "labeled"
                try:
                    df.at[j, "selected_iter"] = int(r.get("iter", df.at[j, "selected_iter"]))
                except Exception:
                    pass
                df.at[j, "job_dir"] = str(r.get("job_dir", df.at[j, "job_dir"]))
                for src, dst in [
                    ("e_hartree", "label_E_hartree"),
                    ("e_kcalmol_abs", "label_E_kcalmol"),
                    ("e_kcalmol_rel", "label_E_kcalmol_rel"),
                ]:
                    try:
                        df.at[j, dst] = float(r.get(src))
                    except Exception:
                        pass
                if iid not in self.state.labeled_ids:
                    self.state.labeled_ids.append(iid)

        self.state.labeled = int(df["status"].astype(str).eq("labeled").sum())
        # Budget is counted as successful/attempted DFT labels in existing code.
        # Preserve the larger value so submitted-budget semantics are not lost.
        try:
            counted_ids = prepared_ids | set(self.state.labeled_ids)
            self.state.dft_used = max(int(self.state.dft_used), len(counted_ids - set(self.state.inherited_label_ids or [])))
        except Exception:
            self.state.dft_used = len(set(self.state.labeled_ids) - set(self.state.inherited_label_ids or []))

        try:
            self.pool_dir = self.paths.pool_dir
            df.to_csv(self.paths.pool_dir / "pool_index.csv", index=False)
        except Exception:
            pass

    def _fit_surrogate(self, training: pd.DataFrame, filename: str, stage: str):
        id_to_row = {str(iid): j for j, iid in enumerate(self.pool.df["id"])}
        rows = [id_to_row[str(iid)] for iid in training["id"] if str(iid) in id_to_row]
        if not rows:
            return None
        if len(rows) != len(training):
            raise ValueError("Training labels do not align with the candidate pool")
        X_train = self.pool.X[rows]
        y_train, target = self._resolve_surrogate_target(training)
        self.logger.info("%s Fitting %s | labels=%d target=%s", stage, self.cfg.surrogate.kind, len(rows), target)
        surrogate = build_surrogate(self.cfg.surrogate)
        surrogate.fit(X_train, y_train)
        snapshot = self.paths.run_dir / "surrogate" / filename
        temporary = snapshot.with_suffix(".pkl.tmp")
        surrogate.save(temporary)
        temporary.replace(snapshot)
        self.logger.info("%s Saved %s", stage, snapshot)
        return surrogate, X_train

    # Selection / submission
    def _select_next_batch(self, k: int) -> List[int]:
        """Return selected pool indices (row positions in df)."""
        df = self.pool.df.copy()
        X = self.pool.X

        # eligible unlabeled
        mask_u = self._eligible_unlabeled_mask()
        idx_u = np.where(mask_u)[0]
        if idx_u.size == 0:
            return []

        # Build training set from labeled ids excluding holdout
        df_labels = self.labels.load()
        if df_labels.empty:
            # cold start: select initial labels from seeds + anchors for coverage
            return self._cold_start_select(k, idx_u)

        # training ids (exclude holdout)
        holdout_ids = set(self.state.holdout_ids or [])
        df_train = df_labels[~df_labels["id"].astype(str).isin(holdout_ids)].copy()
        fitted = self._fit_surrogate(df_train, f"model_iter_{self.state.iter:03d}.pkl", self._round_tag(self.state.iter))
        if fitted is None:
            return self._cold_start_select(k, idx_u)
        surrogate, X_train = fitted

        X_u = X[idx_u]
        mu, sigma = surrogate.predict(X_u)

        # Use Acquisition module: expects a dataframe with 'sigma' and aligned X (candidate subset)
        df_cand = df.iloc[idx_u].copy().reset_index(drop=False)  # keep original row index in 'index'
        labeled_counts = dict(pd.Series(df_train["species_id"].astype(str)).value_counts())
        df_cand["mu"] = mu
        df_cand["sigma"] = sigma
        policy = self.cfg.acquisition.representative
        df_scored = self.acq.compute(
            df=df_cand, X=X_u, labeled_X=X_train, labeled_species_counts=labeled_counts,
            stable_scaling=policy is not None,
        )
        if policy is None:
            self.acq.cfg.k_select = int(k)
            sel_rows = self.acq.select_indices(df_scored=df_scored, X_scored=X_u)
        else:
            informative = has_informative_uncertainty(sigma) and np.isfinite(df_scored["score"]).all()
            selection_mode = "representative_gp" if informative else "representative_coverage"
            sel_rows = select_representatives(
                df_scored, X_u, k, X_train, set(labeled_counts),
                min_support_frames=policy.min_support_frames,
                new_species_slots=policy.new_species_slots,
                scores=df_scored["score"].to_numpy() if informative else None,
                preselect_factor=self.cfg.acquisition.preselect_factor,
            )
            df_scored["selection_mode"] = selection_mode
            df_scored["selected"] = np.isin(np.arange(len(df_scored)), sel_rows)
            self.logger.info(
                "%s %s | selected=%d sigma_min=%.6g sigma_max=%.6g",
                self._round_tag(self.state.iter), selection_mode, len(sel_rows),
                np.min(sigma), np.max(sigma),
            )

        selected_idx = idx_u[sel_rows].astype(int).tolist()
        if not selected_idx and policy is None:
            # fallback: top-k by sigma
            order = np.argsort(-sigma)[:k]
            selected_idx = idx_u[order].astype(int).tolist()

        # write per-iter scored.csv (full scored table)
        it_dir = self.paths.iter_dir(self.state.iter)
        it_dir.mkdir(parents=True, exist_ok=True)
        df_scored["mu_kcalmol"] = mu
        df_scored["sigma_kcalmol"] = sigma
        df_scored.sort_values("score", ascending=False).to_csv(it_dir / "scored.csv", index=False)

        return selected_idx[:k]

    def _cold_start_select(self, k: int, idx_u: np.ndarray) -> List[int]:
        """Initialize oracle labels when none exist (cold start).

        Goal: produce a diverse, chemically reasonable initial labeled set.
        Strategy:
          - prioritize seed geometries
          - fill remaining by farthest-point sampling (FPS) in descriptor space
        """
        df = self.pool.df
        X = self.pool.X

        def fps(Xsub: np.ndarray, ksub: int) -> List[int]:
            if ksub <= 0 or Xsub.size == 0:
                return []
            # deterministic start: point with largest norm
            norms = np.linalg.norm(Xsub, axis=1)
            first = int(np.argmax(norms))
            return fps_indices(Xsub, ksub, start=first)

        seed_mask = df["is_seed"].to_numpy(dtype=bool)
        idx_seed_u = idx_u[seed_mask[idx_u]]

        selected: List[int] = []

        policy = self.cfg.acquisition.representative
        k_seed = min(len(idx_seed_u), k if policy is None else min(k, policy.seed_slots))
        if k_seed > 0:
            sel = fps(X[idx_seed_u], k_seed)
            selected.extend(idx_seed_u[np.array(sel, dtype=int)].tolist())

        k_rem = k - len(selected)
        if policy is not None:
            remaining = idx_u[~seed_mask[idx_u]]
            paired = set(df.iloc[selected]["species_id"].astype(str))
            trajectory = select_representatives(
                df.iloc[remaining], X[remaining], k_rem, X[selected], set(),
                min_support_frames=policy.min_support_frames,
                new_species_slots=policy.new_species_slots,
                paired_species=paired,
            )
            self.logger.info(
                "%s Representative bootstrap | seeds=%d trajectory=%d",
                self._round_tag(self.state.iter), len(selected), len(trajectory),
            )
            selected.extend(remaining[trajectory].astype(int).tolist())
        elif k_rem > 0:
            remaining = idx_u[~np.isin(idx_u, selected)]
            sel = fps(X[remaining], k_rem)
            selected.extend(remaining[np.array(sel, dtype=int)].tolist())

        return selected[:k]

    def _export_cumulative(self) -> None:
        """Write cumulative exports for human inspection."""
        df = self.pool.df
        frames = self.pool.frames
        from tsgen.data.xyz import write_xyz_multiframe

        # filtered pool
        try:
            passed = df.get("passed_gating", pd.Series([True] * len(df))).to_numpy(dtype=bool)
            write_xyz_multiframe(self.paths.filtered_pool_xyz, [fr for fr, ok in zip(frames, passed) if bool(ok)])
        except Exception:
            pass

        # selected cumulative
        try:
            sel_mask = df["selected_iter"].ge(0).to_numpy()
            write_xyz_multiframe(self.paths.selected_xyz, [fr for fr, ok in zip(frames, sel_mask) if ok],
                                 precision=16)
        except (OSError, ValueError, KeyError) as exc:
            self.logger.warning("[export] Could not write %s | %s", self.paths.selected_xyz, exc)

        # labeled ok cumulative
        try:
            lab_mask = df.get("status", pd.Series(["unlabeled"] * len(df))).astype(str).eq("labeled").to_numpy()
            write_xyz_multiframe(self.paths.labeled_ok_xyz, [fr for fr, ok in zip(frames, lab_mask) if bool(ok)])
        except Exception:
            pass

        # trainset: seeds + labeled_ok unique by id
        try:
            keep = df["is_seed"].to_numpy(dtype=bool) | df.get("status", pd.Series(["unlabeled"] * len(df))).astype(
                str
            ).eq("labeled").to_numpy()
            keep &= ~df['id'].astype(str).isin(self.state.holdout_ids or []).to_numpy()
            seen = set()
            out_frames = []
            for fr, rid, ok in zip(frames, df["id"].astype(str).tolist(), keep):
                if not ok:
                    continue
                if rid in seen:
                    continue
                seen.add(rid)
                out_frames.append(fr)
            write_xyz_multiframe(self.paths.trainset_xyz, out_frames)
        except Exception:
            pass

    def _write_exports(self) -> None:
        """Write the run's structure and training-set exports."""
        try:
            self._export_cumulative()
        except Exception:
            pass

        if self.reference_oracle is not None:
            from tsgen.explore.references import export_optimized_references

            try:
                export_optimized_references(self.labels, self.paths.optimized_references_dir)
            except (OSError, ValueError) as exc:
                self.logger.warning("[export] Could not write optimized references | %s", exc)

        # Read completed labels directly so exports also work on resumed runs.
        try:
            self._export_validated_new_species_from_labels(self.paths.new_species_xyz_dir)
        except Exception as exc:
            try:
                self.logger.warning(f"Validated new-species export failed: {exc}")
            except Exception:
                pass

        # Backward-compatible multi-frame export for quick preview
        try:
            df = self.pool.df
            frames = self.pool.frames
            if df is not None and frames is not None and len(df) == len(frames):
                mask = (
                    df.get("status", pd.Series(["unlabeled"] * len(df))).astype(str).eq("labeled")
                    & df.get("selected_iter", pd.Series([-1] * len(df))).astype(int).ge(0)
                    & (~df.get("is_seed", pd.Series([False] * len(df))).astype(bool))
                    & (~df['id'].astype(str).isin(self.state.holdout_ids or []))
                )
                from tsgen.data.xyz import write_xyz_multiframe
                write_xyz_multiframe(self.paths.new_species_xyz, [fr for fr, ok in zip(frames, mask.to_numpy()) if bool(ok)])
        except Exception:
            pass

        # Optional trainset post-processing: de-dup representatives, optional
        # Gaussian optimization jobs, and ReaxFF GEO export. Controlled by
        # cfg.export.scheme so the existing AL behavior remains the default.
        try:
            export_trainset_products(self.paths.run_dir, self.cfg, logger=self.logger)
        except Exception as exc:
            try:
                self.logger.warning(f"Trainset post-processing export failed: {exc}")
            except Exception:
                pass

    def _resolve_job_dir_from_label(self, label_row: pd.Series) -> Path | None:
        """Resolve absolute or legacy relative paths to labeled Gaussian jobs."""
        iid = str(label_row.get("id", ""))
        raw = str(label_row.get("job_dir", "") or "")
        candidates: list[Path] = []
        if raw:
            p = Path(raw)
            candidates.append(p)
            if not p.is_absolute():
                candidates.append(self.paths.run_dir / raw)
                candidates.append(self.paths.run_dir.parent / raw)
        if iid:
            candidates.extend(sorted(self.paths.run_dir.glob(f"iter_*/gaussian/jobs/job_{iid}")))
        for c in candidates:
            try:
                if c.exists() and c.is_dir():
                    return c
            except Exception:
                continue
        return None

    def _export_validated_new_species_from_labels(self, out_dir: Path) -> None:
        """Export one sampled XYZ per successfully labeled non-seed configuration."""
        import csv
        import re
        import shutil

        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / "index.csv"
        readme_path = out_dir / "README.txt"

        df_labels = self.labels.load()
        cols = [
            "id", "species_id", "iter", "is_seed", "e_hartree", "e_kcalmol_abs",
            "e_kcalmol_rel", "job_dir", "xyz_file"
        ]
        if not df_labels.empty:
            df_labels=df_labels[~df_labels['id'].astype(str).isin(self.state.holdout_ids or [])]

        # Build id -> is_seed map from pool if available. Missing IDs are treated
        # as non-seed because labels come from actively submitted jobs.
        seed_map: dict[str, bool] = {}
        try:
            if getattr(self.pool, "df", None) is not None and "id" in self.pool.df.columns:
                seed_map = dict(
                    zip(
                        self.pool.df["id"].astype(str),
                        self.pool.df.get("is_seed", pd.Series([False] * len(self.pool.df))).astype(bool),
                    )
                )
        except Exception:
            seed_map = {}

        exported = 0
        skipped_seed = 0
        skipped_missing_structure = 0
        total_ok = 0

        with index_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)

            if df_labels.empty:
                readme_path.write_text(
                    "No validated structures were available to export because labels/labels.csv is empty.\n"
                    "Run or resume the active-learning loop until Gaussian jobs finish and are collected.\n",
                    encoding="utf-8",
                )
                return

            # Only successful labels are validated structures.
            if "status" in df_labels.columns:
                df_ok = df_labels[df_labels["status"].astype(str).str.lower().isin(["ok", "labeled", "success", "done"])].copy()
            else:
                df_ok = df_labels.copy()
            total_ok = int(len(df_ok))

            for _, r in df_ok.iterrows():
                iid = str(r.get("id", ""))
                if not iid:
                    continue
                is_seed = bool(seed_map.get(iid, False))
                if is_seed:
                    skipped_seed += 1
                    continue

                job_dir = self._resolve_job_dir_from_label(r)
                xyz_src = (job_dir / "structure.xyz") if job_dir is not None else None
                if xyz_src is None or not xyz_src.exists():
                    skipped_missing_structure += 1
                    continue

                sp = str(r.get("species_id", "unknown"))
                safe_sp = re.sub(r"[^A-Za-z0-9_.-]+", "_", sp)[:80]
                xyz_name = f"{safe_sp}__{iid}.xyz"
                xyz_dst = out_dir / xyz_name
                shutil.copyfile(xyz_src, xyz_dst)

                w.writerow([
                    iid,
                    sp,
                    r.get("iter", ""),
                    is_seed,
                    r.get("e_hartree", ""),
                    r.get("e_kcalmol_abs", ""),
                    r.get("e_kcalmol_rel", ""),
                    str(job_dir) if job_dir is not None else "",
                    xyz_name,
                ])
                exported += 1

        readme_path.write_text(
            "Labeled non-seed configurations from TSGen Explore.\n"
            f"Total successful labels observed: {total_ok}\n"
            f"Exported non-seed structures: {exported}\n"
            f"Skipped seed structures: {skipped_seed}\n"
            f"Skipped records with missing structure.xyz: {skipped_missing_structure}\n"
            "\nDefinition: exported structures are successful Gaussian labels whose source pool entry is not marked as seed.\n"
            "Coordinates come from the sampled geometries used for single-point energies.\n"
            "See ../optimized_references/ for the separately optimized species geometries.\n",
            encoding="utf-8",
        )

    def _save_state(self) -> None:
        self.state.labeled = int(
            self.pool.df.get("status", pd.Series(["unlabeled"] * len(self.pool.df))).astype(str).eq("labeled").sum()
        )
        self.state.selected = int(
            self.pool.df.get("selected_iter", pd.Series([-1] * len(self.pool.df))).astype(int).ge(0).sum()
        )
        save_state(self.paths.run_dir / "state.json", self.state)
    # Long-running controls
    def _should_stop(self) -> bool:
        stop = self.cfg.stop
        max_iters = int(stop.max_iters or self.cfg.run.max_iters)
        if self.state.iter >= max_iters:
            return True
        budget = int(stop.dft_budget or self.cfg.run.dft_budget)
        if self.state.dft_used >= budget:
            return True

        # soft stops (only if configured and surrogate has been trained at least once)
        metrics_csv = self.paths.run_dir / "metrics.csv"
        if not metrics_csv.exists():
            return False
        try:
            dfm = pd.read_csv(metrics_csv)
            if len(dfm) == 0:
                return False
            last = dfm.iloc[-1]
            if stop.sigma_mean_below is not None and pd.notna(last.get("mean_sigma_pool")):
                if float(last["mean_sigma_pool"]) <= float(stop.sigma_mean_below):
                    return True
            if stop.no_new_species_rounds is not None and self.state.recent_new_species:
                w = int(stop.no_new_species_rounds)
                if len(self.state.recent_new_species) >= w and all(int(x) == 0 for x in self.state.recent_new_species[-w:]):
                    return True
        except Exception:
            return False
        return False

    def _update_stage(self) -> None:
        # Determine stage based on labeled counts and species growth
        df_labels = self.labels.load()
        labeled_ids = set(df_labels["id"].astype(str).tolist()) if not df_labels.empty else set()
        n_labels = len(labeled_ids - set(self.state.holdout_ids or []))
        n_species = int(df_labels["species_id"].astype(str).nunique()) if not df_labels.empty else 0

        if n_labels < int(self.cfg.stages.bootstrap_min_labels) or n_species < int(self.cfg.stages.bootstrap_min_species):
            self.state.stage = "BOOTSTRAP"
        else:
            # refine heuristic: if new species has stalled recently
            if self.state.recent_new_species and len(self.state.recent_new_species) >= 5 and sum(self.state.recent_new_species[-5:]) == 0:
                self.state.stage = "REFINE"
            else:
                self.state.stage = "EXPLORE"

        # stage-specific acquisition override (non-invasive)
        if self.state.stage == "EXPLORE":
            if self.cfg.stages.explore_method is not None:
                self.cfg.acquisition.method = self.cfg.stages.explore_method
            if self.cfg.stages.explore_ucb_kappa is not None:
                self.cfg.acquisition.ucb_kappa = float(self.cfg.stages.explore_ucb_kappa)
        elif self.state.stage == "REFINE":
            if self.cfg.stages.refine_method is not None:
                self.cfg.acquisition.method = self.cfg.stages.refine_method
            if self.cfg.stages.refine_ucb_kappa is not None:
                self.cfg.acquisition.ucb_kappa = float(self.cfg.stages.refine_ucb_kappa)

    def _count_labeled_ids(self, ids: set[str]) -> int:
        if not ids:
            return 0
        df_labels = self.labels.load()
        if df_labels.empty:
            return 0
        return int(df_labels["id"].astype(str).isin(list(ids)).sum())

    def _select_holdout_unlabeled(self, k: int) -> list[int]:
        """Select k unlabeled holdout points (by id) for oracle labeling."""
        if not self.state.holdout_ids:
            return []
        holdout = set(self.state.holdout_ids)
        df = self.pool.df
        df_labels = self.labels.load()
        labeled = set(df_labels["id"].astype(str).tolist()) if not df_labels.empty else set()

        idx = []
        for j, iid in enumerate(df["id"].astype(str).tolist()):
            if self.reference_oracle is not None and df.at[j, "species_id"] in self.reference_oracle.failures:
                continue
            if iid in holdout and iid not in labeled:
                idx.append(j)
            if len(idx) >= k:
                break
        return idx

    def _submit_selected(self, selected_pool_rows: list[int]) -> int:
        """Prepare + submit Gaussian jobs for selected pool row indices."""
        it = int(self.state.iter)
        it_dir = self.paths.iter_dir(it)
        it_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir = self.paths.jobs_dir(it)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        tag = self._round_tag(it)
        mode = self.cfg.labeling.gaussian.mode
        self.logger.info("%s Preparing Gaussian inputs | selected=%d mode=%s", tag, len(selected_pool_rows), mode)

        # Record the batch before submission.
        try:
            from tsgen.data.xyz import write_xyz_multiframe
            self.pool.df.iloc[selected_pool_rows].to_csv(it_dir / "selected.csv", index=False)
            write_xyz_multiframe(it_dir / "selected.xyz", [self.pool.frames[i] for i in selected_pool_rows])
        except Exception:
            pass

        jobs: list[dict] = []
        submitted = 0
        df = self.pool.df
        reference_starts = {}
        if self.reference_oracle is not None and self.cfg.acquisition.representative is not None:
            available = df["passed_gating"] & ~df["is_seed"]
            available &= ~df["id"].astype(str).isin(self.state.holdout_ids or [])
            by_support = df.loc[available].sort_values("support_frames", ascending=False, kind="stable")
            for idx, row in by_support.drop_duplicates("species_id").iterrows():
                reference_starts[str(row["species_id"])] = self.pool.frames[idx]

        for number, ridx in enumerate(selected_pool_rows, start=1):
            iid = str(df.iloc[ridx]["id"])
            sid = str(df.iloc[ridx].get("species_id", "unknown"))
            job_dir = jobs_dir / f"job_{iid}"
            job_dir.mkdir(parents=True, exist_ok=True)

            # Keep pool bookkeeping current immediately; later label collection
            # will upgrade submitted/prepared -> labeled.
            df.at[ridx, "selected_iter"] = it
            df.at[ridx, "job_dir"] = str(job_dir)

            # Idempotency: if the job was already prepared, do not recreate it.
            if (job_dir / "input.log").exists() or (job_dir / "input.gjf").exists():
                st = self.oracle.job_status(job_dir)
                status = "labeled" if st == "ok" else ("failed" if st == "failed" else "prepared")
                if status != "labeled":
                    df.at[ridx, "status"] = status
                jobs.append({"id": iid, "species_id": sid, "job_dir": str(job_dir), "jobid": None, "status": status})
                continue

            frame = self.pool.frames[ridx]
            spec = GaussianJobSpec(id=iid, species_id=sid, frame=frame)
            if mode != "manual":
                self.logger.info("%s Gaussian candidate %d/%d | id=%s species=%s",
                                 tag, number, len(selected_pool_rows), iid, sid)
            if self.reference_oracle is not None:
                if sid in reference_starts and not bool(df.at[ridx, "is_seed"]):
                    self.reference_oracle.prepare(spec, reference_frame=reference_starts[sid])
                else:
                    self.reference_oracle.prepare(spec)
            self.oracle.write_job(job_dir, spec)
            jobid = None
            try:
                jobid = self.oracle.submit(job_dir)
            except Exception as exc:
                self.logger.error(f"Gaussian submission failed for {job_dir}: {exc}")
                if getattr(self,'strict_submission',False):raise
                jobid = None

            # In local/manual modes submit() may return None but the input files
            # are still valid products. In slurm mode None means prepared but not
            # confirmed submitted; the job can be inspected/retried.
            status = "submitted" if jobid else "prepared"
            df.at[ridx, "status"] = status
            if jobid is not None:
                df.at[ridx, "slurm_jobid"] = str(jobid)
            jobs.append({"id": iid, "species_id": sid, "job_dir": str(job_dir), "jobid": jobid, "status": status})
            submitted += 1

        df_jobs = pd.DataFrame(jobs)
        # Backward compatible manifest + stable submitted/status files.
        df_jobs.to_csv(it_dir / "jobs_manifest.csv", index=False)
        df_jobs[[c for c in ["id", "species_id", "job_dir", "jobid"] if c in df_jobs.columns]].to_csv(
            it_dir / "jobs_submitted.csv", index=False
        )
        df_jobs.to_csv(it_dir / "jobs_status.csv", index=False)
        self.logger.info("%s Gaussian batch prepared | sp_inputs=%d slurm_job_ids=%d",
                         tag, len(jobs), int(df_jobs["jobid"].notna().sum()))
        self.logger.info("%s Selected structures | %s", tag, it_dir / "selected.xyz")

        self.state.selected += len(selected_pool_rows)
        self.state.submitted += submitted
        self.state.dft_used += submitted

        # Persist synchronized pool index for resume/export/debug.
        try:
            df.to_csv(self.paths.pool_dir / "pool_index.csv", index=False)
        except Exception:
            pass
        return submitted

    def _write_metrics(self, submitted_this_iter: int, collected_ok: int, collected_fail: int) -> None:
        df_labels = self.labels.load()
        holdout_ids = set(self.state.holdout_ids or [])
        if df_labels.empty:
            labeled_train = 0
            labeled_holdout = 0
            n_species = 0
        else:
            ids = df_labels["id"].astype(str)
            labeled_holdout = int(ids.isin(list(holdout_ids)).sum())
            labeled_train = int((~ids.isin(list(holdout_ids))).sum())
            n_species = int(df_labels["species_id"].astype(str).nunique())

        # update recent_new_species (for stage/stop heuristics)
        if df_labels.empty:
            new_sp = 0
        else:
            # count species with at least one label
            sp_set = set(df_labels["species_id"].astype(str).tolist())
            prev_sp = set(self.state.previous_species or [])
            new_sp = len(sp_set - prev_sp)
            self.state.previous_species = sorted(sp_set)
        if self.state.recent_new_species is None:
            self.state.recent_new_species = []
        self.state.recent_new_species.append(int(new_sp))
        if len(self.state.recent_new_species) > 50:
            self.state.recent_new_species = self.state.recent_new_species[-50:]

        # mean sigma pool from last scored table if exists
        mean_sigma_pool = None
        max_sigma_pool = None
        mean_sigma_selected = None
        q95_sigma_pool = None
        mean_sigma_pool_over_target_std = None
        q95_sigma_pool_over_target_std = None

        # Attempt compute surrogate metrics only if we can train (>=1 labeled train)
        mae = rmse = nll = cov1 = cov2 = None
        if labeled_train >= 1 and self._eligible_unlabeled_mask().any():
            # Use current surrogate trained on train labels; evaluate on holdout if available
            try:
                # rebuild training data
                df = self.pool.df
                X = self.pool.X
                id_to_row = {str(iid): j for j, iid in enumerate(df["id"].astype(str).tolist())}

                df_train = df_labels[~df_labels["id"].astype(str).isin(list(holdout_ids))].copy()
                df_hold = df_labels[df_labels["id"].astype(str).isin(list(holdout_ids))].copy()

                train_rows = [id_to_row[str(i)] for i in df_train["id"].astype(str).tolist() if str(i) in id_to_row]
                hold_rows = [id_to_row[str(i)] for i in df_hold["id"].astype(str).tolist() if str(i) in id_to_row]

                if len(train_rows) > 0:
                    X_train = X[np.array(train_rows, dtype=int)]
                    y_train, _ = self._resolve_surrogate_target(df_train)
                    surrogate = build_surrogate(self.cfg.surrogate, snapshot=None)
                    surrogate.fit(X_train, y_train)

                    # pool uncertainty summary on eligible unlabeled
                    mask_u = self._eligible_unlabeled_mask()
                    idx_u = np.where(mask_u)[0]
                    if idx_u.size > 0:
                        _, sigma_u = surrogate.predict(X[idx_u])
                        mean_sigma_pool = float(np.mean(sigma_u))
                        max_sigma_pool = float(np.max(sigma_u))
                        q95_sigma_pool = float(np.quantile(sigma_u, 0.95))
                        target_std = float(np.std(y_train)) or 1.0
                        mean_sigma_pool_over_target_std = mean_sigma_pool / target_std
                        q95_sigma_pool_over_target_std = q95_sigma_pool / target_std

                    if len(hold_rows) > 0:
                        X_h = X[np.array(hold_rows, dtype=int)]
                        y_h, _ = self._resolve_surrogate_target(df_hold)
                        mu_h, sigma_h = surrogate.predict(X_h)
                        r = evaluate_holdout(y_h, mu_h, sigma_h)
                        mae, rmse, nll, cov1, cov2 = r["mae"], r["rmse"], r["nll"], r["cov_1sigma"], r["cov_2sigma"]
            except Exception:
                pass

        budget = int(self.cfg.stop.dft_budget or self.cfg.run.dft_budget)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()

        m = ALMetrics(
            iter=int(self.state.iter),
            timestamp=ts,
            stage=str(self.state.stage),
            dft_used=int(self.state.dft_used),
            dft_budget=int(budget),
            n_pool=int(len(self.pool.df)),
            labeled_train=int(labeled_train),
            labeled_holdout=int(labeled_holdout),
            n_species_labeled=int(n_species),
            mae_holdout=mae,
            rmse_holdout=rmse,
            nll_holdout=nll,
            cov_1sigma=cov1,
            cov_2sigma=cov2,
            mean_sigma_selected=mean_sigma_selected,
            mean_sigma_pool=mean_sigma_pool,
            max_sigma_pool=max_sigma_pool,
            q95_sigma_pool=q95_sigma_pool,
            mean_sigma_pool_over_target_std=mean_sigma_pool_over_target_std,
            q95_sigma_pool_over_target_std=q95_sigma_pool_over_target_std,
            submitted_this_iter=int(submitted_this_iter),
            collected_ok_this_iter=int(collected_ok),
            collected_fail_this_iter=int(collected_fail),
        )
        append_metrics_csv(self.paths.run_dir / "metrics.csv", m)
