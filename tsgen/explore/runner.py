from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tsgen.explore.config import ExploreCfg
from tsgen.explore.pool import CandidatePool
from tsgen.explore.gating import PhysicsGater
from tsgen.explore.acquisition import Acquisition
from tsgen.explore.anchors import AnchorManager
from tsgen.explore.io import ExplorePaths, ExploreState, load_state, save_state, write_config_snapshot
from tsgen.explore.label_store import LabelStore, LabelRecord, HARTREE_TO_KCALMOL
from tsgen.explore.oracle_gaussian import GaussianOracle, GaussianJobSpec
from tsgen.explore.surrogate import build_surrogate
from tsgen.explore.metrics import ALMetrics, evaluate_holdout, append_metrics_csv, update_plots
from tsgen.logging import setup_logging
from tsgen.explore.trainset_export import export_trainset_products
from tsgen.selection.fps import fps_indices


class ExploreRunner:
    """One-command, recoverable exploration + oracle-in-the-loop active learning.

    Behavior (HPC-friendly):
      - Each invocation first COLLECTS completed Gaussian jobs (if any) and updates labels.
      - Then, if DFT budget remains, it SELECTS the next batch using surrogate uncertainty and SUBMITS jobs.
      - The user can re-run the same command after jobs finish; the run is fully resumable.

    This runner intentionally avoids interactive waiting/polling by default.
    """

    def __init__(self, cfg: ExploreCfg, paths: ExplorePaths):
        self.cfg = cfg
        self.paths = paths
        self.paths.ensure()
        self.logger = setup_logging(self.paths.run_dir / "run.log.jsonl", console=True)

        # state
        self.state = load_state(self.paths.run_dir / "state.json")

        # ---- release-grade state defaults (avoid AttributeError on resume/migrations)
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

        # components
        self.pool = CandidatePool(cfg.candidate_pool, cfg.descriptor, cfg.species, pool_dir=self.paths.pool_dir)
        self.gater = PhysicsGater(cfg.gating)
        self.acq = Acquisition(cfg.acquisition)
        self.anchors = AnchorManager(cfg.anchors)

        # ✅ correct gaussian config path
        self.oracle = GaussianOracle(cfg.labeling.gaussian)
        self.labels = LabelStore(self.paths.labels_jsonl)

    # -------------------------
    # Target resolution helpers
    # -------------------------
    def _resolve_surrogate_target(self, df_labels: pd.DataFrame) -> Tuple[np.ndarray, str]:
        """
        Resolve surrogate training targets in a robust, release-grade way.

        Returns:
          y (np.ndarray): training targets (kcal/mol by default)
          y_name (str): a human-readable name used for logs/diagnostics

        Supported cfg.surrogate.target values (recommended):
          - "dft_energy"        -> uses df_labels["e_kcalmol_abs"] (kcal/mol)
          - "e_kcalmol_abs"     -> uses df_labels["e_kcalmol_abs"]
          - "e_kcalmol_rel"     -> uses df_labels["e_kcalmol_rel"]
          - "e_hartree"         -> converted to kcal/mol (Hartree * 627.5095...)
        """
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

        # For e_hartree, convert to kcal/mol to keep downstream columns (mu_kcalmol/sigma_kcalmol) consistent
        if key == "e_hartree":
            y = y * float(HARTREE_TO_KCALMOL)
            return y, "e_hartree(kcal/mol)"

        return y, f"{key}(kcal/mol)"

    # -------------------------
    # Public entry
    # -------------------------
    
    def run(self) -> None:
        """Run explore in either single-step or auto-loop mode (config-driven)."""
        self.logger.info("Starting Explore run (oracle-in-the-loop active learning).")

        # snapshot config for reproducibility (CLI writes config.yaml; keep safe)
        try:
            if not (self.paths.run_dir / "config.yaml").exists():
                write_config_snapshot(self.paths.run_dir, "")
        except Exception:
            pass

        # Ensure pool and auxiliary artifacts exist.
        self._ensure_pool()

        # Important: a resumed run may already have completed Gaussian logs even
        # when the DFT budget is exhausted. Always do a final collection/export
        # pass before applying stop checks.
        try:
            self._collect_oracle_labels()
            self._write_exports()
            save_state(self.paths.run_dir / "state.json", self.state)
        except Exception as exc:
            self.logger.warning(f"Initial collect/export pass failed: {exc}")

        # If this run is already at a hard stop (budget/iters), do not start a loop.
        # The final collect/export pass above makes the run_dir inspectable.
        if self._should_stop():
            try:
                budget = int(self.cfg.stop.dft_budget or self.cfg.run.dft_budget)
                if self.state.dft_used >= budget:
                    self.logger.info(f"DFT budget exhausted: {self.state.dft_used}/{budget}.")
            except Exception:
                pass
            self.logger.info("Stop condition reached. Exiting.")
            return

        mode = getattr(self.cfg.loop, "mode", "step")
        if mode == "auto":
            self.logger.info("Loop mode: auto (collect→select→submit→wait).")
            # Long-running loop: step + wait until stop condition met
            while True:
                did_submit, it = self.step_once()

                # If a batch was submitted, wait first, then do a post-wait
                # collection/export pass. This prevents the common failure mode
                # where the budget is reached immediately after submission and
                # the last completed Gaussian logs are never collected.
                if it is not None:
                    self.wait_iter_done(it)
                    try:
                        self._collect_oracle_labels()
                        self._write_exports()
                        save_state(self.paths.run_dir / "state.json", self.state)
                    except Exception as exc:
                        self.logger.warning(f"Post-wait collect/export pass failed: {exc}")
                else:
                    # no eligible samples: wait a bit then re-collect
                    self._sleep(self.cfg.loop.poll_seconds)

                # Always write plots after each step / post-wait collection.
                try:
                    update_plots(self.paths.run_dir / "metrics.csv", self.paths.run_dir / "plots")
                except Exception:
                    pass

                if self._should_stop():
                    try:
                        self._collect_oracle_labels()
                        self._write_exports()
                        save_state(self.paths.run_dir / "state.json", self.state)
                    except Exception as exc:
                        self.logger.warning(f"Final collect/export pass failed: {exc}")
                    self.logger.info("Stop condition reached. Exiting.")
                    return
        else:
            self.logger.info("Loop mode: step (collect→select→submit).")
            self.step_once()
            try:
                update_plots(self.paths.run_dir / "metrics.csv", self.paths.run_dir / "plots")
            except Exception:
                pass
            return

    def _sleep(self, seconds: int) -> None:
        try:
            import time
            time.sleep(max(0, int(seconds)))
        except Exception:
            pass

    def step_once(self) -> tuple[bool, int | None]:
        """Execute a single recoverable AL step.

        Returns:
            (did_submit, iter_submitted_or_None)
        """
        # 1) Collect completed oracle jobs (any iter)
        collect_ok, collect_fail = self._collect_oracle_labels()

        # 2) Ensure anchors/holdout are initialized
        self._ensure_pool()

        # 3) Stage determination (logical, config-driven)
        self._update_stage()

        # 4) Budget checks
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

        # 5) If holdout enabled, prioritize labeling holdout until min_labeled reached
        submitted_this_iter = 0
        it = self.state.iter
        if self.cfg.anchors.enabled:
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

        # 6) Normal selection via surrogate + acquisition
        k = min(int(self.cfg.acquisition.k_select), budget - self.state.dft_used)
        if k <= 0:
            self._write_metrics(submitted_this_iter=0, collected_ok=collect_ok, collected_fail=collect_fail)
            save_state(self.paths.run_dir / "state.json", self.state)
            try:
                self._write_exports()
            except Exception:
                pass
            return False, None

        selected = self._select_next_batch(k=k)
        if not selected:
            self.logger.info("No eligible unlabeled candidates to submit.")
            self._write_metrics(submitted_this_iter=0, collected_ok=collect_ok, collected_fail=collect_fail)
            save_state(self.paths.run_dir / "state.json", self.state)
            try:
                self._write_exports()
            except Exception:
                pass
            return False, None

        submitted_this_iter = self._submit_selected(selected)

        # 7) Advance iteration counter only after attempted submission (idempotent)
        self.state.last_iter_submitted = it
        self.state.iter += 1

        self._write_metrics(submitted_this_iter=submitted_this_iter, collected_ok=collect_ok, collected_fail=collect_fail)
        save_state(self.paths.run_dir / "state.json", self.state)
        try:
            self._write_exports()
        except Exception:
            pass
        return submitted_this_iter > 0, it

    def wait_iter_done(self, it: int) -> None:
        """Poll an iteration's jobs until completion criteria met (auto-loop mode).

        Engineering note:
        We keep a *stable* submitted list (jobs_submitted.csv) and write polling
        results to jobs_status.csv. This avoids confusing logs where ok/pending
        appear to go "backwards" if a manifest is regenerated.
        """
        loop = self.cfg.loop
        it_dir = self.paths.iter_dir(it)

        submitted_csv = it_dir / "jobs_submitted.csv"
        status_csv = it_dir / "jobs_status.csv"
        # Backward compatible fallback
        src_csv = submitted_csv if submitted_csv.exists() else (it_dir / "jobs_manifest.csv")
        if not src_csv.exists():
            return

        import time

        t0 = time.time()
        poll = max(5, int(loop.poll_seconds))
        max_wait = int(loop.max_wait_seconds)

        df0 = pd.read_csv(src_csv)
        if "job_dir" not in df0.columns or len(df0) == 0:
            return
        job_dirs = [Path(p) for p in df0["job_dir"].astype(str).tolist()]

        n_total = len(job_dirs)
        last_logged = None

        while True:
            statuses = [self.oracle.job_status(jd) for jd in job_dirs]
            df = df0.copy()
            df["status"] = statuses
            df.to_csv(status_csv, index=False)

            n_ok = int((df["status"] == "ok").sum())
            n_fail = int((df["status"] == "failed").sum())
            n_done = n_ok + n_fail
            n_pending = n_total - n_done
            ok_frac = (n_ok / n_total) if n_total > 0 else 0.0
            snapshot = (n_ok, n_fail, n_pending)

            if snapshot != last_logged:
                self.logger.info(
                    f"Iter {it:03d} wait: ok={n_ok}, failed={n_fail}, pending={n_pending}, ok_frac={ok_frac:.2f}"
                )
                last_logged = snapshot

            # completion criteria
            if n_done == n_total:
                return
            if (not loop.require_all_ok) and (ok_frac >= float(loop.min_ok_frac)) and (n_done > 0):
                return

            if (time.time() - t0) > max_wait:
                self.logger.info(f"Max wait exceeded ({max_wait}s). Returning to allow resume.")
                return

            time.sleep(poll)

    def _ensure_pool(self) -> None:
        # Build df/X/frames if not present
        if (
            getattr(self.pool, "df", None) is None
            or getattr(self.pool, "X", None) is None
            or getattr(self.pool, "frames", None) is None
        ):
            # CandidatePool.initialize persists pool_index/features; safe for resume.
            self.pool.initialize(self.paths.pool_dir)

        # Apply physics gating once; store filtered mask
        df = self.pool.df
        if "passed_gating" not in df.columns:
            passed = np.ones((len(df),), dtype=bool)
            try:
                kept_frames, df_gate = self.gater.filter(self.pool.frames, df)
                passed = df_gate.get('keep', pd.Series([True]*len(df))).to_numpy(dtype=bool)
                # keep df_gate columns for audit
                for c in df_gate.columns:
                    if c not in df.columns:
                        df[c] = df_gate[c].values
                # if filtering removed frames, keep the original pool.frames unchanged; passed_gating mask controls eligibility
            except Exception:
                passed = np.ones((len(df),), dtype=bool)
            self.pool.df["passed_gating"] = passed

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

    def _eligible_unlabeled_mask(self) -> np.ndarray:
        df = self.pool.df
        passed = df.get("passed_gating", pd.Series([True] * len(df))).to_numpy(dtype=bool)
        status = df.get("status", pd.Series(["unlabeled"] * len(df))).astype(str).to_numpy()
        unlabeled = status == "unlabeled"
        return passed & unlabeled

    # -------------------------
    # Oracle collection
    # -------------------------
    def _collect_oracle_labels(self) -> tuple[int, int]:
        """Scan all iteration job directories and append newly available labels.

        Returns:
            (n_ok_new, n_failed_seen)

        Release-grade behavior:
          - scan actual ``iter_*`` directories on disk, not only ``state.iter``;
          - resync ``pool.df`` from ``labels.csv/jsonl`` even when no new labels
            are found;
          - rewrite the CSV label view with refreshed relative energies;
          - persist ``pool/pool_index.csv`` so downstream exports do not depend
            on stale status/selected_iter fields.
        """
        existing = self.labels.load()
        existing_ids = set(existing["id"].astype(str).tolist()) if not existing.empty else set()

        # Find all iteration directories that exist on disk. This is more robust
        # than relying on state.iter because the last submitted iteration may be
        # waiting when a run is stopped/resumed.
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
                st = self.oracle.job_status(job_dir)
                if st == "failed":
                    n_failed += 1

            recs = self.oracle.collect_labels(jobs_dir, iter_idx=it)
            for r in recs:
                if r.id and r.id not in existing_ids:
                    new_records.append(r)
                    existing_ids.add(r.id)
                    max_collected_iter = it if max_collected_iter is None else max(max_collected_iter, it)

        if new_records:
            self.labels.append(new_records)

        # Always reload and rewrite the tabular label view. This matters for
        # resumed runs where jsonl already contains labels but pool_index.csv was
        # stale in an older version.
        df_labels = self.labels.load()
        if not df_labels.empty:
            try:
                df_labels["e_kcalmol_rel"] = LabelStore.compute_relative_energies(
                    df_labels, reference="per_species_min"
                )
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

        return int(len(new_records)), int(n_failed)

    def _sync_pool_with_jobs_and_labels(self, df_labels: pd.DataFrame | None = None) -> None:
        """Synchronize pool.df status/selected_iter/provenance from disk artifacts.

        This fixes the important export bug where labels and Gaussian logs exist
        but ``pool/pool_index.csv`` still says every candidate is unlabeled.
        """
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
            self.state.dft_used = max(int(self.state.dft_used), len(set(self.state.labeled_ids)))
        except Exception:
            self.state.dft_used = len(set(self.state.labeled_ids))

        try:
            self.pool_dir = self.paths.pool_dir
            df.to_csv(self.paths.pool_dir / "pool_index.csv", index=False)
        except Exception:
            pass

    # -------------------------
    # Selection / submission
    # -------------------------
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
        if df_train.empty:
            return self._cold_start_select(k, idx_u)

        # map id->row idx in pool
        id_to_row = {str(iid): j for j, iid in enumerate(self.pool.df["id"].astype(str).tolist())}
        train_rows = [id_to_row[str(iid)] for iid in df_train["id"].astype(str).tolist() if str(iid) in id_to_row]
        if len(train_rows) == 0:
            return self._cold_start_select(k, idx_u)

        X_train = X[np.array(train_rows, dtype=int)]

        # ✅ resolve y_train based on cfg.surrogate.target (with aliases + unit normalization)
        y_train, y_name = self._resolve_surrogate_target(df_train)
        if y_train.size != X_train.shape[0]:
            raise RuntimeError(
                f"Training set size mismatch: X_train has {X_train.shape[0]} rows but y_train has {y_train.size}. "
                "This indicates an ID mapping / label alignment bug."
            )
        self.logger.info(f"Training surrogate on target={y_name}, n_train={len(y_train)}")

        surrogate = build_surrogate(self.cfg.surrogate, snapshot=None)
        surrogate.fit(X_train, y_train)

        # persist snapshot per iter (model used to select for the current iter)
        snap = self.paths.run_dir / "surrogate" / f"model_iter_{self.state.iter:03d}.pkl"
        try:
            surrogate.save(snap)
        except Exception:
            pass

        X_u = X[idx_u]
        mu, sigma = surrogate.predict(X_u)

        # Score table
        scored = pd.DataFrame(
            {
                "id": df.iloc[idx_u]["id"].astype(str).to_numpy(),
                "species_id": df.iloc[idx_u]["species_id"].astype(str).to_numpy(),
                "mu_kcalmol": mu,
                "sigma_kcalmol": sigma,
            }
        )
        scored["acq_score"] = scored["sigma_kcalmol"].astype(float)

        # Use Acquisition module: expects a dataframe with 'sigma' and aligned X (candidate subset)
        df_cand = df.iloc[idx_u].copy().reset_index(drop=False)  # keep original row index in 'index'
        df_cand["sigma"] = sigma
        labeled_counts = dict(pd.Series(df_train["species_id"].astype(str)).value_counts())
        df_cand["mu"] = mu
        df_cand["sigma"] = sigma
        df_scored = self.acq.compute(df=df_cand, X=X_u, labeled_X=X_train, labeled_species_counts=labeled_counts)
        # stage-aware k
        self.acq.cfg.k_select = int(k)
        sel_rows = self.acq.select_indices(df_scored=df_scored, X_scored=X_u)
        selected_idx = df_scored.iloc[sel_rows]["index"].astype(int).tolist()

        if len(selected_idx) == 0:
            # fallback: top-k by sigma
            order = np.argsort(-sigma)[:k]
            selected_idx = idx_u[order].astype(int).tolist()

        # write per-iter scored.csv (full scored table)
        it_dir = self.paths.iter_dir(self.state.iter)
        it_dir.mkdir(parents=True, exist_ok=True)
        scored.sort_values("acq_score", ascending=False).to_csv(it_dir / "scored.csv", index=False)

        # holdout metrics, if holdout labels exist
        self._maybe_write_metrics(surrogate)

        return selected_idx[:k]

    def _cold_start_select(self, k: int, idx_u: np.ndarray) -> List[int]:
        """Initialize oracle labels when none exist (cold start).

        Goal: produce a diverse, chemically reasonable initial labeled set.
        Strategy:
          - prefer seeds for stability (up to ~half)
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
        idx_other_u = idx_u[~seed_mask[idx_u]]

        selected: List[int] = []

        k_seed = min(len(idx_seed_u), max(0, k // 2))
        if k_seed > 0:
            sel = fps(X[idx_seed_u], k_seed)
            selected.extend(idx_seed_u[np.array(sel, dtype=int)].tolist())

        k_rem = k - len(selected)
        if k_rem > 0:
            remaining = idx_u[~np.isin(idx_u, selected)]
            sel = fps(X[remaining], k_rem)
            selected.extend(remaining[np.array(sel, dtype=int)].tolist())

        return selected[:k]

    def _submit_batch(self, selected_rows: List[int]) -> int:
        it = int(self.state.iter)
        it_dir = self.paths.iter_dir(it)
        jobs_dir = self.paths.jobs_dir(it)
        jobs_dir.mkdir(parents=True, exist_ok=True)

        df = self.pool.df
        frames = self.pool.frames

        # export selected xyz/csv
        selected_frames = [frames[i] for i in selected_rows]
        selected_csv = it_dir / "selected.csv"
        selected_xyz = it_dir / "selected.xyz"
        df.iloc[selected_rows].to_csv(selected_csv, index=False)
        from tsgen.data.xyz import write_xyz_multiframe

        write_xyz_multiframe(selected_xyz, selected_frames)

        # ✅ correct path for gaussian config (fixes ExploreCfg has no attribute 'gaussian')
        gcfg = self.cfg.labeling.gaussian

        n_attempt = 0
        n_ok = 0
        submit_failures: list[tuple[str, str]] = []
        for j, row_idx in enumerate(selected_rows):
            rid = str(df.at[row_idx, "id"])
            sid = str(df.at[row_idx, "species_id"])
            job_dir = jobs_dir / f"job_{j:04d}"
            self.oracle.write_job(job_dir, GaussianJobSpec(id=rid, species_id=sid, frame=frames[row_idx]))
            n_attempt += 1

            jid = self.oracle.submit(job_dir)
            if gcfg.mode == "slurm":
                # In SLURM mode, only mark as submitted if sbatch succeeded.
                if jid is None:
                    submit_failures.append((rid, str(job_dir)))
                    # Keep as unlabeled so it can be retried; record the intended iteration.
                    df.at[row_idx, "status"] = "unlabeled"
                    df.at[row_idx, "selected_iter"] = it
                    continue

            # local mode or successful SLURM submit
            df.at[row_idx, "status"] = "submitted" if gcfg.mode == "slurm" else "prepared"
            df.at[row_idx, "selected_iter"] = it
            if jid is not None:
                df.at[row_idx, "slurm_jobid"] = str(jid)
            n_ok += 1

        # 'selected' counts decisions; 'submitted' counts successful submissions.
        self.state.selected += int(n_attempt)
        self.state.submitted += int(n_ok)
        self.state.iter += 1  # advance iteration after submission

        # Persist a per-iteration failure report (release-grade transparency)
        if submit_failures:
            try:
                rep = it_dir / "submit_failures.txt"
                rep.write_text(
                    "\n".join([f"id={rid} job_dir={jdir}" for rid, jdir in submit_failures]) + "\n",
                    encoding="utf-8",
                )
                self.logger.warning(
                    f"SLURM submission failed for {len(submit_failures)}/{n_attempt} jobs. " f"See {rep}."
                )
            except Exception:
                pass

        # persist pool index for resume/debug
        try:
            df.to_csv(self.paths.pool_dir / "pool_index.csv", index=False)
        except Exception:
            pass

        return n_ok

    # -------------------------
    # Metrics / exports
    # -------------------------
    def _maybe_write_metrics(self, surrogate) -> None:
        # If holdout ids are defined and labeled, compute metrics on them.
        if not self.state.holdout_ids:
            return
        df_labels = self.labels.load()
        if df_labels.empty:
            return
        holdout_ids = set(self.state.holdout_ids)
        df_hold = df_labels[df_labels["id"].astype(str).isin(holdout_ids)].copy()
        if df_hold.empty:
            return

        # map holdout ids to pool rows
        id_to_row = {str(iid): j for j, iid in enumerate(self.pool.df["id"].astype(str).tolist())}
        rows = [id_to_row[str(iid)] for iid in df_hold["id"].astype(str).tolist() if str(iid) in id_to_row]
        if not rows:
            return
        Xh = self.pool.X[np.array(rows, dtype=int)]

        # ✅ use same target definition for metrics as for training
        yh, _ = self._resolve_surrogate_target(df_hold)

        mu, sigma = surrogate.predict(Xh)
        met = evaluate_holdout(yh, mu, sigma)

        # pool sigma mean (eligible unlabeled)
        mask_u = self._eligible_unlabeled_mask()
        Xu = self.pool.X[mask_u]
        mu_u, sigma_u = surrogate.predict(Xu)

        budget = int(self.cfg.stop.dft_budget or self.cfg.run.dft_budget)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        m = ALMetrics(
            iter=int(self.state.iter),
            timestamp=ts,
            stage=str(getattr(self.state, "stage", "explore")),
            dft_used=int(self.state.dft_used),
            dft_budget=int(budget),
            n_pool=int(len(self.pool.df)),
            labeled_train=int((df_labels[~df_labels["id"].astype(str).isin(holdout_ids)].shape[0])),
            labeled_holdout=int(df_hold.shape[0]),
            n_species_labeled=int(df_labels["species_id"].astype(str).nunique()) if "species_id" in df_labels.columns else 0,
            mae_holdout=float(met["mae"]),
            rmse_holdout=float(met["rmse"]),
            nll_holdout=float(met["nll"]),
            cov_1sigma=float(met.get("cov_1sigma", met.get("cov1"))),
            cov_2sigma=float(met.get("cov_2sigma", met.get("cov2"))),
            mean_sigma_pool=float(np.mean(sigma_u)) if getattr(sigma_u, 'size', 0) else None,

        )
        append_metrics_csv(self.paths.metrics_csv, m)

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
            sel_mask = (
                df.get("status", pd.Series(["unlabeled"] * len(df))).astype(str).isin(["submitted", "labeled"]).to_numpy()
            )
            write_xyz_multiframe(self.paths.selected_xyz, [fr for fr, ok in zip(frames, sel_mask) if bool(ok)])
        except Exception:
            pass

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
        """Write all user-facing exports.

        This is intentionally best-effort: exports are convenience artifacts and
        should never crash the AL loop.
        """
        try:
            self._export_cumulative()
        except Exception:
            pass

        # Per-structure validated non-seed exports (primary AL artifact).
        # Use labels.csv/jsonl + job_dir/structure.xyz as the source of truth,
        # not pool_index.status, because pool_index may be stale in older runs.
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
        """Resolve a label row's job_dir robustly.

        label_store historically saved job_dir as a relative path from the TSGen
        working directory. For portability, try several locations and finally
        search by job id under this run directory.
        """
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
        """Export validated, non-seed structures as one XYZ file per structure.

        Source of truth:
          labels/labels.csv or labels.jsonl for validation status and provenance;
          iter_*/gaussian/jobs/job_*/structure.xyz for coordinates.

        This avoids the previous failure mode where ``pool/pool_index.csv`` was
        stale (all rows unlabeled) while Gaussian logs and labels were already
        available.
        """
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
            "Validated non-seed XYZ export generated by TSGen Explore.\n"
            f"Total successful labels observed: {total_ok}\n"
            f"Exported non-seed structures: {exported}\n"
            f"Skipped seed structures: {skipped_seed}\n"
            f"Skipped records with missing structure.xyz: {skipped_missing_structure}\n"
            "\nDefinition: exported structures are successful Gaussian labels whose source pool entry is not marked as seed.\n"
            "Coordinates are copied from iter_*/gaussian/jobs/job_*/structure.xyz.\n",
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
    # -------------------------
    # Long-running controls
    # -------------------------
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

        # User-facing per-iteration selected structures/tables. These are useful
        # even if submission later fails or auto mode exits early.
        try:
            from tsgen.data.xyz import write_xyz_multiframe
            self.pool.df.iloc[selected_pool_rows].to_csv(it_dir / "selected.csv", index=False)
            write_xyz_multiframe(it_dir / "selected.xyz", [self.pool.frames[i] for i in selected_pool_rows])
        except Exception:
            pass

        jobs: list[dict] = []
        submitted = 0
        df = self.pool.df

        for ridx in selected_pool_rows:
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
            self.oracle.write_job(job_dir, spec)
            jobid = None
            try:
                jobid = self.oracle.submit(job_dir)
            except Exception:
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
            prev_sp = set(getattr(self.state, "_prev_species_set", set()))
            new_sp = len(sp_set - prev_sp)
            self.state._prev_species_set = sp_set  # runtime only (not persisted)
        if self.state.recent_new_species is None:
            self.state.recent_new_species = []
        self.state.recent_new_species.append(int(new_sp))
        if len(self.state.recent_new_species) > 50:
            self.state.recent_new_species = self.state.recent_new_species[-50:]

        # mean sigma pool from last scored table if exists
        mean_sigma_pool = None
        max_sigma_pool = None
        mean_sigma_selected = None

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
            submitted_this_iter=int(submitted_this_iter),
            collected_ok_this_iter=int(collected_ok),
            collected_fail_this_iter=int(collected_fail),
        )
        append_metrics_csv(self.paths.run_dir / "metrics.csv", m)
