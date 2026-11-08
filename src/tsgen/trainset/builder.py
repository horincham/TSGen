from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from tsgen.data.xyz import Frame, write_xyz_multiframe
from tsgen.trainset.analysis import write_diagnostic_tables, write_report
from tsgen.trainset.chem import (
    angle_deg,
    dihedral_deg,
    family_from_graph,
    formula_from_symbols,
    infer_bonds,
    passes_chemistry,
    resolve_charge_multiplicity,
)
from tsgen.trainset.charges import load_reference_charges
from tsgen.trainset.config import TrainsetBuildCfg, load_trainset_config, save_trainset_config
from tsgen.trainset.gaussian import HARTREE_TO_KCALMOL, GaussianScanRunner, collect_job_table, collect_scan_point_table, wait_for_jobs
from tsgen.trainset.generate import generate_scan_points
from tsgen.trainset.io import (
    TrainsetPaths,
    clean_dir,
    next_run_dir,
    read_first_frame,
    resolve_input_structures,
    load_input_frames,
    sanitize_id,
    write_dataframe,
    write_json,
    write_single_xyz,
)
from tsgen.trainset.models import CoordinateSpec, EquilibriumStructure, ScanPoint
from tsgen.trainset.reaxff_io import (
    sanitize_reaxff_id,
    restraint_record,
    validate_legacy_reaxff_inputs,
    write_bgf_structure,
    write_geo_file,
    write_models_in,
    write_trainset_in,
)
from tsgen.trainset.selection import select_coordinates
from tsgen.trainset.gaussian_scan_parser import validate_scan_record_alignment
from tsgen.trainset.provenance import verify_optimized_record
from tsgen.utils.geometry import composition_key, pair_distance_fingerprint, fingerprint_distance


def _pair_distance_fingerprint_distance(a: Frame, b: Frame) -> float:
    if composition_key(a) != composition_key(b):
        return float("inf")
    return fingerprint_distance(pair_distance_fingerprint(a), pair_distance_fingerprint(b))

def _canonical_equilibrium_ids(items: list[tuple[str, Path, Frame, str]]) -> list[tuple[str, str, str]]:
    """Return (parent_id, scan_prefix, source_base) for valid input frames.

    Identifiers are intentionally formula-based, not TSGen species-hash based,
    because they appear in ReaxFF DESCRP and trainset.in ENERGY expressions.
    """
    from collections import Counter, defaultdict
    formula_counts = Counter(formula for _, _, _, formula in items)
    seen: dict[str, int] = defaultdict(int)
    out: list[tuple[str, str, str]] = []
    for serial, (name, path, _fr, formula) in enumerate(items, 1):
        seen[formula] += 1
        idx = seen[formula]
        if formula_counts[formula] == 1:
            prefix = formula
        else:
            prefix = f"{formula}{chr(96 + idx)}" if idx <= 26 else f"{formula}i{idx}"
        if len(prefix) > 6:
            prefix = f"M{serial:04d}"
        parent = f"{prefix}_opt"
        source_base = sanitize_reaxff_id(name or Path(path).stem, max_len=72)
        out.append((parent, prefix, source_base))
    return out


def create_run(config_path: Path) -> Path:
    cfg = load_trainset_config(config_path)
    run_dir = next_run_dir(Path(cfg.run.out_dir), cfg.run.name)
    paths = TrainsetPaths(run_dir)
    paths.ensure()
    save_trainset_config(cfg, paths.config_file)
    return run_dir


class TrainsetBuilder:
    def __init__(self, cfg: TrainsetBuildCfg, run_dir: Path, echo=print):
        self.cfg = cfg
        self.paths = TrainsetPaths(run_dir)
        self.paths.ensure()
        self.echo = echo
        self.gaussian = GaussianScanRunner(cfg.gaussian, scan_mode=cfg.scan_generation.mode)
        self.workflow_file = self.paths.exports_dir / "workflow.json"

    def _config_signature(self) -> str:
        data = self.cfg.model_dump(mode="json")
        # Submission and reporting preferences may change safely on resume.
        data.pop("execution", None)
        data["outputs"].pop("write_report", None)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _workflow(self) -> dict:
        if not self.workflow_file.exists():
            raise ValueError("Legacy workflow cannot be resumed with equilibrium-first scans; start a new run")
        state = json.loads(self.workflow_file.read_text(encoding="utf-8"))
        if state.get("config_sha256") != self._config_signature():
            raise ValueError("Calculation settings changed; start a new run to avoid stale Gaussian results")
        return state

    def _save_workflow(self, stage: str) -> None:
        write_json(self.workflow_file, {"version": 2, "stage": stage, "config_sha256": self._config_signature()})

    @classmethod
    def from_config(cls, config_path: Path, echo=print) -> "TrainsetBuilder":
        cfg = load_trainset_config(config_path)
        run_dir = next_run_dir(Path(cfg.run.out_dir), cfg.run.name)
        b = cls(cfg, run_dir, echo=echo)
        save_trainset_config(cfg, b.paths.config_file)
        return b

    @classmethod
    def from_run_dir(cls, run_dir: Path, echo=print) -> "TrainsetBuilder":
        cfg = load_trainset_config(Path(run_dir) / "config.yaml")
        return cls(cfg, Path(run_dir), echo=echo)

    def plan(self) -> None:
        self.echo(f"[trainset] planning run_dir={self.paths.run_dir}")
        if self.workflow_file.exists():
            self._workflow()
            self.echo("[trainset] existing plan retained; use resume to advance")
            return
        if any(self.paths.jobs_dir.iterdir()):
            raise ValueError("Existing Gaussian jobs belong to an older run; start a new run")
        self._prepare_clean_outputs(keep_jobs=True)
        eqs, input_audit = self._load_equilibrium_structures()
        if len({e.charge for e in eqs}) != 1:
            raise ValueError("Mixed total charge dataset needs separate runs: legacy ReaxFF control uses one syscha")
        points, _ = generate_scan_points(eqs, [], self.cfg.scan_generation)
        self._write_plan(eqs, [], points, input_audit, [], [])
        self._write_scan_structures(points)
        self._write_gaussian_inputs(points)
        self._save_workflow("equilibrium")
        self._write_keyword_report(collect_job_table(self.paths.jobs_dir))
        self._write_summary(stage="awaiting_equilibrium")
        self.echo(f"[trainset] prepared {len(eqs)} equilibrium Opt jobs; scans await converged Opt geometries")

    def run(self) -> None:
        self.plan()
        self._execute_available_phases()

    def resume(self) -> None:
        self.echo(f"[trainset] resume run_dir={self.paths.run_dir}")
        # If no manifest yet, reconstruct plan.  Otherwise only collect/finalize.
        if not (self.paths.tables_dir / "scan_manifest.csv").exists():
            self.plan()
        self._workflow()
        self._execute_available_phases()

    def _execute_available_phases(self) -> None:
        for _ in range(2):
            before = self._workflow()["stage"]
            if self.cfg.execution.submit:
                self.submit_missing()
                if self.cfg.execution.wait_for_completion:
                    wait_for_jobs(self.paths.jobs_dir, self.cfg.execution, echo=self.echo)
            self.collect_and_finalize()
            if before == self._workflow()["stage"] or self._workflow()["stage"] == "finalized":
                break

    def submit_missing(self) -> None:
        self._workflow()
        manifest_p = self.paths.tables_dir / "scan_manifest.csv"
        if not manifest_p.exists():
            raise FileNotFoundError(f"Missing scan_manifest.csv; run plan first: {manifest_p}")
        df = pd.read_csv(manifest_p)
        submitted = 0
        for _, row in df.iterrows():
            jd = Path(row["job_dir"])
            st_rows = collect_job_table(jd.parent)
            # Avoid resubmitting ok jobs.
            if self.cfg.execution.reuse_completed:
                local = [r for r in st_rows if r.get("id") == row["id"]]
                if local and local[0].get("status") == "ok":
                    continue
            if self.cfg.gaussian.mode == "manual":
                continue
            if (jd / "sbatch.stdout").exists() and "Submitted batch job" in (jd / "sbatch.stdout").read_text(errors="ignore"):
                continue
            log = jd / "input.log"
            if log.exists() and self.cfg.execution.reuse_completed:
                continue
            jobid = self.gaussian.submit(jd)
            if jobid:
                submitted += 1
        self.echo(f"[trainset] submitted jobs: {submitted}")

    def collect_and_finalize(self) -> None:
        state = self._workflow()
        # A new collection must not publish products from an older job state,
        # including paths which return early while waiting or fail strict policy.
        self._invalidate_current_products()
        quality_path = self.paths.tables_dir / 'export_quality.json'
        if quality_path.exists():
            quality_path.unlink()
        self.echo("[trainset] collecting Gaussian scan jobs")
        job_rows = collect_job_table(self.paths.jobs_dir)
        pd.DataFrame(job_rows).to_csv(self.paths.tables_dir / "scan_jobs.csv", index=False)
        self._write_keyword_report(job_rows)
        if state["stage"] == "equilibrium":
            refs = [r for r in job_rows if r.get("scan_type") == "equilibrium"]
            expected_refs = set(pd.read_csv(self.paths.tables_dir / "equilibrium_structures.csv")["parent_id"])
            if set(r["parent_id"] for r in refs) != expected_refs or any(r["status"] not in {"ok", "failed"} for r in refs):
                self._write_summary(stage="awaiting_equilibrium")
                return
            good = {r["parent_id"]: r for r in refs if r["status"] == "ok"}
            if not good:
                self._write_summary(stage="equilibrium_failed")
                self.echo("[trainset] no converged equilibrium Opt; scans were not generated")
                return
            if self.cfg.execution.require_all_ok and len(good) != len(refs):
                raise ValueError("Equilibrium Opt failures with require_all_ok=true; inspect scan_jobs.csv")
            eqs = []
            eq_table = pd.read_csv(self.paths.tables_dir / "equilibrium_structures.csv")
            for _, e in eq_table.iterrows():
                result = good.get(e["parent_id"])
                if result is None:
                    continue
                fr = read_first_frame(Path(result["optimized_xyz"]))
                if len(fr.symbols) != int(e["n_atoms"]):
                    raise ValueError(f"Atom count changed in equilibrium Opt {e['parent_id']}")
                family = family_from_graph(fr, infer_bonds(fr))
                eqs.append(EquilibriumStructure(str(e["parent_id"]), str(e["source_path"]), str(e["formula"]),
                    str(e["species_id"]), family, fr, int(e["charge"]), int(e["multiplicity"]), int(e["n_electrons"]), str(e["scan_prefix"])))
            coords, coord_audit = select_coordinates(eqs, self.cfg.coordinate_selection)
            points, point_audit = generate_scan_points(eqs, coords, self.cfg.scan_generation)
            audit = pd.read_csv(self.paths.tables_dir / "input_structure_audit.csv").to_dict("records")
            self._write_plan(eqs, coords, points, audit, coord_audit, point_audit)
            scans = [p for p in points if p.scan_type != "equilibrium"]
            self._write_scan_structures(points)
            self._write_gaussian_inputs(scans)
            self._save_workflow("scans")
            self._write_summary(stage="awaiting_scans")
            self.echo(f"[trainset] prepared {len(scans)} scans from {len(eqs)} converged Opt geometries")
            return
        active = pd.read_csv(self.paths.tables_dir / "scan_manifest.csv")
        active_ids = set(active["id"])
        current = {r["id"]: r for r in job_rows}
        if any(i not in current or current[i]["status"] not in {"ok", "failed"} for i in active_ids):
            self._write_summary(stage="awaiting_scans")
            return
        scans = [current[r["id"]] for r in active.to_dict("records") if r["scan_type"] != "equilibrium"]
        active_refs=[current[r['id']] for r in active.to_dict('records') if r['scan_type']=='equilibrium']
        good_parents={r['parent_id'] for r in active_refs if r['status']=='ok'}
        eligible_scans={r['id'] for r in scans if r['status']=='ok' and r.get('parent_id') in good_parents}
        n_ok = len(eligible_scans)
        fraction = n_ok / len(scans) if scans else 1.
        failed_refs = [r for r in active_refs if r['status'] != 'ok']
        excluded = [dict(r,parse_note=(r.get('parse_note','') if r['status']!='ok' else 'excluded: no successful converged parent reference'))
                    for r in scans if r['id'] not in eligible_scans]
        partial = bool(excluded or failed_refs)
        below = fraction < self.cfg.execution.min_ok_frac
        message = (f"Scan success={n_ok}/{len(scans)} ({fraction:.1%}), target={self.cfg.execution.min_ok_frac:.1%}; "
                   f"excluding {len(excluded)} failed/incomplete curves. Export contains verified converged records only.")
        quality = dict(export_state='checking',scan_jobs_ok=n_ok,scan_jobs_total=len(scans),success_fraction=fraction,
            min_ok_frac=self.cfg.execution.min_ok_frac,below_min_ok_frac=below,partial=partial,
            quality_failure_action=self.cfg.execution.quality_failure_action,
            excluded_scan_jobs=[r['id'] for r in excluded],failed_reference_jobs=[r['id'] for r in failed_refs],
            warning=message if partial or below else '')
        write_json(self.paths.tables_dir/'export_quality.json',quality)
        write_dataframe(self.paths.tables_dir/'excluded_scan_jobs.csv',[
            {k:r.get(k,'') for k in ['id','parent_id','scan_type','status','parse_note','job_dir']} for r in excluded])
        if (self.cfg.execution.require_all_ok and partial) or (below and self.cfg.execution.quality_failure_action=='error'):
            quality['export_state']='blocked_by_policy'
            write_json(quality_path,quality)
            self._write_summary(stage="scan_quality_failed")
            raise ValueError(f"Scan quality policy failed: success={n_ok}/{len(scans)}, required fraction={self.cfg.execution.min_ok_frac}; inspect scan_jobs.csv")
        if partial or below:
            self.echo('[WARNING] '+message)
        self._update_manifest_with_energies(job_rows)
        expanded_rows = collect_scan_point_table(self.paths.jobs_dir)
        expanded_rows=[r for r in expanded_rows if
            (r.get('scan_type')=='equilibrium' and r.get('id') in active_ids and r.get('status')=='ok')
            or r.get('scan_job_id',r.get('id')) in eligible_scans]
        self._write_expanded_energy_table(expanded_rows)
        try:
            if not good_parents or not expanded_rows:
                raise ValueError('No active converged equilibrium reference; refusing training-file export')
            self._write_reaxff_products()
        except (ValueError,FileNotFoundError) as exc:
            self._invalidate_current_products(preserve=False)
            quality.update(export_state='failed',hard_error=str(exc))
            write_json(self.paths.tables_dir/'export_quality.json',quality)
            self._write_summary(stage='export_integrity_failed')
            raise
        quality['export_state']='completed'
        write_json(self.paths.tables_dir/'export_quality.json',quality)
        self._write_summary(stage="finalized_partial" if partial or below else "finalized")
        self._save_workflow("finalized")
        write_diagnostic_tables(self.paths.tables_dir)
        if self.cfg.outputs.write_report:
            summary = json.loads(self.paths.summary_json.read_text(encoding="utf-8")) if self.paths.summary_json.exists() else {}
            write_report(self.paths.report_md, summary)
        self.echo(f"[trainset] finalized products in {self.paths.exports_dir}")

    def status(self) -> dict:
        rows = collect_job_table(self.paths.jobs_dir)
        n = len(rows)
        ok = sum(1 for r in rows if r.get("status") == "ok")
        failed = sum(1 for r in rows if r.get("status") == "failed")
        pending = n - ok - failed
        summary = {"run_dir": str(self.paths.run_dir), "jobs_total": n, "jobs_ok": ok, "jobs_failed": failed, "jobs_pending_or_running": pending}
        if self.paths.summary_json.exists():
            try:
                summary.update(json.loads(self.paths.summary_json.read_text(encoding="utf-8")))
            except Exception:
                pass
        return summary

    def _write_keyword_report(self, rows: list[dict]) -> None:
        table = []
        groups: dict[tuple[str,str],int] = {}
        for row in rows:
            path=Path(row['job_dir'])/'input.gjf'
            actual=''
            if path.exists():
                chunks=[]; started=False
                for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
                    if line.lstrip().startswith('#'): started=True
                    if started:
                        if not line.strip(): break
                        chunks.append(line.strip())
                actual=' '.join(chunks)
            role='equilibrium' if row.get('scan_type')=='equilibrium' else 'scan'
            groups[(role,actual)]=groups.get((role,actual),0)+1
            table.append(dict(id=row.get('id'),role=role,status=row.get('status'),actual_route=actual,
                metadata_route=row.get('route',''),charge=row.get('charge'),multiplicity=row.get('multiplicity'),input_file=str(path)))
        write_dataframe(self.paths.tables_dir/'gaussian_keywords.csv',table)
        lines=['# Actual Gaussian input keywords','', 'Read from existing input.gjf files; existing computed jobs are not rewritten.','']
        for (role,route), count in sorted(groups.items()):
            lines.extend([f'## {role} ({count} jobs)','',f'```text\n{route}\n```',''])
        (self.paths.exports_dir/'gaussian_keywords.md').write_text('\n'.join(lines),encoding='utf-8')

    def _invalidate_current_products(self, preserve: bool = True) -> None:
        artifacts=[self.paths.geo_file,self.paths.trainset_in,self.paths.models_in,
                   self.paths.report_md,self.paths.exports_dir/'control.scan_recommended']
        artifacts.extend(self.paths.tables_dir/name for name in [
            'optimization_provenance.csv','structure_name_map.csv','charge_targets.csv',
            'geometry_targets.csv','energy_targets.csv','trainset_entries.csv'])
        for artifact in artifacts:
            if artifact.is_file():
                if preserve:
                    shutil.copy2(artifact,artifact.with_name(artifact.name+'.previous'))
                artifact.unlink()
        # Only generated BGF files are replaced; Gaussian inputs/logs are untouched.
        backup_dir=self.paths.bgf_dir.with_name('bgf.previous')
        for artifact in self.paths.bgf_dir.glob('*.bgf'):
            if preserve:
                backup_dir.mkdir(parents=True,exist_ok=True)
                shutil.copy2(artifact,backup_dir/artifact.name)
            artifact.unlink()

    def _prepare_clean_outputs(self, keep_jobs: bool = True) -> None:
        # Keep gaussian_jobs by default so rerun/resume does not delete expensive completed logs.
        for p in [self.paths.structures_dir, self.paths.tables_dir, self.paths.bgf_dir]:
            clean_dir(p)
        for p in [self.paths.eq_structures_dir, self.paths.bond_structures_dir, self.paths.angle_structures_dir, self.paths.torsion_structures_dir]:
            p.mkdir(parents=True, exist_ok=True)
        if not keep_jobs:
            clean_dir(self.paths.jobs_dir)
        else:
            self.paths.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _load_equilibrium_structures(self) -> tuple[list[EquilibriumStructure], list[dict]]:
        raw_inputs = load_input_frames(self.cfg)
        valid_items: list[tuple[str, Path, Frame, str]] = []
        audit: list[dict] = []

        # First pass: chemistry validation and formula detection.  Final ReaxFF
        # identifiers are assigned after we know how many duplicate formulae are
        # present, so they can become e.g. C9H18_opt_01 / C9H18_opt_02.
        for name, p, fr in raw_inputs:
            try:
                # Charge/spin filtering must use the final resolved state below,
                # not the electron parity of a nominal neutral formula.
                ok, reason = passes_chemistry(fr, self.cfg.chemistry.model_copy(update={"exclude_open_shell": False}))
                formula = formula_from_symbols(fr.symbols)
                if not ok:
                    audit.append({"source_path": str(p), "source_id": name, "formula": formula, "decision": "drop", "reason": reason})
                    continue
                valid_items.append((name, Path(p), fr, formula))
            except Exception as exc:
                audit.append({"source_path": str(p), "source_id": name, "decision": "drop", "reason": f"read_error: {exc}"})

        if not valid_items:
            raise RuntimeError("No equilibrium structures passed chemistry/input filters.")

        ids = _canonical_equilibrium_ids(valid_items)
        eqs: list[EquilibriumStructure] = []
        for (name, p, fr, formula), (pid, scan_prefix, source_base) in zip(valid_items, ids):
            try:
                overrides = self.cfg.gaussian.state_overrides
                override = overrides.get(Path(p).stem, overrides.get(pid, overrides.get(formula, {})))
                q, m, ne = resolve_charge_multiplicity(fr, override.get("charge", self.cfg.gaussian.charge), override.get("multiplicity", self.cfg.gaussian.multiplicity))
                if self.cfg.chemistry.exclude_open_shell and m != 1:
                    audit.append({"source_path": str(p), "source_id": name, "formula": formula, "decision": "drop", "reason": f"open_shell resolved multiplicity={m}, charge={q}"})
                    continue
                bonds = infer_bonds(fr)
                family = family_from_graph(fr, bonds)
                comment = f"id={pid} source={Path(p).name} source_id={source_base} formula={formula} family={family} charge={q} multiplicity={m}"
                fr = Frame(symbols=list(fr.symbols), coords=np.array(fr.coords, dtype=float), comment=comment)
                eqs.append(EquilibriumStructure(pid, str(p), formula, species_id=source_base, family=family, frame=fr, charge=q, multiplicity=m, n_electrons=ne, scan_prefix=scan_prefix))
                audit.append({
                    "source_path": str(p),
                    "source_id": name,
                    "parent_id": pid,
                    "scan_prefix": scan_prefix,
                    "formula": formula,
                    "family": family,
                    "n_atoms": len(fr.symbols),
                    "decision": "keep",
                    "reason": "ok",
                    "charge": q, "multiplicity": m,
                    "state_note": "explicit_override" if override else "input_metadata_or_known_species_then_parity_guess; ground_state_unverified",
                })
            except Exception as exc:
                audit.append({"source_path": str(p), "source_id": name, "formula": formula, "decision": "drop", "reason": f"read_error: {exc}"})

        if getattr(self.cfg.input, "deduplicate_structures", True):
            eqs, dedup_audit = self._deduplicate_equilibrium_structures(eqs)
            audit.extend(dedup_audit)
        if not eqs:
            write_dataframe(self.paths.tables_dir / "input_structure_audit.csv", audit)
            raise RuntimeError("No structures passed electronic-state validation; inspect input_structure_audit.csv")
        return eqs, audit

    def _deduplicate_equilibrium_structures(self, eqs: list[EquilibriumStructure]) -> tuple[list[EquilibriumStructure], list[dict]]:
        kept: list[EquilibriumStructure] = []
        audit: list[dict] = []
        tol = float(getattr(self.cfg.input, "dedup_fingerprint_tol", 0.03))
        for e in eqs:
            duplicate_of = None
            for k in kept:
                if (e.formula != k.formula or len(e.frame.symbols) != len(k.frame.symbols)
                        or (e.charge, e.multiplicity) != (k.charge, k.multiplicity)):
                    continue
                d = _pair_distance_fingerprint_distance(e.frame, k.frame)
                if d <= tol:
                    duplicate_of = (k.parent_id, d)
                    break
            if duplicate_of is None:
                kept.append(e)
            else:
                audit.append({
                    "source_path": e.source_path,
                    "parent_id": e.parent_id,
                    "formula": e.formula,
                    "decision": "drop",
                    "reason": f"duplicate optimized structure of {duplicate_of[0]} fingerprint_distance={duplicate_of[1]:.6f}",
                })
        return kept, audit

    def _write_plan(self, eqs: list[EquilibriumStructure], coords: list[CoordinateSpec], points: list[ScanPoint], input_audit: list[dict], coord_audit: list[dict], point_audit: list[dict]) -> None:
        eq_rows = []
        for e in eqs:
            eq_rows.append({
                "parent_id": e.parent_id,
                "source_path": e.source_path,
                "formula": e.formula,
                "species_id": e.species_id,
                "scan_prefix": e.scan_prefix,
                "family": e.family,
                "n_atoms": len(e.frame.symbols),
                "charge": e.charge,
                "multiplicity": e.multiplicity,
                "n_electrons": e.n_electrons,
            })
        write_dataframe(self.paths.tables_dir / "equilibrium_structures.csv", eq_rows)
        write_dataframe(self.paths.tables_dir / "input_structure_audit.csv", input_audit)

        coord_rows = []
        for c in coords:
            coord_rows.append({
                "coord_id": c.coord_id,
                "parent_id": c.parent_id,
                "scan_type": c.scan_type,
                "family": c.family,
                "atom_indices_1based": " ".join(str(a + 1) for a in c.atom_indices),
                "atom_symbols": "-".join(c.atom_symbols),
                "eq_value": c.eq_value,
                "priority": c.priority,
                "reason": c.reason,
            })
        write_dataframe(self.paths.tables_dir / "selected_coordinates.csv", coord_rows)
        write_dataframe(self.paths.tables_dir / "coordinate_selection_audit.csv", coord_audit)
        write_dataframe(self.paths.tables_dir / "scan_generation_audit.csv", point_audit)

        manifest_rows = []
        for p in points:
            job_dir = self.paths.jobs_dir / p.id
            p.job_dir = str(job_dir)
            manifest_rows.append(self._point_to_row(p))
        write_dataframe(self.paths.tables_dir / "scan_manifest.csv", manifest_rows)

    def _write_scan_structures(self, points: list[ScanPoint]) -> None:
        for p in points:
            if p.scan_type == "equilibrium":
                out = self.paths.eq_structures_dir / f"{p.id}.xyz"
            elif p.scan_type == "bond":
                out = self.paths.bond_structures_dir / f"{p.id}.xyz"
            elif p.scan_type == "angle":
                out = self.paths.angle_structures_dir / f"{p.id}.xyz"
            else:
                out = self.paths.torsion_structures_dir / f"{p.id}.xyz"
            write_single_xyz(out, p.frame)

    def _write_gaussian_inputs(self, points: list[ScanPoint]) -> None:
        for p in points:
            job_dir = self.paths.jobs_dir / p.id
            self.gaussian.write_job(job_dir, p)

    def _update_manifest_with_energies(self, job_rows: list[dict]) -> None:
        p = self.paths.tables_dir / "scan_manifest.csv"
        if not p.exists():
            return
        df = pd.read_csv(p)
        jobs = pd.DataFrame(job_rows)
        if not jobs.empty:
            cols = ["id", "status", "e_hartree", "e_kcalmol_abs", "has_optimized_geometry", "optimized_xyz", "parse_note"]
            if "jobid" in jobs.columns:
                cols.append("jobid")
            df = df.drop(columns=[c for c in cols if c in df.columns and c != "id"], errors="ignore")
            df = df.merge(jobs[[c for c in cols if c in jobs.columns]], on="id", how="left")
        df.to_csv(p, index=False)

        # Relative energies against equilibrium parent.
        if "e_kcalmol_abs" in df.columns:
            ref = df[(df["scan_type"] == "equilibrium") & (df["status"] == "ok")].set_index("parent_id")["e_kcalmol_abs"].to_dict()
            rel = []
            for _, r in df.iterrows():
                e = r.get("e_kcalmol_abs")
                pe = ref.get(r.get("parent_id"))
                if pd.isna(e) or pe is None or pd.isna(pe):
                    rel.append(np.nan)
                else:
                    rel.append(float(e) - float(pe))
            df["e_kcalmol_rel"] = rel
            df.to_csv(p, index=False)
            # Job-level table only.  Per-point relaxed scan rows are written by
            # _write_expanded_energy_table().

    def _write_expanded_energy_table(self, rows: list[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        if "e_kcalmol_abs" in df.columns:
            refs = df[(df["scan_type"] == "equilibrium") & (df["status"] == "ok")].set_index("parent_id")["e_kcalmol_abs"].to_dict()
            rel = []
            for _, r in df.iterrows():
                e = r.get("e_kcalmol_abs")
                pe = refs.get(r.get("parent_id"))
                if pd.isna(e) or pe is None or pd.isna(pe):
                    rel.append(np.nan)
                else:
                    rel.append(float(e) - float(pe))
            df["e_kcalmol_rel"] = rel
        df.to_csv(self.paths.tables_dir / "scan_energy_curves.csv", index=False)

    def _write_reaxff_products(self) -> None:
        energy_p = self.paths.tables_dir / "scan_energy_curves.csv"
        manifest_p = self.paths.tables_dir / "scan_manifest.csv"
        if energy_p.exists():
            df = pd.read_csv(energy_p)
        elif manifest_p.exists():
            df = pd.read_csv(manifest_p)
        else:
            return
        self._invalidate_current_products()
        ok = df[df.get("status", "") == "ok"].copy() if "status" in df.columns else pd.DataFrame()
        if ok.empty:
            # Still write empty placeholders for visibility.
            write_trainset_in(self.paths.trainset_in, [], [], [])
            return

        # Reconstruct frames from Gaussian outputs for all OK rows included in
        # ReaxFF geo.  DESCRP identifiers must match trainset.in exactly, so use
        # the stable reaxff_id column rather than job-directory names.
        geo_rows: list[dict] = []
        bgf_paths: list[Path] = []
        proofs: list[dict] = []
        proof_cache: dict = {}
        for _, r in ok.iterrows():
            jd = Path(r["job_dir"])
            opt_col = str(r.get("optimized_xyz", "")) if "optimized_xyz" in r else ""
            xyz = Path(opt_col) if opt_col else (jd / "optimized_structure.xyz")
            if not xyz.exists():
                raise ValueError(f"Missing converged optimized geometry for {r.get('id')}; refusing GEO export")
            fr = read_first_frame(xyz)
            row = dict(r)
            if str(row.get('scan_type')) in {'bond', 'angle', 'torsion'}:
                atoms = [int(x) for x in str(row.get('atom_indices_1based', '')).split()]
                validate_scan_record_alignment(
                    [(float(row.get('e_hartree', 0.)), fr)], [float(row['target_value'])],
                    str(row['scan_type']), atoms,
                )
            proof=verify_optimized_record(row,fr,proof_cache)
            proof['optimized_xyz']=str(xyz)
            proofs.append(proof)
            ident = sanitize_reaxff_id(str(row.get("reaxff_id") or row.get("id")))
            row["id"] = ident
            row["reaxff_id"] = ident
            row["frame"] = fr
            geo_rows.append(row)
            if self.cfg.outputs.write_bgf:
                gcfg = self.cfg.outputs.geo
                rutype = gcfg.equilibrium_rutype if str(row.get("scan_type")) == "equilibrium" else gcfg.scan_rutype
                bgf_p = self.paths.bgf_dir / f"{ident.replace('.', 'p')}.bgf"
                write_bgf_structure(
                    bgf_p,
                    ident,
                    fr,
                    rutype=rutype,
                    remark=gcfg.remark,
                    include_format_lines=gcfg.include_format_lines,
                    periodic=gcfg.periodic,
                    cell=gcfg.cell,
                    restraint=restraint_record(row, gcfg.restraints.model_dump()),
                )
                bgf_paths.append(bgf_p)

        write_dataframe(self.paths.tables_dir/'optimization_provenance.csv',proofs)

        if self.cfg.outputs.write_geo:
            gcfg = self.cfg.outputs.geo
            write_geo_file(
                self.paths.geo_file,
                geo_rows,
                include_format_lines=gcfg.include_format_lines,
                periodic=gcfg.periodic,
                cell=gcfg.cell,
                remark=gcfg.remark,
                equilibrium_rutype=gcfg.equilibrium_rutype,
                scan_rutype=gcfg.scan_rutype,
                restraints=gcfg.restraints.model_dump(),
            )
        if self.cfg.outputs.write_models_in:
            write_models_in(self.paths.models_in, bgf_paths)

        reference_charges = set(int(v) for v in ok.loc[ok["scan_type"] == "equilibrium", "charge"]) if "charge" in ok else {0}
        if len(reference_charges) != 1:
            raise ValueError("Mixed reference charges cannot share a legacy ReaxFF syscha")
        common_charge = next(iter(reference_charges))
        (self.paths.exports_dir / "control.scan_recommended").write_text(
            "# Merge into your validated control; retain other engine-specific settings.\n"
            "# RUTYPE is omitted: imetho=1 must select minimization.\n"
            "      1 imetho\n      1 igeofo\n 80.000 axis1\n 80.000 axis2\n 80.000 axis3\n"
            f"      4 icharg\n      1 ichaen\n{common_charge:.5f} syscha\n"
            "0.10000 endmm\n   2000 imaxit\n      0 icelop\n      0 igeopt\n"
            "      1 iout4\n      0 iout5\n"
            "# Check convergence, restraint-coordinate deviation and exclusion of restraint energy from ENERGY.\n",
            encoding="utf-8")

        geometry_rows = self._build_geometry_targets(ok)
        write_dataframe(self.paths.tables_dir / "structure_name_map.csv", [
            {k: r.get(k, "") for k in ("reaxff_id", "parent_id", "coord_id", "scan_type", "family", "atom_indices_1based", "target_value", "job_dir")}
            for r in geo_rows])
        energy_rows = self._build_energy_targets(ok)
        charge_rows = self._build_charge_targets(ok)
        entries = []
        for r in charge_rows:
            rr = dict(r); rr["section"] = "CHARGE"; entries.append(rr)
        for r in geometry_rows:
            rr = dict(r); rr["section"] = "GEOMETRY"; entries.append(rr)
        for r in energy_rows:
            rr = dict(r); rr["section"] = "ENERGY"; entries.append(rr)
        write_dataframe(self.paths.tables_dir / "charge_targets.csv", charge_rows)
        write_dataframe(self.paths.tables_dir / "geometry_targets.csv", geometry_rows)
        write_dataframe(self.paths.tables_dir / "energy_targets.csv", energy_rows)
        write_dataframe(self.paths.tables_dir / "trainset_entries.csv", entries)

        if self.cfg.outputs.write_trainset_in:
            write_trainset_in(
                self.paths.trainset_in,
                geometry_rows,
                energy_rows,
                charge_rows,
                skip_zero_targets=self.cfg.trainset.energy.skip_zero_targets,
                zero_epsilon=self.cfg.trainset.energy.zero_epsilon_kcalmol,
                geometry_zero_epsilon=self.cfg.trainset.geometry.zero_epsilon,
            )
        if self.paths.geo_file.exists() and self.paths.trainset_in.exists():
            compatibility_issues = validate_legacy_reaxff_inputs(
                self.paths.geo_file.read_text(encoding="utf-8", errors="ignore"),
                self.paths.trainset_in.read_text(encoding="utf-8", errors="ignore"),
            )
            compatibility_report = {
                "legacy_reaxff_identifier_max_length": 19,
                "charge_records": len(charge_rows),
                "geometry_records": len(geometry_rows),
                "energy_records": len(energy_rows),
                "total_records": len(charge_rows) + len(geometry_rows) + len(energy_rows),
                "issues": compatibility_issues,
            }
            write_json(
                self.paths.tables_dir / "legacy_reaxff_compatibility.json",
                compatibility_report,
            )
            if compatibility_issues:
                raise ValueError(
                    "Legacy ReaxFF GEO/trainset compatibility validation failed:\n"
                    + "\n".join(compatibility_issues)
                )

    def _build_charge_targets(self, ok: pd.DataFrame) -> list[dict]:
        ccfg = self.cfg.trainset.charge
        if not ccfg.enabled:
            return []
        rows: list[dict] = []
        for _, r in ok.iterrows():
            stype = str(r.get("scan_type", ""))
            if stype == "equilibrium" and not ccfg.include_equilibrium:
                continue
            if stype != "equilibrium" and not ccfg.include_scan_points:
                continue
            jd = Path(r["job_dir"])
            opt_col = str(r.get("optimized_xyz", "")) if "optimized_xyz" in r else ""
            xyz = Path(opt_col) if opt_col else (jd / "optimized_structure.xyz")
            if not xyz.exists():
                xyz = jd / "input_guess.xyz"
            if not xyz.exists():
                continue
            fr = read_first_frame(xyz)
            try:
                charges, source = load_reference_charges(
                    jd,
                    natoms=len(fr.symbols),
                    method=ccfg.method,
                    fallback_method=ccfg.fallback_method,
                    adch_command=ccfg.adch_command,
                    adch_wavefunction_patterns=ccfg.adch_wavefunction_patterns,
                    adch_output_files=ccfg.adch_output_files,
                    adch_timeout_seconds=ccfg.adch_timeout_seconds,
                )
            except Exception as exc:
                raise ValueError(f"Missing/invalid reference charge table for {r.get('id')}: {exc}") from exc
            expected_charge = float(r.get("charge", 0))
            if not np.isfinite(charges).all() or abs(sum(charges) - expected_charge) > ccfg.total_charge_tolerance:
                raise ValueError(f"Charge conservation failed for {r.get('id')}: sum={sum(charges):.6f}, expected={expected_charge}")
            ident = sanitize_reaxff_id(str(r.get("reaxff_id") or r.get("id")))
            for idx, (sym, q) in enumerate(zip(fr.symbols, charges), start=1):
                rows.append({
                    "id": ident,
                    "parent_id": r.get("parent_id", ""),
                    "scan_type": stype,
                    "atom_index_1based": idx,
                    "atom_symbol": sym,
                    "charge": float(q),
                    "weight": float(ccfg.weight),
                    "charge_method": source,
                    "job_dir": str(jd),
                })
        return rows

    def _build_geometry_targets(self, ok: pd.DataFrame) -> list[dict]:
        if not self.cfg.trainset.geometry.enabled:
            return []
        coord_p = self.paths.tables_dir / "selected_coordinates.csv"
        if not coord_p.exists():
            return []
        try:
            coords = pd.read_csv(coord_p)
        except pd.errors.EmptyDataError:
            return []
        rows: list[dict] = []
        # Only equilibrium reference structures appear in GEOMETRY section.
        eq_ok = ok[ok["scan_type"] == "equilibrium"].copy()
        eq_ids = set(eq_ok["parent_id"].astype(str))
        for _, c in coords.iterrows():
            parent = str(c["parent_id"])
            if parent not in eq_ids:
                continue
            stype = c["scan_type"]
            if stype == "bond" and not self.cfg.trainset.geometry.include_equilibrium_bonds:
                continue
            if stype == "angle" and not self.cfg.trainset.geometry.include_equilibrium_angles:
                continue
            if stype == "torsion" and not self.cfg.trainset.geometry.include_equilibrium_torsions:
                continue
            erow = eq_ok[eq_ok["parent_id"].astype(str) == parent].iloc[0]
            ident = sanitize_reaxff_id(str(erow.get("reaxff_id") or erow.get("id")))
            weight = getattr(self.cfg.trainset.geometry.weights, stype)
            rows.append({
                "id": ident,
                "parent_id": parent,
                "coord_id": c["coord_id"],
                "target_type": stype,
                "family": c["family"],
                "atom_indices_1based": c["atom_indices_1based"],
                "value": float(c["eq_value"]),
                "weight": float(weight),
            })
        return rows

    def _build_energy_targets(self, ok: pd.DataFrame) -> list[dict]:
        if not self.cfg.trainset.energy.enabled or "e_kcalmol_rel" not in ok.columns:
            return []
        eq_tmp = ok[ok["scan_type"] == "equilibrium"].copy()
        if "reaxff_id" not in eq_tmp.columns:
            eq_tmp["reaxff_id"] = eq_tmp["id"]
        ref_ids = eq_tmp.set_index("parent_id")["reaxff_id"].to_dict()
        rows: list[dict] = []
        for _, r in ok.iterrows():
            stype = str(r["scan_type"])
            if stype == "equilibrium":
                continue
            rel = r.get("e_kcalmol_rel")
            if pd.isna(rel):
                continue
            rel = float(rel)
            if self.cfg.trainset.energy.max_rel_kcalmol is not None and rel > float(self.cfg.trainset.energy.max_rel_kcalmol):
                continue
            if self.cfg.trainset.energy.min_rel_kcalmol is not None and rel < float(self.cfg.trainset.energy.min_rel_kcalmol):
                continue
            ref = ref_ids.get(r["parent_id"])
            if not ref:
                continue
            base_weight = float(getattr(self.cfg.trainset.energy.weights, f"{stype}_scan"))
            weight = base_weight
            if bool(getattr(self.cfg.trainset.energy, "dynamic_weight_by_energy", False)):
                scale = max(float(getattr(self.cfg.trainset.energy, "dynamic_weight_scale_kcalmol", 50.0)), 1e-12)
                weight = base_weight * (1.0 + abs(float(rel)) / scale)
                weight = max(float(getattr(self.cfg.trainset.energy, "min_dynamic_weight", 0.5)), weight)
                weight = min(float(getattr(self.cfg.trainset.energy, "max_dynamic_weight", 25.0)), weight)
            rows.append({
                "id": sanitize_reaxff_id(str(r.get("reaxff_id") or r.get("id"))),
                "reference_id": sanitize_reaxff_id(str(ref)),
                "parent_id": r["parent_id"],
                "coord_id": r["coord_id"],
                "scan_type": stype,
                "family": r["family"],
                "scan_group_comment": r.get("scan_group_comment", r.get("family", "scan")),
                "target_value": r["target_value"],
                "eq_value": r["eq_value"],
                "target_kcalmol": rel,
                "weight": float(weight),
                "base_weight": float(base_weight),
                "weighting_note": "dynamic_by_abs_energy" if bool(getattr(self.cfg.trainset.energy, "dynamic_weight_by_energy", False)) else "constant",
            })
        return rows

    def _write_summary(self, stage: str) -> None:
        def nrows(name: str) -> int:
            p = self.paths.tables_dir / name
            if not p.exists():
                return 0
            try:
                return len(pd.read_csv(p))
            except Exception:
                return 0
        jobs = collect_job_table(self.paths.jobs_dir)
        summary = {
            "stage": stage,
            "run_dir": str(self.paths.run_dir),
            "exports_dir": str(self.paths.exports_dir),
            "equilibrium_structures": nrows("equilibrium_structures.csv"),
            "selected_coordinates": nrows("selected_coordinates.csv"),
            "scan_structures_total": nrows("scan_manifest.csv"),
            "gaussian_jobs_total": len(jobs),
            "gaussian_jobs_ok": sum(1 for r in jobs if r.get("status") == "ok"),
            "gaussian_jobs_failed": sum(1 for r in jobs if r.get("status") == "failed"),
            "trainset_geometry_entries": nrows("geometry_targets.csv"),
            "trainset_energy_entries": nrows("energy_targets.csv"),
            "trainset_charge_entries": nrows("charge_targets.csv"),
            "geo_written": self.paths.geo_file.exists(),
            "trainset_in_written": self.paths.trainset_in.exists(),
        }
        quality_file=self.paths.tables_dir/'export_quality.json'
        if quality_file.exists():
            quality=json.loads(quality_file.read_text(encoding='utf-8'))
            summary.update(export_partial=quality['partial'],scan_success_fraction=quality['success_fraction'],
                export_warning=quality['warning'],excluded_scan_jobs=quality['excluded_scan_jobs'])
        write_json(self.paths.summary_json, summary)

    def _point_to_row(self, p: ScanPoint) -> dict:
        return {
            "id": p.id,
            "parent_id": p.parent_id,
            "coord_id": p.coord_id,
            "scan_type": p.scan_type,
            "family": p.family,
            "point_index": p.point_index,
            "target_value": p.target_value,
            "eq_value": p.eq_value,
            "scan_start": getattr(p, "scan_start", None),
            "scan_stop": getattr(p, "scan_stop", None),
            "scan_step": getattr(p, "scan_step", None),
            "scan_nsteps": getattr(p, "scan_nsteps", None),
            "scan_values": json.dumps(getattr(p, "scan_values", []), ensure_ascii=False),
            "scan_prefix": getattr(p, "scan_prefix", ""),
            "scan_group_comment": getattr(p, "scan_group_comment", ""),
            "scan_unit": getattr(p, "scan_unit", ""),
            "reaxff_id": getattr(p, "reaxff_id", p.id),
            "atom_indices_1based": " ".join(str(a + 1) for a in p.atom_indices),
            "job_dir": p.job_dir,
            "status": p.status,
            "jobid": p.jobid,
            "e_hartree": p.e_hartree,
            "e_kcalmol_abs": p.e_kcalmol_abs,
            "e_kcalmol_rel": p.e_kcalmol_rel,
            "parse_note": p.parse_note,
        }
