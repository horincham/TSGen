"""Reuse ExploreRunner in an independent, fixed pool for each outer cycle."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from tsgen.data.xyz import Frame, write_xyz_multiframe
from tsgen.explore.config import load_explore_config, save_explore_config, CandidatePoolCfg, SeedXYZSourceCfg, TrajXYZSourceCfg
from tsgen.explore.runner import ExploreRunner
from tsgen.explore.io import ExplorePaths, save_state
from tsgen.explore.label_store import LabelRecord, HARTREE_TO_KCALMOL
from tsgen.outerloop.archive import frames_from_records, load_archive, save_archive, species_reference_energies
from tsgen.outerloop.io import atomic_json, read_json
from tsgen.outerloop.fitting import geometry_key
from tsgen.outerloop.methods import _method_identity


def initialize_explore(directory: Path, cfg, archive, candidates, protected, budget: int):
    directory.mkdir(parents=True,exist_ok=True)
    cfg=cfg.model_copy(deep=True)
    forbidden={geometry_key(f) for f in frames_from_records(protected)}
    old_ids={r['id'] for r in archive}
    fresh=[r for r in candidates if r['id'] not in old_ids and geometry_key(Frame(r['symbols'],np.asarray(r['coords']),'')) not in forbidden]
    cfg.anchors.enabled=False
    cfg.holdout.size = 0
    cfg.gating.enabled = False
    cfg.export.scheme='validated'
    cfg.loop.mode='step'
    cfg.run.dft_budget=budget
    cfg.stop.dft_budget=budget
    cfg.stop.max_iters=cfg.run.max_iters=max(cfg.run.max_iters,budget)
    inherited=directory/'inherited.xyz';proposal=directory/'proposals.xyz'
    if not inherited.exists():
        write_xyz_multiframe(inherited, frames_from_records(archive), precision=16)
    if not proposal.exists():
        write_xyz_multiframe(proposal, frames_from_records(fresh), precision=16)
    cfg.candidate_pool=CandidatePoolCfg(sources=[SeedXYZSourceCfg(path=str(inherited)),TrajXYZSourceCfg(path=str(proposal))],split_fragments=False)
    # The fixed pool has already been screened and has no frame support counts.
    cfg.acquisition.representative = None
    save_explore_config(cfg,directory/'config.yaml')
    runner=ExploreRunner(cfg,ExplorePaths(directory))
    if not runner.state.inherited_label_ids:
        runner.state.inherited_label_ids=[r['id'] for r in archive]
        runner.state.previous_species=sorted({r['species_id'] for r in archive})
        runner.state.holdout_ids=[]
        save_state(directory/'state.json',runner.state)
    if not runner.paths.labels_jsonl.exists():
        references = species_reference_energies(
            archive, require_optimized=cfg.labeling.gaussian.reference_mode == 'optimized'
        )
        atomic_json(directory/'energy_references.json', {
            sid: value * HARTREE_TO_KCALMOL for sid, value in references.items()
        })
        reference_metadata = {}
        for record in archive:
            source = record.get('reference_provenance_path')
            if not isinstance(source, str) or not source.strip():
                continue
            metadata = read_json(Path(source))
            sid = record['species_id']
            if sid not in metadata or not np.isclose(
                metadata[sid]['energy_hartree'], references[sid], rtol=0., atol=1e-9
            ):
                raise ValueError(f'Inherited reference provenance differs from the DFT archive for {sid}')
            if sid in reference_metadata and reference_metadata[sid] != metadata[sid]:
                raise ValueError(f'Inherited optimized reference geometries disagree for {sid}')
            reference_metadata[sid] = metadata[sid]
        if reference_metadata:
            atomic_json(runner.labels.reference_metadata_path, reference_metadata)
        runner.labels.append([LabelRecord(r['id'],r['species_id'],r['e_hartree'],r['e_hartree']*HARTREE_TO_KCALMOL,0.,-1,
                              r.get('job_dir',''),r.get('log_path',''),route=r['method'],charge=r['charge'],multiplicity=r['multiplicity']) for r in archive])
    runner._ensure_pool()
    runner._sync_pool_with_jobs_and_labels()
    save_state(directory/'state.json',runner.state)
    return runner


def run_explore(task: Path, outer_cfg, request: dict):
    archive=load_archive(Path(request['files']['archive']['path']))
    candidates=read_json(Path(request['files']['candidates']['path']))
    tests=load_archive(Path(request['files']['protected_test_archive']['path']))
    cfg=load_explore_config(outer_cfg.explore_config)
    cfg.descriptor=outer_cfg.descriptor.model_copy(deep=True)
    cfg.species.bond_tol=outer_cfg.md.bond_tol
    cfg.species.fingerprint_version=outer_cfg.md.fingerprint_version
    cfg.labeling.gaussian.charge=outer_cfg.fragment_charge
    cfg.labeling.gaussian.multiplicity=outer_cfg.fragment_multiplicity
    if _method_identity(cfg.labeling.gaussian.route) != _method_identity(archive[0]['method']):
        raise ValueError('Explore Gaussian route must match the historical DFT method before submitting')
    runner=initialize_explore(task/'explore',cfg,archive,candidates,tests,request['data']['label_budget'])
    runner.strict_submission=True
    # Collect first and never submit a second batch while any prior jobs are pending.
    runner._collect_oracle_labels()
    reconcile_jobs(runner)
    if cfg.labeling.gaussian.mode=='local':
        run_local_jobs(runner)
        runner._collect_oracle_labels()
        reconcile_jobs(runner)
    pending=runner.pool.df['status'].astype(str).isin(['submitted','prepared']).any()
    if not pending and not runner._should_stop():
        runner.step_once()
        if cfg.labeling.gaussian.mode=='local':
            run_local_jobs(runner)
        runner._collect_oracle_labels()
        reconcile_jobs(runner)
    pending=runner.pool.df['status'].astype(str).isin(['submitted','prepared']).any()
    save_state(runner.paths.state_file,runner.state)
    if pending:return None
    available=runner._eligible_unlabeled_mask().any()
    if not runner._should_stop() and available:return None
    old_ids={r['id'] for r in archive}
    labels=runner.labels.load()
    index={str(row['id']):i for i,row in runner.pool.df.iterrows()}
    refs=read_json(runner.paths.run_dir/'energy_references.json')
    records=[]
    for row in labels.to_dict('records'):
        if row['id'] in old_ids:continue
        frame=runner.pool.frames[index[row['id']]]
        record = dict(id=row['id'],symbols=frame.symbols,coords=frame.coords.tolist(),e_hartree=row['e_hartree'],
                      method=row['route'],species_id=row['species_id'],charge=row['charge'],multiplicity=row['multiplicity'],
                      log_path=row['log_path'],job_dir=row['job_dir'],reference_hartree=refs[row['species_id']]/HARTREE_TO_KCALMOL)
        for key in ('reference_provenance_path', 'reference_geometry_path', 'reference_geometry_id',
                    'reference_log_path', 'reference_route'):
            if isinstance(row.get(key), str) and row[key].strip():
                record[key] = row[key]
        records.append(record)
    out=task/'acquired_labels.json';save_archive(out,records)
    return {'labels':out},{'attempted':int(runner.state.dft_used),'inner_run_dir':str(runner.paths.run_dir)}


def reconcile_jobs(runner):
    """Recover reservations from actual job directories, including mid-submit crashes."""
    attempted=set();index={str(i):j for j,i in enumerate(runner.pool.df['id'])}
    labeled = set(runner.labels.load()['id'].astype(str))
    failed_references = runner.reference_oracle.failures if runner.reference_oracle is not None else {}
    for job in runner.paths.run_dir.glob('iter_*/gaussian/jobs/job_*'):
        if not (job/'meta.json').exists():continue
        metadata=read_json(job/'meta.json');rid=str(metadata['id'])
        if rid not in index:raise ValueError('Gaussian job lies outside fixed pool')
        attempted.add(rid)
        status=runner.oracle.job_status(job)
        text=(job/'input.log').read_text(encoding='utf8',errors='replace') if (job/'input.log').exists() else ''
        if status=='failed' or text.rfind('Error termination')>text.rfind('Normal termination'):
            raise RuntimeError(f'Gaussian task failed: {job}; inspect output or import an explicitly curated acquisition result')
        marker=job/'local_execution.json'
        if marker.exists() and read_json(marker).get('returncode',-1)!=0:
            raise RuntimeError(f'Gaussian local execution failed or was interrupted: {job}')
        species_id = metadata['species_id']
        if status == 'ok' and rid not in labeled and species_id in failed_references:
            raise RuntimeError(
                f'Species reference optimization failed for {species_id}: {failed_references[species_id]}'
            )
        j=index[rid]
        runner.pool.df.at[j,'job_dir']=str(job)
        runner.pool.df.at[j,'status']='labeled' if rid in labeled else 'prepared'
    attempted-=set(runner.state.inherited_label_ids or [])
    runner.state.dft_used=max(runner.state.dft_used,len(attempted))
    save_state(runner.paths.state_file,runner.state)


def run_local_jobs(runner):
    jobs = sorted(runner.paths.run_dir.glob('references/jobs/job_*'))
    jobs.extend(sorted(runner.paths.run_dir.glob('iter_*/gaussian/jobs/job_*')))
    for job in jobs:
        runner.oracle.submit(job)
