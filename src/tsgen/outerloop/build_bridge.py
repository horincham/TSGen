"""Drive Opt/scan jobs, verify their provenance and build cumulative fitting files."""
from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from tsgen.data.xyz import Frame
from tsgen.trainset.builder import TrainsetBuilder
from tsgen.trainset.config import load_trainset_config, save_trainset_config
from tsgen.trainset.gaussian import collect_job_table, job_status
from tsgen.trainset.gaussian_scan_parser import parse_completed_opt_records
from tsgen.outerloop.archive import load_archive
from tsgen.outerloop.io import atomic_json, read_json, sha256
from tsgen.outerloop.fitting import validate_fit_pair


def _write_precise_xyz(path: Path, record: dict):
    lines=[str(len(record['symbols'])),f'id={record["id"]} charge={record["charge"]} multiplicity={record["multiplicity"]}']
    lines.extend(f'{s} {x:.12f} {y:.12f} {z:.12f}' for s,(x,y,z) in zip(record['symbols'],record['coords']))
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')


def fitting_control_charge(path: Path) -> float:
    """Legacy ReaxFF applies a single global syscha, default zero."""
    values=[]
    for raw in path.read_text(encoding='utf8').splitlines():
        if raw.lstrip().startswith(('#','!')):continue
        if raw[8:14].strip()=='syscha':
            value=float(raw[:8].strip().replace('D','E').replace('d','e'))
            if not np.isfinite(value):raise ValueError('Nonfinite fitting syscha')
            values.append(value)
        elif 'syscha' in raw.split('#',1)[0].lower():
            raise ValueError('Malformed legacy fitting syscha control record')
    if len(values)>1:raise ValueError('Ambiguous fitting syscha control records')
    return values[0] if values else 0.


def _advance(builder: TrainsetBuilder, limit: int):
    """Run nonblocking phases and reserve every submission before its side effect."""
    for _ in range(3):
        jobs=sorted(p for p in builder.paths.jobs_dir.iterdir() if p.is_dir())
        if len(jobs)>limit:
            raise ValueError(f'Scan job budget exceeded: planned {len(jobs)}, limit {limit}')
        for job in jobs:
            status=job_status(job)[0]
            if status=='ok':continue
            if status=='failed':
                # Let the builder's configured partial-success policy decide;
                # never resubmit a failed job without an explicit new attempt.
                continue
            if builder.cfg.gaussian.mode=='manual':continue
            marker=job/'outer_submission.json'
            if marker.exists():
                execution=read_json(marker)
                if execution.get('status')=='FAILED':
                    raise RuntimeError(f'Gaussian submission failed: {job}; inspect logs before retrying')
                continue
            if (job/'input.log').exists() or (job/'sbatch.stdout').exists():continue
            atomic_json(marker,dict(status='STARTED',input_sha256=sha256(job/'input.gjf')))
            try:
                jobid=builder.gaussian.submit(job)
            except Exception as exc:
                atomic_json(marker,dict(status='FAILED',error=str(exc)))
                raise
            if builder.cfg.gaussian.mode=='slurm' and not jobid:
                atomic_json(marker,dict(status='FAILED',error='No scheduler job ID'))
                raise RuntimeError(f'Gaussian submission produced no job ID: {job}')
            atomic_json(marker,dict(status='RETURNED',jobid=jobid))
        before=builder._workflow()['stage']
        builder.collect_and_finalize()
        stage=builder._workflow()['stage']
        if stage=='finalized':return True
        summary=read_json(builder.paths.summary_json)
        if summary.get('stage') in {'equilibrium_failed','scan_quality_failed','export_integrity_failed'}:
            raise RuntimeError(f'Trainset construction failed: {summary.get("stage")}')
        if stage==before:return False
    return False


def _lineage(job: Path, index: int, prefix: str, outputs: dict, transform=None):
    input_key,log_key=prefix+'_input',prefix+'_log'
    outputs[input_key]=job/'input.gjf';outputs[log_key]=job/'input.log'
    result=dict(input_artifact=input_key,log_artifact=log_key,record_index=index)
    if transform is not None:result['transform']=transform
    return result


def run_trainset_build(task: Path, outer_cfg, request: dict):
    from tsgen.outerloop.merge import merge_fitting_inputs
    records=load_archive(Path(request['files']['new_labels']['path']))
    if not records:raise ValueError('No new labels available for fitting-set construction')
    net_charge=fitting_control_charge(Path(request['files']['control']['path']))
    if any(r['charge']!=net_charge for r in records):
        raise ValueError('New label charge does not match the legacy fitting control syscha')
    source_dir=task/'new_structures';source_dir.mkdir(parents=True,exist_ok=True)
    mapping={}
    for record in records:
        p=source_dir/f'label_{record["id"]}.xyz'
        if not p.exists():_write_precise_xyz(p,record)
        mapping[str(p.resolve())]=record['id']
    builder_dir=task/'trainset'
    if (builder_dir/'config.yaml').exists():
        builder=TrainsetBuilder.from_run_dir(builder_dir)
    else:
        cfg=load_trainset_config(outer_cfg.trainset_config)
        if cfg.scan_generation.mode!='relaxed_modredundant':
            raise ValueError('Outer trainset bridge currently requires verified relaxed_modredundant scans')
        cfg.input.source='xyz_dir';cfg.input.xyz_dir=str(source_dir.resolve())
        cfg.input.deduplicate_structures=True
        cfg.execution.submit=False;cfg.execution.wait_for_completion=False
        # Submission is managed by _advance to write durable reservations.
        cfg.outputs.write_geo=True;cfg.outputs.write_trainset_in=True
        cfg.outputs.write_report=False
        cfg.gaussian.state_overrides={**cfg.gaussian.state_overrides,
            **{f'label_{r["id"]}':dict(charge=r['charge'],multiplicity=r['multiplicity']) for r in records}}
        builder=TrainsetBuilder(cfg,builder_dir)
        save_trainset_config(cfg,builder.paths.config_file)
    from tsgen.outerloop.methods import validate_scan_methods
    validate_scan_methods([r['method'] for r in records],builder.cfg)
    builder.plan()
    complete=_advance(builder,int(outer_cfg.scan_job_budget))
    jobs=collect_job_table(builder.paths.jobs_dir)
    attempted=sum((Path(r['job_dir'])/'outer_submission.json').exists() or
                  (Path(r['job_dir'])/'input.log').exists() or (Path(r['job_dir'])/'sbatch.stdout').exists() for r in jobs)
    atomic_json(task/'build_progress.json',dict(stage=builder._workflow()['stage'],
        jobs_planned=len(jobs),scan_jobs_attempted=attempted,jobs_ok=sum(r['status']=='ok' for r in jobs)))
    if not complete:return None
    tests=load_archive(Path(request['files']['protected_test_archive']['path']))
    protected=[Frame(r['symbols'],np.asarray(r['coords']),'') for r in tests]
    validate_fit_pair(builder.paths.geo_file,builder.paths.trainset_in,protected)
    geo,trainset,id_map=merge_fitting_inputs(Path(request['files']['geo']['path']),Path(request['files']['trainset']['path']),
        builder.paths.geo_file,builder.paths.trainset_in,task/'cumulative')
    validate_fit_pair(geo,trainset,protected,(Path(request['files']['geo']['path']),Path(request['files']['trainset']['path'])))
    outputs={'geo':geo,'trainset':trainset,'control':Path(request['files']['control']['path'])}
    equilibrium=pd.read_csv(builder.paths.tables_dir/'equilibrium_structures.csv')
    parents={str(row['parent_id']):mapping[str(Path(row['source_path']).resolve())] for row in equilibrium.to_dict('records')}
    eq_jobs={r['parent_id']:Path(r['job_dir']) for r in jobs if r.get('scan_type')=='equilibrium' and r['status']=='ok'}
    curves=pd.read_csv(builder.paths.tables_dir/'scan_energy_curves.csv')
    provenance=[];lineage_dir=task/'lineage';lineage_dir.mkdir(exist_ok=True)
    for serial,row in enumerate(curves.to_dict('records')):
        old_id=str(row.get('reaxff_id') or row['id'])
        if old_id not in id_map:continue
        parent=str(row['parent_id']);source_id=parents[parent];eq=eq_jobs[parent]
        eq_records=parse_completed_opt_records((eq/'input.log').read_text(encoding='utf8',errors='replace'))
        chain=[_lineage(eq,len(eq_records)-1,f'p{serial}_eq',outputs)]
        if row['scan_type']!='equilibrium':
            job=Path(row['job_dir']);meta=read_json(job/'meta.json')
            transform=dict(scan_type=meta['scan_type'],atoms_0based=[int(i)-1 for i in meta['atom_indices_1based']],
                           target=float(meta['scan_values'][0]))
            chain.append(_lineage(job,int(row['point_index']),f'p{serial}_scan',outputs,transform))
        key=f'lineage_{serial}';path=lineage_dir/f'{key}.json';atomic_json(path,chain);outputs[key]=path
        provenance.append(dict(geo_id=id_map[old_id],source_label_id=source_id,lineage_artifact=key))
    if not provenance:raise ValueError('No verified new fitting targets were constructed')
    p=task/'fit_provenance.csv';pd.DataFrame(provenance).to_csv(p,index=False);outputs['fit_provenance']=p
    outputs['scan_quality']=builder.paths.tables_dir/'export_quality.json'
    return outputs,dict(included_label_ids=sorted({r['source_label_id'] for r in provenance}),
        scan_jobs_attempted=int(attempted),trainset_run_dir=str(builder_dir))
