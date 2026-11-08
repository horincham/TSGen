"""Native adapters for the inspected INDEEDopt and legacy ffopt layouts.

Settings (plain JSON): kind; reaxff_argv (shell-free argv, {exe} refers to
staged executable); seed (positive integer); atom_capacity (compiled limit).
Asset selectors reaxff_asset/md_control_asset/md_geometry_asset name immutable
request files, default adapter_reaxff/adapter_md_control/adapter_md_geo.
INDEEDopt also requires template_dir and optimizer_argv, and uses the inspected
ffield.optimized + bruteforce_best_error.dat outputs. Its folder/exe shim uses
a Python shebang (POSIX optimizer runtime required for real ./exe invocations).
MD requires coordinates='unwrapped', itrout=1 and ignotr=0 (default zero
is accepted). It imports diff_traj.xyz, which the inspected source writes using
image-corrected cp coordinates; raw wrapped xmolout is preserved separately.
MD stdout must contain the inspected ffopt normal-completion marker; a zero
exit code and partial trajectory alone do not establish success.
No protocol edits occur.
Evaluation uses generated SINGLE POINT BGF records and unit-weight relative
ENERGY rows with a 0.0001 dummy target (avoids legacy zero-target bug). Only the
ReaxFF predicted relative values are reported, never absolute energies.
Optional stage_files maps safe workdir basenames to immutable request asset keys
(e.g. {"tregime.in":"adapter_tregime"}); applies to all ReaxFF runs.
Charged evaluation records are rejected: this BGF exporter has no verified net
charge encoding. Reference and candidate electronic states must match.
No validation observable is invented: empty validation metrics cannot establish
outer-loop convergence. timeout_seconds defaults to 3600 per subprocess.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from tsgen.outerloop.io import atomic_json, read_json, sha256
from tsgen.outerloop.tasks import file_spec, verify_files
from tsgen.sensitivity.fort_parser import build_evaluation_result
from tsgen.sensitivity.trainset_map import parse_trainset, write_reaxff_compatible_geo


def _positive(s, key):
    v=s.get(key)
    if isinstance(v,bool) or not isinstance(v,int) or v<=0:
        raise ValueError(f'{key} must be an explicitly configured positive integer')
    return v


def _asset(request, name):
    if name not in request['files']:
        raise ValueError(f'Missing native request asset: {name}')
    return Path(request['files'][name]['path'])


def _argv(s, key, exe=None):
    args=s.get(key)
    if not isinstance(args,list) or not args or any(not isinstance(x,str) or not x for x in args):
        raise ValueError(f'{key} must be an explicit nonempty argv list')
    return [v.replace('{exe}',str(exe)) if exe else v for v in args]


def _run(directory, args, timeout):
    marker=directory/'execution.json'
    if marker.exists():
        state=read_json(marker)
        if state.get('status')=='RETURNED' and state.get('returncode')==0:
            return
        raise RuntimeError(f'Native execution failed or uncertain; refusing duplicate: {marker}')
    # Exclusive create prevents concurrent submissions, including interrupted runs.
    with marker.open('x',encoding='utf-8') as handle:
        json.dump(dict(status='STARTED',argv=args),handle)
    try:
        with (directory/'stdout.log').open('wb') as out, (directory/'stderr.log').open('wb') as err:
            result=subprocess.run(args,cwd=directory,stdout=out,stderr=err,timeout=timeout,check=False,shell=False)
        atomic_json(marker,dict(status='RETURNED',argv=args,returncode=result.returncode))
    except BaseException as exc:
        atomic_json(marker,dict(status='UNCERTAIN',argv=args,error=str(exc)))
        raise
    if result.returncode:
        raise RuntimeError(f'Native execution failed ({result.returncode}); see {directory}')


def _capacity(geo, limit):
    counts=[];n=0
    for raw in geo.read_text(encoding='utf-8').splitlines():
        word=raw.split()[:1]
        if word and word[0] in ('ATOM','HETATM'):n+=1
        if raw.strip()=='END':counts.append(n);n=0
    if n:counts.append(n)
    if not counts or min(counts)<=0 or max(counts)>limit:
        raise ValueError(f'Geometry empty or exceeds configured atom_capacity={limit}')
    return max(counts)


def _control(path):
    values={}
    for raw in path.read_text(encoding='utf-8').splitlines():
        if not raw.strip() or raw.lstrip().startswith(('#','!')):continue
        # Actual legacy reader uses (8x,a6), not arbitrary whitespace tokens.
        key=raw[8:14].strip()
        try:value=float(raw[:8].strip().replace('D','E').replace('d','e'))
        except ValueError:raise ValueError(f'Invalid legacy control record: {raw!r}')
        if not key or key in values or not math.isfinite(value):
            raise ValueError(f'Ambiguous legacy control record: {raw!r}')
        values[key]=value
    return values


def _stage_extras(directory, request, s):
    mapping=s.get('stage_files',{})
    if not isinstance(mapping,dict):raise ValueError('stage_files must be a filename to request asset mapping')
    reserved={'control','geo','ffield','trainset.in','reaxff.exe','exe','iopt','execution.json','stdout.log','stderr.log','xmolout'}
    for name,key in mapping.items():
        if not isinstance(name,str) or not name or Path(name).name!=name or '/' in name or '\\' in name or name in ('.','..') or name.startswith('fort.') or name in reserved:
            raise ValueError(f'Unsafe or reserved stage_files filename: {name}')
        shutil.copy2(_asset(request,key),directory/name)


def _stage_reax(directory, request, s, field, geo, control):
    directory.mkdir(parents=True)
    _stage_extras(directory,request,s)
    exe=directory/'reaxff.exe'
    shutil.copy2(_asset(request,s.get('reaxff_asset','adapter_reaxff')),exe)
    shutil.copy2(field,directory/'ffield');shutil.copy2(field,directory/'fort.4')
    shutil.copy2(control,directory/'control')
    write_reaxff_compatible_geo(geo,directory/'geo')
    shutil.copy2(directory/'geo',directory/'fort.3')
    (directory/'fort.20').write_text('  0\n',encoding='ascii')
    (directory/'fort.35').write_text(str(s['seed'])+'\n',encoding='ascii')
    return _argv(s,'reaxff_argv',exe.resolve())


def _objective(directory, field):
    return build_evaluation_result('native',directory/'fort.13',directory/'fort.99',
        parse_trainset(directory/'trainset.in'),0.,0,sha256(field))


def _md(work, request, s):
    control=_asset(request,s.get('md_control_asset','adapter_md_control'))
    geo=_asset(request,s.get('md_geometry_asset','adapter_md_geo'))
    values=_control(control)
    if values.get('imetho')!=0:raise ValueError('MD requires explicit imetho=0 (no minimization)')
    if values.get('ivels')!=0:raise ValueError('MD requires explicit ivels=0 (nonzero initial velocities)')
    if s.get('coordinates')!='unwrapped':raise ValueError('MD requires explicitly verified unwrapped coordinates')
    for key in ('nmdit','tstep','mdtemp','iout2'):
        if key=='iout2' and key not in values:continue
        if values.get(key,0)<=0:raise ValueError(f'MD requires positive {key}')
    if values.get('imdmet') not in (1,2,3,4,5,6,7,8):raise ValueError('Unsupported or missing MD imdmet')
    if values.get('igeofo')!=1 or values.get('ixmolo')!=1:
        raise ValueError('MD requires BIOGRF igeofo=1 and XYZ ixmolo=1')
    if any(raw.strip().startswith('RUTYPE') and raw.strip()!='RUTYPE NORMAL RUN' for raw in geo.read_text().splitlines()):
        raise ValueError('MD geometry contains a fitting/minimization RUTYPE override')
    if values.get('itrout')!=1:raise ValueError('MD requires explicit itrout=1 for image-corrected diff_traj.xyz')
    if values.get('ignotr',0)!=0:raise ValueError('MD requires ignotr=0 to preserve every atom')
    atoms=_capacity(geo,s['atom_capacity'])
    hashes=request.get('data',{}).get('md_input_hashes',{})
    if hashes:
        for key,digest in [('system',sha256(geo)),('protocol',sha256(control))]:
            if hashes.get(key)!=digest:raise ValueError('Native MD assets differ from declared initial MD inputs')
    else:hashes=dict(system=sha256(geo),protocol=sha256(control))
    outputs={}
    streams = [('new', 'new_ffield')]
    if request.get('data', {}).get('paired', True):
        streams.insert(0, ('old', 'ffield'))
    for side,key in streams:
        d=work/side
        args=_stage_reax(d,request,s,_asset(request,key),geo,control)
        _run(d,args,s.get('timeout_seconds',3600))
        if 'Normal end of MD-simulation' not in (d/'stdout.log').read_text(encoding='utf-8',errors='replace'):
            raise ValueError('Native MD did not report normal completion; partial trajectory rejected')
        trajectory=d/'diff_traj.xyz'
        from tsgen.data.xyz import read_xyz_multiframe
        frames=read_xyz_multiframe(trajectory)
        if not frames or any(len(f.symbols)!=atoms for f in frames):raise ValueError('Invalid native MD trajectory')
        outputs[side+'_xyz']=trajectory
        outputs[side+'_xmolout']=d/'xmolout'
        outputs[side+'_stdout']=d/'stdout.log';outputs[side+'_stderr']=d/'stderr.log'
    conditions=dict(seed=s['seed'],control_sha256=sha256(control),system_sha256=sha256(geo),control=values,extra_inputs={name:sha256(_asset(request,key)) for name,key in s.get('stage_files',{}).items()})
    atomic_json(work/'conditions.json', dict(
        {side: conditions for side, _ in streams},
        input_hashes=hashes,
        coordinates='unwrapped',
    ))
    outputs.update(conditions=work/'conditions.json',md_system=geo,md_protocol=control)
    return outputs,{}


def _template(source, destination):
    if not isinstance(source,str) or not source.strip():
        raise ValueError('template_dir must explicitly name a complete optimizer directory')
    source=Path(source).resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError('template_dir must not contain the native work directory')
    if not source.is_dir():raise ValueError('template_dir must be an existing complete optimizer directory')
    manifest={}
    for p in sorted(source.rglob('*')):
        if p.is_symlink():raise ValueError('Optimizer template symlinks are unsupported')
        if p.is_file() and '__pycache__' not in p.parts:
            manifest[p.relative_to(source).as_posix()]=sha256(p)
    if 'ffield.optimized' in manifest or 'bruteforce_best_error.dat' in manifest:
        raise ValueError('Optimizer template contains stale final outputs')
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    copied={name:sha256(destination/name) for name in manifest}
    if copied!=manifest:raise ValueError('Optimizer template changed while staging')
    atomic_json(destination.parent/'template_manifest.json',manifest)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


def _optimize(work, request, s):
    d=work/'optimizer'
    fingerprint=_template(s.get('template_dir',''),d)
    folder=d/'folder';folder.mkdir(exist_ok=True)
    for key,name in [('ffield','ffield'),('geo','geo'),('trainset','trainset.in'),('control','control'),('params','params-ml')]:
        shutil.copy2(_asset(request,key),folder/name)
    # Optimizer trials and independent recomputation must feed the identical
    # fixed-column dialect to the inspected legacy reader.
    write_reaxff_compatible_geo(_asset(request,'geo'),folder/'geo')
    shutil.copy2(folder/'params-ml',d/'params-ml')
    _capacity(folder/'geo',s['atom_capacity'])
    _stage_extras(folder,request,s)
    exe=folder/'reaxff.exe';shutil.copy2(_asset(request,s.get('reaxff_asset','adapter_reaxff')),exe)
    argv=_argv(s,'reaxff_argv',exe.resolve())
    # Each optimizer evaluation copies folder; construct argv from cwd so each
    # trial runs its own copied executable, never the original folder's files.
    argv=[v.replace(str(exe.resolve()),'./reaxff.exe') for v in argv]
    shim='#!'+sys.executable+'\nimport pathlib, shutil, subprocess\n'
    shim+="shutil.copy2('geo','fort.3'); shutil.copy2('ffield','fort.4')\n"
    shim+="pathlib.Path('fort.20').write_text('  0\\n'); pathlib.Path('fort.35').write_text("+repr(str(s['seed'])+'\n')+")\n"
    shim+="for name in ('fort.13','fort.99'): pathlib.Path(name).unlink(missing_ok=True)\n"
    shim+='raise SystemExit(subprocess.run('+repr(argv)+', shell=False).returncode)\n'
    (folder/'exe').write_text(shim,encoding='utf-8');(folder/'exe').chmod(0o755)
    (folder/'iopt').write_text('  0\n')
    _run(d,_argv(s,'optimizer_argv'),s.get('timeout_seconds',3600))
    final=d/'ffield.optimized'
    if not final.is_file():raise ValueError('INDEEDopt did not write ffield.optimized')
    claimed=float((d/'bruteforce_best_error.dat').read_text().strip())
    calc=work/'fit_recompute'
    args=_stage_reax(calc,request,s,final,_asset(request,'geo'),_asset(request,'control'))
    shutil.copy2(_asset(request,'trainset'),calc/'trainset.in')
    _run(calc,args,s.get('timeout_seconds',3600))
    result=_objective(calc,final)
    if not math.isfinite(claimed) or claimed<0 or not math.isclose(claimed,result.total_loss,rel_tol=1e-6,abs_tol=1e-8):
        raise ValueError('INDEEDopt objective differs from independently recomputed final force field')
    atomic_json(work/'fit_report.json',dict(objective=claimed,recomputed_objective=result.total_loss,
        independent_reaxff=True,n_items=len(result.item_losses),ffield_sha256=sha256(final)))
    outputs=dict(ffield=final,fit_report=work/'fit_report.json',fit_total=calc/'fort.13',fit_items=calc/'fort.99',
        optimizer_stdout=d/'stdout.log',optimizer_stderr=d/'stderr.log',fit_stdout=calc/'stdout.log',
        template_manifest=work/'template_manifest.json')
    return outputs,dict(template_sha256=fingerprint)


def _evaluate(work, request, s):
    import numpy as np
    from tsgen.data.xyz import Frame
    from tsgen.trainset.reaxff_io import bgf_block
    candidates=read_json(_asset(request,'candidates'));references=read_json(_asset(request,'reference_geometries'))
    if not candidates or len({r['id'] for r in candidates})!=len(candidates):raise ValueError('Empty or duplicate candidates')
    refs={r['species_id']:r for r in references}
    if len(refs)!=len(references):raise ValueError('Duplicate species references')
    for r in candidates+references:
        if r.get('charge',0)!=0:raise ValueError('Native evaluation of charged records needs a verified charge-aware engine adapter')
    records={r['id']:r for r in references}
    for r in candidates:
        if r['id'] in records and (r['symbols']!=records[r['id']]['symbols'] or r['coords']!=records[r['id']]['coords']):
            raise ValueError('Candidate/reference identity collision')
        records[r['id']]=r
        ref=refs.get(r['species_id'])
        if ref is None or sorted(r['symbols'])!=sorted(ref['symbols']) or r.get('multiplicity',1)!=ref.get('multiplicity',1):raise ValueError('Missing or incompatible species/electronic-state reference')
    names={ident:f'g{i:08d}' for i,ident in enumerate(records)}
    blocks=[]
    for ident,r in records.items():
        coords=np.asarray(r['coords'],dtype=float)
        if coords.shape!=(len(r['symbols']),3) or not np.isfinite(coords).all():raise ValueError('Invalid evaluation coordinates')
        blocks.append(bgf_block(names[ident],Frame(r['symbols'],coords,''),rutype='SINGLE POINT'))
    geo=work/'evaluation.geo';geo.write_text('\n\n'.join(blocks)+'\n',encoding='utf-8')
    _capacity(geo,s['atom_capacity'])
    train=work/'evaluation.trainset'
    train.write_text('ENERGY\n'+''.join(f"1 + {names[r['id']]}/1 - {names[refs[r['species_id']]['id']]}/1 0.0001\n" for r in candidates)+'ENDENERGY\n')
    control=_asset(request,s.get('evaluation_control_asset','control'))
    predictions={};outputs={}
    for side,key in [('old','ffield'),('new','new_ffield')]:
        d=work/side;args=_stage_reax(d,request,s,_asset(request,key),geo,control)
        shutil.copy2(train,d/'trainset.in');_run(d,args,s.get('timeout_seconds',3600))
        result=_objective(d,_asset(request,key))
        expected=[names[r['id']]+'+'+names[refs[r['species_id']]['id']] for r in candidates]
        if [item.structure_id for item in result.item_losses]!=expected:raise ValueError('Evaluation energy row order/identity mismatch')
        predictions[side]=[item.reaxff_value for item in result.item_losses]
        outputs[side+'_items']=d/'fort.99';outputs[side+'_total']=d/'fort.13';outputs[side+'_stdout']=d/'stdout.log'
    table=work/'energies.csv'
    with table.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','species_id','reference_id','old_energy','new_energy','old_reference','new_reference']);writer.writeheader()
        for i,r in enumerate(candidates):
            writer.writerow(dict(id=r['id'],species_id=r['species_id'],reference_id=refs[r['species_id']]['id'],old_energy=predictions['old'][i],new_energy=predictions['new'][i],old_reference=0.,new_reference=0.))
    atomic_json(work/'validation.json',dict(metrics={},protocol='',reason='No independent validation observable configured; cannot establish convergence'))
    atomic_json(work/'energy_identity.json',names)
    outputs.update(energies=table,validation=work/'validation.json',evaluation_geo=geo,evaluation_trainset=train,energy_identity=work/'energy_identity.json')
    return outputs,dict(energy_semantics='shared-reference relative kcal/mol',reference_zero='algebraic origin only; predicted fort.99 values are already E(candidate)-E(reference)',dummy_target_kcalmol=.0001)


def run_native_engine(task: Path, stage: str, settings: dict) -> tuple[dict[str,Path],dict] | None:
    """Execute a frozen native task once; interrupted/failed tasks need inspection.

    Successful receipts may be read repeatedly after verifying every output.
    An incomplete top-level marker always refuses execution, including a process
    that returned but whose collection was interrupted. This avoids unsafe retry.
    """
    if not settings:return None
    task=Path(task).resolve();s=dict(settings)
    kind=s.get('kind');expected={'indeedopt':'optimize','reaxff_md':'md','reaxff_evaluate':'evaluate'}
    if kind not in expected or expected[kind]!=stage:raise ValueError('Native kind does not match stage')
    request=read_json(task/'request.json');verify_files(request['files'])
    if request.get('data',{}).get('native_settings',s)!=s:raise ValueError('Native settings changed from request')
    fingerprint=hashlib.sha256(json.dumps(dict(settings=s,request_sha256=sha256(task/'request.json')),sort_keys=True).encode()).hexdigest()
    work=task/'native';receipt=work/'receipt.json';marker=task/'native_execution.json'
    if marker.exists():
        state=read_json(marker)
        if state.get('fingerprint')!=fingerprint:raise ValueError('Native settings/request changed')
        if state.get('status')!='COMPLETE' or not receipt.exists():raise RuntimeError('Native execution incomplete or uncertain; refusing duplicate execution')
        saved=read_json(receipt);verify_files(saved['outputs'])
        return {key:Path(v['path']) for key,v in saved['outputs'].items()},saved['metadata']
    s['seed']=_positive(s,'seed');s['atom_capacity']=_positive(s,'atom_capacity')
    _argv(s,'reaxff_argv')
    _asset(request,s.get('reaxff_asset','adapter_reaxff'))
    if work.exists():raise RuntimeError('Unowned native work directory exists; refusing stale outputs')
    with marker.open('x',encoding='utf-8') as handle:json.dump(dict(status='STARTED',fingerprint=fingerprint),handle)
    work.mkdir()
    outputs,metadata={'indeedopt':_optimize,'reaxff_md':_md,'reaxff_evaluate':_evaluate}[kind](work,request,s)
    metadata.update(native_kind=kind,request_sha256=sha256(task/'request.json'))
    verify_files(request['files'])
    atomic_json(receipt,dict(outputs={key:file_spec(p) for key,p in outputs.items()},metadata=metadata))
    atomic_json(marker,dict(status='COMPLETE',fingerprint=fingerprint))
    return outputs,metadata
