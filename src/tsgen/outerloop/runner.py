"""Iterative force-field refinement and dataset expansion."""
from __future__ import annotations
import shutil
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
from tsgen.data.xyz import Frame
from tsgen.trainset.io import next_run_dir, read_frames_from_reaxff_geo
from tsgen.explore.species import species_signature
from tsgen.outerloop.archive import (
    load_archive, save_archive, merge_records, frames_from_records, species_reference_energies,
)
from tsgen.outerloop.assessment import energy_disagreement
from tsgen.outerloop.candidates import extract
from tsgen.outerloop.config import OuterConfig, AdapterConfig
from tsgen.outerloop.convergence import decide
from tsgen.outerloop.fitting import validate_fit_pair, validate_ffield, geometry_key
from tsgen.outerloop.io import atomic_json, read_json, run_lock, sha256
from tsgen.outerloop.methods import _method_identity
from tsgen.outerloop.tasks import prepare_task, collect_task, execute_once, file_spec, verify_files, import_result

# D2_READY remains terminal for runs completed by the previous release.
TERMINAL = {'CONVERGED', 'BUDGET_EXHAUSTED', 'MAX_CYCLES', 'NO_NEW_LABELS', 'D2_READY', 'DATASET_UPDATED'}


def _archive_species(record, frame, md):
    signature = species_signature(frame, md.bond_tol, fingerprint_version=md.fingerprint_version)
    original = record['species_id']
    if re.search(r'_(?:[0-9a-f]{8}|v2[0-9a-f]{16})$', original) and original != signature:
        raise ValueError('Archive species IDs differ from the configured graph or fingerprint_version; use the original version or rebuild the reference archive')
    return signature


class OuterRunner:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        data = read_json(self.root/'config.json')
        data['md'].setdefault('fingerprint_version', 'legacy')
        self.cfg = OuterConfig.model_validate(data)

    @classmethod
    def create(cls, cfg: OuterConfig):
        root = cfg.run_dir.resolve() if cfg.run_dir is not None else None
        if root is not None and root.exists():
            raise ValueError(f'Run directory already exists; use resume: {root}')
        records = load_archive(cfg.initial.labels,project_root=cfg.project_root)
        if not records:
            raise ValueError('Initial archive is empty')
        if len({_method_identity(r['method']) for r in records})!=1:
            raise ValueError('Initial archive must use one consistent DFT method/route')
        tests = load_archive(cfg.initial.test_labels,project_root=cfg.project_root) if cfg.initial.test_labels else []
        if cfg.initial.holdout_ids:
            holdout_table=pd.read_csv(cfg.initial.holdout_ids,dtype={'id':str})
            if 'id' not in holdout_table:raise ValueError('Holdout table must contain id')
            excluded=set(holdout_table['id'])
            extracted=[r for r in records if r['id'] in excluded]
            tests=merge_records(tests,extracted,set())
            records=[r for r in records if r['id'] not in excluded]
            if not records:raise ValueError('No training labels remain after excluding holdout IDs')
        if any(_method_identity(r['method']) != _method_identity(records[0]['method']) for r in tests):
            raise ValueError('Test DFT method must match the reference training protocol')
        ec = None
        if cfg.explore_config:
            from tsgen.explore.config import load_explore_config, save_explore_config
            ec = load_explore_config(cfg.explore_config)
            if ec.species.fingerprint_version != cfg.md.fingerprint_version:
                raise ValueError('Explore and outer MD must use the same species fingerprint_version')
            if cfg.mode != 'legacy' and ec.labeling.backend != 'gaussian':
                raise ValueError('Outer acquisition requires the Gaussian labeling backend')
            if cfg.mode != 'legacy' and _method_identity(ec.labeling.gaussian.route) != _method_identity(records[0]['method']):
                raise ValueError('Explore Gaussian route must match historical labels')
        protected_frames = frames_from_records(tests)
        forbidden = {geometry_key(f) for f in protected_frames}
        for r, f in zip(records, frames_from_records(records)):
            if geometry_key(f) in forbidden:
                raise ValueError('Initial archive contains protected test geometry')
            r['species_id'] = _archive_species(r, f, cfg.md)
        references = species_reference_energies(
            records,
            require_optimized=ec is not None and ec.labeling.gaussian.reference_mode == 'optimized',
        )
        for r in records:r['reference_hartree']=references[r['species_id']]
        for r, f in zip(tests, protected_frames):
            r['species_id'] = _archive_species(r, f, cfg.md)
        explicit_test_refs = species_reference_energies(
            [r for r in tests if pd.notna(r.get('reference_hartree'))], require_optimized=True,
        )
        for sid in references.keys() & explicit_test_refs.keys():
            if not np.isclose(references[sid], explicit_test_refs[sid], atol=1e-9, rtol=0):
                raise ValueError(f'Test and training reference_hartree differ for {sid}')
        for r in tests:
            r['reference_hartree'] = references.get(r['species_id'], explicit_test_refs.get(r['species_id']))
        validate_fit_pair(cfg.initial.geo,cfg.initial.trainset,protected_frames)
        if cfg.existing_md is None:
            validate_ffield(cfg.initial.ffield,cfg.initial.ffield,cfg.initial.params)
        else:
            validate_ffield(cfg.existing_md.ffield, cfg.initial.ffield, params=None)
        initial_files = ['ffield', 'geo', 'trainset', 'control']
        if cfg.initial.params is not None:
            initial_files.append('params')
        if cfg.initial.holdout_ids is not None:
            initial_files.append('holdout_ids')
        all_paths = [getattr(cfg.initial,k) for k in initial_files] + list(cfg.md.inputs.values())
        if cfg.existing_md is not None:
            all_paths.extend([cfg.existing_md.ffield, *cfg.existing_md.trajectories])
        if cfg.explore_config:
            all_paths.append(cfg.explore_config)
            if ec.gating.reference_bonds_file:
                all_paths.append(Path(ec.gating.reference_bonds_file))
        if cfg.trainset_config:all_paths.append(cfg.trainset_config)
        for adapter in cfg.adapters.values():all_paths.extend(adapter.inputs.values())
        for p in all_paths:
            if not p.is_file():
                raise FileNotFoundError(p)
        # Validate internal calculation modes before creating a run or launching
        # any costly task. Automatic means Gaussian submission is configured too.
        sc=None
        if cfg.trainset_config:
            from tsgen.trainset.config import load_trainset_config,save_trainset_config
            sc=load_trainset_config(cfg.trainset_config)
            from tsgen.outerloop.methods import validate_scan_methods
            validate_scan_methods([r['method'] for r in records],sc)
            if sc.scan_generation.mode!='relaxed_modredundant':
                raise ValueError('Outer trainset construction requires relaxed_modredundant scans')
            from tsgen.outerloop.build_bridge import fitting_control_charge
            net_charge=fitting_control_charge(cfg.initial.control)
            if any(r['charge']!=net_charge for r in records) or cfg.fragment_charge!=net_charge:
                raise ValueError('Archive and fragment charges must match legacy fitting control syscha')
        if cfg.mode=='automatic' and (ec.labeling.gaussian.mode=='manual' or sc.gaussian.mode=='manual'):
            raise ValueError('Automatic mode requires local or slurm Gaussian for Explore and trainset construction')
        for stage,adapter in cfg.adapters.items():
            if adapter.native:
                expected={'optimize':'indeedopt','md':'reaxff_md','evaluate':'reaxff_evaluate'}
                if adapter.native.get('kind')!=expected.get(stage):
                    raise ValueError(f'Native adapter kind does not match {stage}')
                from tsgen.outerloop.engines import _positive,_argv
                _positive(adapter.native,'seed');_positive(adapter.native,'atom_capacity')
                _argv(adapter.native,'reaxff_argv')
                assets={'adapter_'+k for k in adapter.inputs}
                required=[adapter.native.get('reaxff_asset','adapter_reaxff')]
                if stage=='md':required.extend([adapter.native.get('md_control_asset','adapter_md_control'),adapter.native.get('md_geometry_asset','adapter_md_geo')])
                required.extend(adapter.native.get('stage_files',{}).values())
                if any(key not in assets for key in required):
                    raise ValueError(f'Missing native adapter assets for {stage}: {required}')
                if stage=='optimize':
                    _argv(adapter.native,'optimizer_argv')
                    template=adapter.native.get('template_dir')
                    if not template or not Path(template).is_dir():
                        raise ValueError('Native INDEEDopt needs a complete template_dir')
        if root is None:
            root = next_run_dir(cfg.out_dir.resolve(), 'outerloop')
        root.mkdir(parents=True)
        inputs = root/'inputs';inputs.mkdir()
        cfg = cfg.model_copy(deep=True)
        cfg.run_dir = root
        for name in initial_files:
            src = getattr(cfg.initial,name); dst = inputs/name
            shutil.copyfile(src,dst);setattr(cfg.initial,name,dst)
        if cfg.existing_md is not None:
            md_snapshot = inputs/'existing_md'
            md_snapshot.mkdir()
            cfg.existing_md.ffield = cls._copy(cfg.existing_md.ffield, md_snapshot/'ffield')
            cfg.existing_md.trajectories = [cls._copy(path, md_snapshot/f'{index:03d}_{path.name}')
                                            for index, path in enumerate(cfg.existing_md.trajectories, 1)]
        if cfg.initial.holdout_ids:
            cfg.initial.holdout_ids = cls._copy(cfg.initial.holdout_ids, inputs/'source_holdout_ids.csv')
        cfg.md.inputs = {k: cls._copy(p,inputs/('md_'+k+p.suffix)) for k,p in cfg.md.inputs.items()}
        for stage,adapter in cfg.adapters.items():
            if any(not k.replace('_','').isalnum() for k in adapter.inputs):
                raise ValueError('Adapter input keys must be simple names')
            adapter.inputs={k:cls._copy(p,inputs/('adapter_'+stage+'_'+k+p.suffix)) for k,p in adapter.inputs.items()}
            if adapter.native and stage=='optimize':
                from tsgen.outerloop.engines import _template
                destination=inputs/'optimizer_template'
                _template(adapter.native['template_dir'],destination)
                adapter.native['template_dir']=str(destination)
        if cfg.explore_config:
            if ec.gating.reference_bonds_file:
                reference_bonds = cls._copy(Path(ec.gating.reference_bonds_file), inputs/'reference_bonds.json')
                ec.gating.reference_bonds_file = str(reference_bonds)
            save_explore_config(ec,inputs/'explore.yaml')
            cfg.explore_config = inputs/'explore.yaml'
        if cfg.trainset_config:
            save_trainset_config(sc,inputs/'trainset_config.yaml')
            cfg.trainset_config=inputs/'trainset_config.yaml'
        save_archive(inputs/'archive.json',records)
        save_archive(inputs/'test_archive.json',tests)
        cfg.initial.labels = inputs/'archive.json'
        cfg.initial.test_labels = inputs/'test_archive.json'
        atomic_json(root/'config.json',cfg.model_dump(mode='json'))
        first_stage = 'existing_md' if cfg.existing_md is not None else 'optimize'
        state = dict(schema_version=1,cycle=1,stage=first_stage,status='READY',dft_used=0,
                     archive=str(cfg.initial.labels),ffield=str(cfg.initial.ffield),
                     geo=str(cfg.initial.geo),trainset=str(cfg.initial.trainset),control=str(cfg.initial.control),
                     history=[],pending_task=None,tracked={},ff_references={},scan_jobs_used=0,dft_jobs_used=0)
        for r in records:
            sid=r['species_id']
            if sid not in state['ff_references'] or r['e_hartree']<state['ff_references'][sid]['e_hartree']:
                state['ff_references'][sid]=r
        for p in [root/'config.json',*(p for p in inputs.rglob('*') if p.is_file())]:
            state['tracked'][str(p)] = file_spec(p)
        from tsgen.outerloop.reporting import write_surrogate_report
        write_surrogate_report(root/'surrogate_reports/cycle_000',records,tests,cfg)
        for p in (root/'surrogate_reports').rglob('*'):
            if p.is_file():state['tracked'][str(p)]=file_spec(p)
        atomic_json(root/'state.json',state)
        return cls(root)

    @staticmethod
    def _copy(src: Path, dst: Path) -> Path:
        shutil.copy2(src,dst)
        return dst

    def run_until_pause(self,wait: bool | None = None):
        """Automatic mode can keep polling internal/scheduler work; manual imports pause."""
        wait=self.cfg.mode=='automatic' if wait is None else wait
        start=time.monotonic()
        while True:
            result=self.resume()
            if result['status']!='WAITING' or not wait:return result
            if self.cfg.mode=='manual' and result['stage'] in {'optimize','md'}:return result
            remaining=self.cfg.wait_timeout_seconds-(time.monotonic()-start)
            if remaining<=0:return result
            time.sleep(min(self.cfg.poll_seconds,remaining))

    def status(self) -> dict:
        return read_json(self.root/'state.json')

    def _save(self):
        atomic_json(self.root/'state.json',self.state)

    def _track(self, *paths):
        for p in paths:
            self.state['tracked'][str(p)] = file_spec(p)

    def _protected(self):
        return frames_from_records(load_archive(self.cfg.initial.test_labels))

    def _finish(self, reason):
        self.state['status'] = reason
        self.state['pending_task'] = None
        if 'new_ffield' in self.state:
            self.state['ffield'] = self.state['new_ffield']
        result = dict(reason=reason, cycle=self.state['cycle'], archive=self.state['archive'],
                      ffield=self.state['ffield'], geo=self.state['geo'], trainset=self.state['trainset'],
                      dft_used=self.state['dft_used'])
        if 'ffield_fit_archive' in self.state:
            result['ffield_fit_archive'] = self.state['ffield_fit_archive']
        if reason == 'DATASET_UPDATED':
            result['next_initial'] = {
                'ffield': self.state['ffield'],
                'geo': self.state['geo'],
                'trainset': self.state['trainset'],
                'control': self.state['control'],
                'labels': self.state['archive'],
                'test_labels': str(self.cfg.initial.test_labels),
            }
            if self.cfg.initial.holdout_ids:
                result['next_initial']['holdout_ids'] = str(self.cfg.initial.holdout_ids)
        atomic_json(self.root/'final.json', result)

    def resume(self) -> dict:
        with run_lock(self.root):
            self.state = self.status()
            try:
                verify_files(self.state['tracked'])
                self.cfg = OuterConfig.model_validate(read_json(self.root/'config.json'))
                if self.state['status'] in TERMINAL:
                    return self.state
                self.state['status'] = 'RUNNING'
                self.state.pop('error',None)
                while self.state['status'] == 'RUNNING':
                    self.directory = self.root/f'cycle_{self.state["cycle"]:03d}'
                    self.directory.mkdir(exist_ok=True)
                    getattr(self,'_'+self.state['stage'])()
                    self._save()
            except Exception as exc:
                self.state['status']='FAILED'
                self.state['error']=f'{type(exc).__name__}: {exc}'
                self._save()
                raise
            return self.state

    def _task(self, stage, extra_files=None, data=None):
        files = {k:Path(self.state[k]) for k in ('archive','ffield','geo','trainset','control')}
        if self.cfg.initial.params is not None:
            files['params']=self.cfg.initial.params
        files['protected_test_archive']=self.cfg.initial.test_labels
        files.update(extra_files or {})
        adapter=self.cfg.adapters.get(stage,AdapterConfig())
        files.update({'adapter_'+k:p for k,p in adapter.inputs.items()})
        task = self.directory/stage
        task_data=dict(cycle=self.state['cycle'],**(data or {}))
        # Preserve request hashes for pre-existing manual/command tasks.
        if adapter.native:task_data['native_settings']=adapter.native
        prepare_task(task,dict(stage=stage,run_dir=str(self.root),files=files,data=task_data))
        self.state['pending_task']=str(task)
        self._track(task/'request.json')
        self._save()  # durable before any external side effect
        result = collect_task(task)
        if result is None:
            if stage=='acquire' and self.cfg.explore_config:
                from tsgen.outerloop.bridge import run_explore
                product = run_explore(task,self.cfg,read_json(task/'request.json'))
                if product:
                    import_result(task,product[0],product[1])
            elif stage=='build' and self.cfg.trainset_config:
                from tsgen.outerloop.build_bridge import run_trainset_build
                product=run_trainset_build(task,self.cfg,read_json(task/'request.json'))
                if product:import_result(task,product[0],product[1])
            elif adapter.native:
                from tsgen.outerloop.engines import run_native_engine
                product=run_native_engine(task,stage,adapter.native)
                if product:import_result(task,product[0],product[1])
            else:
                execute_once(task,self.cfg.adapters.get(stage,AdapterConfig()))
            result = collect_task(task)
        if result is None:
            self.state['status']='WAITING'
            self.state['next_action']=('Import the completed optimizer result (FF and fit report)' if stage=='optimize' else
                'Import updated-force-field MD trajectories and simulation conditions' if stage=='md' else
                'TSGen is collecting/submitting Gaussian labels' if stage=='acquire' else
                'TSGen is optimizing/scanning new structures and merging cumulative targets' if stage=='build' else
                'Collect paired force-field energy evaluation')
            progress=task/'build_progress.json'
            if progress.exists():self.state['build_progress']=read_json(progress)
        else:
            self.state.pop('next_action',None)
            self._track(task/'result.json',*(Path(v['path']) for v in result['outputs'].values()))
        return result

    @staticmethod
    def _output(result,key):
        if key not in result['outputs']:
            raise ValueError(f'Task result missing output: {key}')
        return Path(result['outputs'][key]['path'])

    def _optimize(self):
        result=self._task('optimize')
        if result is None:return
        ff=self._output(result,'ffield')
        rows=validate_ffield(ff,Path(self.state['ffield']),self.cfg.initial.params)
        report=read_json(self._output(result,'fit_report'))
        actual=float(report['recomputed_objective']);claimed=float(report['objective'])
        if not np.isfinite([actual,claimed]).all() or actual<0 or not np.isclose(actual,claimed,rtol=1e-6,atol=1e-8):
            raise ValueError('Optimized objective was not consistently recomputed')
        pd.DataFrame(rows).to_csv(self.directory/'parameter_changes.csv',index=False)
        self._track(self.directory/'parameter_changes.csv')
        self.state['new_ffield']=str(ff)
        self.state['ffield_fit_archive'] = self.state['archive']
        self.state['stage']='md'

    def _md(self):
        extra={'new_ffield':Path(self.state['new_ffield'])}
        extra.update({'md_'+k:p for k,p in self.cfg.md.inputs.items()})
        hashes={k:sha256(p) for k,p in self.cfg.md.inputs.items()}
        request_path = self.directory/'md/request.json'
        paired = read_json(request_path)['data']['paired'] if request_path.is_file() else False
        result=self._task('md',extra,dict(md_input_hashes=hashes,coordinates='unwrapped',paired=paired))
        if result is None:return
        conditions=read_json(self._output(result,'conditions'))
        if not hashes:
            imported={k[3:]:v for k,v in result['outputs'].items() if k.startswith('md_')}
            if not {'system','protocol'}.issubset(imported):
                raise ValueError('Manual MD result requires md_system and md_protocol artifacts when no initial MD files were configured')
            hashes={k:v['sha256'] for k,v in imported.items()}
        if not conditions.get('new') or conditions.get('input_hashes') != hashes:
            raise ValueError('MD stream requires declared conditions matched to initial inputs')
        if 'old_xyz' in result['outputs'] and conditions.get('old') != conditions['new']:
            raise ValueError('Paired diagnostic MD streams must have matched conditions')
        if conditions.get('coordinates')!='unwrapped':
            raise ValueError('MD trajectories must be explicitly unwrapped before fragment extraction')
        streams = {
            name: self._output(result, name)
            for name in sorted(result['outputs'])
            if name == 'new_xyz' or name.startswith('new_xyz_')
        }
        if not streams:
            raise ValueError('MD result is missing updated-force-field trajectory new_xyz')
        self._extract_md_candidates(streams)

    def _existing_md(self):
        existing = self.cfg.existing_md
        self.state['new_ffield'] = str(existing.ffield)
        streams = {f'new_{index:03d}': path for index, path in enumerate(existing.trajectories, 1)}
        self._extract_md_candidates(streams)

    def _extract_md_candidates(self, streams):
        protected_ids = None
        if self.cfg.initial.holdout_ids:
            protected_ids = set(pd.read_csv(self.cfg.initial.holdout_ids, dtype={'id': str})['id'])
        candidates = extract(streams, load_archive(Path(self.state['archive'])), self._protected(),
                             self.cfg, self.directory, protected_ids=protected_ids)
        for r in candidates:self.state['ff_references'].setdefault(r['species_id'],r)
        references=[self.state['ff_references'][sid] for sid in sorted({r['species_id'] for r in candidates})]
        atomic_json(self.directory/'reference_geometries.json',references)
        self._track(*(p for p in self.directory.iterdir() if p.is_file()))
        self._track(*(p for p in (self.directory/'preprocessing').rglob('*') if p.is_file()))
        self.state['stage']='evaluate'

    def _evaluate(self):
        candidates=read_json(self.directory/'candidates.json')
        # Keep each species' FF reference geometry fixed across outer cycles.
        refs={r['species_id']:r['id'] for r in read_json(self.directory/'reference_geometries.json')}
        result=self._task('evaluate',dict(new_ffield=Path(self.state['new_ffield']),
                                         candidates=self.directory/'candidates.json',candidate_xyz=self.directory/'candidates.xyz',
                                         reference_geometries=self.directory/'reference_geometries.json'),
                          dict(references=refs,energy_units='kcal/mol',validation_limits=self.cfg.validation_limits))
        if result is None:return
        energy=pd.read_csv(self._output(result,'energies'),dtype={'id':str,'reference_id':str,'species_id':str})
        per,delta=energy_disagreement(energy,{r['id']:r['species_id'] for r in candidates},refs)
        per.to_csv(self.directory/'energy_disagreement.csv',index=False)
        report=read_json(self._output(result,'validation'))
        metrics=report.get('metrics',{})
        passed=bool(self.cfg.validation_limits) and bool(report.get('protocol'))
        unavailable=not metrics and bool(report.get('reason'))
        if unavailable:passed=False
        for key,limit in ([] if unavailable else self.cfg.validation_limits.items()):
            value=float(metrics.get(key,float('nan')))
            if not np.isfinite(value) or value<0:
                raise ValueError(f'Invalid validation metric: {key}')
            passed = passed and value<=limit
        d=np.array([r['d_nn'] for r in candidates])
        new_species=sorted({r['species_id'] for r in candidates if r['new_species']})
        assessment=dict(cycle=self.state['cycle'],n_candidates=len(candidates),new_species=len(new_species),
                        novelty_median=float(np.median(d)),novelty_mean=float(np.mean(d)),novelty_q95=float(np.quantile(d,.95)),
                        delta_ff=delta,validation_passed=bool(passed),dft_used=self.state['dft_used'],
                        old_ffield_sha256=sha256(Path(self.state['ffield'])),new_ffield_sha256=sha256(Path(self.state['new_ffield'])),
                        reference_archive_sha256=sha256(Path(self.state['archive'])))
        atomic_json(self.directory/'assessment.json',assessment)
        atomic_json(self.directory/'new_species.json',new_species)
        self.state['history']=[m for m in self.state['history'] if m['cycle']!=self.state['cycle']]+[assessment]
        pd.DataFrame(self.state['history']).to_csv(self.root/'summary.csv',index=False)
        self._track(self.directory/'assessment.json',self.directory/'energy_disagreement.csv',self.directory/'new_species.json')
        if self.cfg.existing_md is not None:
            self.state['stage']='acquire'
            return
        decision=decide(self.state['history'],self.cfg.stop,self.state['dft_used'])
        if decision == 'CONTINUE' or (decision == 'MAX_CYCLES' and self.cfg.stop.acquire_on_final_cycle):
            self.state['stage']='acquire'
        else:
            self._finish(decision)

    def _acquire(self):
        budget=min(self.cfg.stop.labels_per_cycle,self.cfg.stop.dft_budget-self.state['dft_used'])
        result=self._task('acquire',dict(candidates=self.directory/'candidates.json',candidate_xyz=self.directory/'candidates.xyz'),
                          dict(label_budget=budget))
        if result is None:return
        old=load_archive(Path(self.state['archive']));old_ids={r['id'] for r in old}
        candidates={r['id']:r for r in read_json(self.directory/'candidates.json')}
        incoming=load_archive(self._output(result,'labels'))
        expected_method = _method_identity(old[0]['method'])
        for r in incoming:
            if r['id'] not in candidates:
                raise ValueError('Acquired label is outside requested candidate domain')
            expected=candidates[r['id']]
            if (r['charge'],r['multiplicity'])!=(expected['charge'],expected['multiplicity']):
                raise ValueError('Acquired electronic state does not match requested charge/multiplicity')
            r['species_id']=candidates[r['id']]['species_id']
            if _method_identity(r['method']) != expected_method:
                raise ValueError('DFT method changed across cycles')
        forbidden={geometry_key(f) for f in self._protected()}
        if any(geometry_key(f) in forbidden for f in frames_from_records(incoming)):
            raise ValueError('Acquired label is protected test geometry')
        merged=merge_records(old,incoming,{r['id'] for r in load_archive(self.cfg.initial.test_labels)})
        from tsgen.explore.config import load_explore_config
        require_optimized = (
            self.cfg.explore_config is not None
            and load_explore_config(self.cfg.explore_config).labeling.gaussian.reference_mode == 'optimized'
        )
        references = species_reference_energies(old + incoming, require_optimized=require_optimized)
        for r in merged:r['reference_hartree']=references[r['species_id']]
        new=[r for r in merged if r['id'] not in old_ids]
        attempts=result['metadata'].get('attempted')
        if not isinstance(attempts,int) or isinstance(attempts,bool) or not len(new)<=attempts<=budget:
            raise ValueError('Acquisition attempted count is invalid or exceeds budget')
        save_archive(self.directory/'archive_next.json',merged)
        atomic_json(self.directory/'new_labels.json',new)
        from tsgen.outerloop.reporting import write_surrogate_report
        write_surrogate_report(self.root/f'surrogate_reports/cycle_{self.state["cycle"]:03d}',
                               merged,load_archive(self.cfg.initial.test_labels),self.cfg)
        for p in (self.root/f'surrogate_reports/cycle_{self.state["cycle"]:03d}').rglob('*'):
            if p.is_file():self._track(p)
        self._track(self.directory/'archive_next.json',self.directory/'new_labels.json')
        # Charge once: the stage transition and accounting are persisted together.
        self.state['dft_used']+=attempts
        self.state['dft_jobs_used']=self.state['dft_used']+self.state.get('scan_jobs_used',0)
        if not new:
            self._finish('BUDGET_EXHAUSTED' if self.state['dft_used']>=self.cfg.stop.dft_budget else 'NO_NEW_LABELS')
        else:
            self.state['stage']='build'

    def _build(self):
        result=self._task('build',dict(archive=self.directory/'archive_next.json',new_labels=self.directory/'new_labels.json',
                                      new_ffield=Path(self.state['new_ffield'])))
        if result is None:return
        geo,trainset,control=[self._output(result,k) for k in ('geo','trainset','control')]
        geometries=validate_fit_pair(geo,trainset,self._protected(),(Path(self.state['geo']),Path(self.state['trainset'])))
        added={r['id'] for r in read_json(self.directory/'new_labels.json')}
        included=set(result['metadata'].get('included_label_ids',[]))
        if not included or not included.issubset(added):
            raise ValueError('Updated fit must record which newly acquired labels generated fitting targets')
        provenance=pd.read_csv(self._output(result,'fit_provenance'),dtype=str)
        if not {'geo_id','source_label_id'}.issubset(provenance.columns):
            raise ValueError('Fitting provenance requires geo_id and source_label_id')
        from tsgen.sensitivity.trainset_map import parse_trainset
        used={i for item in parse_trainset(trainset) for i in item.structure_ids}
        mapping=dict(geometries)
        old_geometry_ids={i for i,_ in read_frames_from_reaxff_geo(Path(self.state['geo']),preserve_ids=True)}
        new_records={r['id']:r for r in read_json(self.directory/'new_labels.json')}
        traced=set()
        for row in provenance.to_dict('records'):
            gid,lid=row['geo_id'],row['source_label_id']
            if lid not in included or gid not in mapping or gid not in used or gid in old_geometry_ids:
                raise ValueError('Invalid new fitting geometry provenance')
            if sorted(mapping[gid].symbols)!=sorted(new_records[lid]['symbols']):
                raise ValueError('Fitting provenance elemental composition mismatch')
            from tsgen.outerloop.fitting import verify_derivation
            verify_derivation(row,mapping[gid],new_records[lid],result['outputs'])
            traced.add(lid)
        if traced!=included:
            raise ValueError('Every included label requires actual new fitting target provenance')
        scan_jobs=result['metadata'].get('scan_jobs_attempted',0)
        if not isinstance(scan_jobs,int) or not 0<=scan_jobs<=self.cfg.scan_job_budget:
            raise ValueError('Invalid scan job accounting')
        self.state['scan_jobs_used']=self.state.get('scan_jobs_used',0)+scan_jobs
        self.state['dft_jobs_used']=self.state['dft_used']+self.state['scan_jobs_used']
        atomic_json(self.directory/'dataset_update.json',dict(
            archive_before_sha256=sha256(Path(self.state['archive'])),archive_after_sha256=sha256(self.directory/'archive_next.json'),
            geo_before_sha256=sha256(Path(self.state['geo'])),geo_after_sha256=sha256(geo),
            trainset_before_sha256=sha256(Path(self.state['trainset'])),trainset_after_sha256=sha256(trainset),
            new_labels=len(added),fitted_new_labels=len(included),scan_jobs_attempted=scan_jobs))
        self._track(self.directory/'dataset_update.json')
        self.state.update(archive=str(self.directory/'archive_next.json'),geo=str(geo),trainset=str(trainset),control=str(control),
                          ffield=self.state.pop('new_ffield'),pending_task=None)
        if self.cfg.existing_md is not None:
            self._finish('DATASET_UPDATED')
        elif self.state['cycle'] >= self.cfg.stop.max_cycles:
            self._finish('MAX_CYCLES')
        else:
            self.state.update(cycle=self.state['cycle']+1,stage='optimize')
