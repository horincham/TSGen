"""Surrogate predictions for a fixed independent test set."""
from pathlib import Path
import numpy as np
import pandas as pd
from tsgen.descriptors.features import compute_descriptor_matrix
from tsgen.explore.config import SurrogateCfg, load_explore_config
from tsgen.explore.surrogate import build_surrogate
from tsgen.explore.label_store import HARTREE_TO_KCALMOL
from tsgen.outerloop.archive import frames_from_records
from tsgen.outerloop.io import atomic_json


def write_surrogate_report(directory: Path, archive: list[dict], tests: list[dict], cfg):
    if not tests:return
    if {r['id'] for r in archive} & {r['id'] for r in tests}:
        raise ValueError('Protected test leaked into surrogate training')
    directory.mkdir(parents=True,exist_ok=True)
    X=compute_descriptor_matrix(frames_from_records(archive+tests),cfg.descriptor)
    norm=np.linalg.norm(X,axis=1,keepdims=True)
    if np.any(norm<=1e-15) or not np.isfinite(X).all():
        raise ValueError('Invalid reporting descriptor')
    X=X/norm
    surrogate_cfg=load_explore_config(cfg.explore_config).surrogate if cfg.explore_config else SurrogateCfg(target='e_kcalmol_rel')
    absolute=surrogate_cfg.target in {'dft_energy','e_kcalmol_abs','e_hartree'}
    def relative(r):
        if r.get('reference_hartree') is None:return None
        return (r['e_hartree']-r['reference_hartree'])*HARTREE_TO_KCALMOL
    y=np.array([r['e_hartree']*HARTREE_TO_KCALMOL if absolute else relative(r) for r in archive])
    model=build_surrogate(surrogate_cfg)
    model.fit(X[:len(archive)],y)
    mu,sigma=model.predict(X[len(archive):])
    if absolute:
        mu=mu-np.array([r['reference_hartree']*HARTREE_TO_KCALMOL if r.get('reference_hartree') is not None else np.nan for r in tests])
    rows=[]
    for r,m,s in zip(tests,mu,sigma):
        target=relative(r)
        rows.append(dict(id=r['id'],species_id=r['species_id'],reference_hartree=r['reference_hartree'],
                         dft_relative_kcalmol=target,predicted_relative_kcalmol=float(m) if target is not None else None,
                         sigma_kcalmol=float(s),absolute_error_kcalmol=abs(float(m)-target) if target is not None else None,
                         status='ok' if target is not None else 'missing_training_reference'))
    table=pd.DataFrame(rows);table.to_csv(directory/'predictions.csv',index=False)
    errors=table.loc[table['status']=='ok','absolute_error_kcalmol'].to_numpy(dtype=float)
    atomic_json(directory/'metrics.json',dict(n_train=len(archive),n_test=len(tests),
                n_scored=len(errors),missing_reference_ids=table.loc[table['status']!='ok','id'].tolist(),
                mae=float(errors.mean()) if len(errors) else None,rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                q95=float(np.quantile(errors,.95)) if len(errors) else None,
                reference_source='frozen training references',descriptor=cfg.descriptor.model_dump(mode='json'),
                surrogate=surrogate_cfg.model_dump(mode='json'),role='reporting only; excluded from stopping'))
    model.save(directory/'model.pkl')
