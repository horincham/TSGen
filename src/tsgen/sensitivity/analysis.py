from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.hsic import repeated_hsic, student_t_interval
from tsgen.sensitivity.models import ParameterCoordinate
from tsgen.sensitivity.params_ml import parse_params_ml, write_params_ml,optimizer_bounds
from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.parameter_validity import exclusion_reason
from tsgen.sensitivity.trainset_map import parse_geo_inventory,parse_trainset
from tsgen.trainset.reaxff_io import validate_legacy_reaxff_inputs
from tsgen.sensitivity.paths import SensitivityPaths


TARGET_COLUMNS = {
    "TOTAL": "total_loss",
    "ENERGY": "energy_loss",
    "GEOMETRY": "geometry_loss",
    "CHARGE": "charge_loss",
}


def _digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def select_parameters(
    summary: pd.DataFrame,
    threshold: float,
    force_include: set[str],
    force_exclude: set[str],
    method: str = 'cumulative',top_k: int = 20,
) -> pd.DataFrame:
    values=summary['hsic_mean'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values<0) or not np.any(values>0):
        raise ValueError('No finite positive HSIC signal; cannot select parameters')
    ranked = summary.sort_values(["hsic_mean", "parameter_id"], ascending=[False, True]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["cumulative_fraction"] = ranked["hsic_mean"].cumsum()
    crossing = int(np.searchsorted(ranked["cumulative_fraction"].to_numpy(), threshold, side="left"))
    if method not in {'cumulative','top_k'}:raise ValueError(f'Unknown selection method: {method}')
    count = min(crossing + 1, np.count_nonzero(values)) if method == "cumulative" else top_k
    automatic = set(ranked.iloc[: min(count, len(ranked))]["parameter_id"])
    decisions = []
    for row in ranked.itertuples(index=False):
        parameter_id = row.parameter_id
        physically_valid = bool(getattr(row, "physically_valid", True))
        if parameter_id in force_exclude:
            selected, reason = False, "force_exclude"
        elif not physically_valid:
            selected, reason = False, "physical_validity"
        elif parameter_id in force_include:
            selected, reason = True, "force_include"
        elif parameter_id in automatic:
            selected, reason = True, "cumulative_threshold" if method=='cumulative' else 'top_k'
        else:
            selected, reason = False, "below_threshold"
        record = row._asdict()
        record.update({"selected": selected, "reason": reason})
        decisions.append(record)
    return pd.DataFrame(decisions)


def _summary_from_repeats(repeats: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for parameter_id, group in repeats.groupby("parameter_id", sort=False):
        mean, standard_deviation, low, high = student_t_interval(
            group["normalized_hsic"].to_numpy()
        )
        rows.append(
            {
                "target": target,
                "parameter_id": parameter_id,
                "hsic_mean": mean,
                "raw_hsic_mean": float(group.raw_hsic.mean()),
                "raw_hsic_std": float(group.raw_hsic.std(ddof=1)),
                "selection_frequency": float(group["selected"].mean()),
                "hsic_std": standard_deviation,
                "hsic_ci_low": low,
                "hsic_ci_high": high,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["hsic_mean", "parameter_id"], ascending=[False, True]
    )
    result["rank_within_target"] = np.arange(1, len(result) + 1)
    return result


def _parameter_metadata(catalog: pd.DataFrame) -> pd.DataFrame:
    metadata = catalog.copy()
    metadata["parameter_label"] = (
        metadata["parameter_id"].astype(str)
        + " "
        + metadata["parameter_name"].astype(str)
        + metadata["block_key"].fillna("").astype(str).map(lambda value: f" [{value}]" if value else "")
    )
    metadata["parameter_group"] = (
        metadata["section_name"].astype(str) + ":" + metadata["block_key"].fillna("").astype(str)
    ).str.rstrip(":")
    return metadata[
        [
            "parameter_id",
            "parameter_label",
            "parameter_group",
            "section_name",
            "block_key",
        ]
    ]


def _rank_convergence(
    matrix: np.ndarray,
    loss: np.ndarray,
    parameter_ids: list[str],
    cfg: SensitivityConfig,
) -> pd.DataFrame:
    sizes = sorted(size for size in cfg.analysis.convergence_sizes if size <= len(loss))
    columns = [
        "sample_count",
        "comparison_sample_count",
        "spearman_all",
        "spearman_top_k",
        "top_k_overlap",
    ]
    if len(sizes) < 2:
        return pd.DataFrame(columns=columns)
    rankings: dict[int, list[str]] = {}
    for size in sizes:
        repeats = repeated_hsic(
            matrix[:size],
            loss[:size],
            parameter_ids,
            cfg.analysis.repeats,
            min(cfg.analysis.subsample_size, size),
            cfg.run.seed + size,
            cfg.analysis.parameter_sigma,
            cfg.analysis.loss_gamma,
        )
        means = repeats.groupby("parameter_id")["normalized_hsic"].mean().clip(lower=0)
        rankings[size] = list(means.sort_values(ascending=False).index)
    rows = []
    for first, second in zip(sizes[:-1], sizes[1:], strict=True):
        rank_first = {parameter: rank for rank, parameter in enumerate(rankings[first], 1)}
        rank_second = {parameter: rank for rank, parameter in enumerate(rankings[second], 1)}
        ordered = parameter_ids
        all_correlation = float(
            spearmanr(
                [rank_first[parameter] for parameter in ordered],
                [rank_second[parameter] for parameter in ordered],
            ).statistic
        )
        top_k = min(cfg.analysis.rank_top_k, len(parameter_ids))
        top_first = rankings[first][:top_k]
        top_second = rankings[second][:top_k]
        union = list(dict.fromkeys(top_first + top_second))
        top_correlation = float(
            spearmanr(
                [rank_first[parameter] for parameter in union],
                [rank_second[parameter] for parameter in union],
            ).statistic
        )
        rows.append(
            {
                "sample_count": first,
                "comparison_sample_count": second,
                "spearman_all": all_correlation,
                "spearman_top_k": top_correlation,
                "top_k_overlap": len(set(top_first) & set(top_second)) / top_k,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def analyze_run(paths: SensitivityPaths, cfg: SensitivityConfig) -> dict[str,pd.DataFrame]:
    try:
        return _analyze_run_impl(paths,cfg)
    except Exception:
        for name in ['params-ml.selected','params-ml.selected.full_precision.json',
                     'params-ml.rounding_audit.csv','params-ml.diagnostic_only','params-ml.diagnostic_only.full_precision.json']:
            artifact=paths.exports_dir/name
            if artifact.is_file():artifact.unlink()
        method_path=paths.tables_dir/'analysis_method.json'
        if method_path.exists():
            method=json.loads(method_path.read_text(encoding='utf-8'))
            method.update(production_ready=False,analysis_state='failed')
            method_path.write_text(json.dumps(method,indent=2),encoding='utf-8')
        raise


def _analyze_run_impl(paths: SensitivityPaths, cfg: SensitivityConfig) -> dict[str, pd.DataFrame]:
    # Invalidate optimizer handoffs before any validation can fail. Older
    # artifacts are retained under explicit previous names, never as current.
    for artifact in [paths.selected_params_ml,paths.exports_dir/'params-ml.selected.full_precision.json',
                     paths.exports_dir/'params-ml.rounding_audit.csv',paths.tables_dir/'analysis_method.json']:
        if artifact.is_file():
            shutil.copy2(artifact,artifact.with_name(artifact.name+'.previous'));artifact.unlink()
    if paths.indeedopt_bundle.exists():
        previous=paths.indeedopt_bundle.with_name('indeedopt_bundle.previous')
        suffix=1
        while previous.exists():
            suffix+=1;previous=paths.indeedopt_bundle.with_name(f'indeedopt_bundle.previous_{suffix}')
        if not all(p.resolve().is_relative_to(paths.exports_dir.resolve()) for p in [paths.indeedopt_bundle,previous]):
            raise ValueError('Optimizer bundle archive must remain inside exports')
        paths.indeedopt_bundle.rename(previous)
    samples = pd.read_csv(paths.parameter_samples)
    losses = pd.read_csv(paths.sample_losses)
    successful = losses.loc[losses["run_status"] == "success"].copy()
    inventory=parse_geo_inventory(paths.inputs_dir/'geo')
    items=parse_trainset(paths.inputs_dir/'trainset.in')
    input_issues=validate_legacy_reaxff_inputs((paths.inputs_dir/'geo').read_text(encoding='utf-8'),
        (paths.inputs_dir/'trainset.in').read_text(encoding='utf-8'))
    coverage={section:dict(declared=sum(item.section==section for item in items),
        evaluated_counts=sorted(set(int(v) for v in successful.get('n_items_'+section.lower(),pd.Series(dtype=int)).dropna())))
        for section in ['ENERGY','GEOMETRY','CHARGE']}
    incomplete=any(values['evaluated_counts']!=[values['declared']] for values in coverage.values() if values['declared'])
    quality=dict(diagnostic_only=cfg.analysis.diagnostic_only,input_issues=input_issues,objective_coverage=coverage,
        production_ready=not input_issues and not incomplete)
    (paths.tables_dir/'analysis_quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
    if not quality['production_ready'] and not cfg.analysis.diagnostic_only:
        raise ValueError('Incomplete objective coverage or unsafe legacy inputs; correct the training inputs and rerun ReaxFF, '
                         'or use --diagnostic-only for retrospective ranking without optimizer export')
    success_fraction = len(successful) / len(samples)
    if len(successful) < cfg.analysis.subsample_size:
        raise ValueError(
            f"HSIC requires {cfg.analysis.subsample_size} successful samples; observed {len(successful)}"
        )
    if success_fraction < cfg.analysis.minimum_success_fraction:
        raise ValueError(
            f"Successful sample fraction {success_fraction:.3f} is below "
            f"{cfg.analysis.minimum_success_fraction:.3f}"
        )
    merged = samples.merge(successful, on="sample_id", how="inner", validate="one_to_one")
    parameter_columns = [column for column in samples if column.startswith("normalized__")]
    parameter_ids = [column.split("__", 1)[1] for column in parameter_columns]
    matrix = merged[parameter_columns].to_numpy(dtype=float)
    document=FFieldDocument.from_path(paths.inputs_dir/'ffield')
    definitions={p.coordinate.parameter_id:p for p in document.parameters}
    reasons={pid:exclusion_reason(definitions[pid],document,inventory,items,cfg.parameter_space.exclude_inactive_parameters)
             for pid in parameter_ids}
    pd.DataFrame([dict(parameter_id=pid,eligible=not reason,reason=reason)
        for pid,reason in reasons.items()]).to_csv(paths.tables_dir/'analysis_parameter_eligibility.csv',index=False)
    keep=[i for i,pid in enumerate(parameter_ids) if not reasons[pid]]
    if not keep:raise ValueError('No physically eligible parameter candidates remain')
    parameter_ids=[parameter_ids[i] for i in keep]
    matrix=matrix[:,keep]
    catalog = pd.read_csv(paths.parameter_catalog)
    metadata = _parameter_metadata(catalog)
    repeat_frames = []
    summary_frames = []
    available_targets = []
    target_status=[]
    for target in cfg.analysis.targets:
        column = TARGET_COLUMNS[target]
        available = column in merged and merged[column].notna().all()
        if not available:
            target_status.append(dict(target=target,status='unavailable_or_missing_values'))
            continue
        vector=merged[column].to_numpy(dtype=float)
        if np.isfinite(vector).all() and np.ptp(vector)<=np.finfo(float).eps:
            target_status.append(dict(target=target,status='constant_loss_no_sensitivity'))
            continue
        target_repeats = repeated_hsic(
            matrix,
            merged[column].to_numpy(dtype=float),
            parameter_ids,
            cfg.analysis.repeats,
            cfg.analysis.subsample_size,
            cfg.run.seed,
            cfg.analysis.parameter_sigma,
            cfg.analysis.loss_gamma,
            selection_threshold=cfg.analysis.cumulative_threshold,
        )
        target_repeats.insert(1, "target", target)
        repeat_frames.append(target_repeats)
        repeat_totals = target_repeats.groupby("repeat")["normalized_hsic"].sum()
        if repeat_totals.eq(0).any():
            target_status.append(dict(target=target, status="repeat_without_positive_hsic"))
            continue
        summary_frames.append(_summary_from_repeats(target_repeats, target))
        available_targets.append(target)
        target_status.append(dict(target=target,status='analyzed'))
    pd.DataFrame(target_status).to_csv(paths.tables_dir/'analysis_target_status.csv',index=False)
    if "TOTAL" not in available_targets:
        raise ValueError("TOTAL sensitivity is unavailable or lacks positive HSIC in a repeat")
    subsamples = pd.concat(repeat_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True).merge(
        metadata, on="parameter_id", how="left", validate="many_to_one"
    )
    summary['physical_exclusion_reason']=summary.parameter_id.map(reasons)
    summary['physically_valid']=summary.physical_exclusion_reason.eq('')
    total = summary.loc[summary["target"] == "TOTAL"].copy()
    force_include = set(cfg.parameter_space.force_include)
    force_exclude = set(cfg.parameter_space.force_exclude)
    decisions = select_parameters(
        total,
        cfg.analysis.cumulative_threshold,
        force_include,
        force_exclude,
        cfg.analysis.selection_method,cfg.analysis.selection_top_k,
    )
    candidate_bounds = parse_params_ml(paths.inputs_dir / "params-ml.candidates")
    selected_ids = set(decisions.loc[decisions["selected"], "parameter_id"])
    selected_bounds=[bound for bound in candidate_bounds if bound.coordinate.parameter_id in selected_ids]
    if not selected_bounds:
        raise ValueError("No candidate parameters were selected for export")
    export_document = FFieldDocument.from_path(paths.inputs_dir / 'ffield')
    initial_values = {p.coordinate: p.initial_value for p in export_document.parameters}
    selected_output=paths.selected_params_ml if not cfg.analysis.diagnostic_only else paths.exports_dir/'params-ml.diagnostic_only'
    write_params_ml(
        selected_output,
        selected_bounds,style='indeedopt',initial_values=initial_values,
    )
    precise=[dict(parameter_id=b.coordinate.parameter_id,delta=b.delta,lower_bound=b.lower_bound,upper_bound=b.upper_bound)
             for b in selected_bounds]
    precise_name='params-ml.selected.full_precision.json' if not cfg.analysis.diagnostic_only else 'params-ml.diagnostic_only.full_precision.json'
    (paths.exports_dir/precise_name).write_text(json.dumps(precise,indent=2),encoding='utf-8')
    rounded={b.coordinate:b for b in optimizer_bounds(selected_bounds, initial_values)}
    pd.DataFrame([dict(**row,export_delta=rounded[b.coordinate].delta,
        export_lower_bound=rounded[b.coordinate].lower_bound,export_upper_bound=rounded[b.coordinate].upper_bound)
        for b,row in zip(selected_bounds,precise)]).to_csv(paths.exports_dir/'params-ml.rounding_audit.csv',index=False)
    (paths.tables_dir/'analysis_method.json').write_text(json.dumps(dict(
        loss_scaling='log/minmax on the full successful loss vector before subsampling',
        aggregation="within each repeat, clip signed HSIC at zero, normalize by positive score sum, then average over repeats",
        selection_frequency="fraction of repeats reaching the cumulative threshold with the parameter retained",
        uncertainty='Student-t interval across overlapping subsamples is descriptive repeat variability, not a confidence or significance guarantee',
        selection_method=cfg.analysis.selection_method,selection_is_heuristic=True,
        diagnostic_only=cfg.analysis.diagnostic_only,production_ready=False,analysis_state='building',
        cumulative_threshold=cfg.analysis.cumulative_threshold,
        selected_count=len(selected_bounds),selected_score_fraction=float(decisions.loc[decisions.selected,'hsic_mean'].sum()),
        physically_excluded=[pid for pid,reason in reasons.items() if reason]),indent=2),encoding='utf-8')
    group_rows = []
    for group_type, column in (("section", "section_name"), ("interaction", "parameter_group")):
        for label, group in total.groupby(column, dropna=False):
            group_rows.append(
                {
                    "group_type": group_type,
                    "group_label": str(label),
                    "sensitivity_sum": float(group["hsic_mean"].sum()),
                    "parameter_count": len(group),
                }
            )
    group_summary = pd.DataFrame(group_rows).sort_values(
        ["group_type", "sensitivity_sum"], ascending=[True, False]
    )
    convergence = _rank_convergence(
        matrix,
        merged["total_loss"].to_numpy(dtype=float),
        parameter_ids,
        cfg,
    )
    outputs = {
        "hsic_subsample_results": subsamples,
        "parameter_sensitivity_summary": summary,
        "parameter_group_summary": group_summary,
        "rank_convergence": convergence,
        "selection_decisions": decisions,
    }
    for key, frame in outputs.items():
        frame.to_csv(getattr(paths, key), index=False)
    method_path=paths.tables_dir/'analysis_method.json'
    method=json.loads(method_path.read_text(encoding='utf-8'))
    tracked=[paths.inputs_dir/name for name in ['ffield','geo','trainset.in','control','params-ml.candidates']]
    tracked.extend([paths.parameter_samples,paths.sample_losses,selected_output])
    method.update(production_ready=quality['production_ready'] and not cfg.analysis.diagnostic_only,
        analysis_state='completed',analysis_config=cfg.analysis.model_dump(mode='json'),
        artifacts_sha256={str(path.relative_to(paths.run_dir)): _digest(path) for path in tracked if path.is_file()})
    method_path.write_text(json.dumps(method,indent=2),encoding='utf-8')
    return outputs
