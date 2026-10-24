from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tsgen.sensitivity.config import SensitivityConfig
from tsgen.sensitivity.hsic import repeated_hsic, student_t_interval
from tsgen.sensitivity.models import ParameterCoordinate
from tsgen.sensitivity.params_ml import parse_params_ml, write_params_ml
from tsgen.sensitivity.paths import SensitivityPaths


TARGET_COLUMNS = {
    "TOTAL": "total_loss",
    "ENERGY": "energy_loss",
    "GEOMETRY": "geometry_loss",
    "CHARGE": "charge_loss",
}


def select_parameters(
    summary: pd.DataFrame,
    threshold: float,
    force_include: set[str],
    force_exclude: set[str],
) -> pd.DataFrame:
    ranked = summary.sort_values(["hsic_mean", "parameter_id"], ascending=[False, True]).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["cumulative_fraction"] = ranked["hsic_mean"].cumsum()
    crossing = int(np.searchsorted(ranked["cumulative_fraction"].to_numpy(), threshold, side="left"))
    automatic = set(ranked.iloc[: min(crossing + 1, len(ranked))]["parameter_id"])
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
            selected, reason = True, "cumulative_threshold"
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
        means = repeats.groupby("parameter_id")["normalized_hsic"].mean()
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


def analyze_run(paths: SensitivityPaths, cfg: SensitivityConfig) -> dict[str, pd.DataFrame]:
    samples = pd.read_csv(paths.parameter_samples)
    losses = pd.read_csv(paths.sample_losses)
    successful = losses.loc[losses["run_status"] == "success"].copy()
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
    catalog = pd.read_csv(paths.parameter_catalog)
    metadata = _parameter_metadata(catalog)
    repeat_frames = []
    summary_frames = []
    available_targets = []
    for target in cfg.analysis.targets:
        column = TARGET_COLUMNS[target]
        available = column in merged and merged[column].notna().all()
        if not available:
            continue
        target_repeats = repeated_hsic(
            matrix,
            merged[column].to_numpy(dtype=float),
            parameter_ids,
            cfg.analysis.repeats,
            cfg.analysis.subsample_size,
            cfg.run.seed + sum(ord(char) for char in target),
            cfg.analysis.parameter_sigma,
            cfg.analysis.loss_gamma,
        )
        target_repeats.insert(1, "target", target)
        repeat_frames.append(target_repeats)
        summary_frames.append(_summary_from_repeats(target_repeats, target))
        available_targets.append(target)
    if "TOTAL" not in available_targets:
        raise ValueError("TOTAL loss is unavailable; cannot select sensitive parameters")
    bootstrap = pd.concat(repeat_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True).merge(
        metadata, on="parameter_id", how="left", validate="many_to_one"
    )
    summary["physically_valid"] = True
    total = summary.loc[summary["target"] == "TOTAL"].copy()
    force_include = set(cfg.parameter_space.force_include)
    force_exclude = set(cfg.parameter_space.force_exclude)
    decisions = select_parameters(
        total,
        cfg.analysis.cumulative_threshold,
        force_include,
        force_exclude,
    )
    candidate_bounds = parse_params_ml(paths.inputs_dir / "params-ml.candidates")
    selected_ids = set(decisions.loc[decisions["selected"], "parameter_id"])
    write_params_ml(
        paths.selected_params_ml,
        [bound for bound in candidate_bounds if bound.coordinate.parameter_id in selected_ids],
    )
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
    ranking = decisions[
        [
            "parameter_id",
            "parameter_label",
            "parameter_group",
            "hsic_mean",
            "hsic_ci_low",
            "hsic_ci_high",
            "rank",
            "selected",
        ]
    ].head(cfg.analysis.rank_top_k)
    panel_c_rows = []
    for parameter_id in decisions["parameter_id"]:
        parameter_label = metadata.loc[
            metadata["parameter_id"] == parameter_id, "parameter_label"
        ].iloc[0]
        for target in cfg.analysis.targets:
            match = summary.loc[
                (summary["parameter_id"] == parameter_id) & (summary["target"] == target)
            ]
            panel_c_rows.append(
                {
                    "parameter_id": parameter_id,
                    "parameter_label": parameter_label,
                    "target": target,
                    "normalized_hsic": None if match.empty else float(match["hsic_mean"].iloc[0]),
                    "rank_within_target": None
                    if match.empty
                    else int(match["rank_within_target"].iloc[0]),
                    "available": not match.empty,
                }
            )
    panel_c = pd.DataFrame(panel_c_rows)
    panel_d = decisions[
        ["rank", "parameter_id", "hsic_mean", "cumulative_fraction", "selected"]
    ].rename(columns={"hsic_mean": "normalized_hsic"})
    panel_d["threshold"] = cfg.analysis.cumulative_threshold
    convergence = _rank_convergence(
        matrix,
        merged["total_loss"].to_numpy(dtype=float),
        parameter_ids,
        cfg,
    )
    outputs = {
        "hsic_bootstrap_results": bootstrap,
        "parameter_sensitivity_summary": summary,
        "parameter_group_summary": group_summary,
        "rank_convergence": convergence,
        "selection_decisions": decisions,
    }
    for key, frame in outputs.items():
        frame.to_csv(getattr(paths, key), index=False)
    ranking.to_csv(paths.panel_a_source, index=False)
    group_summary.loc[group_summary["group_type"] == "interaction"].to_csv(
        paths.panel_b_source, index=False
    )
    panel_c.to_csv(paths.panel_c_source, index=False)
    panel_d.to_csv(paths.panel_d_source, index=False)
    return outputs
