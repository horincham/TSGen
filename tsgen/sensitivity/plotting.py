from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsgen.sensitivity.paths import SensitivityPaths


NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#B8B8B8"


def _finish_axis(axis) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)


def plot_sensitivity_figure(paths: SensitivityPaths) -> dict[str, Path]:
    ranking = pd.read_csv(paths.panel_a_source)
    groups = pd.read_csv(paths.panel_b_source)
    targets = pd.read_csv(paths.panel_c_source)
    cumulative = pd.read_csv(paths.panel_d_source)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(NATURE_RC):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(180 / 25.4, 118 / 25.4),
            constrained_layout=True,
        )
        axis_a, axis_b, axis_c, axis_d = axes.ravel()

        ordered = ranking.sort_values("rank", ascending=False)
        colors = [BLUE if value else GRAY for value in ordered["selected"]]
        y_positions = np.arange(len(ordered))
        axis_a.barh(
            y_positions,
            ordered["hsic_mean"].to_numpy(dtype=float),
            color=colors,
            edgecolor="black",
            linewidth=0.4,
        )
        for y_position, low, high in zip(
            y_positions,
            ordered["hsic_ci_low"].to_numpy(dtype=float),
            ordered["hsic_ci_high"].to_numpy(dtype=float),
            strict=True,
        ):
            axis_a.hlines(y_position, low, high, color="black", linewidth=0.6)
            axis_a.vlines([low, high], y_position - 0.08, y_position + 0.08, color="black", linewidth=0.6)
        axis_a.set_yticks(y_positions, ordered["parameter_label"])
        axis_a.set_xlabel("Normalized HSIC")
        axis_a.set_ylabel("ReaxFF parameter")

        group_order = groups.sort_values("sensitivity_sum", ascending=False)
        bars = axis_b.bar(
            np.arange(len(group_order)),
            group_order["sensitivity_sum"],
            color=ORANGE,
            edgecolor="black",
            linewidth=0.4,
        )
        axis_b.set_xticks(
            np.arange(len(group_order)),
            group_order["group_label"],
            rotation=45,
            ha="right",
        )
        axis_b.set_ylabel("Summed normalized HSIC")
        for bar, count in zip(bars, group_order["parameter_count"], strict=True):
            axis_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=6,
            )

        target_order = [target for target in ("TOTAL", "ENERGY", "GEOMETRY", "CHARGE") if target in targets["target"].unique()]
        parameter_order = list(
            ranking.sort_values("rank")["parameter_id"]
        )
        heat = (
            targets.loc[targets["parameter_id"].isin(parameter_order)]
            .pivot(index="parameter_id", columns="target", values="normalized_hsic")
            .reindex(index=parameter_order, columns=target_order)
        )
        image = axis_c.imshow(
            np.ma.masked_invalid(heat.to_numpy(dtype=float)),
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=max(1e-12, float(np.nanmax(heat.to_numpy(dtype=float)))),
        )
        label_map = dict(zip(ranking["parameter_id"], ranking["parameter_label"], strict=True))
        axis_c.set_xticks(np.arange(len(target_order)), target_order, rotation=30, ha="right")
        axis_c.set_yticks(
            np.arange(len(parameter_order)),
            [label_map[parameter] for parameter in parameter_order],
        )
        colorbar = figure.colorbar(image, ax=axis_c, fraction=0.046, pad=0.03)
        colorbar.set_label("Normalized HSIC")

        axis_d.plot(
            cumulative["rank"],
            cumulative["cumulative_fraction"],
            marker="o",
            markersize=2.8,
            linewidth=1.0,
            color=GREEN,
        )
        threshold = float(cumulative["threshold"].iloc[0])
        axis_d.axhline(threshold, color="black", linestyle="--", linewidth=0.7)
        selected = cumulative.loc[cumulative["selected"]]
        if not selected.empty:
            crossing_rank = int(selected["rank"].max())
            axis_d.axvline(crossing_rank, color=ORANGE, linestyle=":", linewidth=0.8)
        axis_d.set_xlabel("Parameter rank")
        axis_d.set_ylabel("Cumulative normalized HSIC")
        axis_d.set_ylim(0, 1.04)

        for label, axis in zip("abcd", axes.ravel(), strict=True):
            _finish_axis(axis)
            axis.text(
                -0.18,
                1.08,
                label,
                transform=axis.transAxes,
                fontsize=9,
                fontweight="bold",
                va="top",
            )
        base = paths.figures_dir / "reaxff_parameter_sensitivity"
        outputs = {
            "pdf": base.with_suffix(".pdf"),
            "svg": base.with_suffix(".svg"),
            "png": base.with_suffix(".png"),
        }
        figure.savefig(outputs["pdf"], bbox_inches="tight")
        figure.savefig(outputs["svg"], bbox_inches="tight")
        figure.savefig(outputs["png"], dpi=600, bbox_inches="tight")
        plt.close(figure)
    return outputs
