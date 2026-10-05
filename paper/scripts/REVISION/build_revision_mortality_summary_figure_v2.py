#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from build_revision_mortality_summary_figure_v1 import (
    COHORT_ORDER,
    EXPOSURE_ORDER,
    EXPOSURE_SHORT,
    MODEL_ORDER,
    OUTDIR,
    draw_context,
    draw_forest,
    load_cox_results,
    load_followup_summary,
    panel_label,
    plt,
)


MODEL_COLORS = {
    "M1": "#7a7a7a",
    "M2": "#2f6f73",
    "M3": "#b85c38",
    "M4": "#6f55a4",
}
MODEL_OFFSETS = {
    "M1": 0.24,
    "M2": 0.08,
    "M3": -0.08,
    "M4": -0.24,
}


def draw_adjustment_forest(ax: plt.Axes, cox, cohort: str, label: str) -> None:
    sub = cox[cox["cohort_label"].eq(cohort) & cox["status"].eq("ok")].copy()
    y_base = {exp: i for i, exp in enumerate(EXPOSURE_ORDER[::-1])}
    ax.axvline(1.0, color="#222222", lw=1.0, ls="--", zorder=0)

    for model in MODEL_ORDER:
        dat = sub[sub["model"].eq(model)].copy()
        for _, row in dat.iterrows():
            exposure = str(row["exposure"])
            y = y_base[exposure] + MODEL_OFFSETS[model]
            color = MODEL_COLORS[model]
            significant = bool(row["p_value"] < 0.05)
            ax.errorbar(
                row["hr_per_sd"],
                y,
                xerr=[
                    [row["hr_per_sd"] - row["ci_lower"]],
                    [row["ci_upper"] - row["hr_per_sd"]],
                ],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.5,
                markersize=4.8,
                markerfacecolor=color if significant else "white",
                markeredgewidth=1.2,
                zorder=3,
                label=model,
            )

    ax.set_yticks([y_base[exp] for exp in EXPOSURE_ORDER])
    ax.set_yticklabels([EXPOSURE_SHORT[exp] for exp in EXPOSURE_ORDER])
    ax.set_xscale("log")
    ax.set_xlim(0.80, 1.35)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.80, 0.90, 1.00, 1.10, 1.20, 1.30]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["0.80", "0.90", "1.00", "1.10", "1.20", "1.30"]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("HR per SD")
    ax.set_title(f"{cohort}: adjustment-specific estimates")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.15)

    for exposure in EXPOSURE_ORDER:
        ax.axhline(y_base[exposure] + 0.39, color="#f1f1f1", lw=0.8, zorder=0)
        ax.axhline(y_base[exposure] - 0.39, color="#f1f1f1", lw=0.8, zorder=0)

    panel_label(ax, label)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cox = load_cox_results()
    followup = load_followup_summary()

    cox.to_csv(OUTDIR / "figure7_mortality_all_study_type_summary_source.csv", index=False)
    followup.to_csv(OUTDIR / "figure7_mortality_all_study_type_followup_source.csv", index=False)

    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.88, 1.22])
    ax_context = fig.add_subplot(gs[0, 0])
    ax_forest = fig.add_subplot(gs[0, 1])
    ax_bidmc = fig.add_subplot(gs[1, 0])
    ax_mros = fig.add_subplot(gs[1, 1], sharey=ax_bidmc)

    draw_context(ax_context, followup)
    draw_forest(ax_forest, cox)
    draw_adjustment_forest(ax_bidmc, cox, "BIDMC", "C")
    draw_adjustment_forest(ax_mros, cox, "MrOS", "D")

    fig.suptitle(
        "Mortality associations of breathing stability and comparator sleep metrics",
        fontsize=13,
        y=0.97,
    )
    fig.text(
        0.5,
        0.02,
        "Model 1: unadjusted; Model 2: age + sex; Model 3: age + sex + AHI; Model 4: mutually adjusted. "
        "Comparator Model 4 includes the BSI mortality score. Filled markers denote p<0.05.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    model_handles = [
        Line2D([0], [0], marker="o", color=MODEL_COLORS[model], lw=1.4, markersize=5, label=model)
        for model in MODEL_ORDER
    ]
    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        frameon=False,
        ncol=4,
        title="Adjustment model",
        title_fontsize=8,
        fontsize=8,
    )
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.08, right=0.92, hspace=0.56, wspace=0.28)

    stem = "figure7_mortality_all_study_type_summary_v2"
    fig.savefig(OUTDIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTDIR / f'{stem}.png'}")
    print(f"Wrote {OUTDIR / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
