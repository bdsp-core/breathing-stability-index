#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = REPO_ROOT / "REVISION" / "figures_v2"
SELECTION = REPO_ROOT / "REVISION" / "mortality_v2" / "selection_one_row_per_subject.csv"
BSI_PREDICTIONS = (
    REPO_ROOT
    / "REVISION"
    / "mortality_bsi_score_cv_all_study_type_original_style_50fold_v1"
    / "bsi_score_oof_predictions.csv"
)
BSI_RESULTS = (
    REPO_ROOT
    / "REVISION"
    / "mortality_bsi_score_cv_all_study_type_original_style_50fold_v1"
    / "bsi_score_cv_results.csv"
)
COMPARATOR_RESULTS = (
    REPO_ROOT
    / "REVISION"
    / "mortality_comparator_all_study_type_v1"
    / "comparator_all_study_type_results.csv"
)

SELECTION_MODE = "all_study_type_sensitivity"
BSI_FEATURE_SET = "original_style_quantile_windows"
BSI_MODEL_FOR_MERGE = "m4_age_sex_comparators"
COHORTS = {"I0002": "BIDMC", "mros": "MrOS"}
COHORT_ORDER = ["BIDMC", "MrOS"]
MODEL_MAP = {
    "m1_unadjusted": "M1",
    "m2_age_sex": "M2",
    "m3_age_sex_ahi": "M3",
    "m4_age_sex_comparators": "M4",
    "m4_age_sex_bsi_other_comparators": "M4",
}
MODEL_ORDER = ["M1", "M2", "M3", "M4"]
EXPOSURE_ORDER = ["Breathing stability score", "SS", "AHI", "HB", "ArI"]
EXPOSURE_SHORT = {
    "Breathing stability score": "BSI score",
    "SS": "SS",
    "AHI": "AHI",
    "HB": "HB",
    "ArI": "ArI",
}
EXPOSURE_COLORS = {
    "Breathing stability score": "#2f6f73",
    "SS": "#b85c38",
    "AHI": "#356c80",
    "HB": "#5d8a3f",
    "ArI": "#6f55a4",
}
COHORT_COLORS = {"BIDMC": "#356c80", "MrOS": "#b85c38"}


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7.5,
    }
)


def load_cox_results() -> pd.DataFrame:
    bsi = pd.read_csv(BSI_RESULTS)
    bsi = bsi[
        bsi["selection_mode"].eq(SELECTION_MODE)
        & bsi["feature_set"].eq(BSI_FEATURE_SET)
        & bsi["model_name"].isin(MODEL_MAP)
    ].copy()
    bsi["exposure"] = "Breathing stability score"
    bsi["model"] = bsi["model_name"].map(MODEL_MAP)
    bsi["source"] = "bsi_score"

    comparators = pd.read_csv(COMPARATOR_RESULTS)
    comparators = comparators[
        comparators["selection_mode"].eq(SELECTION_MODE) & comparators["model_name"].isin(MODEL_MAP)
    ].copy()
    comparators["exposure"] = comparators["analysis_exposure_label"]
    comparators["model"] = comparators["model_name"].map(MODEL_MAP)
    comparators["source"] = "comparator"

    common = [
        "cohort_label",
        "exposure",
        "model",
        "source",
        "n",
        "n_events",
        "status",
        "hr_per_sd",
        "ci_lower",
        "ci_upper",
        "p_value",
    ]
    cox = pd.concat([bsi[common], comparators[common]], ignore_index=True)
    cox["cohort_label"] = pd.Categorical(cox["cohort_label"], COHORT_ORDER, ordered=True)
    cox["exposure"] = pd.Categorical(cox["exposure"], EXPOSURE_ORDER, ordered=True)
    cox["model"] = pd.Categorical(cox["model"], MODEL_ORDER, ordered=True)
    return cox.sort_values(["cohort_label", "exposure", "model"]).copy()


def load_followup_summary() -> pd.DataFrame:
    selected = pd.read_csv(
        SELECTION,
        usecols=["selection_mode", "fileid", "sid", "cohort", "followup_days", "event"],
        low_memory=False,
    )
    selected = selected[selected["selection_mode"].eq(SELECTION_MODE) & selected["cohort"].isin(COHORTS)].copy()
    score = pd.read_csv(BSI_PREDICTIONS)
    score = score[
        score["feature_set"].eq(BSI_FEATURE_SET)
        & score["model_name"].eq(BSI_MODEL_FOR_MERGE)
        & score["cohort"].isin(COHORTS)
    ][["fileid", "sid", "cohort", "bsi_score_oof"]].copy()
    data = selected.merge(score, on=["fileid", "sid", "cohort"], how="inner")
    data["cohort_label"] = data["cohort"].map(COHORTS)
    data["followup_years"] = pd.to_numeric(data["followup_days"], errors="coerce") / 365.25
    data["event"] = pd.to_numeric(data["event"], errors="coerce")

    rows: list[dict[str, object]] = []
    for cohort_label in COHORT_ORDER:
        dat = data[data["cohort_label"].eq(cohort_label)].copy()
        q = dat["followup_years"].quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "cohort_label": cohort_label,
                "n": int(len(dat)),
                "n_events": int(dat["event"].sum()),
                "event_percent": float(dat["event"].mean() * 100.0),
                "followup_median_years": float(q.loc[0.5]),
                "followup_q1_years": float(q.loc[0.25]),
                "followup_q3_years": float(q.loc[0.75]),
            }
        )
    return pd.DataFrame(rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_context(ax: plt.Axes, followup: pd.DataFrame) -> None:
    context_lines: list[str] = []
    y_positions = np.arange(len(COHORT_ORDER))[::-1]
    for y, cohort in zip(y_positions, COHORT_ORDER):
        row = followup[followup["cohort_label"].eq(cohort)].iloc[0]
        context_lines.append(f"{cohort}: n={row['n']:,}; deaths={row['n_events']:,} ({row['event_percent']:.1f}%)")
        color = COHORT_COLORS[cohort]
        ax.plot(
            [row["followup_q1_years"], row["followup_q3_years"]],
            [y, y],
            color=color,
            lw=8,
            solid_capstyle="round",
            alpha=0.22,
        )
        ax.plot(row["followup_median_years"], y, "o", color=color, ms=7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(COHORT_ORDER)
    ax.set_xlim(0, 21)
    ax.set_xlabel("Follow-up, years")
    ax.set_title("Analytic survival cohorts")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.text(
        0.98,
        0.96,
        "\n".join(context_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#333333",
    )
    ax.text(0.02, -0.28, "Point: median; band: IQR", transform=ax.transAxes, fontsize=8, color="#555555")
    panel_label(ax, "A")


def draw_forest(ax: plt.Axes, cox: pd.DataFrame) -> None:
    sub = cox[cox["model"].eq("M4") & cox["status"].eq("ok")].copy()
    y_base = {exp: i for i, exp in enumerate(EXPOSURE_ORDER[::-1])}
    offsets = {"BIDMC": 0.13, "MrOS": -0.13}
    ax.axvline(1.0, color="#222222", lw=1.0, ls="--", zorder=0)
    for cohort in COHORT_ORDER:
        dat = sub[sub["cohort_label"].eq(cohort)].copy()
        for _, row in dat.iterrows():
            y = y_base[str(row["exposure"])] + offsets[cohort]
            color = COHORT_COLORS[cohort]
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
                elinewidth=1.4,
                capsize=3,
                markersize=6,
                markerfacecolor=color if significant else "white",
                markeredgewidth=1.5,
                zorder=3,
                label=cohort,
            )
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(
        dedup.values(),
        dedup.keys(),
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title="Cohort",
        title_fontsize=7.5,
        borderaxespad=0.0,
    )
    ax.set_yticks([y_base[exp] for exp in EXPOSURE_ORDER])
    ax.set_yticklabels([EXPOSURE_SHORT[exp] for exp in EXPOSURE_ORDER])
    ax.set_xscale("log")
    ax.set_xlim(0.80, 1.25)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.80, 0.90, 1.00, 1.10, 1.20]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["0.80", "0.90", "1.00", "1.10", "1.20"]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Fully adjusted HR per SD")
    ax.set_title("Mutually adjusted mortality associations")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.15)
    ax.text(
        0.02,
        -0.28,
        "Filled markers denote p<0.05; open markers denote p>=0.05.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    panel_label(ax, "B")


def draw_trajectory(ax: plt.Axes, cox: pd.DataFrame, cohort: str, label: str) -> None:
    sub = cox[cox["cohort_label"].eq(cohort) & cox["status"].eq("ok")].copy()
    x = np.arange(len(MODEL_ORDER))
    ax.axhline(1.0, color="#222222", lw=1.0, ls="--", zorder=0)
    for exposure in EXPOSURE_ORDER:
        dat = sub[sub["exposure"].eq(exposure)].sort_values("model").copy()
        if dat.empty:
            continue
        color = EXPOSURE_COLORS[exposure]
        y = dat["hr_per_sd"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - dat["ci_lower"].to_numpy(dtype=float),
                dat["ci_upper"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            lw=1.4,
            elinewidth=0.8,
            capsize=2,
            alpha=0.95,
            marker="o",
            markersize=5,
            label=EXPOSURE_SHORT[exposure],
        )
        nonsig = dat["p_value"].to_numpy(dtype=float) >= 0.05
        if np.any(nonsig):
            ax.scatter(
                x[nonsig],
                y[nonsig],
                s=28,
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
                zorder=5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_yscale("log")
    ax.set_ylim(0.82, 1.33)
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.85, 0.90, 1.00, 1.10, 1.20, 1.30]))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(["0.85", "0.90", "1.00", "1.10", "1.20", "1.30"]))
    ax.set_ylabel("HR per SD")
    ax.set_title(f"{cohort}: adjustment trajectory")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.15)
    if cohort == "BIDMC":
        ax.legend(frameon=False, loc="upper right", ncol=1)
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
    draw_trajectory(ax_bidmc, cox, "BIDMC", "C")
    draw_trajectory(ax_mros, cox, "MrOS", "D")
    ax_mros.set_ylabel("")

    fig.suptitle(
        "Mortality associations of breathing stability and comparator sleep metrics",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Model 1: unadjusted; Model 2: age + sex; Model 3: age + sex + AHI; Model 4: mutually adjusted. "
        "Comparator Model 4 includes the BSI mortality score.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.08, right=0.92, hspace=0.56, wspace=0.28)

    stem = "figure7_mortality_all_study_type_summary_v1"
    fig.savefig(OUTDIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTDIR / f'{stem}.png'}")
    print(f"Wrote {OUTDIR / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
