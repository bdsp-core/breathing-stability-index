#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from build_revision_mortality_summary_figure_v1 import (
    BSI_FEATURE_SET,
    BSI_MODEL_FOR_MERGE,
    BSI_PREDICTIONS,
    BSI_RESULTS,
    COMPARATOR_RESULTS,
    COHORTS,
    OUTDIR,
    SELECTION,
    SELECTION_MODE,
    EXPOSURE_ORDER,
    plt,
)
from build_revision_mortality_summary_figure_v2 import MODEL_COLORS, MODEL_ORDER


BSI_PREDICTIONS_EXTRA = [
    Path(__file__).resolve().parents[1]
    / "REVISION"
    / "mortality_bsi_score_cv_all_study_type_original_style_50fold_s0001_probe_v1"
    / "bsi_score_oof_predictions.csv"
]
BSI_RESULTS_EXTRA = [
    Path(__file__).resolve().parents[1]
    / "REVISION"
    / "mortality_bsi_score_cv_all_study_type_original_style_50fold_s0001_probe_v1"
    / "bsi_score_cv_results.csv"
]
COMPARATOR_RESULTS_EXTRA = [
    Path(__file__).resolve().parents[1]
    / "REVISION"
    / "mortality_comparator_all_study_type_s0001_probe_v1"
    / "comparator_all_study_type_results.csv"
]
CORR_COLUMNS = {
    "bsi_score_oof": "Breathing Stability",
    "ss_percent_main": "Periodic Breathing",
    "ahi": "AHI",
    "hypoxic_burden": "Hypoxic Burden",
    "arousal_index": "Arousal Index",
}
CORR_ORDER = list(CORR_COLUMNS.values())
EXPOSURE_DISPLAY = {
    "Breathing stability score": "Breathing Stability",
    "SS": "Periodic Breathing",
    "AHI": "AHI",
    "HB": "Hypoxic Burden",
    "ArI": "Arousal Index",
}
COHORT_LABELS = {"I0002": "BIDMC", "mros": "MrOS", "S0001": "MGH"}
COHORTS_ALL = {**COHORTS, "S0001": "MGH"}
COHORT_FOLLOWUP_ORDER = ["MrOS", "BIDMC", "MGH"]
COHORT_COLORS = {"BIDMC": "#356c80", "MGH": "#4f6f52", "MrOS": "#b85c38"}
COHORT_BAND_COLORS = {"BIDMC": "#d7e1e6", "MGH": "#dce6dc", "MrOS": "#ead7cf"}
MODEL_OFFSETS = {
    "M1": 0.24,
    "M2": 0.08,
    "M3": -0.08,
    "M4": -0.24,
}
MODEL_LEGEND_LABELS = {
    "M1": "M1: unadjusted",
    "M2": "M2: age + sex",
    "M3": "M3: age + sex + AHI",
    "M4": "M4: age + sex + all other metrics",
}


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if Path(path).exists()]
    if not frames:
        raise FileNotFoundError("None of the requested mortality result files exist.")
    return pd.concat(frames, ignore_index=True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_correlation_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(
        SELECTION,
        usecols=["selection_mode", "fileid", "sid", "cohort", *[c for c in CORR_COLUMNS if c != "bsi_score_oof"]],
        low_memory=False,
    )
    selected = selected[selected["selection_mode"].eq(SELECTION_MODE) & selected["cohort"].isin(COHORTS_ALL)].copy()
    score = read_concat([BSI_PREDICTIONS, *BSI_PREDICTIONS_EXTRA])
    score = score[
        score["feature_set"].eq(BSI_FEATURE_SET)
        & score["model_name"].eq(BSI_MODEL_FOR_MERGE)
        & score["cohort"].isin(COHORTS_ALL)
    ][["fileid", "sid", "cohort", "bsi_score_oof"]].copy()
    data = selected.merge(score, on=["fileid", "sid", "cohort"], how="inner")
    for col in CORR_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.rename(columns=CORR_COLUMNS)

    rows: list[dict[str, object]] = []
    for cohort, dat in data.groupby("cohort", sort=False):
        corr = dat[CORR_ORDER].corr(method="spearman")
        for row_name in CORR_ORDER:
            for col_name in CORR_ORDER:
                rows.append(
                    {
                        "cohort": cohort,
                        "cohort_label": COHORT_LABELS[cohort],
                        "row_variable": row_name,
                        "column_variable": col_name,
                        "spearman_rho": corr.loc[row_name, col_name],
                    }
                )
    return pd.DataFrame(rows), data[["cohort", *CORR_ORDER]].copy()


def draw_context_tight(ax: plt.Axes, followup: pd.DataFrame) -> None:
    y_positions = {label: 3.3 - 1.1 * i for i, label in enumerate(COHORT_FOLLOWUP_ORDER)}
    for cohort_label in COHORT_FOLLOWUP_ORDER:
        y = y_positions[cohort_label]
        row = followup[followup["cohort_label"].eq(cohort_label)].iloc[0]
        color = COHORT_COLORS[cohort_label]
        ax.plot(
            [row["followup_q1_years"], row["followup_q3_years"]],
            [y, y],
            color=COHORT_BAND_COLORS[cohort_label],
            lw=8,
            solid_capstyle="round",
        )
        ax.plot(row["followup_median_years"], y, "o", color=color, ms=7)

    ax.set_yticks([y_positions[label] for label in COHORT_FOLLOWUP_ORDER])
    ax.set_yticklabels(COHORT_FOLLOWUP_ORDER)
    ax.set_ylim(0.0, 4.5)
    ax.set_xlim(0, 21)
    ax.set_xlabel("Follow-up, years")
    ax.set_title("Analytic survival cohorts")
    ax.grid(axis="x", color="#e9e9e9", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.text(0.02, 0.06, "Median with IQR", transform=ax.transAxes, fontsize=8, color="#555555")
    panel_label(ax, "A")


def load_cox_results_all() -> pd.DataFrame:
    model_map = {
        "m1_unadjusted": "M1",
        "m2_age_sex": "M2",
        "m3_age_sex_ahi": "M3",
        "m4_age_sex_comparators": "M4",
        "m4_age_sex_bsi_other_comparators": "M4",
    }
    bsi = read_concat([BSI_RESULTS, *BSI_RESULTS_EXTRA])
    bsi = bsi[
        bsi["selection_mode"].eq(SELECTION_MODE)
        & bsi["feature_set"].eq(BSI_FEATURE_SET)
        & bsi["model_name"].isin(model_map)
    ].copy()
    bsi["exposure"] = "Breathing stability score"
    bsi["model"] = bsi["model_name"].map(model_map)
    bsi["source"] = "bsi_score"

    comparators = read_concat([COMPARATOR_RESULTS, *COMPARATOR_RESULTS_EXTRA])
    comparators = comparators[
        comparators["selection_mode"].eq(SELECTION_MODE) & comparators["model_name"].isin(model_map)
    ].copy()
    comparators["exposure"] = comparators["analysis_exposure_label"]
    comparators["model"] = comparators["model_name"].map(model_map)
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
    cox["cohort_label"] = cox["cohort_label"].replace({"MGH/S0001": "MGH"})
    cox["cohort_label"] = pd.Categorical(cox["cohort_label"], ["MrOS", "BIDMC", "MGH"], ordered=True)
    cox["exposure"] = pd.Categorical(cox["exposure"], EXPOSURE_ORDER, ordered=True)
    cox["model"] = pd.Categorical(cox["model"], MODEL_ORDER, ordered=True)
    return cox.sort_values(["cohort_label", "exposure", "model"]).copy()


def load_followup_summary_all() -> pd.DataFrame:
    selected = pd.read_csv(
        SELECTION,
        usecols=["selection_mode", "fileid", "sid", "cohort", "followup_days", "event"],
        low_memory=False,
    )
    selected = selected[selected["selection_mode"].eq(SELECTION_MODE) & selected["cohort"].isin(COHORTS_ALL)].copy()
    score = read_concat([BSI_PREDICTIONS, *BSI_PREDICTIONS_EXTRA])
    score = score[
        score["feature_set"].eq(BSI_FEATURE_SET)
        & score["model_name"].eq(BSI_MODEL_FOR_MERGE)
        & score["cohort"].isin(COHORTS_ALL)
    ][["fileid", "sid", "cohort", "bsi_score_oof"]].copy()
    data = selected.merge(score, on=["fileid", "sid", "cohort"], how="inner")
    data["cohort_label"] = data["cohort"].map(COHORTS_ALL)
    data["followup_years"] = pd.to_numeric(data["followup_days"], errors="coerce") / 365.25
    data["event"] = pd.to_numeric(data["event"], errors="coerce")

    rows: list[dict[str, object]] = []
    for cohort_label in COHORT_FOLLOWUP_ORDER:
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


def draw_split_correlation_matrix(ax: plt.Axes, corr_long: pd.DataFrame) -> None:
    n = len(CORR_ORDER)
    values = np.full((n, n), np.nan)
    annotations = np.full((n, n), "", dtype=object)

    for i, row_name in enumerate(CORR_ORDER):
        for j, col_name in enumerate(CORR_ORDER):
            if i == j:
                continue
            cohort_label = "MrOS" if i < j else "BIDMC"
            val = corr_long[
                corr_long["cohort_label"].eq(cohort_label)
                & corr_long["row_variable"].eq(row_name)
                & corr_long["column_variable"].eq(col_name)
            ]["spearman_rho"].iloc[0]
            values[i, j] = abs(val)
            annotations[i, j] = f"{val:.2f}"

    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("#f7f7f7")
    im = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=0.9, interpolation="nearest", aspect="auto")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "1.00", ha="center", va="center", fontsize=8, color="#777777")
                continue
            color = "white" if values[i, j] >= 0.55 else "#222222"
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(CORR_ORDER, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n))
    ax.set_yticklabels(CORR_ORDER)
    ax.set_title("Exposure correlations\nBIDMC lower; MrOS upper")
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    panel_label(ax, "B")


def draw_adjustment_forest(
    ax: plt.Axes,
    cox: pd.DataFrame,
    cohort: str,
    label: str,
    *,
    show_ylabel: bool = True,
    xlim: tuple[float, float] = (0.80, 1.35),
    xticks: tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20, 1.30),
) -> None:
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
                elinewidth=1.0,
                capsize=2.2,
                markersize=4.2,
                markerfacecolor=color if significant else "white",
                markeredgewidth=1.1,
                zorder=3,
            )

    ax.set_yticks([y_base[exp] for exp in EXPOSURE_ORDER])
    ax.set_yticklabels([EXPOSURE_DISPLAY[exp] for exp in EXPOSURE_ORDER])
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(mticker.FixedLocator(list(xticks)))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{tick:.2f}" for tick in xticks]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("HR per SD")
    ax.set_title(f"{cohort} mortality")
    ax.grid(axis="x", color="#e9e9e9", linewidth=0.8)
    ax.grid(axis="y", color="#f0f0f0", linewidth=0.8)

    for exposure in EXPOSURE_ORDER:
        ax.axhline(y_base[exposure] + 0.39, color="#f1f1f1", lw=0.8, zorder=0)
        ax.axhline(y_base[exposure] - 0.39, color="#f1f1f1", lw=0.8, zorder=0)

    panel_label(ax, label)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cox = load_cox_results_all()
    followup = load_followup_summary_all()
    corr_long, corr_source = load_correlation_source()

    cox.to_csv(OUTDIR / "figure7_mortality_all_study_type_summary_source.csv", index=False)
    followup.to_csv(OUTDIR / "figure7_mortality_all_study_type_followup_source.csv", index=False)
    corr_long.to_csv(OUTDIR / "figure7_mortality_all_study_type_correlation_source.csv", index=False)
    corr_source.to_csv(OUTDIR / "figure7_mortality_all_study_type_correlation_analysis_rows.csv", index=False)

    fig = plt.figure(figsize=(6.63, 8.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.86, 1.12, 1.12])
    ax_context = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_mros = fig.add_subplot(gs[1, 0])
    ax_bidmc = fig.add_subplot(gs[1, 1])
    ax_mgh = fig.add_subplot(gs[2, 0])
    ax_legend = fig.add_subplot(gs[2, 1])
    ax_legend.axis("off")

    draw_context_tight(ax_context, followup)
    draw_split_correlation_matrix(ax_corr, corr_long)
    draw_adjustment_forest(ax_mros, cox, "MrOS", "C", show_ylabel=True)
    draw_adjustment_forest(ax_bidmc, cox, "BIDMC", "D", show_ylabel=False)
    draw_adjustment_forest(
        ax_mgh,
        cox,
        "MGH",
        "E",
        show_ylabel=True,
        xlim=(0.75, 2.15),
        xticks=(0.80, 1.00, 1.20, 1.50, 2.00),
    )

    fig.suptitle("Mortality associations of breathing stability and comparator sleep metrics", fontsize=11, y=0.955)
    model_handles = [
        Line2D([0], [0], marker="o", color=MODEL_COLORS[model], lw=1.4, markersize=5, label=MODEL_LEGEND_LABELS[model])
        for model in MODEL_ORDER
    ]
    ax_legend.legend(
        handles=model_handles,
        loc="center left",
        bbox_to_anchor=(0.12, 0.56),
        frameon=False,
        ncol=1,
        title="Adjustment model",
        title_fontsize=8,
        fontsize=8,
        handlelength=1.6,
        labelspacing=0.8,
    )
    ax_legend.text(
        0.12,
        0.23,
        "For comparator rows, M4\n"
        "also includes\n"
        "the BSI mortality score.\n"
        "Filled markers denote p<0.05.",
        transform=ax_legend.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
        linespacing=1.25,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.955), h_pad=1.1, w_pad=1.4)

    stem = "figure7_mortality_all_study_type_summary_v3"
    fig.savefig(OUTDIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.eps", bbox_inches="tight")
    fig.savefig(
        OUTDIR / f"{stem}.jpg",
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95},
    )
    plt.close(fig)
    print(f"Wrote {OUTDIR / f'{stem}.png'}")
    print(f"Wrote {OUTDIR / f'{stem}.pdf'}")
    print(f"Wrote {OUTDIR / f'{stem}.eps'}")
    print(f"Wrote {OUTDIR / f'{stem}.jpg'}")


if __name__ == "__main__":
    main()
