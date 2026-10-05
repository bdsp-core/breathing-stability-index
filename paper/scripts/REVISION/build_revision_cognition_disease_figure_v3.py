#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_revision_cognition_disease_target_bsi_v1 import (
    COGNITION_TARGETS,
    COMPARATORS,
    MASTER,
    cognition_model,
    oof_ridge_score,
    prediction_matrix,
    safe_numeric,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "REVISION" / "cognition_disease_target_bsi_v1"
OUTDIR = REPO_ROOT / "REVISION" / "figures_v2"
STEM = "figure6_cognition_disease_integrated_v3"

COGNITION_ORDER = ["Fluid cognition", "Crystallized cognition", "Total cognition"]
DISEASE_ORDER = [
    "Dementia",
    "MCI",
    "Symptomatic",
    "Atrial fibrillation",
    "Myocardial infarction",
    "Diabetes II",
    "Hypertension",
    "Bipolar disorder",
    "Depression",
]
COHORT_ORDER = ["mros", "mgh-cog"]
COHORT_LABELS = {"mros": "MrOS", "mgh-cog": "MGH-Cog"}

EXPOSURE_LABELS_PHYS = {
    "bsi_score": "Breathing stability",
    "ss_percent_main": "Periodic breathing",
    "ahi": "Apnea-hypopnea index",
    "hypoxic_burden": "Hypoxic burden",
    "arousal_index": "Arousal index",
}
EXPOSURE_ORDER = list(EXPOSURE_LABELS_PHYS.values())
EXPOSURE_COLORS = {
    "Breathing stability": "#2f6c9f",
    "Periodic breathing": "#8b5a3c",
    "Apnea-hypopnea index": "#4f6f52",
    "Hypoxic burden": "#b07c2a",
    "Arousal index": "#6f55a4",
}
COGNITION_DISPLAY_CONVENTION = (
    "Cognition panels are oriented so positive beta indicates worse cognition "
    "per 1 SD less favorable physiology."
)
COGNITION_EXPOSURE_SIGN = {
    "bsi_score": -1.0,
    "ss_percent_main": 1.0,
    "ahi": 1.0,
    "hypoxic_burden": 1.0,
    "arousal_index": 1.0,
}
PERFORMANCE_META = {
    "cognition": [
        ("M0_demographics", "Demographics", "#9e9e9e", "o", "-"),
        ("BSI_features", "Breathing stability", "#222222", "o", "-"),
        ("M0_plus_BSI_features", "Demographics + breathing stability", "#2f6c9f", "D", "--"),
    ],
    "disease": [
        ("M0_demographics", "Demographics", "#9e9e9e", "o", "-"),
        ("BSI_features", "Breathing stability", "#222222", "o", "-"),
    ],
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
        "axes.grid": False,
    }
)


def add_panel_label(ax: plt.Axes, label: str) -> None:
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


def draw_interval(
    ax: plt.Axes,
    x: float,
    low: float,
    high: float,
    y: float,
    *,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    ax.hlines(y, low, high, color=color, linestyle=linestyle, linewidth=1.8)
    ax.vlines([low, high], y - 0.045, y + 0.045, color=color, linewidth=1.2)
    ax.plot(
        x,
        y,
        marker=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=4.2,
        linestyle="",
    )


def model_offsets(model_sets: list[str]) -> dict[str, float]:
    if len(model_sets) == 2:
        return {model_sets[0]: -0.12, model_sets[1]: 0.12}
    if len(model_sets) == 3:
        return {model_sets[0]: -0.18, model_sets[1]: 0.0, model_sets[2]: 0.18}
    values = np.linspace(-0.22, 0.22, len(model_sets))
    return dict(zip(model_sets, values))


def plot_performance(ax: plt.Axes, perf: pd.DataFrame, endpoint_type: str, x_label: str, title: str) -> pd.DataFrame:
    meta = PERFORMANCE_META[endpoint_type]
    displayed_sets = [item[0] for item in meta]
    sub = perf[perf["endpoint_type"].eq(endpoint_type) & perf["model_set"].isin(displayed_sets)].copy()
    targets = COGNITION_ORDER if endpoint_type == "cognition" else DISEASE_ORDER
    sub["target_label"] = pd.Categorical(sub["target_label"], categories=targets, ordered=True)
    sub = sub.dropna(subset=["target_label"]).copy()
    base_y = {label: idx for idx, label in enumerate(targets[::-1])}
    offsets = model_offsets(displayed_sets)

    ax.axvline(0.0 if endpoint_type == "cognition" else 0.5, color="#666666", linestyle="--", linewidth=1)
    for model_set, _, color, marker, linestyle in meta:
        mod = sub[sub["model_set"].eq(model_set)]
        for _, row in mod.iterrows():
            y = base_y[str(row["target_label"])] + offsets[model_set]
            draw_interval(
                ax,
                row["estimate"],
                row["ci_lower"],
                row["ci_upper"],
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
            )

    ax.set_yticks(list(base_y.values()))
    ax.set_yticklabels(list(base_y.keys()))
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if endpoint_type == "cognition":
        ax.set_xlim(-0.08, max(0.40, float(sub["ci_upper"].max()) + 0.03))
    else:
        ax.set_xlim(0.44, max(0.72, float(sub["ci_upper"].max()) + 0.03))
    return sub


def load_bsi_feature_columns() -> list[str]:
    path = ANALYSIS_DIR / "bsi_feature_columns.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def add_delta_vs_m0_by_cohort(performance: pd.DataFrame) -> pd.DataFrame:
    m0 = (
        performance[performance["model_set"].eq("M0_demographics")]
        [["endpoint_type", "cohort", "target", "estimate"]]
        .rename(columns={"estimate": "m0_estimate"})
    )
    out = performance.merge(m0, on=["endpoint_type", "cohort", "target"], how="left")
    out["delta_vs_m0"] = out["estimate"] - out["m0_estimate"]
    return out.drop(columns=["m0_estimate"])


def load_cognition_prediction_by_cohort_results() -> pd.DataFrame:
    cached = OUTDIR / f"{STEM}_performance_source.csv"
    if cached.exists():
        cached_perf = pd.read_csv(cached)
        if {"endpoint_type", "cohort", "model_set"}.issubset(cached_perf.columns):
            cached_cog = cached_perf[
                cached_perf["endpoint_type"].eq("cognition")
                & cached_perf["cohort"].isin(COHORT_ORDER)
                & cached_perf["model_set"].isin([item[0] for item in PERFORMANCE_META["cognition"]])
            ].copy()
            expected_rows = len(COHORT_ORDER) * len(COGNITION_TARGETS) * len(PERFORMANCE_META["cognition"])
            if len(cached_cog) == expected_rows:
                return cached_cog

    bsi_cols = load_bsi_feature_columns()
    outcomes = [target for target, _ in COGNITION_TARGETS]
    master = pd.read_csv(
        MASTER,
        usecols=["fileid", "sid", "cohort", "age", "sex", *outcomes, *bsi_cols],
        low_memory=False,
    )
    for col in ["age", "sex", *outcomes, *bsi_cols]:
        master[col] = safe_numeric(master[col])
    master["analysis_subject_id"] = np.where(
        master["cohort"].astype(str).eq("mgh-cog"),
        master["fileid"].astype(str),
        master["cohort"].astype(str) + ":" + master["sid"].astype(str),
    )

    rows = []
    for outcome, outcome_label in COGNITION_TARGETS:
        dat = master.dropna(subset=[outcome]).copy()
        dat = dat[dat[bsi_cols].notna().any(axis=1)].copy()
        for cohort in COHORT_ORDER:
            cohort_dat = dat[dat["cohort"].eq(cohort)].copy()
            if len(cohort_dat) < 50:
                continue
            bsi_score = oof_ridge_score(
                cohort_dat[bsi_cols],
                cohort_dat[outcome],
                cohort_dat["analysis_subject_id"],
                seed=101,
            )
            m0_x = prediction_matrix(cohort_dat, ["age", "sex"])
            m0_score = oof_ridge_score(
                m0_x,
                cohort_dat[outcome],
                cohort_dat["analysis_subject_id"],
                seed=151,
            )
            demo_bsi_x = pd.concat([m0_x, cohort_dat[bsi_cols]], axis=1)
            demo_bsi_score = oof_ridge_score(
                demo_bsi_x,
                cohort_dat[outcome],
                cohort_dat["analysis_subject_id"],
                seed=171,
            )
            rows.extend(
                [
                    {
                        "endpoint_type": "cognition",
                        "cohort": cohort,
                        "cohort_label": COHORT_LABELS[cohort],
                        "target": outcome,
                        "target_label": outcome_label,
                        "model_set": "M0_demographics",
                        "n": int(len(cohort_dat)),
                        "n_cases": np.nan,
                        "n_controls": np.nan,
                        "metric": "pearson_r",
                        "estimate": m0_score.metric,
                        "ci_lower": m0_score.ci_lower,
                        "ci_upper": m0_score.ci_upper,
                        "note": "OOF age + sex benchmark within cohort.",
                    },
                    {
                        "endpoint_type": "cognition",
                        "cohort": cohort,
                        "cohort_label": COHORT_LABELS[cohort],
                        "target": outcome,
                        "target_label": outcome_label,
                        "model_set": "BSI_features",
                        "n": int(len(cohort_dat)),
                        "n_cases": np.nan,
                        "n_controls": np.nan,
                        "metric": "pearson_r",
                        "estimate": bsi_score.metric,
                        "ci_lower": bsi_score.ci_lower,
                        "ci_upper": bsi_score.ci_upper,
                        "note": "OOF target-specific BSI feature score within cohort.",
                    },
                    {
                        "endpoint_type": "cognition",
                        "cohort": cohort,
                        "cohort_label": COHORT_LABELS[cohort],
                        "target": outcome,
                        "target_label": outcome_label,
                        "model_set": "M0_plus_BSI_features",
                        "n": int(len(cohort_dat)),
                        "n_cases": np.nan,
                        "n_controls": np.nan,
                        "metric": "pearson_r",
                        "estimate": demo_bsi_score.metric,
                        "ci_lower": demo_bsi_score.ci_lower,
                        "ci_upper": demo_bsi_score.ci_upper,
                        "note": "OOF age + sex plus target-specific BSI features within cohort.",
                    },
                ]
            )
    return add_delta_vs_m0_by_cohort(pd.DataFrame(rows))


def plot_cognition_performance_by_cohort(
    ax: plt.Axes,
    perf: pd.DataFrame,
    cohort: str,
    panel_label: str,
    *,
    show_ylabel: bool,
) -> pd.DataFrame:
    sub = perf[perf["cohort"].eq(cohort)].copy()
    source = plot_performance(ax, sub, "cognition", "Pearson r", f"{COHORT_LABELS[cohort]} cognition\nprediction")
    if not show_ylabel:
        ax.set_yticklabels([])
    add_panel_label(ax, panel_label)
    return source


def load_cognition_by_cohort_results() -> pd.DataFrame:
    scores = pd.read_csv(ANALYSIS_DIR / "cognition_target_bsi_oof_scores.csv")
    comparator_cols = list(COMPARATORS.keys())
    master = pd.read_csv(
        MASTER,
        usecols=["fileid", "sid", "cohort", "age", "sex", *comparator_cols],
        low_memory=False,
    )
    for frame in (scores, master):
        frame["fileid_key"] = frame["fileid"].astype(str)
        frame["sid_key"] = frame["sid"].astype(str)
        frame["cohort_key"] = frame["cohort"].astype(str)
    master_covars = master.drop(columns=["fileid", "sid", "cohort"])
    dat = scores.merge(master_covars, on=["fileid_key", "sid_key", "cohort_key"], how="left", validate="many_to_one")
    for col in ["age", "sex", *comparator_cols, "outcome_value", "bsi_score"]:
        dat[col] = safe_numeric(dat[col])

    rows = []
    for outcome, outcome_label in COGNITION_TARGETS:
        outcome_dat = dat[dat["outcome"].eq(outcome)].copy()
        outcome_dat["worse_cognition"] = -outcome_dat["outcome_value"]
        for cohort in COHORT_ORDER:
            cohort_dat = outcome_dat[outcome_dat["cohort"].eq(cohort)].copy()
            for exposure in ["bsi_score", *comparator_cols]:
                display_exposure = f"{exposure}_worse_physiology"
                exposure_sign = COGNITION_EXPOSURE_SIGN[exposure]
                cohort_dat[display_exposure] = exposure_sign * cohort_dat[exposure]
                row = cognition_model(cohort_dat, "worse_cognition", display_exposure, ["age", "sex"], "M2")
                if row is None:
                    continue
                row["outcome"] = outcome
                row["exposure"] = exposure
                row["outcome_label"] = outcome_label
                row["cohort"] = cohort
                row["cohort_label"] = COHORT_LABELS[cohort]
                row["exposure_label"] = EXPOSURE_LABELS_PHYS[exposure]
                row["display_outcome"] = "worse_cognition"
                row["display_exposure"] = display_exposure
                row["outcome_sign"] = -1.0
                row["exposure_sign"] = exposure_sign
                row["display_convention"] = COGNITION_DISPLAY_CONVENTION
                rows.append(row)
    return pd.DataFrame(rows)


def panel_cognition_forest(
    ax: plt.Axes,
    cog: pd.DataFrame,
    cohort: str,
    panel_label: str,
    xlim: tuple[float, float],
    *,
    show_ylabel: bool,
) -> None:
    sub = cog[cog["cohort"].eq(cohort)].copy()
    sub["outcome_label"] = pd.Categorical(sub["outcome_label"], categories=COGNITION_ORDER, ordered=True)
    sub["exposure_label"] = pd.Categorical(sub["exposure_label"], categories=EXPOSURE_ORDER, ordered=True)
    sub = sub.sort_values(["outcome_label", "exposure_label"])
    base_y = {label: idx for idx, label in enumerate(COGNITION_ORDER[::-1])}
    offsets = {
        "Breathing stability": 0.28,
        "Periodic breathing": 0.14,
        "Apnea-hypopnea index": 0.0,
        "Hypoxic burden": -0.14,
        "Arousal index": -0.28,
    }
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    for _, row in sub.iterrows():
        y = base_y[str(row["outcome_label"])] + offsets[str(row["exposure_label"])]
        color = EXPOSURE_COLORS[str(row["exposure_label"])]
        markerface = color if row["p_value"] < 0.05 else "white"
        ax.errorbar(
            row["beta_std"],
            y,
            xerr=[[row["beta_std"] - row["ci_lower"]], [row["ci_upper"] - row["beta_std"]]],
            fmt="o",
            color=color,
            ecolor=color,
            markerfacecolor=markerface,
            markeredgecolor=color,
            markersize=4,
            capsize=2.5,
        )
    ax.set_yticks(list(base_y.values()))
    ax.set_yticklabels(list(base_y.keys()) if show_ylabel else [])
    ax.set_xlim(*xlim)
    ax.set_xlabel("Beta for worse cognition\nper 1 SD worse physiology")
    ax.set_title(f"{COHORT_LABELS[cohort]} cognition\nage/sex-adjusted linear model")
    add_panel_label(ax, panel_label)


def panel_disease_forest(ax: plt.Axes, disease: pd.DataFrame, *, show_ylabel: bool = True) -> pd.DataFrame:
    sub = disease[(disease["model"].eq("M2")) & (disease["exposure"].eq("bsi_score"))].copy()
    sub["disease"] = pd.Categorical(sub["disease"], categories=DISEASE_ORDER, ordered=True)
    sub = sub.sort_values("disease")
    y_positions = np.arange(len(DISEASE_ORDER))[::-1]
    y_map = {label: y for label, y in zip(DISEASE_ORDER, y_positions)}
    ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1)
    for _, row in sub.iterrows():
        y = y_map[str(row["disease"])]
        markerface = "#2f6c9f" if row["p_value"] < 0.05 else "white"
        ax.errorbar(
            row["odds_ratio"],
            y,
            xerr=[[row["odds_ratio"] - row["ci_lower"]], [row["ci_upper"] - row["odds_ratio"]]],
            fmt="o",
            color="#2f6c9f",
            ecolor="#2f6c9f",
            markerfacecolor=markerface,
            markeredgecolor="#2f6c9f",
            markersize=4,
            capsize=2.5,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(DISEASE_ORDER if show_ylabel else [])
    ax.set_xlim(0.75, max(sub["ci_upper"].max() + 0.14, 1.92))
    ax.set_xlabel("Odds ratio per 1 SD worse\nbreathing-stability score")
    ax.set_title("Disease association\nage/sex-adjusted logistic model")
    add_panel_label(ax, "F")
    return sub


def save_figure(fig: plt.Figure) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{STEM}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{STEM}.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{STEM}.eps", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{STEM}.jpg", bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(ANALYSIS_DIR / "performance_benchmark_results.csv")
    disease = pd.read_csv(ANALYSIS_DIR / "disease_model_ladder_results.csv")
    cognition_perf_by_cohort = load_cognition_prediction_by_cohort_results()
    cog_by_cohort = load_cognition_by_cohort_results()

    cognition_lim = max(abs(cog_by_cohort["ci_lower"]).max(), abs(cog_by_cohort["ci_upper"]).max(), 0.18) + 0.02
    cognition_xlim = (-cognition_lim, cognition_lim)

    fig = plt.figure(figsize=(7.0, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.95, 1.18], width_ratios=[1.0, 1.0])
    ax_perf_mros = fig.add_subplot(gs[0, 0])
    ax_perf_mgh = fig.add_subplot(gs[0, 1])
    ax_cog_mros = fig.add_subplot(gs[1, 0])
    ax_cog_mgh = fig.add_subplot(gs[1, 1])
    ax_perf_dis = fig.add_subplot(gs[2, 0])
    ax_dis = fig.add_subplot(gs[2, 1])

    perf_mros_source = plot_cognition_performance_by_cohort(
        ax_perf_mros,
        cognition_perf_by_cohort,
        "mros",
        "A",
        show_ylabel=True,
    )
    perf_mgh_source = plot_cognition_performance_by_cohort(
        ax_perf_mgh,
        cognition_perf_by_cohort,
        "mgh-cog",
        "B",
        show_ylabel=False,
    )
    perf_dis_source = plot_performance(ax_perf_dis, perf, "disease", "AUROC", "Disease prediction")
    add_panel_label(ax_perf_dis, "E")
    panel_cognition_forest(ax_cog_mros, cog_by_cohort, "mros", "C", cognition_xlim, show_ylabel=True)
    panel_cognition_forest(ax_cog_mgh, cog_by_cohort, "mgh-cog", "D", cognition_xlim, show_ylabel=False)
    disease_source = panel_disease_forest(ax_dis, disease, show_ylabel=False)

    perf_source = pd.concat([perf_mros_source, perf_mgh_source, perf_dis_source], ignore_index=True)
    perf_source.to_csv(OUTDIR / f"{STEM}_performance_source.csv", index=False)
    cog_by_cohort.to_csv(OUTDIR / f"{STEM}_cognition_by_cohort_m2_source.csv", index=False)
    disease_source.to_csv(OUTDIR / f"{STEM}_disease_m2_source.csv", index=False)

    perf_handles = [
        plt.Line2D([0], [0], marker=marker, linestyle=linestyle, color=color, label=label, markersize=4)
        for _, label, color, marker, linestyle in PERFORMANCE_META["cognition"]
    ]
    exposure_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=EXPOSURE_COLORS[label], label=label, markersize=4)
        for label in EXPOSURE_ORDER
    ]
    fig.legend(
        handles=perf_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.935),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.legend(
        handles=exposure_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.035),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
    )
    fig.suptitle(
        "Cross-sectional cognition and disease prediction\nfrom breathing stability features",
        y=0.992,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.91), h_pad=1.7, w_pad=1.4)
    save_figure(fig)
    print(f"Wrote {OUTDIR / (STEM + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
