#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


DEFAULT_MASTER = Path("REVISION/master_analysis_table_primary_cohorts_v4_paper.csv")
DEFAULT_OUTDIR = Path("REVISION/stage_ahi_anova_mixed_v1")

AHI_LABELS = ["Normal", "Mild", "Moderate", "Severe"]
AHI_BINS = [0.0, 5.0, 15.0, 30.0, np.inf]
STAGE_ORDER = ["NREM", "N1", "N2", "N3", "REM"]
MIN_STAGE_MINUTES = 15.0

STAGE_BSI_COLS = {
    "NREM": "bsi_robust_mean_w2_ov0p9_nrem_median_stability",
    "N1": "bsi_robust_mean_w2_ov0p9_n1_median_stability",
    "N2": "bsi_robust_mean_w2_ov0p9_n2_median_stability",
    "N3": "bsi_robust_mean_w2_ov0p9_n3_median_stability",
    "REM": "bsi_robust_mean_w2_ov0p9_rem_median_stability",
}
STAGE_MIN_COLS = {
    "N1": "n1_min",
    "N2": "n2_min",
    "N3": "n3_min",
    "REM": "rem_min",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate the original stage-by-AHI BSI figure with ANOVA and mixed-model summaries."
    )
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-stage-minutes", type=float, default=MIN_STAGE_MINUTES)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the figure from saved source CSVs without recomputing ANOVA or mixed models.",
    )
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def p_text(p_value: float | None) -> str:
    if p_value is None or pd.isna(p_value):
        return "NA"
    if p_value < 1e-6:
        return "<1e-6"
    if p_value < 0.001:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def add_multiplicity_columns(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    out = df.copy()
    ok = out[p_col].notna()
    if not ok.any():
        for col in ["p_holm", "p_bonferroni", "p_bh_fdr"]:
            out[col] = np.nan
        return out

    p_values = out.loc[ok, p_col].astype(float).to_numpy()
    out["p_holm"] = np.nan
    out["p_bonferroni"] = np.nan
    out["p_bh_fdr"] = np.nan
    out.loc[ok, "p_holm"] = multipletests(p_values, method="holm")[1]
    out.loc[ok, "p_bonferroni"] = multipletests(p_values, method="bonferroni")[1]
    out.loc[ok, "p_bh_fdr"] = multipletests(p_values, method="fdr_bh")[1]
    return out


def load_master(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [
        col
        for col in [
            "cohort",
            "sid",
            "fileid",
            "ahi",
            *STAGE_BSI_COLS.values(),
            *STAGE_MIN_COLS.values(),
        ]
        if col in header.columns
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in ["cohort", "sid", "fileid", "ahi", *STAGE_BSI_COLS.values(), *STAGE_MIN_COLS.values()]:
        if col not in df.columns:
            df[col] = np.nan

    df["ahi"] = safe_numeric(df["ahi"])
    for col in [*STAGE_BSI_COLS.values(), *STAGE_MIN_COLS.values()]:
        df[col] = safe_numeric(df[col])

    df["ahi_category"] = pd.cut(
        df["ahi"],
        bins=AHI_BINS,
        labels=AHI_LABELS,
        right=False,
        include_lowest=True,
    )
    df["analysis_subject_id"] = np.where(
        df["cohort"].astype(str).eq("mgh-cog"),
        df["fileid"].astype(str),
        df["sid"].astype(str),
    )
    df["analysis_subject_id"] = df["cohort"].astype(str) + ":" + df["analysis_subject_id"].astype(str)
    df["psg_id"] = df["cohort"].astype(str) + ":" + df["fileid"].astype(str)
    return df


def build_stage_long(df: pd.DataFrame, min_stage_minutes: float) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    nrem_minutes = df[["n1_min", "n2_min", "n3_min"]].sum(axis=1, min_count=1)

    for stage in STAGE_ORDER:
        bsi_col = STAGE_BSI_COLS[stage]
        stage_df = df[
            ["cohort", "fileid", "psg_id", "analysis_subject_id", "ahi", "ahi_category", bsi_col]
        ].copy()
        stage_df = stage_df.rename(columns={bsi_col: "bsi"})
        stage_df["stage"] = stage
        if stage == "NREM":
            stage_df["stage_minutes"] = nrem_minutes
        else:
            stage_df["stage_minutes"] = df[STAGE_MIN_COLS[stage]]
        stage_df["bsi"] = safe_numeric(stage_df["bsi"])
        stage_df["stage_minutes"] = safe_numeric(stage_df["stage_minutes"])
        stage_df = stage_df[
            stage_df["bsi"].notna()
            & stage_df["ahi_category"].notna()
            & stage_df["stage_minutes"].ge(min_stage_minutes)
        ].copy()
        pieces.append(
            stage_df[
                [
                    "cohort",
                    "fileid",
                    "psg_id",
                    "analysis_subject_id",
                    "ahi",
                    "ahi_category",
                    "stage",
                    "stage_minutes",
                    "bsi",
                ]
            ]
        )
    out = pd.concat(pieces, ignore_index=True)
    out["ahi_category"] = pd.Categorical(out["ahi_category"], categories=AHI_LABELS, ordered=True)
    out["stage"] = pd.Categorical(out["stage"], categories=STAGE_ORDER, ordered=True)
    return out


def summarize_bars(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in STAGE_ORDER:
        sub = long_df[long_df["stage"].eq(stage)]
        for ahi_cat in AHI_LABELS:
            grp = sub[sub["ahi_category"].eq(ahi_cat)]
            rows.append(bar_row("A_stage_by_ahi", stage, ahi_cat, grp))
    for ahi_cat in AHI_LABELS:
        sub = long_df[long_df["ahi_category"].eq(ahi_cat)]
        for stage in STAGE_ORDER:
            grp = sub[sub["stage"].eq(stage)]
            rows.append(bar_row("B_ahi_by_stage", ahi_cat, stage, grp))
    return pd.DataFrame(rows)


def bar_row(panel: str, facet: str, x_level: str, grp: pd.DataFrame) -> dict[str, object]:
    values = safe_numeric(grp["bsi"]).dropna()
    return {
        "panel": panel,
        "facet": facet,
        "x_level": x_level,
        "n_rows": int(len(values)),
        "n_psgs": int(grp.loc[values.index, "psg_id"].nunique()) if len(values) else 0,
        "n_subjects": int(grp.loc[values.index, "analysis_subject_id"].nunique()) if len(values) else 0,
        "mean_bsi": float(values.mean()) if len(values) else np.nan,
        "sd_bsi": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
    }


def anova_oneway(model_name: str, data: pd.DataFrame, group_col: str) -> dict[str, object]:
    groups = []
    group_sizes = []
    for _, grp in data.groupby(group_col, observed=False):
        values = safe_numeric(grp["bsi"]).dropna().to_numpy()
        if len(values) > 0:
            groups.append(values)
            group_sizes.append(len(values))

    row: dict[str, object] = {
        "model_name": model_name,
        "method": "one_way_anova",
        "n_rows": int(sum(group_sizes)),
        "n_groups": int(len(groups)),
        "group_col": group_col,
        "df_between": np.nan,
        "df_within": np.nan,
        "f_stat": np.nan,
        "p_value": np.nan,
        "eta_sq": np.nan,
        "status": "insufficient_groups",
    }
    if len(groups) <= 1:
        return row

    f_stat, p_value = stats.f_oneway(*groups)
    all_values = np.concatenate(groups)
    grand_mean = float(np.mean(all_values))
    ss_between = float(sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups))
    ss_within = float(sum(np.sum((g - float(np.mean(g))) ** 2) for g in groups))
    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else np.nan

    row.update(
        {
            "df_between": len(groups) - 1,
            "df_within": len(all_values) - len(groups),
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "eta_sq": float(eta_sq),
            "status": "ok",
        }
    )
    return row


def fixed_effect_r2(result: object) -> float:
    fixed_pred = np.asarray(result.model.exog @ result.fe_params, dtype=float)
    var_fixed = float(np.nanvar(fixed_pred, ddof=1))
    try:
        var_random = float(np.asarray(result.cov_re).diagonal().sum())
    except Exception:
        var_random = 0.0
    var_resid = float(getattr(result, "scale", np.nan))
    denom = var_fixed + max(var_random, 0.0) + max(var_resid, 0.0)
    return var_fixed / denom if denom > 0 else np.nan


def fit_mixed_with_retries(formula: str, data: pd.DataFrame, group_col: str) -> tuple[object, str]:
    errors = []
    for method, maxiter in [("lbfgs", 200), ("powell", 500), ("cg", 500)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, data=data, groups=data[group_col])
                result = model.fit(method=method, reml=False, maxiter=maxiter, disp=False)
            if not np.isfinite(float(result.llf)):
                raise RuntimeError(f"non-finite log-likelihood: {result.llf}")
            return result, method
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    raise RuntimeError(" | ".join(errors))


def fit_ols_lrt(
    model_name: str,
    model_data: pd.DataFrame,
    full_formula: str,
    null_formula: str,
    group_col: str,
    mixed_error: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    row: dict[str, object] = {
        "model_name": model_name,
        "method": "ols_lrt_fallback",
        "n_rows": int(len(model_data)),
        "n_random_groups": int(model_data[group_col].nunique()),
        "group_col": group_col,
        "full_formula": full_formula,
        "null_formula": null_formula,
        "llf_full": np.nan,
        "llf_null": np.nan,
        "df_diff_fixed": np.nan,
        "lr_stat": np.nan,
        "p_value": np.nan,
        "marginal_r2": np.nan,
        "converged_full": False,
        "converged_null": False,
        "optimizer_full": "",
        "optimizer_null": "",
        "status": "failed",
        "error": mixed_error,
    }
    coef_rows: list[dict[str, object]] = []
    try:
        full_result = smf.ols(full_formula, data=model_data).fit()
        null_result = smf.ols(null_formula, data=model_data).fit()
        df_diff = int(full_result.df_model - null_result.df_model)
        lr_stat = float(max(0.0, 2.0 * (full_result.llf - null_result.llf)))
        p_value = float(stats.chi2.sf(lr_stat, df_diff)) if df_diff > 0 else np.nan
        row.update(
            {
                "llf_full": float(full_result.llf),
                "llf_null": float(null_result.llf),
                "df_diff_fixed": df_diff,
                "lr_stat": lr_stat,
                "p_value": p_value,
                "marginal_r2": float(full_result.rsquared),
                "converged_full": True,
                "converged_null": True,
                "status": "ols_lrt_fallback",
            }
        )
        ci = full_result.conf_int()
        for term in full_result.params.index:
            coef_rows.append(
                {
                    "model_name": model_name,
                    "term": term,
                    "coef": float(full_result.params[term]),
                    "std_err": float(full_result.bse[term]),
                    "z": float(full_result.tvalues[term]),
                    "p_value": float(full_result.pvalues[term]),
                    "ci_low": float(ci.loc[term, 0]),
                    "ci_high": float(ci.loc[term, 1]),
                }
            )
    except Exception as exc:
        row["error"] = f"{mixed_error} | ols fallback: {exc}"
    return row, pd.DataFrame(coef_rows)


def fit_mixed_lrt(
    model_name: str,
    data: pd.DataFrame,
    full_formula: str,
    null_formula: str,
    group_col: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    model_data = data[["bsi", "ahi_category", "stage", group_col]].dropna().copy()
    model_data["ahi_category"] = pd.Categorical(model_data["ahi_category"], categories=AHI_LABELS, ordered=True)
    model_data["stage"] = pd.Categorical(model_data["stage"], categories=STAGE_ORDER, ordered=True)
    row: dict[str, object] = {
        "model_name": model_name,
        "method": "mixedlm_lrt",
        "n_rows": int(len(model_data)),
        "n_random_groups": int(model_data[group_col].nunique()),
        "group_col": group_col,
        "full_formula": full_formula,
        "null_formula": null_formula,
        "llf_full": np.nan,
        "llf_null": np.nan,
        "df_diff_fixed": np.nan,
        "lr_stat": np.nan,
        "p_value": np.nan,
        "marginal_r2": np.nan,
        "converged_full": False,
        "converged_null": False,
        "optimizer_full": "",
        "optimizer_null": "",
        "status": "not_fit",
        "error": "",
    }
    coef_rows: list[dict[str, object]] = []
    if len(model_data) < 20 or model_data[group_col].nunique() < 2:
        row["status"] = "insufficient_data"
        return row, pd.DataFrame(coef_rows)

    try:
        full_result, optimizer_full = fit_mixed_with_retries(full_formula, model_data, group_col)
        null_result, optimizer_null = fit_mixed_with_retries(null_formula, model_data, group_col)

        df_diff = int(len(full_result.fe_params) - len(null_result.fe_params))
        lr_stat = float(max(0.0, 2.0 * (full_result.llf - null_result.llf)))
        p_value = float(stats.chi2.sf(lr_stat, df_diff)) if df_diff > 0 else np.nan
        row.update(
            {
                "llf_full": float(full_result.llf),
                "llf_null": float(null_result.llf),
                "df_diff_fixed": df_diff,
                "lr_stat": lr_stat,
                "p_value": p_value,
                "marginal_r2": fixed_effect_r2(full_result),
                "converged_full": bool(getattr(full_result, "converged", False)),
                "converged_null": bool(getattr(null_result, "converged", False)),
                "optimizer_full": optimizer_full,
                "optimizer_null": optimizer_null,
                "status": "ok",
            }
        )

        ci = full_result.conf_int()
        for term in full_result.params.index:
            coef_rows.append(
                {
                    "model_name": model_name,
                    "term": term,
                    "coef": float(full_result.params[term]),
                    "std_err": float(full_result.bse[term]),
                    "z": float(full_result.tvalues[term]),
                    "p_value": float(full_result.pvalues[term]),
                    "ci_low": float(ci.loc[term, 0]),
                    "ci_high": float(ci.loc[term, 1]),
                }
            )
    except Exception as exc:
        return fit_ols_lrt(model_name, model_data, full_formula, null_formula, group_col, str(exc))
    return row, pd.DataFrame(coef_rows)


def build_stats(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anova_rows: list[dict[str, object]] = []
    mixed_rows: list[dict[str, object]] = []
    coef_frames: list[pd.DataFrame] = []

    for stage in STAGE_ORDER:
        sub = long_df[long_df["stage"].eq(stage)].copy()
        model_name = f"A_stage_{stage}_ahi_category"
        anova_rows.append(anova_oneway(model_name, sub, "ahi_category"))
        mixed_row, coef_df = fit_mixed_lrt(
            model_name,
            sub,
            "bsi ~ C(ahi_category, Treatment(reference='Normal'))",
            "bsi ~ 1",
            "analysis_subject_id",
        )
        mixed_rows.append(mixed_row)
        if not coef_df.empty:
            coef_frames.append(coef_df)

    for ahi_cat in AHI_LABELS:
        sub = long_df[long_df["ahi_category"].eq(ahi_cat)].copy()
        model_name = f"B_ahi_{ahi_cat}_stage"
        anova_rows.append(anova_oneway(model_name, sub, "stage"))
        mixed_row, coef_df = fit_mixed_lrt(
            model_name,
            sub,
            "bsi ~ C(stage, Treatment(reference='N2'))",
            "bsi ~ 1",
            "psg_id",
        )
        mixed_rows.append(mixed_row)
        if not coef_df.empty:
            coef_frames.append(coef_df)

    anova = pd.DataFrame(anova_rows)
    mixed = pd.DataFrame(mixed_rows)
    anova = add_multiplicity_columns(anova)
    mixed = add_multiplicity_columns(mixed)
    coefs = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    return anova, mixed, coefs


def mixed_annotation_row(mixed: pd.DataFrame, model_name: str) -> dict[str, object]:
    m = mixed[mixed["model_name"].eq(model_name)]
    if m.empty:
        marginal_r2 = np.nan
        p_value = np.nan
        p_holm = np.nan
        status = "missing"
    else:
        row = m.iloc[0]
        marginal_r2 = row.get("marginal_r2", np.nan)
        p_value = row.get("p_value", np.nan)
        p_holm = row.get("p_holm", np.nan)
        status = row.get("status", "")

    p_for_display = p_holm if pd.notna(p_holm) else p_value
    return {
        "model_name": model_name,
        "marginal_r2": marginal_r2,
        "p_value": p_value,
        "p_holm": p_holm,
        "p_display": p_text(p_for_display),
        "status": status,
        "annotation_mathtext": f"$R^2_m$={marginal_r2:.2f}, p{p_text(p_for_display)}",
    }


def build_plot_annotation_source(mixed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in STAGE_ORDER:
        row = mixed_annotation_row(mixed, f"A_stage_{stage}_ahi_category")
        row.update({"panel": "A_stage_by_ahi", "facet": stage})
        rows.append(row)
    for ahi_cat in AHI_LABELS:
        row = mixed_annotation_row(mixed, f"B_ahi_{ahi_cat}_stage")
        row.update({"panel": "B_ahi_by_stage", "facet": ahi_cat})
        rows.append(row)
    return pd.DataFrame(rows)


def mixed_annotation_text(mixed: pd.DataFrame, model_name: str) -> str:
    return str(mixed_annotation_row(mixed, model_name)["annotation_mathtext"])


def plot_stage_ahi_figure(
    summary: pd.DataFrame,
    mixed: pd.DataFrame,
    outdir: Path,
) -> None:
    fig = plt.figure(figsize=(10.5, 7.4))
    outer = fig.add_gridspec(
        2,
        1,
        left=0.075,
        right=0.99,
        bottom=0.075,
        top=0.895,
        hspace=0.42,
    )
    top_gs = outer[0].subgridspec(1, 5, wspace=0.52)
    bottom_gs = outer[1].subgridspec(1, 4, wspace=0.45)
    y_max = float(summary["mean_bsi"].add(summary["sd_bsi"].fillna(0)).max())
    y_lim = max(2.5, min(4.0, y_max * 1.08))

    for idx, stage in enumerate(STAGE_ORDER):
        ax = fig.add_subplot(top_gs[0, idx])
        sub = summary[(summary["panel"].eq("A_stage_by_ahi")) & (summary["facet"].eq(stage))]
        sub = sub.set_index("x_level").reindex(AHI_LABELS)
        x = np.arange(len(AHI_LABELS))
        ax.bar(
            x,
            sub["mean_bsi"],
            yerr=sub["sd_bsi"],
            capsize=5,
            color="0.35",
            width=0.72,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(AHI_LABELS, rotation=90, ha="center", fontsize=8)
        ax.set_title(stage, fontsize=10, fontweight="bold", loc="left")
        if idx == 0:
            ax.set_ylabel("BSI (mean +/- SD)", fontsize=9)
        ax.set_ylim(0, y_lim)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", color="0.86")
        ax.set_axisbelow(True)
        ax.text(
            0.02,
            0.98,
            mixed_annotation_text(mixed, f"A_stage_{stage}_ahi_category"),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        )

    for idx, ahi_cat in enumerate(AHI_LABELS):
        ax = fig.add_subplot(bottom_gs[0, idx])
        sub = summary[(summary["panel"].eq("B_ahi_by_stage")) & (summary["facet"].eq(ahi_cat))]
        sub = sub.set_index("x_level").reindex(STAGE_ORDER)
        x = np.arange(len(STAGE_ORDER))
        ax.bar(
            x,
            sub["mean_bsi"],
            yerr=sub["sd_bsi"],
            capsize=5,
            color="0.35",
            width=0.72,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_ORDER, rotation=90, ha="center", fontsize=8)
        ax.set_title(f"AHI: {ahi_cat}", fontsize=10, fontweight="bold", loc="left")
        if idx == 0:
            ax.set_ylabel("BSI (mean +/- SD)", fontsize=9)
        ax.set_ylim(0, y_lim)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", color="0.86")
        ax.set_axisbelow(True)
        ax.text(
            0.02,
            0.98,
            mixed_annotation_text(mixed, f"B_ahi_{ahi_cat}_stage"),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        )

    fig.text(0.012, 0.887, "A", fontsize=15, fontweight="bold")
    fig.text(0.012, 0.445, "B", fontsize=15, fontweight="bold")
    fig.suptitle("Breathing stability by sleep stage and AHI", fontsize=13, fontweight="bold", y=0.963)
    fig.savefig(outdir / "figure_stage_ahi_anova_mixed_v1.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "figure_stage_ahi_anova_mixed_v1.pdf", bbox_inches="tight")
    fig.savefig(outdir / "figure_stage_ahi_anova_mixed_v1.eps", bbox_inches="tight")
    fig.savefig(
        outdir / "figure_stage_ahi_anova_mixed_v1.jpg",
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )
    plt.close(fig)


def write_note(outdir: Path, long_df: pd.DataFrame, anova: pd.DataFrame, mixed: pd.DataFrame) -> None:
    lines = [
        "# Stage By AHI Figure v1",
        "",
        "Purpose: reproduce the original publication-style stage-by-AHI BSI figure, while exporting both the legacy one-way ANOVA summaries and mixed-effects omnibus tests requested for revision.",
        "",
        f"Long-format rows: `{len(long_df):,}`",
        f"PSGs: `{long_df['psg_id'].nunique():,}`",
        f"Subjects/cohort-subjects: `{long_df['analysis_subject_id'].nunique():,}`",
        f"Minimum stage duration: `{MIN_STAGE_MINUTES:g}` minutes",
        "",
        "Outputs:",
        "- `figure_stage_ahi_anova_mixed_v1.png`",
        "- `figure_stage_ahi_anova_mixed_v1.pdf`",
        "- `figure_stage_ahi_anova_mixed_v1.eps`",
        "- `figure_stage_ahi_anova_mixed_v1.jpg`",
        "- `stage_ahi_long_source_v1.csv`",
        "- `stage_ahi_bar_source_v1.csv`",
        "- `stage_ahi_plot_annotation_source_v1.csv`",
        "- `stage_ahi_anova_results_v1.csv`",
        "- `stage_ahi_mixed_lrt_results_v1.csv`",
        "- `stage_ahi_mixed_coefficients_v1.csv`",
        "",
        "Interpretation guardrail: the bars retain the original descriptive mean +/- SD display. Figure annotations use mixed-effects marginal R-squared and Holm-adjusted omnibus p-values. The ANOVA rows are retained only for continuity with the original figure.",
        "",
        "Mixed-effects model definitions:",
        "- Panel A: within each stage, `BSI ~ AHI category` with random intercept for cohort-specific subject.",
        "- Panel B: within each AHI category, `BSI ~ sleep stage` with random intercept for PSG.",
        "- Omnibus p-values are likelihood-ratio tests versus random-intercept null models; `R2m` is the fixed-effect marginal R-squared approximation.",
        "- Holm, Bonferroni, and Benjamini-Hochberg adjusted p-values are exported for the nine panel-level omnibus tests; figure annotations show marginal R-squared and Holm-adjusted omnibus p-values.",
        "",
        "ANOVA summary:",
    ]
    for _, row in anova.iterrows():
        lines.append(
            f"- `{row['model_name']}`: eta2=`{row['eta_sq']:.3g}`, p=`{p_text(row['p_value'])}`, Holm p=`{p_text(row['p_holm'])}`, n=`{int(row['n_rows']):,}`"
        )
    lines.append("")
    lines.append("Mixed-effects summary:")
    for _, row in mixed.iterrows():
        lines.append(
            f"- `{row['model_name']}`: R2m=`{row['marginal_r2']:.3g}`, LRT p=`{p_text(row['p_value'])}`, Holm p=`{p_text(row['p_holm'])}`, n=`{int(row['n_rows']):,}`, groups=`{int(row['n_random_groups']):,}`, status=`{row['status']}`"
        )
    (outdir / "stage_ahi_figure_note_v1.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        summary = pd.read_csv(args.output_dir / "stage_ahi_bar_source_v1.csv")
        mixed = pd.read_csv(args.output_dir / "stage_ahi_mixed_lrt_results_v1.csv")
        annotation_source = build_plot_annotation_source(mixed)
        annotation_source.to_csv(args.output_dir / "stage_ahi_plot_annotation_source_v1.csv", index=False)
        plot_stage_ahi_figure(summary, mixed, args.output_dir)
        anova_path = args.output_dir / "stage_ahi_anova_results_v1.csv"
        long_path = args.output_dir / "stage_ahi_long_source_v1.csv"
        if anova_path.exists() and long_path.exists():
            anova = pd.read_csv(anova_path)
            long_df = pd.read_csv(long_path, usecols=["psg_id", "analysis_subject_id"])
            write_note(args.output_dir, long_df, anova, mixed)
        print(f"Wrote {args.output_dir}")
        return

    df = load_master(args.master_csv)
    long_df = build_stage_long(df, args.min_stage_minutes)
    summary = summarize_bars(long_df)
    anova, mixed, coefs = build_stats(long_df)
    annotation_source = build_plot_annotation_source(mixed)

    long_df.to_csv(args.output_dir / "stage_ahi_long_source_v1.csv", index=False)
    summary.to_csv(args.output_dir / "stage_ahi_bar_source_v1.csv", index=False)
    annotation_source.to_csv(args.output_dir / "stage_ahi_plot_annotation_source_v1.csv", index=False)
    anova.to_csv(args.output_dir / "stage_ahi_anova_results_v1.csv", index=False)
    mixed.to_csv(args.output_dir / "stage_ahi_mixed_lrt_results_v1.csv", index=False)
    coefs.to_csv(args.output_dir / "stage_ahi_mixed_coefficients_v1.csv", index=False)
    plot_stage_ahi_figure(summary, mixed, args.output_dir)
    write_note(args.output_dir, long_df, anova, mixed)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
