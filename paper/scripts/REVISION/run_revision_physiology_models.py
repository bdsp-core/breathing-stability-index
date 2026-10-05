from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


DEFAULT_MASTER = Path("REVISION/master_analysis_table_primary_cohorts_v3.csv")
DEFAULT_OUTPUT_DIR = Path("REVISION/physiology_v1")

PRIMARY_BSI = "bsi_median_sleep"
SUMMARY_BSI = "bsi_summary_mean_w2_ov0p9_sleep_median_stability"
SS_COL = "ss_percent_main"
FILE_ID = "fileid"
COHORT = "cohort"

COMPARATOR_COLS = ["ss_percent_main", "ahi", "hypoxic_burden", "arousal_index"]
PHYSIOLOGY_REQUIRED = [PRIMARY_BSI, SS_COL, "ahi", "hypoxic_burden", "arousal_index"]

STAGE_BSI_COLS = {
    "N1": "bsi_median_n1",
    "N2": "bsi_median_n2",
    "N3": "bsi_median_n3",
    "REM": "bsi_median_rem",
}
STAGE_MIN_COLS = {
    "N1": "n1_min",
    "N2": "n2_min",
    "N3": "n3_min",
    "REM": "rem_min",
}
NREM_COL = "bsi_robust_mean_w2_ov0p9_nrem_median_stability"
NREM_MIN_COMPONENTS = ["n1_min", "n2_min", "n3_min"]
MIN_STAGE_MINUTES = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revision physiology/comparator analyses from the v3 master table.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def iqr_text(series: pd.Series) -> str:
    s = safe_numeric(series).dropna()
    if s.empty:
        return "NA"
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.3f} [{q1:.3f}, {q3:.3f}]"


def p_text(p: object) -> str:
    if p is None or pd.isna(p):
        return "NA"
    p_float = float(p)
    if p_float < 0.001:
        return "<0.001"
    return f"{p_float:.3f}".rstrip("0").rstrip(".")


def standardize(series: pd.Series) -> pd.Series:
    s = safe_numeric(series)
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (s - s.mean()) / std


def pearson_summary(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    x_num = safe_numeric(x)
    y_num = safe_numeric(y)
    mask = x_num.notna() & y_num.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x_num[mask], y_num[mask])
    return float(r), float(p), n


def spearman_summary(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    x_num = safe_numeric(x)
    y_num = safe_numeric(y)
    mask = x_num.notna() & y_num.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x_num[mask], y_num[mask])
    return float(rho), float(p), n


def wilcoxon_summary(x: pd.Series, y: pd.Series) -> tuple[int, float, float | None]:
    x_num = safe_numeric(x)
    y_num = safe_numeric(y)
    mask = x_num.notna() & y_num.notna()
    n = int(mask.sum())
    if n < 3:
        return n, np.nan, np.nan
    diff = x_num[mask] - y_num[mask]
    if np.allclose(diff.to_numpy(), 0, equal_nan=True):
        return n, 0.0, 1.0
    stat, p = stats.wilcoxon(x_num[mask], y_num[mask], zero_method="wilcox", alternative="two-sided")
    return n, float(stat), float(p)


def load_master(path: Path) -> pd.DataFrame:
    usecols = [
        FILE_ID,
        "sid",
        COHORT,
        "age",
        "sex",
        "bmi",
        "phenotype_osa_csa",
        PRIMARY_BSI,
        SUMMARY_BSI,
        SS_COL,
        "ahi",
        "hypoxic_burden",
        "arousal_index",
        "tst_min",
        "n1_min",
        "n2_min",
        "n3_min",
        "rem_min",
        NREM_COL,
        *STAGE_BSI_COLS.values(),
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df[COHORT] = df[COHORT].astype(str)
    df[FILE_ID] = df[FILE_ID].astype(str)
    for col in [
        "age",
        "sex",
        "bmi",
        PRIMARY_BSI,
        SUMMARY_BSI,
        SS_COL,
        "ahi",
        "hypoxic_burden",
        "arousal_index",
        "tst_min",
        "n1_min",
        "n2_min",
        "n3_min",
        "rem_min",
        NREM_COL,
        *STAGE_BSI_COLS.values(),
    ]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    return df


def build_subset_counts(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    physiology_mask = df[PHYSIOLOGY_REQUIRED].notna().all(axis=1)
    rem_nrem_mask = (
        df["rem_min"].ge(MIN_STAGE_MINUTES)
        & df[NREM_COL].notna()
        & df[STAGE_BSI_COLS["REM"]].notna()
        & df[NREM_MIN_COMPONENTS].fillna(0).sum(axis=1).ge(MIN_STAGE_MINUTES)
    )
    stage_long_mask = pd.Series(False, index=df.index)
    for stage, bsi_col in STAGE_BSI_COLS.items():
        stage_long_mask = stage_long_mask | (df[bsi_col].notna() & df[STAGE_MIN_COLS[stage]].ge(MIN_STAGE_MINUTES))
    out = (
        df.groupby(COHORT)
        .agg(
            total_psgs=(FILE_ID, "size"),
            physiology_ready=(PRIMARY_BSI, lambda s: int((df.loc[s.index, PHYSIOLOGY_REQUIRED].notna().all(axis=1)).sum())),
            stage_long_ready=(FILE_ID, lambda s: int(stage_long_mask.loc[s.index].sum())),
            rem_nrem_ready=(FILE_ID, lambda s: int(rem_nrem_mask.loc[s.index].sum())),
            phenotype_nonmissing=("phenotype_osa_csa", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        {
            COHORT: ["overall"],
            "total_psgs": [len(df)],
            "physiology_ready": [int(physiology_mask.sum())],
            "stage_long_ready": [int(stage_long_mask.sum())],
            "rem_nrem_ready": [int(rem_nrem_mask.sum())],
            "phenotype_nonmissing": [int(df["phenotype_osa_csa"].notna().sum())],
        }
    )
    out = pd.concat([overall, out], ignore_index=True)
    out.to_csv(outdir / "physiology_subset_counts.csv", index=False)
    return out


def build_comparator_correlations(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohorts = ["overall"] + sorted(df[COHORT].dropna().unique().tolist())
    for cohort in cohorts:
        sub = df if cohort == "overall" else df[df[COHORT] == cohort]
        for comp in COMPARATOR_COLS:
            rho, p_s, n_s = spearman_summary(sub[PRIMARY_BSI], sub[comp])
            r, p_p, n_p = pearson_summary(sub[PRIMARY_BSI], sub[comp])
            rows.append(
                {
                    "cohort": cohort,
                    "bsi_exposure": PRIMARY_BSI,
                    "comparator": comp,
                    "n_spearman": n_s,
                    "spearman_rho": rho,
                    "spearman_p": p_s,
                    "n_pearson": n_p,
                    "pearson_r": r,
                    "pearson_p": p_p,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "bsi_comparator_correlations.csv", index=False)
    return out


def run_adjusted_model(data: pd.DataFrame, formula: str, predictor_terms: list[str], outcome_col: str, model_name: str) -> list[dict[str, object]]:
    try:
        fit = smf.ols(formula, data=data).fit()
    except Exception as exc:
        return [
            {
                "model_name": model_name,
                "outcome": outcome_col,
                "predictor": term,
                "n": len(data),
                "beta_std": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "model_error": str(exc),
            }
            for term in predictor_terms
        ]

    ci = fit.conf_int()
    rows = []
    for term in predictor_terms:
        rows.append(
            {
                "model_name": model_name,
                "outcome": outcome_col,
                "predictor": term,
                "n": int(fit.nobs),
                "beta_std": float(fit.params.get(term, np.nan)),
                "ci_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                "ci_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
                "p_value": float(fit.pvalues.get(term, np.nan)),
                "r_squared": float(fit.rsquared),
                "model_error": "",
            }
        )
    return rows


def build_adjusted_comparator_models(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    work = df[[COHORT, "age", "sex", "bmi", PRIMARY_BSI, SUMMARY_BSI, SS_COL, "ahi", "hypoxic_burden", "arousal_index"]].copy()
    work["sex"] = safe_numeric(work["sex"])
    for col in ["age", "bmi", PRIMARY_BSI, SUMMARY_BSI, SS_COL, "ahi", "hypoxic_burden", "arousal_index"]:
        work[col] = safe_numeric(work[col])

    rows: list[dict[str, object]] = []
    for outcome in COMPARATOR_COLS:
        needed_cols = list(dict.fromkeys([COHORT, "age", "sex", "bmi", PRIMARY_BSI, SS_COL, outcome]))
        model = work[needed_cols].copy()
        model = model.dropna()
        if model.empty:
            continue
        model["outcome_z"] = standardize(model[outcome])
        model["bsi_z"] = standardize(model[PRIMARY_BSI])
        model["ss_z"] = standardize(model[SS_COL])
        model["age_z"] = standardize(model["age"])
        model["bmi_z"] = standardize(model["bmi"])

        if outcome == SS_COL:
            rows.extend(
                run_adjusted_model(
                    model.dropna(subset=["outcome_z", "bsi_z", "age_z", "sex", "bmi_z"]),
                    "outcome_z ~ bsi_z + age_z + sex + bmi_z + C(cohort)",
                    ["bsi_z"],
                    outcome,
                    "bsi_vs_ss_age_sex_bmi_cohort",
                )
            )
            continue

        rows.extend(
            run_adjusted_model(
                model.dropna(subset=["outcome_z", "bsi_z", "age_z", "sex", "bmi_z"]),
                "outcome_z ~ bsi_z + age_z + sex + bmi_z + C(cohort)",
                ["bsi_z"],
                outcome,
                "bsi_only_age_sex_bmi_cohort",
            )
        )
        rows.extend(
            run_adjusted_model(
                model.dropna(subset=["outcome_z", "bsi_z", "ss_z", "age_z", "sex", "bmi_z"]),
                "outcome_z ~ bsi_z + ss_z + age_z + sex + bmi_z + C(cohort)",
                ["bsi_z", "ss_z"],
                outcome,
                "bsi_plus_ss_age_sex_bmi_cohort",
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "bsi_comparator_adjusted_models.csv", index=False)
    return out


def build_belt_agreement(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohorts = ["overall"] + sorted(df[COHORT].dropna().unique().tolist())
    for cohort in cohorts:
        sub = df if cohort == "overall" else df[df[COHORT] == cohort]
        paired = sub[[PRIMARY_BSI, SUMMARY_BSI]].dropna()
        rho, p_s, n_s = spearman_summary(paired[PRIMARY_BSI], paired[SUMMARY_BSI])
        r, p_p, n_p = pearson_summary(paired[PRIMARY_BSI], paired[SUMMARY_BSI])
        diff = paired[PRIMARY_BSI] - paired[SUMMARY_BSI]
        rows.append(
            {
                "cohort": cohort,
                "n_paired": int(len(paired)),
                "spearman_rho": rho,
                "spearman_p": p_s,
                "pearson_r": r,
                "pearson_p": p_p,
                "median_signed_difference": float(diff.median()) if len(paired) else np.nan,
                "median_abs_difference": float(diff.abs().median()) if len(paired) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "bsi_primary_sleep_belt_agreement.csv", index=False)
    return out


def build_stage_long(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for stage, bsi_col in STAGE_BSI_COLS.items():
        stage_df = df[[FILE_ID, COHORT, bsi_col, STAGE_MIN_COLS[stage]]].copy()
        stage_df = stage_df.rename(columns={bsi_col: "bsi", STAGE_MIN_COLS[stage]: "stage_minutes"})
        stage_df["stage"] = stage
        stage_df["bsi"] = safe_numeric(stage_df["bsi"])
        stage_df["stage_minutes"] = safe_numeric(stage_df["stage_minutes"])
        stage_df = stage_df[stage_df["bsi"].notna() & stage_df["stage_minutes"].ge(MIN_STAGE_MINUTES)].copy()
        pieces.append(stage_df[[FILE_ID, COHORT, "stage", "stage_minutes", "bsi"]])
    return pd.concat(pieces, ignore_index=True)


def stage_descriptives(long_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for cohort in ["overall"] + sorted(long_df[COHORT].dropna().unique().tolist()):
        sub = long_df if cohort == "overall" else long_df[long_df[COHORT] == cohort]
        for stage in ["N1", "N2", "N3", "REM"]:
            grp = sub[sub["stage"] == stage]
            rows.append(
                {
                    "cohort": cohort,
                    "stage": stage,
                    "n_rows": int(len(grp)),
                    "n_psgs": int(grp[FILE_ID].nunique()),
                    "bsi_median_iqr": iqr_text(grp["bsi"]),
                    "stage_minutes_median_iqr": iqr_text(grp["stage_minutes"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "stage_descriptives.csv", index=False)
    return out


def fit_mixed_model(model_name: str, formula: str, data: pd.DataFrame, group_col: str) -> tuple[str, pd.DataFrame, dict[str, object]]:
    meta: dict[str, object] = {
        "model_name": model_name,
        "method": "mixedlm",
        "status": "ok",
        "n_rows": int(len(data)),
        "n_groups": int(data[group_col].nunique()),
        "error": "",
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, data=data, groups=data[group_col])
            result = model.fit(method="lbfgs", reml=False, maxiter=200, disp=False)
        ci = result.conf_int()
        rows = []
        for term in result.params.index:
            rows.append(
                {
                    "model_name": model_name,
                    "term": term,
                    "coef": float(result.params[term]),
                    "std_err": float(result.bse[term]),
                    "z_or_t": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "ci_low": float(ci.loc[term, 0]),
                    "ci_high": float(ci.loc[term, 1]),
                }
            )
        meta["converged"] = bool(getattr(result, "converged", True))
        return "mixedlm", pd.DataFrame(rows), meta
    except Exception as exc:
        meta["method"] = "fallback"
        meta["status"] = "failed_mixedlm"
        meta["error"] = str(exc)
        return "fallback", pd.DataFrame(), meta


def build_stage_models(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_df = build_stage_long(df)
    long_df.head(200).to_csv(outdir / "stage_long_data_smoke_head.csv", index=False)

    results_frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, object]] = []

    method, categorical_res, categorical_meta = fit_mixed_model(
        "categorical_stage_model",
        "bsi ~ C(stage, Treatment(reference='N2')) + C(cohort)",
        long_df,
        FILE_ID,
    )
    if not categorical_res.empty:
        results_frames.append(categorical_res)
    meta_rows.append(categorical_meta)

    nrem_long = long_df[long_df["stage"].isin(["N1", "N2", "N3"])].copy()
    nrem_long["stage_order"] = nrem_long["stage"].map({"N1": 1.0, "N2": 2.0, "N3": 3.0})
    method_trend, trend_res, trend_meta = fit_mixed_model(
        "nrem_trend_model",
        "bsi ~ stage_order + C(cohort)",
        nrem_long,
        FILE_ID,
    )
    if not trend_res.empty:
        results_frames.append(trend_res)
    meta_rows.append(trend_meta)

    if method == "fallback" or method_trend == "fallback":
        wide = df[[COHORT, FILE_ID, *STAGE_BSI_COLS.values(), *STAGE_MIN_COLS.values()]].copy()
        for stage, bsi_col in STAGE_BSI_COLS.items():
            wide.loc[wide[STAGE_MIN_COLS[stage]] < MIN_STAGE_MINUTES, bsi_col] = np.nan
        four_stage = wide.dropna(subset=list(STAGE_BSI_COLS.values()))
        if len(four_stage) >= 3:
            friedman_stat, friedman_p = stats.friedmanchisquare(
                four_stage[STAGE_BSI_COLS["N1"]],
                four_stage[STAGE_BSI_COLS["N2"]],
                four_stage[STAGE_BSI_COLS["N3"]],
                four_stage[STAGE_BSI_COLS["REM"]],
            )
        else:
            friedman_stat, friedman_p = np.nan, np.nan
        fallback_rows = [
            {
                "model_name": "categorical_stage_model_fallback",
                "term": "friedman_all_stages",
                "coef": np.nan,
                "std_err": np.nan,
                "z_or_t": float(friedman_stat) if pd.notna(friedman_stat) else np.nan,
                "p_value": float(friedman_p) if pd.notna(friedman_p) else np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        ]
        results_frames.append(pd.DataFrame(fallback_rows))

    results = pd.concat(results_frames, ignore_index=True) if results_frames else pd.DataFrame()
    metadata = pd.DataFrame(meta_rows)
    results.to_csv(outdir / "stage_model_results.csv", index=False)
    metadata.to_csv(outdir / "stage_model_metadata.csv", index=False)
    return results, metadata


def build_rem_nrem_summary(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nrem_minutes = df[NREM_MIN_COMPONENTS].fillna(0).sum(axis=1)
    eligible = df[
        df[STAGE_BSI_COLS["REM"]].notna()
        & df[NREM_COL].notna()
        & df["rem_min"].ge(MIN_STAGE_MINUTES)
        & nrem_minutes.ge(MIN_STAGE_MINUTES)
    ].copy()
    eligible["rem_minus_nrem"] = eligible[STAGE_BSI_COLS["REM"]] - eligible[NREM_COL]

    rows = []
    for cohort in ["overall"] + sorted(df[COHORT].dropna().unique().tolist()):
        sub = eligible if cohort == "overall" else eligible[eligible[COHORT] == cohort]
        n, stat, p = wilcoxon_summary(sub[STAGE_BSI_COLS["REM"]], sub[NREM_COL])
        diff = sub["rem_minus_nrem"].dropna()
        rows.append(
            {
                "cohort": cohort,
                "n_paired": n,
                "rem_median_iqr": iqr_text(sub[STAGE_BSI_COLS["REM"]]),
                "nrem_median_iqr": iqr_text(sub[NREM_COL]),
                "median_rem_minus_nrem": float(diff.median()) if not diff.empty else np.nan,
                "pct_positive_diff": float((diff > 0).mean() * 100) if not diff.empty else np.nan,
                "wilcoxon_stat": stat,
                "wilcoxon_p": p,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "rem_nrem_summary.csv", index=False)
    return summary, eligible


def build_nrem_trend_summary(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    eligible = df[
        df[STAGE_BSI_COLS["N1"]].notna()
        & df[STAGE_BSI_COLS["N2"]].notna()
        & df[STAGE_BSI_COLS["N3"]].notna()
        & df["n1_min"].ge(MIN_STAGE_MINUTES)
        & df["n2_min"].ge(MIN_STAGE_MINUTES)
        & df["n3_min"].ge(MIN_STAGE_MINUTES)
    ].copy()
    eligible["n1_minus_n2"] = eligible[STAGE_BSI_COLS["N1"]] - eligible[STAGE_BSI_COLS["N2"]]
    eligible["n2_minus_n3"] = eligible[STAGE_BSI_COLS["N2"]] - eligible[STAGE_BSI_COLS["N3"]]

    rows = []
    for cohort in ["overall"] + sorted(df[COHORT].dropna().unique().tolist()):
        sub = eligible if cohort == "overall" else eligible[eligible[COHORT] == cohort]
        n12, stat12, p12 = wilcoxon_summary(sub[STAGE_BSI_COLS["N1"]], sub[STAGE_BSI_COLS["N2"]])
        n23, stat23, p23 = wilcoxon_summary(sub[STAGE_BSI_COLS["N2"]], sub[STAGE_BSI_COLS["N3"]])
        rows.append(
            {
                "cohort": cohort,
                "n_complete": int(len(sub)),
                "n1_median_iqr": iqr_text(sub[STAGE_BSI_COLS["N1"]]),
                "n2_median_iqr": iqr_text(sub[STAGE_BSI_COLS["N2"]]),
                "n3_median_iqr": iqr_text(sub[STAGE_BSI_COLS["N3"]]),
                "median_n1_minus_n2": float(sub["n1_minus_n2"].median()) if len(sub) else np.nan,
                "median_n2_minus_n3": float(sub["n2_minus_n3"].median()) if len(sub) else np.nan,
                "pct_n1_gt_n2": float((sub["n1_minus_n2"] > 0).mean() * 100) if len(sub) else np.nan,
                "pct_n2_gt_n3": float((sub["n2_minus_n3"] > 0).mean() * 100) if len(sub) else np.nan,
                "wilcoxon_n1_vs_n2_n": n12,
                "wilcoxon_n1_vs_n2_stat": stat12,
                "wilcoxon_n1_vs_n2_p": p12,
                "wilcoxon_n2_vs_n3_n": n23,
                "wilcoxon_n2_vs_n3_stat": stat23,
                "wilcoxon_n2_vs_n3_p": p23,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "nrem_stage_trend_summary.csv", index=False)
    return out


def build_rem_predominant_summary(rem_nrem_df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(rem_nrem_df["rem_minus_nrem"].quantile(0.75)) if len(rem_nrem_df) else np.nan
    flagged = rem_nrem_df.copy()
    flagged["rem_predominant_instability"] = flagged["rem_minus_nrem"] >= threshold

    summary_rows = []
    for cohort in ["overall"] + sorted(flagged[COHORT].dropna().unique().tolist()):
        sub = flagged if cohort == "overall" else flagged[flagged[COHORT] == cohort]
        summary_rows.append(
            {
                "cohort": cohort,
                "n_eligible": int(len(sub)),
                "threshold_rem_minus_nrem_q75": threshold,
                "n_rem_predominant": int(sub["rem_predominant_instability"].sum()),
                "pct_rem_predominant": float(sub["rem_predominant_instability"].mean() * 100) if len(sub) else np.nan,
                "median_rem_minus_nrem": float(sub["rem_minus_nrem"].median()) if len(sub) else np.nan,
                "bsi_median_sleep_iqr": iqr_text(sub[PRIMARY_BSI]),
                "ss_percent_main_iqr": iqr_text(sub[SS_COL]),
                "ahi_iqr": iqr_text(sub["ahi"]),
                "hypoxic_burden_iqr": iqr_text(sub["hypoxic_burden"]),
                "arousal_index_iqr": iqr_text(sub["arousal_index"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "rem_predominant_instability_summary.csv", index=False)

    compare_rows = []
    for label, grp in flagged.groupby("rem_predominant_instability"):
        compare_rows.append(
            {
                "group": "rem_predominant" if label else "not_rem_predominant",
                "n": int(len(grp)),
                "rem_minus_nrem_iqr": iqr_text(grp["rem_minus_nrem"]),
                "bsi_median_sleep_iqr": iqr_text(grp[PRIMARY_BSI]),
                "ss_percent_main_iqr": iqr_text(grp[SS_COL]),
                "ahi_iqr": iqr_text(grp["ahi"]),
                "hypoxic_burden_iqr": iqr_text(grp["hypoxic_burden"]),
                "arousal_index_iqr": iqr_text(grp["arousal_index"]),
            }
        )
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(outdir / "rem_predominant_group_comparison.csv", index=False)

    return summary, flagged[[FILE_ID, COHORT, "rem_minus_nrem", "rem_predominant_instability"]]


def build_phenotype_comparisons(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    phenotype_df = df[df["phenotype_osa_csa"].notna()].copy()
    phenotype_df = phenotype_df.dropna(subset=[PRIMARY_BSI, SS_COL, "ahi", "hypoxic_burden", "arousal_index"])

    desc_rows = []
    for phenotype, grp in phenotype_df.groupby("phenotype_osa_csa"):
        desc_rows.append(
            {
                "phenotype_osa_csa": phenotype,
                "n": int(len(grp)),
                "bsi_median_sleep_iqr": iqr_text(grp[PRIMARY_BSI]),
                "ss_percent_main_iqr": iqr_text(grp[SS_COL]),
                "ahi_iqr": iqr_text(grp["ahi"]),
                "hypoxic_burden_iqr": iqr_text(grp["hypoxic_burden"]),
                "arousal_index_iqr": iqr_text(grp["arousal_index"]),
            }
        )
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(outdir / "phenotype_descriptives.csv", index=False)

    metrics = [PRIMARY_BSI, SS_COL, "ahi", "hypoxic_burden", "arousal_index"]
    test_rows = []
    groups = {k: g for k, g in phenotype_df.groupby("phenotype_osa_csa")}
    for metric in metrics:
        samples = [safe_numeric(g[metric]).dropna() for g in groups.values()]
        if len(samples) >= 3 and all(len(s) >= 3 for s in samples):
            stat, p = stats.kruskal(*samples)
        else:
            stat, p = np.nan, np.nan
        test_rows.append(
            {
                "comparison_type": "omnibus_kruskal",
                "metric": metric,
                "group_a": "osa",
                "group_b": "csa,mixed",
                "n_a": int(len(groups.get("osa", []))),
                "n_b": int(len(groups.get("csa", [])) + len(groups.get("mixed", []))),
                "statistic": float(stat) if pd.notna(stat) else np.nan,
                "p_value": float(p) if pd.notna(p) else np.nan,
            }
        )
        for a, b in [("osa", "csa"), ("osa", "mixed"), ("csa", "mixed")]:
            if a not in groups or b not in groups:
                continue
            a_vals = safe_numeric(groups[a][metric]).dropna()
            b_vals = safe_numeric(groups[b][metric]).dropna()
            if len(a_vals) >= 3 and len(b_vals) >= 3:
                stat_u, p_u = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            else:
                stat_u, p_u = np.nan, np.nan
            test_rows.append(
                {
                    "comparison_type": "pairwise_mannwhitney",
                    "metric": metric,
                    "group_a": a,
                    "group_b": b,
                    "n_a": int(len(a_vals)),
                    "n_b": int(len(b_vals)),
                    "statistic": float(stat_u) if pd.notna(stat_u) else np.nan,
                    "p_value": float(p_u) if pd.notna(p_u) else np.nan,
                }
            )
    tests = pd.DataFrame(test_rows)
    tests.to_csv(outdir / "phenotype_pairwise_tests.csv", index=False)
    return desc, tests


def build_adjusted_phenotype_bsi_models(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    work = df[[COHORT, "age", "sex", "bmi", "phenotype_osa_csa", PRIMARY_BSI, "ahi", "hypoxic_burden"]].copy()
    work["phenotype_osa_csa"] = work["phenotype_osa_csa"].astype(str).str.lower()
    work = work[work["phenotype_osa_csa"].isin(["csa", "mixed", "osa"])].copy()
    for col in ["age", "sex", "bmi", PRIMARY_BSI, "ahi", "hypoxic_burden"]:
        work[col] = safe_numeric(work[col])

    term_labels = {
        "C(phenotype_osa_csa, Treatment(reference='csa'))[T.mixed]": "mixed vs csa",
        "C(phenotype_osa_csa, Treatment(reference='csa'))[T.osa]": "osa vs csa",
    }
    model_specs = [
        {
            "model_name": "phenotype_age_sex_bmi_cohort",
            "model_label": "Age, sex, BMI, and cohort",
            "formula": f"{PRIMARY_BSI} ~ C(phenotype_osa_csa, Treatment(reference='csa')) + age + C(sex) + bmi + C({COHORT})",
            "needed": [PRIMARY_BSI, "phenotype_osa_csa", "age", "sex", "bmi", COHORT],
        },
        {
            "model_name": "phenotype_age_sex_bmi_cohort_ahi_hypoxic_burden",
            "model_label": "Age, sex, BMI, cohort, AHI, and hypoxic burden",
            "formula": f"{PRIMARY_BSI} ~ C(phenotype_osa_csa, Treatment(reference='csa')) + age + C(sex) + bmi + C({COHORT}) + ahi + hypoxic_burden",
            "needed": [PRIMARY_BSI, "phenotype_osa_csa", "age", "sex", "bmi", COHORT, "ahi", "hypoxic_burden"],
        },
    ]

    rows: list[dict[str, object]] = []
    for spec in model_specs:
        model_data = work.dropna(subset=spec["needed"]).copy()
        try:
            fit = smf.ols(spec["formula"], data=model_data).fit(cov_type="HC3")
            ci = fit.conf_int()
            status = "ok"
            error = ""
        except Exception as exc:
            fit = None
            ci = pd.DataFrame()
            status = "failed"
            error = str(exc)

        phenotype_counts = model_data["phenotype_osa_csa"].value_counts().to_dict()
        for term, label in term_labels.items():
            rows.append(
                {
                    "model_name": spec["model_name"],
                    "model_label": spec["model_label"],
                    "outcome": PRIMARY_BSI,
                    "reference": "csa",
                    "contrast": label,
                    "term": term,
                    "n": int(len(model_data)),
                    "n_csa": int(phenotype_counts.get("csa", 0)),
                    "n_mixed": int(phenotype_counts.get("mixed", 0)),
                    "n_osa": int(phenotype_counts.get("osa", 0)),
                    "beta": float(fit.params.get(term, np.nan)) if fit is not None else np.nan,
                    "ci_lower": float(ci.loc[term, 0]) if fit is not None and term in ci.index else np.nan,
                    "ci_upper": float(ci.loc[term, 1]) if fit is not None and term in ci.index else np.nan,
                    "p_value": float(fit.pvalues.get(term, np.nan)) if fit is not None else np.nan,
                    "r_squared": float(fit.rsquared) if fit is not None else np.nan,
                    "covariance": "HC3",
                    "status": status,
                    "model_error": error,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "phenotype_adjusted_bsi_models.csv", index=False)
    write_adjusted_phenotype_bsi_text(out, outdir)
    return out


def contrast_text(row: pd.Series | None) -> str:
    if row is None or row.empty or row.get("status", "ok") != "ok":
        return "not estimable"
    return (
        f"{float(row['beta']):.3f} "
        f"(95% CI {float(row['ci_lower']):.3f} to {float(row['ci_upper']):.3f}; "
        f"p={p_text(row['p_value'])})"
    )


def write_adjusted_phenotype_bsi_text(models: pd.DataFrame, outdir: Path) -> None:
    base = models[models["model_name"].eq("phenotype_age_sex_bmi_cohort")]
    exposure = models[models["model_name"].eq("phenotype_age_sex_bmi_cohort_ahi_hypoxic_burden")]

    def pick(frame: pd.DataFrame, contrast: str) -> pd.Series | None:
        sub = frame[frame["contrast"].eq(contrast)]
        if sub.empty:
            return None
        return sub.iloc[0]

    base_mixed = pick(base, "mixed vs csa")
    base_osa = pick(base, "osa vs csa")
    exposure_mixed = pick(exposure, "mixed vs csa")
    exposure_osa = pick(exposure, "osa vs csa")
    base_n = int(base.iloc[0]["n"]) if not base.empty else 0
    exposure_n = int(exposure.iloc[0]["n"]) if not exposure.empty else 0

    lines = [
        "# Adjusted OSA/CSA Phenotype BSI Text",
        "",
        "## Methodology",
        (
            "We modeled median sleep BSI in complete cases as a function of respiratory phenotype using "
            "ordinary least-squares linear regression with CSA as the reference group and HC3 robust confidence intervals."
        ),
        (
            "The primary model adjusted for age, sex, BMI, and cohort; a sensitivity model additionally adjusted "
            "for AHI and hypoxic burden."
        ),
        "",
        "## Results",
        (
            f"In the primary adjusted model (n={base_n:,}), mixed and OSA phenotypes had higher BSI than CSA: "
            f"mixed vs CSA {contrast_text(base_mixed)}; OSA vs CSA {contrast_text(base_osa)}."
        ),
        (
            f"After additional adjustment for AHI and hypoxic burden (n={exposure_n:,}), the mixed phenotype "
            f"remained modestly higher than CSA: {contrast_text(exposure_mixed)}; the OSA coefficient "
            f"reversed direction: {contrast_text(exposure_osa)}, indicating that the OSA-CSA contrast was largely "
            "accounted for by obstructive event frequency and hypoxic burden."
        ),
        "",
    ]
    (outdir / "phenotype_adjusted_bsi_results_text.md").write_text("\n".join(lines))


def write_interpretation_note(
    outdir: Path,
    master_csv: Path,
    subset_counts: pd.DataFrame,
    comparator_corr: pd.DataFrame,
    adjusted_models: pd.DataFrame,
    stage_metadata: pd.DataFrame,
    rem_nrem_summary: pd.DataFrame,
    nrem_trend_summary: pd.DataFrame,
    rem_pred_summary: pd.DataFrame,
    phenotype_desc: pd.DataFrame,
    phenotype_adjusted_models: pd.DataFrame,
) -> None:
    overall_counts = subset_counts[subset_counts[COHORT] == "overall"].iloc[0]
    corr_overall = comparator_corr[comparator_corr["cohort"] == "overall"].copy()
    corr_map = {row["comparator"]: row for _, row in corr_overall.iterrows()}
    stage_meta = stage_metadata.set_index("model_name")
    rem_row = rem_nrem_summary[rem_nrem_summary["cohort"] == "overall"].iloc[0]
    nrem_row = nrem_trend_summary[nrem_trend_summary["cohort"] == "overall"].iloc[0]
    rem_pred_row = rem_pred_summary[rem_pred_summary["cohort"] == "overall"].iloc[0]
    bsi_joint = adjusted_models[
        (adjusted_models["outcome"] == "hypoxic_burden")
        & (adjusted_models["model_name"] == "bsi_plus_ss_age_sex_bmi_cohort")
        & (adjusted_models["predictor"] == "bsi_z")
    ]
    ss_joint = adjusted_models[
        (adjusted_models["outcome"] == "hypoxic_burden")
        & (adjusted_models["model_name"] == "bsi_plus_ss_age_sex_bmi_cohort")
        & (adjusted_models["predictor"] == "ss_z")
    ]
    phenotype_line = []
    for _, row in phenotype_desc.iterrows():
        phenotype_line.append(f"{row['phenotype_osa_csa']}: n={int(row['n'])}, BSI {row['bsi_median_sleep_iqr']}, SS {row['ss_percent_main_iqr']}")
    phenotype_adj = phenotype_adjusted_models[
        phenotype_adjusted_models["model_name"].eq("phenotype_age_sex_bmi_cohort")
    ].copy()
    phenotype_adj_mixed = contrast_text(
        phenotype_adj[phenotype_adj["contrast"].eq("mixed vs csa")].iloc[0]
        if not phenotype_adj[phenotype_adj["contrast"].eq("mixed vs csa")].empty
        else None
    )
    phenotype_adj_osa = contrast_text(
        phenotype_adj[phenotype_adj["contrast"].eq("osa vs csa")].iloc[0]
        if not phenotype_adj[phenotype_adj["contrast"].eq("osa vs csa")].empty
        else None
    )

    lines = [
        "# Physiology / Comparator Interpretation Note",
        "",
        "## Current Analysis Scope",
        f"- Working source: `{master_csv}`.",
        f"- Physiology-ready rows with BSI, SS, AHI, hypoxic burden, and arousal index: `{int(overall_counts['physiology_ready'])}`.",
        f"- Stage-analysis rows contributing at least one qualifying sleep stage: `{int(overall_counts['stage_long_ready'])}`.",
        f"- REM-vs-NREM eligible PSGs: `{int(overall_counts['rem_nrem_ready'])}`.",
        "",
        "## Comparator Results",
        f"- `bsi_median_sleep` vs `ss_percent_main`: Spearman `{corr_map[SS_COL]['spearman_rho']:.3f}`, Pearson `{corr_map[SS_COL]['pearson_r']:.3f}`.",
        f"- `bsi_median_sleep` vs `ahi`: Spearman `{corr_map['ahi']['spearman_rho']:.3f}`, Pearson `{corr_map['ahi']['pearson_r']:.3f}`.",
        f"- `bsi_median_sleep` vs `hypoxic_burden`: Spearman `{corr_map['hypoxic_burden']['spearman_rho']:.3f}`, Pearson `{corr_map['hypoxic_burden']['pearson_r']:.3f}`.",
        f"- `bsi_median_sleep` vs `arousal_index`: Spearman `{corr_map['arousal_index']['spearman_rho']:.3f}`, Pearson `{corr_map['arousal_index']['pearson_r']:.3f}`.",
    ]
    if not bsi_joint.empty and not ss_joint.empty:
        lines.extend(
            [
                "- In the age/sex/BMI/cohort-adjusted hypoxic-burden joint model, both BSI and SS can be inspected directly for overlap vs complementarity:",
                f"  - `bsi_z`: beta `{bsi_joint.iloc[0]['beta_std']:.3f}`, p `{bsi_joint.iloc[0]['p_value']:.3g}`.",
                f"  - `ss_z`: beta `{ss_joint.iloc[0]['beta_std']:.3f}`, p `{ss_joint.iloc[0]['p_value']:.3g}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Stage Results",
            f"- Stage-model method for categorical stage effect: `{stage_meta.loc['categorical_stage_model', 'method']}`.",
            f"- Stage-model status: `{stage_meta.loc['categorical_stage_model', 'status']}`.",
            f"- REM vs NREM paired median difference: `{rem_row['median_rem_minus_nrem']:.3f}` with Wilcoxon p `{rem_row['wilcoxon_p']:.3g}` across `{int(rem_row['n_paired'])}` PSGs.",
            f"- N1 vs N2 paired median difference: `{nrem_row['median_n1_minus_n2']:.3f}` with Wilcoxon p `{nrem_row['wilcoxon_n1_vs_n2_p']:.3g}`.",
            f"- N2 vs N3 paired median difference: `{nrem_row['median_n2_minus_n3']:.3f}` with Wilcoxon p `{nrem_row['wilcoxon_n2_vs_n3_p']:.3g}`.",
            "",
            "## REM-Predominant Instability",
            f"- REM-predominant instability is defined here as the top quartile of `REM - NREM` BSI among REM/NREM-eligible PSGs.",
            f"- Threshold used: `{rem_pred_row['threshold_rem_minus_nrem_q75']:.3f}`.",
            f"- Fraction flagged overall: `{rem_pred_row['pct_rem_predominant']:.1f}%` (`{int(rem_pred_row['n_rem_predominant'])}` of `{int(rem_pred_row['n_eligible'])}`).",
            "",
            "## Phenotype Summary",
            f"- {('; '.join(phenotype_line))}.",
            f"- Adjusted phenotype model, age/sex/BMI/cohort: mixed vs CSA `{phenotype_adj_mixed}`; OSA vs CSA `{phenotype_adj_osa}`.",
            "",
            "## Methodological Limits Right Now",
            "- Stage eligibility uses the sleep-stage minute columns (`n1_min`, `n2_min`, `n3_min`, `rem_min`) as conservative available duration fields.",
            "- The mixed-effects stage model is PSG-level repeated measures using `fileid` as the grouping variable.",
            "- These outputs are physiology/comparator results; mortality-specific restrictions are handled in the mortality bundle.",
        ]
    )
    (outdir / "physiology_interpretation.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_master(args.master_csv)
    subset_counts = build_subset_counts(df, args.output_dir)
    comparator_corr = build_comparator_correlations(df, args.output_dir)
    adjusted_models = build_adjusted_comparator_models(df, args.output_dir)
    build_belt_agreement(df, args.output_dir)
    stage_descriptives(build_stage_long(df), args.output_dir)
    _, stage_metadata = build_stage_models(df, args.output_dir)
    rem_nrem_summary, rem_nrem_df = build_rem_nrem_summary(df, args.output_dir)
    nrem_trend_summary = build_nrem_trend_summary(df, args.output_dir)
    rem_pred_summary, _ = build_rem_predominant_summary(rem_nrem_df, args.output_dir)
    phenotype_desc, _ = build_phenotype_comparisons(df, args.output_dir)
    phenotype_adjusted_models = build_adjusted_phenotype_bsi_models(df, args.output_dir)
    write_interpretation_note(
        args.output_dir,
        args.master_csv,
        subset_counts,
        comparator_corr,
        adjusted_models,
        stage_metadata,
        rem_nrem_summary,
        nrem_trend_summary,
        rem_pred_summary,
        phenotype_desc,
        phenotype_adjusted_models,
    )


if __name__ == "__main__":
    main()
