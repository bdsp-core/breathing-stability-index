#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    from statsmodels.duration.hazard_regression import PHReg

    HAVE_PHREG = True
    PHREG_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    HAVE_PHREG = False
    PHREG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
DEFAULT_OUTDIR = REPO_ROOT / "REVISION" / "posture_v1"

PRIMARY_BSI = "bsi_median_sleep"
BSI_CANDIDATES = [
    "bsi_median_sleep",
    "bsi_quantile_75_sleep",
    "bsi_instability_burden_gt_1p5_sleep",
]
STAGE_BSI_COLS = {
    "N1": "bsi_median_n1",
    "N2": "bsi_median_n2",
    "N3": "bsi_median_n3",
    "REM": "bsi_median_rem",
}
NREM_COL = "bsi_robust_mean_w2_ov0p9_nrem_median_stability"
MIN_STAGE_MINUTES = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run posture analyses for the breathing-stability revision.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_bool(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    return as_str.map(
        {
            "true": True,
            "1": True,
            "1.0": True,
            "yes": True,
            "false": False,
            "0": False,
            "0.0": False,
            "no": False,
        }
    )


def safe_sex(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    as_str = series.astype(str).str.strip().str.lower()
    mapped = as_str.map(
        {
            "m": 1.0,
            "male": 1.0,
            "1": 1.0,
            "1.0": 1.0,
            "f": 0.0,
            "female": 0.0,
            "0": 0.0,
            "0.0": 0.0,
        }
    )
    return numeric.combine_first(mapped)


def read_csv_available(path: Path, requested_cols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in requested_cols if col in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in requested_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_master(path: Path) -> pd.DataFrame:
    cols = [
        "fileid",
        "sid",
        "cohort",
        "visitno",
        "age",
        "sex",
        "bmi",
        "paper_psg_type",
        "paper_primary_psg_eligible",
        "posture_available",
        "paper_posture_usable",
        "position_coverage_pct",
        "position_pct_sum",
        "position_tst_sum_min",
        "position_tst_delta_min",
        "position_pct_sum_within_1pct",
        "position_tst_within_5min",
        "position_tst_within_30min",
        "supine_pct_tst",
        "left_pct_tst",
        "right_pct_tst",
        "prone_pct_tst",
        "upright_pct_tst",
        "supine_tst_min",
        "left_tst_min",
        "right_tst_min",
        "prone_tst_min",
        "upright_tst_min",
        "tst_min",
        "n1_min",
        "n2_min",
        "n3_min",
        "rem_min",
        "ahi",
        "hypoxic_burden",
        "arousal_index",
        "ss_percent_main",
        "followup_days",
        "vital_status",
        *BSI_CANDIDATES,
        *STAGE_BSI_COLS.values(),
        NREM_COL,
    ]
    df = read_csv_available(path, cols)
    df["cohort"] = df["cohort"].astype(str)
    df["sid"] = df["sid"].astype(str)
    df["fileid"] = df["fileid"].astype(str)
    df["paper_psg_type"] = df["paper_psg_type"].astype("string").str.strip().str.lower()
    df["paper_primary_psg_eligible_bool"] = safe_bool(df["paper_primary_psg_eligible"])
    df["posture_available_bool"] = safe_bool(df["posture_available"])
    df["paper_posture_usable_bool"] = safe_bool(df["paper_posture_usable"])
    df["sex"] = safe_sex(df["sex"])
    for col in [
        "visitno",
        "age",
        "bmi",
        "position_coverage_pct",
        "position_pct_sum",
        "position_tst_sum_min",
        "position_tst_delta_min",
        "supine_pct_tst",
        "left_pct_tst",
        "right_pct_tst",
        "prone_pct_tst",
        "upright_pct_tst",
        "supine_tst_min",
        "left_tst_min",
        "right_tst_min",
        "prone_tst_min",
        "upright_tst_min",
        "tst_min",
        "n1_min",
        "n2_min",
        "n3_min",
        "rem_min",
        "ahi",
        "hypoxic_burden",
        "arousal_index",
        "ss_percent_main",
        "followup_days",
        *BSI_CANDIDATES,
        *STAGE_BSI_COLS.values(),
        NREM_COL,
    ]:
        df[col] = safe_numeric(df[col])
    df["event"] = df["vital_status"].astype("string").str.lower().eq("dead").astype(float)
    df["visit_sort"] = df["visitno"].fillna(999999)
    df["physiology_ready"] = df[[PRIMARY_BSI, "ss_percent_main", "ahi", "hypoxic_burden", "arousal_index"]].notna().all(axis=1)
    df["mortality_core_ready"] = df[["followup_days", "vital_status", PRIMARY_BSI, "ss_percent_main", "ahi", "hypoxic_burden"]].notna().all(axis=1)
    return df


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else np.nan


def build_coverage(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    strata = {
        "all_retained_psgs": pd.Series(True, index=df.index),
        "strict_primary_psgs": df["paper_primary_psg_eligible_bool"].eq(True),
        "physiology_ready_psgs": df["physiology_ready"],
        "strict_primary_physiology_ready_psgs": df["paper_primary_psg_eligible_bool"].eq(True) & df["physiology_ready"],
        "mortality_core_ready_psgs": df["mortality_core_ready"],
        "strict_primary_mortality_core_ready_psgs": df["paper_primary_psg_eligible_bool"].eq(True) & df["mortality_core_ready"],
    }
    rows: list[dict[str, object]] = []
    for stratum, mask in strata.items():
        base = df[mask].copy()
        groups = {"overall": base}
        for cohort, cohort_df in base.groupby("cohort"):
            groups[str(cohort)] = cohort_df
        for group, grp in groups.items():
            rows.append(
                {
                    "stratum": stratum,
                    "cohort": group,
                    "n_rows": int(len(grp)),
                    "posture_available_n": int(grp["posture_available_bool"].eq(True).sum()),
                    "posture_available_pct": pct(int(grp["posture_available_bool"].eq(True).sum()), int(len(grp))),
                    "paper_posture_usable_n": int(grp["paper_posture_usable_bool"].eq(True).sum()),
                    "paper_posture_usable_pct": pct(int(grp["paper_posture_usable_bool"].eq(True).sum()), int(len(grp))),
                    "position_tst_within_5min_n": int(safe_bool(grp["position_tst_within_5min"]).eq(True).sum()),
                    "supine_pct_nonmissing_n": int(grp["supine_pct_tst"].notna().sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "posture_coverage.csv", index=False)
    return out


def iqr_text(series: pd.Series) -> str:
    s = safe_numeric(series).dropna()
    if s.empty:
        return "NA"
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.3f} [{q1:.3f}, {q3:.3f}]"


def build_supine_descriptives(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    work = df[
        df["paper_primary_psg_eligible_bool"].eq(True)
        & df["paper_posture_usable_bool"].eq(True)
        & df["supine_pct_tst"].notna()
    ].copy()
    work["supine_tertile"] = pd.qcut(
        work["supine_pct_tst"],
        q=3,
        labels=["T1_low_supine", "T2_mid_supine", "T3_high_supine"],
        duplicates="drop",
    )
    rows: list[dict[str, object]] = []
    for group_name, group_df in [("overall", work), *[(str(c), g) for c, g in work.groupby("cohort")]]:
        for tertile, grp in group_df.groupby("supine_tertile", observed=True):
            rows.append(
                {
                    "cohort": group_name,
                    "supine_tertile": str(tertile),
                    "n": int(len(grp)),
                    "supine_pct_tst_iqr": iqr_text(grp["supine_pct_tst"]),
                    "bsi_median_sleep_iqr": iqr_text(grp["bsi_median_sleep"]),
                    "bsi_quantile_75_sleep_iqr": iqr_text(grp["bsi_quantile_75_sleep"]),
                    "bsi_instability_burden_gt_1p5_sleep_iqr": iqr_text(grp["bsi_instability_burden_gt_1p5_sleep"]),
                    "ss_percent_main_iqr": iqr_text(grp["ss_percent_main"]),
                    "ahi_iqr": iqr_text(grp["ahi"]),
                    "hypoxic_burden_iqr": iqr_text(grp["hypoxic_burden"]),
                    "arousal_index_iqr": iqr_text(grp["arousal_index"]),
                    "age_iqr": iqr_text(grp["age"]),
                    "bmi_iqr": iqr_text(grp["bmi"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "posture_bsi_by_supine_tertile.csv", index=False)
    return out


def standardize(series: pd.Series) -> pd.Series:
    s = safe_numeric(series)
    std = s.std(ddof=0)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (s - s.mean()) / std


def fit_ols(data: pd.DataFrame, formula: str, terms: list[str], model_name: str, outcome: str) -> list[dict[str, object]]:
    try:
        fit = smf.ols(formula, data=data).fit()
    except Exception as exc:
        return [
            {
                "outcome": outcome,
                "model_name": model_name,
                "term": term,
                "n": int(len(data)),
                "beta": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "status": "fit_error",
                "status_detail": f"{type(exc).__name__}: {exc}",
            }
            for term in terms
        ]
    ci = fit.conf_int()
    rows = []
    for term in terms:
        rows.append(
            {
                "outcome": outcome,
                "model_name": model_name,
                "term": term,
                "n": int(fit.nobs),
                "beta": float(fit.params.get(term, np.nan)),
                "ci_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                "ci_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
                "p_value": float(fit.pvalues.get(term, np.nan)),
                "r_squared": float(fit.rsquared),
                "status": "ok",
                "status_detail": "",
            }
        )
    return rows


def build_supine_models(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = df[
        df["paper_primary_psg_eligible_bool"].eq(True)
        & df["paper_posture_usable_bool"].eq(True)
        & df["supine_pct_tst"].notna()
    ].copy()
    for outcome in BSI_CANDIDATES:
        work = base[["cohort", outcome, "supine_pct_tst", "age", "sex", "bmi", "ahi", "hypoxic_burden", "arousal_index"]].copy()
        work["outcome_z"] = standardize(work[outcome])
        work["supine_pct_z"] = standardize(work["supine_pct_tst"])
        work["age_z"] = standardize(work["age"])
        work["bmi_z"] = standardize(work["bmi"])
        work["ahi_z"] = standardize(work["ahi"])
        work["hypoxic_burden_z"] = standardize(work["hypoxic_burden"])
        work["arousal_index_z"] = standardize(work["arousal_index"])
        rows.extend(
            fit_ols(
                work.dropna(subset=["outcome_z", "supine_pct_z"]),
                "outcome_z ~ supine_pct_z",
                ["supine_pct_z"],
                "unadjusted",
                outcome,
            )
        )
        rows.extend(
            fit_ols(
                work.dropna(subset=["outcome_z", "supine_pct_z", "age_z", "sex", "bmi_z", "cohort"]),
                "outcome_z ~ supine_pct_z + age_z + sex + bmi_z + C(cohort)",
                ["supine_pct_z"],
                "age_sex_bmi_cohort",
                outcome,
            )
        )
        rows.extend(
            fit_ols(
                work.dropna(
                    subset=[
                        "outcome_z",
                        "supine_pct_z",
                        "age_z",
                        "sex",
                        "bmi_z",
                        "ahi_z",
                        "hypoxic_burden_z",
                        "arousal_index_z",
                        "cohort",
                    ]
                ),
                "outcome_z ~ supine_pct_z + age_z + sex + bmi_z + ahi_z + hypoxic_burden_z + arousal_index_z + C(cohort)",
                ["supine_pct_z"],
                "age_sex_bmi_respiratory_cohort",
                outcome,
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "posture_supine_bsi_models.csv", index=False)
    return out


def build_stage_posture_models(df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nrem_minutes = df[["n1_min", "n2_min", "n3_min"]].fillna(0).sum(axis=1)
    eligible = df[
        df["paper_primary_psg_eligible_bool"].eq(True)
        & df["paper_posture_usable_bool"].eq(True)
        & df["supine_pct_tst"].notna()
        & df["rem_min"].ge(MIN_STAGE_MINUTES)
        & nrem_minutes.ge(MIN_STAGE_MINUTES)
        & df[STAGE_BSI_COLS["REM"]].notna()
        & df[NREM_COL].notna()
    ].copy()
    eligible["rem_minus_nrem"] = eligible[STAGE_BSI_COLS["REM"]] - eligible[NREM_COL]
    eligible["supine_pct_z"] = standardize(eligible["supine_pct_tst"])
    eligible["age_z"] = standardize(eligible["age"])
    eligible["bmi_z"] = standardize(eligible["bmi"])
    eligible["ahi_z"] = standardize(eligible["ahi"])

    rows = []
    for group_name, grp in [("overall", eligible), *[(str(c), g) for c, g in eligible.groupby("cohort")]]:
        rows.append(
            {
                "cohort": group_name,
                "n": int(len(grp)),
                "rem_bsi_iqr": iqr_text(grp[STAGE_BSI_COLS["REM"]]),
                "nrem_bsi_iqr": iqr_text(grp[NREM_COL]),
                "rem_minus_nrem_iqr": iqr_text(grp["rem_minus_nrem"]),
                "supine_pct_tst_iqr": iqr_text(grp["supine_pct_tst"]),
                "pct_rem_minus_nrem_positive": float((grp["rem_minus_nrem"] > 0).mean() * 100) if len(grp) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "posture_stage_summary.csv", index=False)

    model_rows: list[dict[str, object]] = []
    model_rows.extend(
        fit_ols(
            eligible.dropna(subset=["rem_minus_nrem", "supine_pct_z"]),
            "rem_minus_nrem ~ supine_pct_z",
            ["supine_pct_z"],
            "rem_minus_nrem_unadjusted",
            "rem_minus_nrem",
        )
    )
    model_rows.extend(
        fit_ols(
            eligible.dropna(subset=["rem_minus_nrem", "supine_pct_z", "age_z", "sex", "bmi_z", "cohort"]),
            "rem_minus_nrem ~ supine_pct_z + age_z + sex + bmi_z + C(cohort)",
            ["supine_pct_z"],
            "rem_minus_nrem_age_sex_bmi_cohort",
            "rem_minus_nrem",
        )
    )
    model_rows.extend(
        fit_ols(
            eligible.dropna(subset=["rem_minus_nrem", "supine_pct_z", "age_z", "sex", "bmi_z", "ahi_z", "cohort"]),
            "rem_minus_nrem ~ supine_pct_z + age_z + sex + bmi_z + ahi_z + C(cohort)",
            ["supine_pct_z"],
            "rem_minus_nrem_age_sex_bmi_ahi_cohort",
            "rem_minus_nrem",
        )
    )
    models = pd.DataFrame(model_rows)
    models.to_csv(outdir / "posture_stage_models.csv", index=False)
    return summary, models


def select_mortality_primary(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df[
        df["paper_primary_psg_eligible_bool"].eq(True)
        & df[["followup_days", "event", PRIMARY_BSI, "ss_percent_main", "ahi", "hypoxic_burden", "age", "sex", "bmi"]]
        .notna()
        .all(axis=1)
    ].copy()
    candidates = candidates.sort_values(["cohort", "sid", "visit_sort", "fileid"], kind="mergesort")
    candidates["selection_rank_within_subject"] = candidates.groupby(["cohort", "sid"]).cumcount() + 1
    return candidates[candidates["selection_rank_within_subject"].eq(1)].copy()


def zscore(series: pd.Series) -> pd.Series:
    return standardize(series)


def fit_cox_terms(df: pd.DataFrame, covariates: list[str], model_name: str) -> list[dict[str, object]]:
    rows = []
    required = ["followup_days", "event", *covariates]
    dat = df[required].dropna().copy()
    base = {
        "model_name": model_name,
        "n": int(len(dat)),
        "n_events": int(dat["event"].sum()) if len(dat) else 0,
        "status": "not_run",
        "status_detail": "",
    }
    if not HAVE_PHREG:
        return [{**base, "term": term, "hr_per_sd": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan, "status_detail": PHREG_IMPORT_ERROR} for term in covariates]
    if len(dat) < 50 or int(dat["event"].sum()) < 10:
        return [{**base, "term": term, "hr_per_sd": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan, "status_detail": "insufficient_rows_or_events"} for term in covariates]
    exog = pd.DataFrame(index=dat.index)
    order = []
    for cov in covariates:
        if cov == "sex":
            exog[cov] = dat[cov]
        else:
            exog[f"{cov}_z"] = zscore(dat[cov])
        order.append(cov if cov == "sex" else f"{cov}_z")
    keep = exog.notna().all(axis=1)
    dat = dat.loc[keep]
    exog = exog.loc[keep]
    if len(dat) < 50 or int(dat["event"].sum()) < 10:
        return [{**base, "term": term, "hr_per_sd": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan, "status_detail": "insufficient_complete_cases"} for term in covariates]
    try:
        fit = PHReg(dat["followup_days"], exog, status=dat["event"], ties="breslow").fit(disp=0)
        ci = fit.conf_int()
        for cov in covariates:
            name = cov if cov == "sex" else f"{cov}_z"
            idx = order.index(name)
            beta = float(fit.params[idx])
            rows.append(
                {
                    **base,
                    "n": int(len(dat)),
                    "n_events": int(dat["event"].sum()),
                    "term": cov,
                    "hr_per_sd": float(np.exp(beta)),
                    "ci_lower": float(np.exp(ci[idx][0])),
                    "ci_upper": float(np.exp(ci[idx][1])),
                    "p_value": float(fit.pvalues[idx]),
                    "status": "ok",
                    "status_detail": "",
                }
            )
        return rows
    except Exception as exc:
        return [{**base, "term": term, "hr_per_sd": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan, "status": "fit_error", "status_detail": f"{type(exc).__name__}: {exc}"} for term in covariates]


def build_mortality_posture_sensitivity(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    selected = select_mortality_primary(df)
    selected["posture_usable_in_selected"] = selected["paper_posture_usable_bool"].eq(True) & selected["supine_pct_tst"].notna()
    selected.to_csv(outdir / "posture_mortality_selected_subjects.csv", index=False)
    posture = selected[selected["posture_usable_in_selected"]].copy()
    rows: list[dict[str, object]] = []
    model_specs = [
        ("bsi_age_sex_bmi", [PRIMARY_BSI, "age", "sex", "bmi"]),
        ("bsi_age_sex_bmi_supine", [PRIMARY_BSI, "age", "sex", "bmi", "supine_pct_tst"]),
        ("bsi_age_sex_bmi_respiratory", [PRIMARY_BSI, "age", "sex", "bmi", "ahi", "hypoxic_burden", "arousal_index"]),
        ("bsi_age_sex_bmi_respiratory_supine", [PRIMARY_BSI, "age", "sex", "bmi", "ahi", "hypoxic_burden", "arousal_index", "supine_pct_tst"]),
    ]
    for model_name, covariates in model_specs:
        rows.extend(fit_cox_terms(posture, covariates, model_name))
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "posture_mortality_sensitivity.csv", index=False)
    return out


def write_decision_memo(
    outdir: Path,
    coverage: pd.DataFrame,
    supine_models: pd.DataFrame,
    stage_summary: pd.DataFrame,
    mortality_models: pd.DataFrame,
) -> None:
    strict_cov = coverage[
        coverage["stratum"].eq("strict_primary_psgs") & coverage["cohort"].eq("overall")
    ].iloc[0]
    clinical_zero = coverage[
        coverage["stratum"].eq("strict_primary_psgs")
        & coverage["cohort"].isin(["mros", "mgh-cog"])
        & coverage["paper_posture_usable_n"].eq(0)
    ]
    bsi_supine_adj = supine_models[
        supine_models["outcome"].eq(PRIMARY_BSI)
        & supine_models["model_name"].eq("age_sex_bmi_respiratory_cohort")
        & supine_models["term"].eq("supine_pct_z")
    ]
    bsi_supine_text = "not estimable"
    if len(bsi_supine_adj):
        row = bsi_supine_adj.iloc[0]
        bsi_supine_text = f"beta={row['beta']:.3f}, p={row['p_value']:.3g}, n={int(row['n']):,}"

    stage_overall = stage_summary[stage_summary["cohort"].eq("overall")]
    stage_text = "not estimable"
    if len(stage_overall):
        row = stage_overall.iloc[0]
        stage_text = f"REM-NREM {row['rem_minus_nrem_iqr']} across n={int(row['n']):,}"

    bsi_mort = mortality_models[
        mortality_models["term"].eq(PRIMARY_BSI)
        & mortality_models["model_name"].isin(["bsi_age_sex_bmi_respiratory", "bsi_age_sex_bmi_respiratory_supine"])
        & mortality_models["status"].eq("ok")
    ][["model_name", "hr_per_sd", "ci_lower", "ci_upper", "p_value", "n", "n_events"]]
    if len(bsi_mort) >= 2:
        no_sup = bsi_mort[bsi_mort["model_name"].eq("bsi_age_sex_bmi_respiratory")].iloc[0]
        sup = bsi_mort[bsi_mort["model_name"].eq("bsi_age_sex_bmi_respiratory_supine")].iloc[0]
        pct_change = 100.0 * (sup["hr_per_sd"] - no_sup["hr_per_sd"]) / no_sup["hr_per_sd"]
        mortality_text = (
            f"BSI HR {no_sup['hr_per_sd']:.2f} without supine and {sup['hr_per_sd']:.2f} with supine "
            f"({pct_change:+.1f}% change; n={int(sup['n']):,}, events={int(sup['n_events']):,})"
        )
    else:
        mortality_text = "mortality posture sensitivity not estimable"

    if strict_cov["paper_posture_usable_pct"] >= 70 and len(clinical_zero) == 0:
        recommendation = "main_or_supplement_candidate"
    else:
        recommendation = "supplement_sensitivity_with_limitation"

    lines = [
        "# Posture v1 Decision Memo",
        "",
        "## Recommendation",
        f"- Classification: `{recommendation}`",
        "- Use posture as a reviewer-response and supplement/sensitivity analysis rather than a required primary covariate.",
        "- Rationale: posture coverage is strong in BDSP clinical cohorts but absent in MrOS and mgh-cog, so posture adjustment would change the cohort composition of primary analyses.",
        "",
        "## Coverage",
        f"- Strict primary posture-usable rows: `{int(strict_cov['paper_posture_usable_n']):,}` / `{int(strict_cov['n_rows']):,}` (`{strict_cov['paper_posture_usable_pct']:.1f}%`).",
        "- MrOS and mgh-cog have no posture-usable rows in the merged source.",
        "",
        "## Supine burden and BSI",
        f"- Adjusted supine-percent association with `bsi_median_sleep`: `{bsi_supine_text}`.",
        "",
        "## Stage interpretation",
        f"- Posture-usable REM/NREM summary: `{stage_text}`.",
        "",
        "## Mortality interpretation",
        f"- Posture-usable mortality sensitivity: `{mortality_text}`.",
        "",
        "## Reviewer-facing wording",
        "- We can say posture was recovered with high coverage for the BDSP cohorts and internally coherent position percentages.",
        "- We should not imply whole-study posture adjustment, because the community/cognition cohorts lack posture fields.",
        "- The strongest framing is: posture was evaluated as a sensitivity analysis; it is suitable for supplement/reviewer response, and remaining incomplete posture availability is a limitation.",
    ]
    (outdir / "posture_decision_memo.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_master(args.master_csv)

    coverage = build_coverage(df, args.output_dir)
    build_supine_descriptives(df, args.output_dir)
    supine_models = build_supine_models(df, args.output_dir)
    stage_summary, stage_models = build_stage_posture_models(df, args.output_dir)
    mortality_models = build_mortality_posture_sensitivity(df, args.output_dir)
    write_decision_memo(args.output_dir, coverage, supine_models, stage_summary, mortality_models)

    strict_cov = coverage[coverage["stratum"].eq("strict_primary_psgs") & coverage["cohort"].eq("overall")].iloc[0]
    summary = {
        "rows_in_master": int(len(df)),
        "strict_primary_rows": int(strict_cov["n_rows"]),
        "strict_primary_posture_usable_rows": int(strict_cov["paper_posture_usable_n"]),
        "strict_primary_posture_usable_pct": float(strict_cov["paper_posture_usable_pct"]),
        "supine_model_rows": int(len(supine_models)),
        "stage_model_rows": int(len(stage_models)),
        "mortality_sensitivity_rows": int(len(mortality_models)),
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
