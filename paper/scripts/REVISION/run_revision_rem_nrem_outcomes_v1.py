#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

try:
    from statsmodels.duration.hazard_regression import PHReg

    HAVE_PHREG = True
    PHREG_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    HAVE_PHREG = False
    PHREG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
DEFAULT_OUTDIR = REPO_ROOT / "REVISION" / "rem_nrem_outcomes_v1"

REM_BSI = "rem_bsi_median_stability"
NREM_BSI = "nrem_bsi_median_stability"
DELTA_BSI = "rem_minus_nrem_bsi_median_stability"
REM_PREDOMINANT = "rem_predominant_instability_flag"
REM_SOURCE = "bsi_robust_mean_w2_ov0p9_rem_median_stability"
NREM_SOURCE = "bsi_robust_mean_w2_ov0p9_nrem_median_stability"
MIN_STAGE_MINUTES = 15.0

PREDICTORS = {
    REM_BSI: "REM BSI median stability",
    NREM_BSI: "NREM BSI median stability",
    DELTA_BSI: "REM minus NREM BSI median stability",
    REM_PREDOMINANT: "REM-predominant instability flag",
}

COGNITION_OUTCOMES = ["cog_fluid", "cog_crystallized", "cog_total"]

MORTALITY_MODELS = [
    ("m0_unadjusted", "Unadjusted", []),
    ("m1_age_sex_bmi", "Age + sex + BMI", ["age", "sex", "bmi"]),
    ("m2_age_sex_bmi_ahi", "Age + sex + BMI + AHI", ["age", "sex", "bmi", "ahi"]),
    (
        "m3_age_sex_bmi_hypoxic_burden",
        "Age + sex + BMI + hypoxic burden",
        ["age", "sex", "bmi", "hypoxic_burden"],
    ),
    (
        "m4_age_sex_bmi_ahi_hypoxic_burden",
        "Age + sex + BMI + AHI + hypoxic burden",
        ["age", "sex", "bmi", "ahi", "hypoxic_burden"],
    ),
]

COGNITION_MODELS = [
    ("unadjusted", "Unadjusted", [], False),
    ("age_sex", "Age + sex", ["age", "C(sex)", "C(cohort)"], False),
    ("age_sex_bmi", "Age + sex + BMI", ["age", "C(sex)", "C(cohort)", "bmi"], True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R2-2 REM/NREM outcome analyses from v4_paper.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_bool(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    return as_str.map({"true": True, "1": True, "1.0": True, "yes": True, "false": False, "0": False, "0.0": False, "no": False})


def safe_sex(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    as_str = series.astype(str).str.strip().str.lower()
    mapped = as_str.map({"m": 1.0, "male": 1.0, "1": 1.0, "1.0": 1.0, "f": 0.0, "female": 0.0, "0": 0.0, "0.0": 0.0})
    return numeric.combine_first(mapped)


def zscore(series: pd.Series, ddof: int = 0) -> pd.Series:
    values = safe_numeric(series)
    std = values.std(ddof=ddof)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return (values - values.mean()) / std


def read_master(path: Path) -> pd.DataFrame:
    required = [
        "fileid",
        "sid",
        "cohort",
        "visitno",
        "age",
        "sex",
        "bmi",
        "paper_psg_type",
        "paper_primary_psg_eligible",
        "followup_days",
        "vital_status",
        "death_date",
        "censor_date",
        "ahi",
        "hypoxic_burden",
        "rem_min",
        "n1_min",
        "n2_min",
        "n3_min",
        REM_SOURCE,
        NREM_SOURCE,
        *COGNITION_OUTCOMES,
        "cognition_days_from_psg",
    ]
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in required if col in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df["fileid"] = df["fileid"].astype(str)
    df["sid"] = df["sid"].astype(str)
    df["cohort"] = df["cohort"].astype(str)
    df["sex"] = safe_sex(df["sex"])
    df["paper_primary_psg_eligible_bool"] = safe_bool(df["paper_primary_psg_eligible"])
    df["paper_psg_type"] = df["paper_psg_type"].astype("string").str.strip().str.lower()
    for col in [
        "visitno",
        "age",
        "bmi",
        "followup_days",
        "ahi",
        "hypoxic_burden",
        "rem_min",
        "n1_min",
        "n2_min",
        "n3_min",
        REM_SOURCE,
        NREM_SOURCE,
        *COGNITION_OUTCOMES,
        "cognition_days_from_psg",
    ]:
        df[col] = safe_numeric(df[col])

    df[REM_BSI] = df[REM_SOURCE]
    df[NREM_BSI] = df[NREM_SOURCE]
    df[DELTA_BSI] = df[REM_BSI] - df[NREM_BSI]
    df[REM_PREDOMINANT] = np.where(df[DELTA_BSI].notna(), (df[DELTA_BSI] > 0).astype(float), np.nan)
    df["nrem_min"] = df[["n1_min", "n2_min", "n3_min"]].fillna(0).sum(axis=1)
    df["rem_nrem_bsi_ready"] = (
        df["rem_min"].ge(MIN_STAGE_MINUTES)
        & df["nrem_min"].ge(MIN_STAGE_MINUTES)
        & df[REM_BSI].notna()
        & df[NREM_BSI].notna()
    )
    df["event"] = df["vital_status"].astype("string").str.lower().eq("dead").astype(float)
    df["visit_sort"] = df["visitno"].fillna(999999)
    return df


def select_one_row_per_subject(df: pd.DataFrame, base_mask: pd.Series) -> pd.DataFrame:
    sub = df.loc[base_mask].copy()
    sub = sub.sort_values(["cohort", "sid", "visit_sort", "fileid"], kind="mergesort")
    sub["selection_rank_within_subject"] = sub.groupby(["cohort", "sid"]).cumcount() + 1
    return sub[sub["selection_rank_within_subject"].eq(1)].copy()


def build_mortality_cohort(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    mask = (
        df["paper_primary_psg_eligible_bool"].eq(True)
        & df["followup_days"].notna()
        & df["followup_days"].ge(0)
        & df["vital_status"].notna()
        & df["age"].notna()
        & df["sex"].notna()
        & df["rem_nrem_bsi_ready"]
    )
    selected = select_one_row_per_subject(df, mask)
    selected["analysis_cohort"] = "strict_primary_rem_nrem_mortality_ready"
    selected.to_csv(outdir / "mortality_selection_one_row_per_subject.csv", index=False)
    return selected


def cox_exog(dat: pd.DataFrame, predictor: str, covariates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    exog = pd.DataFrame(index=dat.index)
    predictor_term = predictor if predictor == REM_PREDOMINANT else f"{predictor}_z"
    exog[predictor_term] = safe_numeric(dat[predictor]) if predictor == REM_PREDOMINANT else zscore(dat[predictor])
    order = [predictor_term]
    for covariate in covariates:
        if covariate == "sex":
            exog["sex"] = safe_numeric(dat["sex"])
            order.append("sex")
        elif covariate in {"age", "bmi", "ahi", "hypoxic_burden"}:
            name = f"{covariate}_z"
            exog[name] = zscore(dat[covariate])
            order.append(name)
        else:
            exog[covariate] = safe_numeric(dat[covariate])
            order.append(covariate)
    return exog[order], order


def fit_cox(frame: pd.DataFrame, predictor: str, model_name: str, model_label: str, covariates: list[str], group: str) -> dict[str, object]:
    required = ["followup_days", "event", predictor, *covariates]
    dat = frame[["cohort", *required]].dropna(subset=required).copy()
    out = {
        "analysis_group": group,
        "predictor": predictor,
        "predictor_label": PREDICTORS[predictor],
        "model": model_name,
        "model_label": model_label,
        "n": int(len(dat)),
        "n_events": int(dat["event"].sum()) if len(dat) else 0,
        "status": "not_run",
        "status_detail": "",
        "effect_scale": "per SD" if predictor != REM_PREDOMINANT else "flag: REM>NREM",
        "hr": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
    }
    if not HAVE_PHREG:
        out["status_detail"] = f"PHReg unavailable: {PHREG_IMPORT_ERROR}"
        return out
    if len(dat) < 50 or int(dat["event"].sum()) < 10:
        out["status_detail"] = "Too few rows/events for Cox fit."
        return out

    exog, order = cox_exog(dat, predictor, covariates)
    keep = exog.notna().all(axis=1)
    dat = dat.loc[keep].copy()
    exog = exog.loc[keep].copy()
    out["n"] = int(len(dat))
    out["n_events"] = int(dat["event"].sum()) if len(dat) else 0
    if len(dat) < 50 or int(dat["event"].sum()) < 10:
        out["status_detail"] = "Insufficient rows/events after complete-case filtering."
        return out
    try:
        fit = PHReg(
            endog=dat["followup_days"],
            exog=exog,
            status=dat["event"],
            strata=dat["cohort"] if group == "overall_stratified" else None,
            ties="breslow",
        ).fit(disp=0)
        idx = order.index(predictor if predictor == REM_PREDOMINANT else f"{predictor}_z")
        beta = float(fit.params[idx])
        ci = fit.conf_int()[idx]
        out["status"] = "ok"
        out["hr"] = float(np.exp(beta))
        out["ci_lower"] = float(np.exp(ci[0]))
        out["ci_upper"] = float(np.exp(ci[1]))
        out["p_value"] = float(fit.pvalues[idx])
        return out
    except Exception as exc:  # pragma: no cover - data dependent
        out["status"] = "fit_error"
        out["status_detail"] = f"{type(exc).__name__}: {exc}"
        return out


def run_mortality_models(selected: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = {"overall_stratified": selected}
    for cohort, cohort_df in selected.groupby("cohort"):
        groups[str(cohort)] = cohort_df.copy()
    for group, frame in groups.items():
        for predictor in PREDICTORS:
            for model_name, model_label, covariates in MORTALITY_MODELS:
                row = fit_cox(frame, predictor, model_name, model_label, covariates, group)
                rows.append(row)
    results = pd.DataFrame(rows).sort_values(["analysis_group", "predictor", "model"])
    results.to_csv(outdir / "mortality_cox_model_results.csv", index=False)
    return results


def build_cognition_cohort(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    cog_any = df[COGNITION_OUTCOMES].notna().any(axis=1)
    base = cog_any & df["age"].notna() & df["sex"].notna() & df["rem_nrem_bsi_ready"]
    selected = select_one_row_per_subject(df, base)
    selected["analysis_cohort"] = "cross_sectional_rem_nrem_cognition_available"
    selected.to_csv(outdir / "cognition_selection_one_row_per_subject.csv", index=False)
    return selected


def fit_ols(dat: pd.DataFrame, predictor: str, outcome: str, model_name: str, model_label: str, covariates: list[str]) -> dict[str, object]:
    required = [predictor, outcome, "age", "sex", "cohort"] + (["bmi"] if "bmi" in covariates else [])
    sub = dat[required].dropna().copy()
    out = {
        "predictor": predictor,
        "predictor_label": PREDICTORS[predictor],
        "outcome": outcome,
        "model": model_name,
        "model_label": model_label,
        "n": int(len(sub)),
        "status": "not_run",
        "status_detail": "",
        "effect_scale": "standardized beta" if predictor != REM_PREDOMINANT else "standardized outcome beta for flag",
        "beta": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
    }
    if len(sub) < 25:
        out["status_detail"] = "Too few complete rows for OLS fit (<25)."
        return out
    sub["outcome_z"] = zscore(sub[outcome], ddof=1)
    sub["predictor_term"] = safe_numeric(sub[predictor]) if predictor == REM_PREDOMINANT else zscore(sub[predictor], ddof=1)
    sub = sub.dropna(subset=["outcome_z", "predictor_term"])
    out["n"] = int(len(sub))
    if len(sub) < 25:
        out["status_detail"] = "Too few rows after standardization."
        return out
    formula = "outcome_z ~ predictor_term"
    if covariates:
        formula += " + " + " + ".join(covariates)
    try:
        fit = smf.ols(formula, data=sub).fit()
        ci = fit.conf_int().loc["predictor_term"]
        out["status"] = "ok"
        out["beta"] = float(fit.params["predictor_term"])
        out["ci_lower"] = float(ci[0])
        out["ci_upper"] = float(ci[1])
        out["p_value"] = float(fit.pvalues["predictor_term"])
        out["r_squared"] = float(fit.rsquared)
        return out
    except Exception as exc:  # pragma: no cover - data dependent
        out["status"] = "fit_error"
        out["status_detail"] = f"{type(exc).__name__}: {exc}"
        return out


def run_cognition_models(selected: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for predictor in PREDICTORS:
        for outcome in COGNITION_OUTCOMES:
            for model_name, model_label, covariates, _require_bmi in COGNITION_MODELS:
                rows.append(fit_ols(selected, predictor, outcome, model_name, model_label, covariates))
    results = pd.DataFrame(rows).sort_values(["outcome", "predictor", "model"])
    results.to_csv(outdir / "cognition_ols_model_results.csv", index=False)
    return results


def build_counts(master: pd.DataFrame, mortality: pd.DataFrame, cognition: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, frame in [
        ("master_rows", master),
        ("strict_primary_rem_nrem_mortality_selected", mortality),
        ("cross_sectional_rem_nrem_cognition_selected", cognition),
    ]:
        row = {
            "analysis_set": label,
            "n_rows": int(len(frame)),
            "n_subjects_cohort_sid": int(frame[["cohort", "sid"]].drop_duplicates().shape[0]) if {"cohort", "sid"}.issubset(frame.columns) else np.nan,
            "n_events": int(frame["event"].sum()) if "event" in frame.columns else np.nan,
            "n_rem_nrem_ready": int(frame["rem_nrem_bsi_ready"].sum()) if "rem_nrem_bsi_ready" in frame.columns else np.nan,
            "n_with_bmi": int(frame["bmi"].notna().sum()) if "bmi" in frame.columns else np.nan,
            "cohorts": ", ".join(sorted(frame["cohort"].dropna().unique())) if "cohort" in frame.columns else "",
        }
        rows.append(row)
    counts = pd.DataFrame(rows)
    counts.to_csv(outdir / "analysis_set_counts.csv", index=False)

    cohort_rows = []
    for label, frame in [("mortality", mortality), ("cognition", cognition)]:
        for cohort, grp in frame.groupby("cohort"):
            cohort_rows.append(
                {
                    "analysis_set": label,
                    "cohort": cohort,
                    "n": int(len(grp)),
                    "n_events": int(grp["event"].sum()) if "event" in grp.columns else np.nan,
                    "n_with_bmi": int(grp["bmi"].notna().sum()),
                    "n_cog_fluid": int(grp["cog_fluid"].notna().sum()) if "cog_fluid" in grp.columns else np.nan,
                    "n_cog_crystallized": int(grp["cog_crystallized"].notna().sum()) if "cog_crystallized" in grp.columns else np.nan,
                    "n_cog_total": int(grp["cog_total"].notna().sum()) if "cog_total" in grp.columns else np.nan,
                }
            )
    by_cohort = pd.DataFrame(cohort_rows)
    by_cohort.to_csv(outdir / "analysis_set_counts_by_cohort.csv", index=False)
    return counts


def fmt_effect(row: pd.Series, effect_col: str) -> str:
    if row.get("status") != "ok":
        return "not fit"
    return f"{row[effect_col]:.3g} ({row['ci_lower']:.3g}-{row['ci_upper']:.3g}), p={row['p_value']:.3g}, n={int(row['n'])}"


def write_note(outdir: Path, master_path: Path, counts: pd.DataFrame, mortality_results: pd.DataFrame, cognition_results: pd.DataFrame) -> None:
    mortality_row = counts[counts["analysis_set"].eq("strict_primary_rem_nrem_mortality_selected")].iloc[0]
    cognition_row = counts[counts["analysis_set"].eq("cross_sectional_rem_nrem_cognition_selected")].iloc[0]
    primary_mort = mortality_results[
        mortality_results["analysis_group"].eq("overall_stratified")
        & mortality_results["model"].eq("m4_age_sex_bmi_ahi_hypoxic_burden")
    ].copy()
    primary_cog = cognition_results[
        cognition_results["model"].eq("age_sex_bmi")
        & cognition_results["outcome"].eq("cog_total")
    ].copy()

    lines = [
        "# R2-2 REM/NREM Outcome Analysis Note",
        "",
        f"Source master: `{master_path.relative_to(REPO_ROOT)}`",
        "",
        "## Exact analysis Ns",
        "",
        f"- Strict primary REM/NREM mortality-ready one-row-per-subject cohort: `{int(mortality_row['n_rows']):,}` subjects, `{int(mortality_row['n_events']):,}` deaths, cohorts `{mortality_row['cohorts']}`.",
        f"- Cross-sectional REM/NREM cognition cohort: `{int(cognition_row['n_rows']):,}` subjects/rows, cohorts `{cognition_row['cohorts']}`.",
        f"- Minimum stage support for REM/NREM predictors: REM >= `{MIN_STAGE_MINUTES:g}` minutes and NREM >= `{MIN_STAGE_MINUTES:g}` minutes, with both REM and NREM median BSI nonmissing.",
        "",
        "## Mortality interpretation",
        "",
        "Overall stratified Cox models use the strict primary mortality cohort and report REM/NREM effects per SD, except the REM-predominant flag, which contrasts REM>NREM against REM<=NREM. The reviewer-facing primary adjusted model is age, sex, BMI, AHI, and hypoxic burden when complete cases are available.",
        "",
    ]
    for _, row in primary_mort.iterrows():
        lines.append(f"- `{row['predictor']}`: HR {fmt_effect(row, 'hr')}.")

    lines.extend(
        [
            "",
            "## Cognition interpretation",
            "",
            "Cognition models are cross-sectional OLS models for available cognitive composites. Effects are standardized outcome differences per SD predictor, except the REM-predominant flag, which is a binary contrast. Age/sex and age/sex/BMI models include cohort fixed effects where estimable.",
            "",
        ]
    )
    for _, row in primary_cog.iterrows():
        lines.append(f"- `cog_total`, `{row['predictor']}`: beta {fmt_effect(row, 'beta')}.")

    lines.extend(
        [
            "",
            "## Disease outcome timing",
            "",
            "Disease outcome modeling is not promoted in this R2-2 REM/NREM outcome package. The source master supports cross-sectional cognition and mortality follow-up, but disease endpoints do not have uniformly timing-safe incident-event definitions relative to PSG in this analysis table.",
        ]
    )
    (outdir / "rem_nrem_outcomes_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    master = read_master(args.master_csv)
    mortality = build_mortality_cohort(master, args.output_dir)
    mortality_results = run_mortality_models(mortality, args.output_dir)
    cognition = build_cognition_cohort(master, args.output_dir)
    cognition_results = run_cognition_models(cognition, args.output_dir)
    counts = build_counts(master, mortality, cognition, args.output_dir)
    write_note(args.output_dir, args.master_csv, counts, mortality_results, cognition_results)


if __name__ == "__main__":
    main()
