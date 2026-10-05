#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
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
DEFAULT_OUTDIR = REPO_ROOT / "REVISION" / "mortality_v2"
DEFAULT_MGH_DISEASE = Path("/home/wolfgang/repos/sleep_cognition/disease_groups/bdsp_disease_cohorts_sex-age.csv")
DEFAULT_BIDMC_SURVIVAL = Path("/home/wolfgang/repos/sleep_cognition/survival_analysis/data_bidmc_survival.csv")
DEFAULT_MROS_TABLE = Path("/home/wolfgang/repos/sleep_cognition/step0a_prepare_data_cohorts/mros_table_all.csv")

COMMON_REQUIRED = [
    "followup_days",
    "vital_status",
    "age",
    "sex",
    "bsi_median_sleep",
    "ss_percent_main",
    "ahi",
    "hypoxic_burden",
]

EXPOSURE_SPECS = {
    "bsi_median_sleep": {
        "label": "BSI median sleep",
        "family": "raw_bsi",
        "primary": True,
    },
    "bsi_quantile_75_sleep": {
        "label": "BSI quantile 75 sleep",
        "family": "raw_bsi",
        "primary": False,
    },
    "bsi_instability_burden_gt_1p5_sleep": {
        "label": "BSI instability burden >1.5 sleep",
        "family": "raw_bsi",
        "primary": False,
    },
    "bsi_quantile_90_sleep": {
        "label": "BSI quantile 90 sleep",
        "family": "raw_bsi_deemphasized",
        "primary": False,
    },
    "ss_percent_main": {
        "label": "Self-similarity percent main",
        "family": "comparator",
        "primary": True,
    },
    "ahi": {
        "label": "AHI",
        "family": "comparator",
        "primary": True,
    },
    "hypoxic_burden": {
        "label": "Hypoxic burden",
        "family": "comparator",
        "primary": True,
    },
}

BASE_MODEL_SPECS = [
    {
        "model_name": "m0_unadjusted",
        "covariates": [],
        "label": "Unadjusted",
    },
    {
        "model_name": "m1_age_sex_bmi",
        "covariates": ["age", "sex", "bmi"],
        "label": "Age + sex + BMI",
    },
    {
        "model_name": "m2_age_sex_bmi_cardiometabolic",
        "covariates": [
            "age",
            "sex",
            "bmi",
            "cov_hypertension",
            "cov_diabetesii",
            "cov_myocardial_infarction",
            "cov_atrial_fibrillation",
        ],
        "label": "Age + sex + BMI + cardiometabolic covariates",
    },
]

RAW_BSI_ADD_ON_MODEL_SPECS = [
    {
        "model_name": "m3_plus_ahi",
        "covariates": [
            "age",
            "sex",
            "bmi",
            "cov_hypertension",
            "cov_diabetesii",
            "cov_myocardial_infarction",
            "cov_atrial_fibrillation",
            "ahi",
        ],
        "label": "Model 2 + AHI",
    },
    {
        "model_name": "m4_plus_hypoxic_burden",
        "covariates": [
            "age",
            "sex",
            "bmi",
            "cov_hypertension",
            "cov_diabetesii",
            "cov_myocardial_infarction",
            "cov_atrial_fibrillation",
            "hypoxic_burden",
        ],
        "label": "Model 2 + hypoxic burden",
    },
    {
        "model_name": "m5_plus_arousal_index",
        "covariates": [
            "age",
            "sex",
            "bmi",
            "cov_hypertension",
            "cov_diabetesii",
            "cov_myocardial_infarction",
            "cov_atrial_fibrillation",
            "arousal_index",
        ],
        "label": "Model 2 + arousal index",
    },
    {
        "model_name": "m6_plus_ss",
        "covariates": [
            "age",
            "sex",
            "bmi",
            "cov_hypertension",
            "cov_diabetesii",
            "cov_myocardial_infarction",
            "cov_atrial_fibrillation",
            "ss_percent_main",
        ],
        "label": "Model 2 + self-similarity",
    },
    {
        "model_name": "m7_plus_desaturation_index",
        "covariates": [
            "age",
            "sex",
            "bmi",
            "cov_hypertension",
            "cov_diabetesii",
            "cov_myocardial_infarction",
            "cov_atrial_fibrillation",
            "desaturation_index",
        ],
        "label": "Model 2 + desaturation index",
    },
]

MODE_SPECS = {
    "strict_primary": {
        "label": "Strict primary: diagnostic/untreated PSG only",
        "description": "paper_primary_psg_eligible == True",
    },
    "broad_sleep_unspecified": {
        "label": "Sensitivity: strict primary plus sleep_unspecified",
        "description": "paper_primary_psg_eligible == True OR paper_psg_type == sleep_unspecified",
    },
    "all_study_type_sensitivity": {
        "label": "Sensitivity: all mortality-ready study types",
        "description": "All mortality-ready rows; subject selection prioritizes strict primary, then sleep_unspecified, split-night, PAP/titration, and other types.",
    },
}

SURVIVAL_CURVE_EXPOSURES = ["bsi_median_sleep", "ahi", "hypoxic_burden", "ss_percent_main"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final revision mortality analyses from v4_paper.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--mgh-disease-csv", type=Path, default=DEFAULT_MGH_DISEASE)
    parser.add_argument("--bidmc-survival-csv", type=Path, default=DEFAULT_BIDMC_SURVIVAL)
    parser.add_argument("--mros-table-csv", type=Path, default=DEFAULT_MROS_TABLE)
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_binary(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    mapped = as_str.map(
        {
            "1": 1.0,
            "1.0": 1.0,
            "true": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "dead": 1.0,
            "0": 0.0,
            "0.0": 0.0,
            "false": 0.0,
            "no": 0.0,
            "n": 0.0,
            "alive": 0.0,
            "nan": np.nan,
            "none": np.nan,
            "": np.nan,
        }
    )
    numeric = safe_numeric(series)
    return numeric.combine_first(mapped)


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


def extract_mros_token(fileid: object) -> str | None:
    if fileid is None or (isinstance(fileid, float) and pd.isna(fileid)):
        return None
    match = re.search(r"(bi\d{4}|sd\d{4})", str(fileid).lower())
    return match.group(1) if match else None


def read_csv_available(path: Path, requested_cols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in requested_cols if col in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in requested_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_master(path: Path) -> pd.DataFrame:
    columns = [
        "fileid",
        "sid",
        "cohort",
        "visitno",
        "age",
        "sex",
        "bmi",
        "study_type",
        "psg_type",
        "paper_psg_type",
        "paper_primary_psg_eligible",
        "ahi",
        "hypoxic_burden",
        "arousal_index",
        "desaturation_index",
        "bsi_median_sleep",
        "bsi_quantile_75_sleep",
        "bsi_quantile_90_sleep",
        "bsi_instability_burden_gt_1p5_sleep",
        "ss_percent_main",
        "followup_days",
        "vital_status",
        "death_date",
        "censor_date",
    ]
    df = read_csv_available(path, columns)
    df["sid"] = df["sid"].astype(str)
    df["cohort"] = df["cohort"].astype(str)
    df["fileid"] = df["fileid"].astype(str)
    df["age"] = safe_numeric(df["age"])
    df["sex"] = safe_sex(df["sex"])
    df["bmi"] = safe_numeric(df["bmi"])
    for col in [
        "ahi",
        "hypoxic_burden",
        "arousal_index",
        "desaturation_index",
        "bsi_median_sleep",
        "bsi_quantile_75_sleep",
        "bsi_quantile_90_sleep",
        "bsi_instability_burden_gt_1p5_sleep",
        "ss_percent_main",
        "followup_days",
    ]:
        df[col] = safe_numeric(df[col])
    df["vital_status"] = df["vital_status"].astype("string")
    df["event"] = df["vital_status"].str.lower().eq("dead").astype(float)
    df["visitno_numeric"] = safe_numeric(df["visitno"])
    df["visit_sort"] = df["visitno_numeric"].fillna(999999)
    df["paper_psg_type"] = df["paper_psg_type"].astype("string").str.strip().str.lower()
    df["paper_primary_psg_eligible_bool"] = safe_bool(df["paper_primary_psg_eligible"])
    df["common_core_ready"] = df[COMMON_REQUIRED].notna().all(axis=1)
    df["mros_sid_token"] = df["fileid"].map(extract_mros_token)
    return df


def merge_covariates(df: pd.DataFrame, mgh_disease_csv: Path, bidmc_survival_csv: Path, mros_table_csv: Path) -> pd.DataFrame:
    out = df.copy()
    cov_cols = ["cov_hypertension", "cov_diabetesii", "cov_myocardial_infarction", "cov_atrial_fibrillation"]
    for column in cov_cols:
        out[column] = np.nan

    if mgh_disease_csv.exists():
        mgh = pd.read_csv(
            mgh_disease_csv,
            usecols=[
                "fileid",
                "dx-cs-hypertension",
                "dx-cs-diabetesii",
                "dx-cs-myocardial_infarction",
                "dx-cs-atrial_fibrillation",
            ],
        ).drop_duplicates("fileid")
        mgh["fileid"] = mgh["fileid"].str.replace("sub-s0001", "sub-S0001", regex=False)
        mgh = mgh.rename(
            columns={
                "dx-cs-hypertension": "cov_hypertension",
                "dx-cs-diabetesii": "cov_diabetesii",
                "dx-cs-myocardial_infarction": "cov_myocardial_infarction",
                "dx-cs-atrial_fibrillation": "cov_atrial_fibrillation",
            }
        )
        for column in cov_cols:
            mgh[column] = safe_binary(mgh[column])
        out = out.merge(mgh, on="fileid", how="left", suffixes=("", "_mgh"))
        for column in cov_cols:
            merged_col = f"{column}_mgh"
            out[column] = out[column].combine_first(out[merged_col])
            out = out.drop(columns=[merged_col])

    if bidmc_survival_csv.exists():
        bidmc = pd.read_csv(
            bidmc_survival_csv,
            usecols=[
                "fileid",
                "head_dx-tm-hypertension",
                "head_dx-tm-diabetesii",
                "head_dx-tm-myocardial_infarction",
                "head_dx-tm-atrial_fibrillation",
            ],
        ).drop_duplicates("fileid")
        bidmc = bidmc.rename(
            columns={
                "head_dx-tm-hypertension": "cov_hypertension",
                "head_dx-tm-diabetesii": "cov_diabetesii",
                "head_dx-tm-myocardial_infarction": "cov_myocardial_infarction",
                "head_dx-tm-atrial_fibrillation": "cov_atrial_fibrillation",
            }
        )
        for column in cov_cols:
            bidmc[column] = safe_binary(bidmc[column])
        out = out.merge(bidmc, on="fileid", how="left", suffixes=("", "_bidmc"))
        for column in cov_cols:
            merged_col = f"{column}_bidmc"
            out[column] = out[column].combine_first(out[merged_col])
            out = out.drop(columns=[merged_col])

    if mros_table_csv.exists():
        mros = pd.read_csv(mros_table_csv, usecols=["sid", "mhbp", "mhdiab", "mhmi"]).drop_duplicates("sid")
        mros["sid"] = mros["sid"].astype(str)
        mros = mros.rename(
            columns={
                "sid": "mros_sid_token",
                "mhbp": "cov_hypertension",
                "mhdiab": "cov_diabetesii",
                "mhmi": "cov_myocardial_infarction",
            }
        )
        mros["cov_hypertension"] = safe_binary(mros["cov_hypertension"])
        mros["cov_diabetesii"] = safe_binary(mros["cov_diabetesii"])
        mros["cov_myocardial_infarction"] = safe_binary(mros["cov_myocardial_infarction"])
        mros["cov_atrial_fibrillation"] = np.nan
        out = out.merge(mros, on="mros_sid_token", how="left", suffixes=("", "_mros"))
        for column in cov_cols:
            merged_col = f"{column}_mros"
            out[column] = out[column].combine_first(out[merged_col])
            out = out.drop(columns=[merged_col])

    return out


def filter_mode_candidates(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df[df["common_core_ready"]].copy()
    if mode == "strict_primary":
        out = out[out["paper_primary_psg_eligible_bool"].eq(True)].copy()
    elif mode == "broad_sleep_unspecified":
        out = out[
            out["paper_primary_psg_eligible_bool"].eq(True) | out["paper_psg_type"].eq("sleep_unspecified")
        ].copy()
    elif mode == "all_study_type_sensitivity":
        out = out[out["paper_psg_type"].notna()].copy()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported mode: {mode}")

    out["selection_mode"] = mode
    out["mode_label"] = MODE_SPECS[mode]["label"]
    priority_map = {
        "diagnostic": 0,
        "sleep_unspecified": 1,
        "split_night": 2,
        "pap_titration": 3,
        "other_unclear": 4,
        "hsat": 5,
        "mslt": 6,
    }
    out["selection_priority_psg_type"] = np.where(
        out["paper_primary_psg_eligible_bool"].eq(True),
        0,
        out["paper_psg_type"].map(priority_map).fillna(9),
    )
    out["selection_reason"] = np.select(
        [
            out["paper_primary_psg_eligible_bool"].eq(True).fillna(False).to_numpy(dtype=bool),
            out["paper_psg_type"].eq("sleep_unspecified").fillna(False).to_numpy(dtype=bool),
            out["paper_psg_type"].eq("split_night").fillna(False).to_numpy(dtype=bool),
            out["paper_psg_type"].eq("pap_titration").fillna(False).to_numpy(dtype=bool),
        ],
        [
            "paper_primary_psg_eligible",
            "sleep_unspecified_sensitivity",
            "split_night_sensitivity",
            "pap_titration_sensitivity",
        ],
        default="other_study_type_sensitivity",
    )
    return out


def select_one_row_per_subject(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = filter_mode_candidates(df, mode)
    candidates = candidates.sort_values(
        ["cohort", "sid", "selection_priority_psg_type", "visit_sort", "fileid"],
        kind="mergesort",
    ).copy()
    candidates["selection_rank_within_subject"] = candidates.groupby(["cohort", "sid"]).cumcount() + 1
    selected = candidates[candidates["selection_rank_within_subject"].eq(1)].copy()
    selected["selected_one_row_per_subject"] = True
    return candidates, selected


def build_model_specs_for_exposure(exposure: str) -> list[dict[str, object]]:
    specs = list(BASE_MODEL_SPECS)
    if EXPOSURE_SPECS[exposure]["family"].startswith("raw_bsi"):
        specs.extend(RAW_BSI_ADD_ON_MODEL_SPECS)
    return specs


def write_formula_catalog(outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, mode_spec in MODE_SPECS.items():
        for exposure, exposure_spec in EXPOSURE_SPECS.items():
            for model_spec in build_model_specs_for_exposure(exposure):
                covariates = [exposure] + list(model_spec["covariates"])
                rows.append(
                    {
                        "selection_mode": mode,
                        "mode_label": mode_spec["label"],
                        "selection_description": mode_spec["description"],
                        "analysis_exposure": exposure,
                        "analysis_exposure_label": exposure_spec["label"],
                        "exposure_family": exposure_spec["family"],
                        "model_name": model_spec["model_name"],
                        "model_label": model_spec["label"],
                        "formula_like": "Surv(followup_days, event) ~ " + " + ".join(covariates),
                        "covariates": ", ".join(covariates),
                    }
                )
    catalog = pd.DataFrame(rows)
    catalog.to_csv(outdir / "model_formula_catalog.csv", index=False)
    return catalog


def attach_model_flags(df: pd.DataFrame, exposure: str) -> pd.DataFrame:
    out = df.copy()
    for model_spec in build_model_specs_for_exposure(exposure):
        required = ["followup_days", "event", exposure] + list(model_spec["covariates"])
        flag_name = f"complete_for_{model_spec['model_name']}"
        out[flag_name] = out[required].notna().all(axis=1)
    return out


def write_model_ready_datasets(selected_frames: dict[str, pd.DataFrame], outdir: Path) -> None:
    for mode, frame in selected_frames.items():
        for exposure in EXPOSURE_SPECS:
            dataset = attach_model_flags(frame, exposure)
            dataset["analysis_exposure"] = exposure
            dataset["analysis_exposure_label"] = EXPOSURE_SPECS[exposure]["label"]
            filename = f"model_ready__{mode}__{exposure}.csv"
            dataset.to_csv(outdir / filename, index=False)


def build_completeness_audits(selected_frames: dict[str, pd.DataFrame], outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []

    for mode, frame in selected_frames.items():
        for exposure in EXPOSURE_SPECS:
            with_flags = attach_model_flags(frame, exposure)
            model_specs = build_model_specs_for_exposure(exposure)
            groups = {"overall": with_flags}
            for cohort, cohort_df in with_flags.groupby("cohort"):
                groups[str(cohort)] = cohort_df
            for group_name, group_df in groups.items():
                row = {
                    "selection_mode": mode,
                    "analysis_group": group_name,
                    "analysis_exposure": exposure,
                    "n_subjects": int(len(group_df)),
                    "n_events": int(group_df["event"].sum()),
                    "with_bmi": int(group_df["bmi"].notna().sum()),
                    "with_hypertension": int(group_df["cov_hypertension"].notna().sum()),
                    "with_diabetesii": int(group_df["cov_diabetesii"].notna().sum()),
                    "with_myocardial_infarction": int(group_df["cov_myocardial_infarction"].notna().sum()),
                    "with_atrial_fibrillation": int(group_df["cov_atrial_fibrillation"].notna().sum()),
                    "strict_eligible_rows": int(group_df["paper_primary_psg_eligible_bool"].eq(True).sum()),
                    "sleep_unspecified_rows": int(group_df["paper_psg_type"].eq("sleep_unspecified").sum()),
                }
                for model_spec in model_specs:
                    row[f"n_{model_spec['model_name']}"] = int(group_df[f"complete_for_{model_spec['model_name']}"].sum())
                if group_name == "overall":
                    overall_rows.append(row)
                else:
                    cohort_rows.append(row)

    overall = pd.DataFrame(overall_rows).sort_values(["selection_mode", "analysis_exposure"])
    by_cohort = pd.DataFrame(cohort_rows).sort_values(["selection_mode", "analysis_group", "analysis_exposure"])
    overall.to_csv(outdir / "covariate_completeness_overall.csv", index=False)
    by_cohort.to_csv(outdir / "covariate_completeness_by_cohort.csv", index=False)
    return overall, by_cohort


def zscore_series(series: pd.Series) -> pd.Series:
    values = safe_numeric(series)
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return (values - mean) / std


def build_exog_matrix(df: pd.DataFrame, exposure: str, covariates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    exog = pd.DataFrame(index=df.index)
    order: list[str] = []

    exposure_name = f"{exposure}_z"
    exog[exposure_name] = zscore_series(df[exposure])
    order.append(exposure_name)

    z_covariates = {"age", "bmi", "ahi", "hypoxic_burden", "arousal_index", "desaturation_index", "ss_percent_main"}
    for covariate in covariates:
        if covariate == "sex":
            exog["sex"] = safe_numeric(df["sex"])
            order.append("sex")
        else:
            name = f"{covariate}_z" if covariate in z_covariates else covariate
            if name.endswith("_z"):
                exog[name] = zscore_series(df[covariate])
            else:
                exog[name] = safe_numeric(df[covariate])
            order.append(name)
    return exog[order], order


def ph_diagnostic_from_fit(fit: object, dat: pd.DataFrame, exposure_idx: int) -> tuple[float, float, int, str]:
    try:
        schoenfeld = np.asarray(fit.schoenfeld_residuals)
        resid = pd.Series(schoenfeld[:, exposure_idx], index=dat.index)
        mask = dat["event"].eq(1) & resid.notna() & dat["followup_days"].gt(0)
        n_events = int(mask.sum())
        if n_events < 10:
            return np.nan, np.nan, n_events, "too_few_event_residuals"
        rho, p_value = stats.spearmanr(np.log(dat.loc[mask, "followup_days"]), resid.loc[mask])
        return float(rho), float(p_value), n_events, "spearman_schoenfeld_log_time"
    except Exception as exc:  # pragma: no cover - depends on statsmodels internals
        return np.nan, np.nan, 0, f"diagnostic_error: {type(exc).__name__}: {exc}"


def fit_phreg(
    df: pd.DataFrame,
    exposure: str,
    model_spec: dict[str, object],
    analysis_group: str,
) -> dict[str, object]:
    required = ["followup_days", "event", exposure] + list(model_spec["covariates"])
    dat = df[required + ["cohort"]].copy()
    dat = dat.dropna(subset=required)

    result = {
        "analysis_group": analysis_group,
        "analysis_exposure": exposure,
        "analysis_exposure_label": EXPOSURE_SPECS[exposure]["label"],
        "exposure_family": EXPOSURE_SPECS[exposure]["family"],
        "model_name": model_spec["model_name"],
        "model_label": model_spec["label"],
        "n": int(len(dat)),
        "n_events": int(dat["event"].sum()) if len(dat) else 0,
        "status": "not_run",
        "status_detail": "",
        "hr_per_sd": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "ph_test": "",
        "ph_spearman_rho_log_time": np.nan,
        "ph_p_value": np.nan,
        "ph_n_event_residuals": 0,
    }

    if not HAVE_PHREG:
        result["status_detail"] = f"PHReg unavailable: {PHREG_IMPORT_ERROR}"
        return result
    if len(dat) < 50:
        result["status_detail"] = "Too few rows for Cox fit (<50)."
        return result
    if int(dat["event"].sum()) < 10:
        result["status_detail"] = "Too few events for Cox fit (<10)."
        return result

    exog, exog_order = build_exog_matrix(dat, exposure, list(model_spec["covariates"]))
    keep = exog.notna().all(axis=1)
    dat = dat.loc[keep].copy()
    exog = exog.loc[keep].copy()
    result["n"] = int(len(dat))
    result["n_events"] = int(dat["event"].sum()) if len(dat) else 0
    if len(dat) < 50 or int(dat["event"].sum()) < 10:
        result["status_detail"] = "Insufficient rows/events after standardization and complete-case filtering."
        return result

    try:
        strata = dat["cohort"] if analysis_group == "overall_stratified" else None
        fit = PHReg(
            endog=dat["followup_days"],
            exog=exog,
            status=dat["event"],
            strata=strata,
            ties="breslow",
        ).fit(disp=0)
        idx = exog_order.index(f"{exposure}_z")
        beta = float(fit.params[idx])
        ci = fit.conf_int()[idx]
        p_value = float(fit.pvalues[idx])
        rho, ph_p, ph_n, ph_test = ph_diagnostic_from_fit(fit, dat, idx)
        result["status"] = "ok"
        result["hr_per_sd"] = float(np.exp(beta))
        result["ci_lower"] = float(np.exp(ci[0]))
        result["ci_upper"] = float(np.exp(ci[1]))
        result["p_value"] = p_value
        result["ph_test"] = ph_test
        result["ph_spearman_rho_log_time"] = rho
        result["ph_p_value"] = ph_p
        result["ph_n_event_residuals"] = ph_n
        return result
    except Exception as exc:  # pragma: no cover - depends on data realization
        result["status"] = "fit_error"
        result["status_detail"] = f"{type(exc).__name__}: {exc}"
        return result


def run_models(selected_frames: dict[str, pd.DataFrame], outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, frame in selected_frames.items():
        groups: dict[str, pd.DataFrame] = {"overall_stratified": frame}
        for cohort, cohort_df in frame.groupby("cohort"):
            groups[str(cohort)] = cohort_df.copy()
        for group_name, group_df in groups.items():
            for exposure in EXPOSURE_SPECS:
                for model_spec in build_model_specs_for_exposure(exposure):
                    row = fit_phreg(group_df, exposure, model_spec, group_name)
                    row["selection_mode"] = mode
                    row["mode_label"] = MODE_SPECS[mode]["label"]
                    row["selection_description"] = MODE_SPECS[mode]["description"]
                    rows.append(row)
    results = pd.DataFrame(rows).sort_values(["selection_mode", "analysis_group", "analysis_exposure", "model_name"])
    results.to_csv(outdir / "cox_model_results.csv", index=False)
    return results


def write_selection_outputs(candidate_frames: dict[str, pd.DataFrame], selected_frames: dict[str, pd.DataFrame], outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.concat(candidate_frames.values(), ignore_index=True).sort_values(
        ["selection_mode", "cohort", "sid", "selection_rank_within_subject", "fileid"]
    )
    selected = pd.concat(selected_frames.values(), ignore_index=True).sort_values(
        ["selection_mode", "cohort", "sid", "fileid"]
    )
    candidates.to_csv(outdir / "selection_candidate_rows.csv", index=False)
    selected.to_csv(outdir / "selection_one_row_per_subject.csv", index=False)
    return candidates, selected


def km_curve(group: pd.DataFrame) -> pd.DataFrame:
    dat = group[["followup_days", "event"]].dropna().copy()
    dat = dat[dat["followup_days"].ge(0)].sort_values("followup_days")
    if dat.empty:
        return pd.DataFrame()
    rows = [{"time_days": 0.0, "survival": 1.0, "n_risk": int(len(dat)), "n_events_at_time": 0, "n_censored_at_time": 0}]
    survival = 1.0
    for time, at_time in dat.groupby("followup_days", sort=True):
        n_risk = int((dat["followup_days"] >= time).sum())
        n_events = int(at_time["event"].eq(1).sum())
        n_censored = int(at_time["event"].eq(0).sum())
        if n_risk > 0 and n_events > 0:
            survival *= 1.0 - (n_events / n_risk)
        rows.append(
            {
                "time_days": float(time),
                "survival": float(survival),
                "n_risk": n_risk,
                "n_events_at_time": n_events,
                "n_censored_at_time": n_censored,
            }
        )
    return pd.DataFrame(rows)


def assign_tertiles(series: pd.Series) -> pd.Series:
    values = safe_numeric(series)
    try:
        return pd.qcut(values, q=3, labels=["T1_low", "T2_mid", "T3_high"], duplicates="drop")
    except ValueError:
        return pd.Series(pd.NA, index=series.index, dtype="object")


def build_tertile_survival(selected_frames: dict[str, pd.DataFrame], outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for mode, frame in selected_frames.items():
        for exposure in SURVIVAL_CURVE_EXPOSURES:
            dat = frame[["cohort", "sid", "followup_days", "event", exposure]].dropna().copy()
            dat = dat[dat["followup_days"].gt(0)].copy()
            dat["tertile"] = assign_tertiles(dat[exposure])
            dat = dat[dat["tertile"].notna()].copy()
            for tertile, grp in dat.groupby("tertile", observed=True):
                exposure_values = safe_numeric(grp[exposure])
                summary_rows.append(
                    {
                        "selection_mode": mode,
                        "analysis_exposure": exposure,
                        "analysis_exposure_label": EXPOSURE_SPECS[exposure]["label"],
                        "tertile": str(tertile),
                        "n_subjects": int(len(grp)),
                        "n_events": int(grp["event"].sum()),
                        "exposure_min": float(exposure_values.min()),
                        "exposure_median": float(exposure_values.median()),
                        "exposure_max": float(exposure_values.max()),
                        "followup_median_days": float(grp["followup_days"].median()),
                    }
                )
                curve = km_curve(grp)
                if not curve.empty:
                    curve["selection_mode"] = mode
                    curve["analysis_exposure"] = exposure
                    curve["analysis_exposure_label"] = EXPOSURE_SPECS[exposure]["label"]
                    curve["tertile"] = str(tertile)
                    curve["n_subjects"] = int(len(grp))
                    curve["n_events"] = int(grp["event"].sum())
                    curve_rows.append(curve)

    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    curves.to_csv(outdir / "tertile_survival_source.csv", index=False)
    summary.to_csv(outdir / "tertile_survival_summary.csv", index=False)
    return curves, summary


def build_selection_summary(master: pd.DataFrame, selected_frames: dict[str, pd.DataFrame], outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, frame in selected_frames.items():
        rows.append(
            {
                "selection_mode": mode,
                "mode_label": MODE_SPECS[mode]["label"],
                "n_selected_subjects": int(len(frame)),
                "n_events": int(frame["event"].sum()),
                "n_cohorts": int(frame["cohort"].nunique()),
                "cohorts": ", ".join(sorted(frame["cohort"].dropna().astype(str).unique())),
                "n_strict_eligible": int(frame["paper_primary_psg_eligible_bool"].eq(True).sum()),
                "n_sleep_unspecified": int(frame["paper_psg_type"].eq("sleep_unspecified").sum()),
            }
        )
    for psg_type, grp in master.groupby("paper_psg_type", dropna=False):
        rows.append(
            {
                "selection_mode": "master_psg_type_inventory",
                "mode_label": str(psg_type),
                "n_selected_subjects": int(len(grp)),
                "n_events": int(grp["event"].sum()),
                "n_cohorts": int(grp["cohort"].nunique()),
                "cohorts": ", ".join(sorted(grp["cohort"].dropna().astype(str).unique())),
                "n_strict_eligible": int(grp["paper_primary_psg_eligible_bool"].eq(True).sum()),
                "n_sleep_unspecified": int(grp["paper_psg_type"].eq("sleep_unspecified").sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "selection_summary.csv", index=False)
    return out


def fmt_hr(row: pd.Series | None) -> str:
    if row is None or row.empty or row.get("status") != "ok":
        return "not estimable"
    return f"{row['hr_per_sd']:.2f} ({row['ci_lower']:.2f}-{row['ci_upper']:.2f}), p={row['p_value']:.3g}"


def write_note(outdir: Path, master_path: Path, selected: pd.DataFrame, cox_results: pd.DataFrame, curve_summary: pd.DataFrame) -> None:
    strict_selected = selected[selected["selection_mode"].eq("strict_primary")]
    broad_selected = selected[selected["selection_mode"].eq("broad_sleep_unspecified")]
    all_type_selected = selected[selected["selection_mode"].eq("all_study_type_sensitivity")]
    strict_bsi_m2 = cox_results[
        cox_results["selection_mode"].eq("strict_primary")
        & cox_results["analysis_group"].eq("overall_stratified")
        & cox_results["analysis_exposure"].eq("bsi_median_sleep")
        & cox_results["model_name"].eq("m2_age_sex_bmi_cardiometabolic")
    ]
    strict_ahi_m2 = cox_results[
        cox_results["selection_mode"].eq("strict_primary")
        & cox_results["analysis_group"].eq("overall_stratified")
        & cox_results["analysis_exposure"].eq("ahi")
        & cox_results["model_name"].eq("m2_age_sex_bmi_cardiometabolic")
    ]
    bsi_hr = fmt_hr(strict_bsi_m2.iloc[0]) if len(strict_bsi_m2) else "not estimable"
    ahi_hr = fmt_hr(strict_ahi_m2.iloc[0]) if len(strict_ahi_m2) else "not estimable"

    ok = int(cox_results["status"].eq("ok").sum()) if not cox_results.empty else 0
    total = int(len(cox_results))
    ph_flags = cox_results[
        cox_results["status"].eq("ok") & cox_results["ph_p_value"].notna() & cox_results["ph_p_value"].lt(0.05)
    ]

    lines = [
        "# Mortality v2 Status Note",
        "",
        f"Source master: `{master_path}`",
        "",
        "## Primary selection",
        f"- Strict primary one-row-per-subject rows: `{len(strict_selected):,}`",
        f"- Strict primary deaths: `{int(strict_selected['event'].sum()):,}`",
        f"- Strict primary cohorts: `{', '.join(sorted(strict_selected['cohort'].dropna().astype(str).unique()))}`",
        f"- Broad sleep-unspecified sensitivity rows: `{len(broad_selected):,}`",
        f"- Broad sleep-unspecified sensitivity deaths: `{int(broad_selected['event'].sum()):,}`",
        f"- All-study-type sensitivity rows: `{len(all_type_selected):,}`",
        f"- All-study-type sensitivity deaths: `{int(all_type_selected['event'].sum()):,}`",
        "",
        "## Primary overall stratified results",
        f"- `bsi_median_sleep`, model 2: `{bsi_hr}`",
        f"- `ahi`, model 2: `{ahi_hr}`",
        "",
        "## Model execution",
        f"- Cox result rows: `{total:,}`",
        f"- Successful Cox fits: `{ok:,}`",
        f"- PH diagnostic rows with exposure residual/log-time Spearman p<0.05: `{len(ph_flags):,}`",
        "",
        "## Figure-ready exports",
        f"- Tertile survival curve rows: `{len(curve_summary):,}` exposure-tertile summaries",
        "- Survival curve source data include BSI median, AHI, hypoxic burden, and SS.",
        "",
        "## Interpretation guardrails",
        "- The primary mortality cohort now uses `paper_primary_psg_eligible == True`; old provisional missing-psg-type assumptions are no longer used.",
        "- Split-night and PAP/titration rows are not part of the primary analysis; they are admitted only in the all-study-type sensitivity mode.",
        "- Cardiometabolic model complete-case Ns vary by cohort because atrial fibrillation remains unavailable for MrOS in the merged covariate source.",
        "- `bsi_quantile_90_sleep` is retained only as a sensitivity/de-emphasized raw BSI candidate because it failed the prior REM>NREM face-validity screen.",
    ]
    (outdir / "mortality_v2_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = load_master(args.master_csv)
    merged = merge_covariates(master, args.mgh_disease_csv, args.bidmc_survival_csv, args.mros_table_csv)

    candidate_frames: dict[str, pd.DataFrame] = {}
    selected_frames: dict[str, pd.DataFrame] = {}
    for mode in MODE_SPECS:
        candidates, selected = select_one_row_per_subject(merged, mode)
        candidate_frames[mode] = candidates
        selected_frames[mode] = selected

    _, selected = write_selection_outputs(candidate_frames, selected_frames, args.output_dir)
    build_selection_summary(merged, selected_frames, args.output_dir)
    completeness_overall, completeness_by_cohort = build_completeness_audits(selected_frames, args.output_dir)
    write_formula_catalog(args.output_dir)
    write_model_ready_datasets(selected_frames, args.output_dir)
    cox_results = run_models(selected_frames, args.output_dir)
    _, curve_summary = build_tertile_survival(selected_frames, args.output_dir)
    write_note(args.output_dir, args.master_csv, selected, cox_results, curve_summary)

    summary = {
        "rows_in_master": int(len(master)),
        "strict_selected_subject_rows": int(len(selected_frames["strict_primary"])),
        "strict_events": int(selected_frames["strict_primary"]["event"].sum()),
        "broad_selected_subject_rows": int(len(selected_frames["broad_sleep_unspecified"])),
        "broad_events": int(selected_frames["broad_sleep_unspecified"]["event"].sum()),
        "all_study_type_selected_subject_rows": int(len(selected_frames["all_study_type_sensitivity"])),
        "all_study_type_events": int(selected_frames["all_study_type_sensitivity"]["event"].sum()),
        "cox_rows": int(len(cox_results)),
        "cox_ok_rows": int(cox_results["status"].eq("ok").sum()) if not cox_results.empty else 0,
        "ph_flag_rows_p_lt_0p05": int(
            (
                cox_results["status"].eq("ok")
                & cox_results["ph_p_value"].notna()
                & cox_results["ph_p_value"].lt(0.05)
            ).sum()
        )
        if not cox_results.empty
        else 0,
        "strict_m2_bsi_complete_n": int(
            completeness_overall[
                completeness_overall["selection_mode"].eq("strict_primary")
                & completeness_overall["analysis_exposure"].eq("bsi_median_sleep")
            ]["n_m2_age_sex_bmi_cardiometabolic"].iloc[0]
        ),
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
