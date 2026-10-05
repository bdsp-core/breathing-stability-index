#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPO_ROOT / "REVISION" / "mortality_v2" / "selection_one_row_per_subject.csv"
DEFAULT_BSI_SCORE = (
    REPO_ROOT
    / "REVISION"
    / "mortality_bsi_score_cv_all_study_type_original_style_50fold_v1"
    / "bsi_score_oof_predictions.csv"
)
DEFAULT_OUTDIR = REPO_ROOT / "REVISION" / "mortality_comparator_all_study_type_v1"

SELECTION_MODE = "all_study_type_sensitivity"
BSI_SCORE_FEATURE_SET = "original_style_quantile_windows"
BSI_SCORE_MODEL = "m4_age_sex_comparators"

COHORTS = {
    "I0002": "BIDMC",
    "mros": "MrOS",
}

EXPOSURES = {
    "ss_percent_main": "SS",
    "ahi": "AHI",
    "hypoxic_burden": "HB",
    "arousal_index": "ArI",
}

MODEL_SPECS = {
    "m1_unadjusted": {
        "label": "Model 1: unadjusted",
        "short_label": "M1",
        "covariates": [],
    },
    "m2_age_sex": {
        "label": "Model 2: age + sex",
        "short_label": "M2",
        "covariates": ["age", "sex"],
    },
    "m3_age_sex_ahi": {
        "label": "Model 3: age + sex + AHI",
        "short_label": "M3",
        "covariates": ["age", "sex", "ahi"],
    },
    "m4_age_sex_bsi_other_comparators": {
        "label": "Model 4: age + sex + BSI score + other comparators",
        "short_label": "M4",
        "covariates": "mutual_with_bsi",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all-study-type cohort-specific mortality models for comparator exposures."
    )
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--bsi-score-csv", type=Path, default=DEFAULT_BSI_SCORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def zscore(series: pd.Series) -> pd.Series:
    values = safe_numeric(series)
    std = values.std(ddof=0)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return (values - values.mean()) / std


def strip_z_suffix(name: str) -> str:
    return name[:-2] if name.endswith("_z") else name


def covariates_for_model(model_name: str, exposure: str) -> tuple[list[str], str]:
    spec = MODEL_SPECS[model_name]
    note = ""
    if spec["covariates"] == "mutual_with_bsi":
        covariates = ["age", "sex", "bsi_score_oof"] + [col for col in EXPOSURES if col != exposure]
    else:
        covariates = list(spec["covariates"])

    if exposure in covariates:
        covariates = [col for col in covariates if col != exposure]
        note = f"Self-adjustment omitted for exposure `{exposure}`."
    return covariates, note


def build_exog(dat: pd.DataFrame, exposure: str, covariates: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    exog = pd.DataFrame(index=dat.index)
    order: list[str] = []
    dropped: list[str] = []

    exposure_name = f"{exposure}_z"
    exog[exposure_name] = zscore(dat[exposure])
    order.append(exposure_name)

    for covariate in covariates:
        values = safe_numeric(dat[covariate])
        if values.dropna().nunique() <= 1:
            dropped.append(covariate)
            continue
        if covariate == "sex":
            exog[covariate] = values
            order.append(covariate)
        else:
            name = f"{covariate}_z"
            exog[name] = zscore(values)
            order.append(name)

    return exog[order], order, dropped


def ph_diagnostic(fit: object, dat: pd.DataFrame, exposure_idx: int) -> tuple[float, float, int]:
    try:
        residuals = np.asarray(fit.schoenfeld_residuals)
        resid = pd.Series(residuals[:, exposure_idx], index=dat.index)
        mask = dat["event"].eq(1) & resid.notna() & dat["followup_days"].gt(0)
        n_events = int(mask.sum())
        if n_events < 10:
            return np.nan, np.nan, n_events
        rho, p_value = stats.spearmanr(np.log(dat.loc[mask, "followup_days"]), resid.loc[mask])
        return float(rho), float(p_value), n_events
    except Exception:
        return np.nan, np.nan, 0


def fit_model(frame: pd.DataFrame, cohort: str, exposure: str, model_name: str) -> dict[str, object]:
    requested_covariates, model_note = covariates_for_model(model_name, exposure)
    required = ["followup_days", "event", exposure] + requested_covariates
    dat = frame[required].dropna(subset=required).copy()
    result = {
        "selection_mode": SELECTION_MODE,
        "bsi_score_source": str(DEFAULT_BSI_SCORE),
        "bsi_score_feature_set": BSI_SCORE_FEATURE_SET,
        "bsi_score_model": BSI_SCORE_MODEL,
        "cohort": cohort,
        "cohort_label": COHORTS[cohort],
        "analysis_exposure": exposure,
        "analysis_exposure_label": EXPOSURES[exposure],
        "model_name": model_name,
        "model_label": MODEL_SPECS[model_name]["label"],
        "requested_covariates": ", ".join(requested_covariates),
        "used_covariates": "",
        "dropped_constant_covariates": "",
        "n": int(len(dat)),
        "n_events": int(dat["event"].sum()) if len(dat) else 0,
        "status": "not_run",
        "status_detail": model_note,
        "hr_per_sd": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "ph_spearman_rho_log_time": np.nan,
        "ph_p_value": np.nan,
        "ph_n_event_residuals": 0,
    }
    if len(dat) < 50 or int(dat["event"].sum()) < 10:
        result["status_detail"] = "; ".join(filter(None, [model_note, "Too few rows/events for Cox fit."]))
        return result

    exog, exog_order, dropped = build_exog(dat, exposure, requested_covariates)
    keep = exog.notna().all(axis=1)
    dat = dat.loc[keep].copy()
    exog = exog.loc[keep].copy()
    result["n"] = int(len(dat))
    result["n_events"] = int(dat["event"].sum()) if len(dat) else 0
    result["used_covariates"] = ", ".join(strip_z_suffix(name) for name in exog_order[1:])
    result["dropped_constant_covariates"] = ", ".join(dropped)
    if len(dat) < 50 or int(dat["event"].sum()) < 10:
        result["status_detail"] = "; ".join(
            filter(None, [model_note, "Insufficient rows/events after complete-case filtering."])
        )
        return result

    try:
        fit = PHReg(dat["followup_days"], exog, status=dat["event"], ties="breslow").fit(disp=0)
        idx = exog_order.index(f"{exposure}_z")
        ci = fit.conf_int()[idx]
        rho, ph_p, ph_n = ph_diagnostic(fit, dat, idx)
        result["status"] = "ok"
        result["hr_per_sd"] = float(np.exp(fit.params[idx]))
        result["ci_lower"] = float(np.exp(ci[0]))
        result["ci_upper"] = float(np.exp(ci[1]))
        result["p_value"] = float(fit.pvalues[idx])
        result["ph_spearman_rho_log_time"] = rho
        result["ph_p_value"] = ph_p
        result["ph_n_event_residuals"] = ph_n
    except Exception as exc:
        result["status"] = "fit_error"
        result["status_detail"] = "; ".join(filter(None, [model_note, f"{type(exc).__name__}: {exc}"]))
    return result


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def fmt_hr(row: pd.Series) -> str:
    if row["status"] != "ok":
        return "not estimable"
    return f"{row['hr_per_sd']:.2f} ({row['ci_lower']:.2f}-{row['ci_upper']:.2f}), p={fmt_p(row['p_value'])}"


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_markdown(outdir: Path, results: pd.DataFrame) -> None:
    lines = [
        "# All-Study-Type Comparator Mortality Models v1",
        "",
        "All HRs are per 1 SD of the exposure. M4 adjusts for age, sex, the 50-fold out-of-fold BSI mortality score, and the other comparator exposures.",
        "",
    ]
    for cohort_label, cohort_df in results.groupby("cohort_label", sort=False):
        table_rows: list[dict[str, object]] = []
        for exposure_label, exp_df in cohort_df.groupby("analysis_exposure_label", sort=False):
            row = {
                "Exposure": exposure_label,
                "n": int(exp_df["n"].iloc[0]),
                "events": int(exp_df["n_events"].iloc[0]),
            }
            for model_name in MODEL_SPECS:
                model_row = exp_df[exp_df["model_name"].eq(model_name)].iloc[0]
                row[MODEL_SPECS[model_name]["short_label"]] = fmt_hr(model_row)
            table_rows.append(row)
        lines.extend([f"## {cohort_label}", "", markdown_table(pd.DataFrame(table_rows)), ""])
    (outdir / "comparator_all_study_type_results.md").write_text("\n".join(lines).rstrip() + "\n")


def write_note(outdir: Path, args: argparse.Namespace, results: pd.DataFrame) -> None:
    lines = [
        "# All-Study-Type Comparator Mortality Models v1 Note",
        "",
        f"- Selection: `{args.selection_csv}`",
        f"- Selection mode: `{SELECTION_MODE}`",
        f"- BSI score source: `{args.bsi_score_csv}`",
        f"- BSI score feature set/model: `{BSI_SCORE_FEATURE_SET}` / `{BSI_SCORE_MODEL}`",
        "- M1: unadjusted.",
        "- M2: age + sex.",
        "- M3: age + sex + AHI; for AHI exposure, self-adjustment is omitted.",
        "- M4: age + sex + out-of-fold BSI score + the other comparators among SS, AHI, HB, and ArI.",
        "- Continuous exposures and covariates are z-scored within the fitted cohort/model complete-case data.",
        "- MrOS has no sex variation, so sex is dropped from adjusted fits.",
    ]
    non_ok = results[results["status"].ne("ok")]
    if not non_ok.empty:
        lines.extend(["", "## Non-ok fits", ""])
        for _, row in non_ok.iterrows():
            lines.append(
                f"- {row['cohort_label']} {row['analysis_exposure_label']} {row['model_name']}: {row['status']} {row['status_detail']}"
            )
    (outdir / "comparator_all_study_type_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_cols = [
        "selection_mode",
        "fileid",
        "sid",
        "cohort",
        "followup_days",
        "event",
        "age",
        "sex",
        *EXPOSURES.keys(),
    ]
    selected = pd.read_csv(args.selection_csv, usecols=selected_cols, low_memory=False)
    selected = selected[selected["selection_mode"].eq(SELECTION_MODE) & selected["cohort"].isin(COHORTS)].copy()

    score = pd.read_csv(args.bsi_score_csv)
    score = score[
        score["feature_set"].eq(BSI_SCORE_FEATURE_SET)
        & score["model_name"].eq(BSI_SCORE_MODEL)
        & score["cohort"].isin(COHORTS)
    ][["fileid", "sid", "cohort", "bsi_score_oof"]].copy()
    data = selected.merge(score, on=["fileid", "sid", "cohort"], how="inner")
    data["cohort_label"] = data["cohort"].map(COHORTS)
    for col in ["followup_days", "event", "age", "sex", *EXPOSURES.keys(), "bsi_score_oof"]:
        data[col] = safe_numeric(data[col])

    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        frame = data[data["cohort"].eq(cohort)].copy()
        for exposure in EXPOSURES:
            for model_name in MODEL_SPECS:
                rows.append(fit_model(frame, cohort, exposure, model_name))

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "comparator_all_study_type_results.csv", index=False)
    write_markdown(args.output_dir, results)
    write_note(args.output_dir, args, results)
    print(f"Wrote {args.output_dir / 'comparator_all_study_type_results.csv'}")


if __name__ == "__main__":
    main()
