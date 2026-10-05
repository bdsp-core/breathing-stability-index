#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BSI_RESULTS = (
    REPO_ROOT
    / "REVISION"
    / "mortality_bsi_score_cv_all_study_type_original_style_50fold_v1"
    / "bsi_score_cv_results.csv"
)
BSI_RESULTS_EXTRA = [
    REPO_ROOT
    / "REVISION"
    / "mortality_bsi_score_cv_all_study_type_original_style_50fold_s0001_probe_v1"
    / "bsi_score_cv_results.csv"
]
COMPARATOR_RESULTS = (
    REPO_ROOT
    / "REVISION"
    / "mortality_comparator_all_study_type_v1"
    / "comparator_all_study_type_results.csv"
)
COMPARATOR_RESULTS_EXTRA = [
    REPO_ROOT
    / "REVISION"
    / "mortality_comparator_all_study_type_s0001_probe_v1"
    / "comparator_all_study_type_results.csv"
]
OUTDIR = REPO_ROOT / "REVISION" / "mortality_comparator_all_study_type_v1"

SELECTION_MODE = "all_study_type_sensitivity"
BSI_FEATURE_SET = "original_style_quantile_windows"

MODEL_MAP = {
    "m1_unadjusted": "M1",
    "m2_age_sex": "M2",
    "m3_age_sex_ahi": "M3",
    "m4_age_sex_comparators": "M4",
    "m4_age_sex_bsi_other_comparators": "M4",
}
MODEL_ORDER = ["M1", "M2", "M3", "M4"]
EXPOSURE_ORDER = [
    "Breathing stability score",
    "SS",
    "AHI",
    "HB",
    "ArI",
]
COHORT_ORDER = ["MrOS", "BIDMC", "MGH"]


def read_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("None of the requested mortality result files exist.")
    return pd.concat(frames, ignore_index=True)


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


def load_bsi() -> pd.DataFrame:
    bsi = read_concat([BSI_RESULTS, *BSI_RESULTS_EXTRA])
    bsi = bsi[
        bsi["selection_mode"].eq(SELECTION_MODE)
        & bsi["feature_set"].eq(BSI_FEATURE_SET)
        & bsi["model_name"].isin(MODEL_MAP)
    ].copy()
    bsi["Exposure"] = "Breathing stability score"
    bsi["Model"] = bsi["model_name"].map(MODEL_MAP)
    bsi["events"] = bsi["n_events"]
    return bsi[
        [
            "cohort_label",
            "Exposure",
            "Model",
            "n",
            "events",
            "status",
            "hr_per_sd",
            "ci_lower",
            "ci_upper",
            "p_value",
        ]
    ]


def load_comparators() -> pd.DataFrame:
    comparators = read_concat([COMPARATOR_RESULTS, *COMPARATOR_RESULTS_EXTRA])
    comparators = comparators[
        comparators["selection_mode"].eq(SELECTION_MODE) & comparators["model_name"].isin(MODEL_MAP)
    ].copy()
    comparators["Exposure"] = comparators["analysis_exposure_label"]
    comparators["Model"] = comparators["model_name"].map(MODEL_MAP)
    comparators["events"] = comparators["n_events"]
    return comparators[
        [
            "cohort_label",
            "Exposure",
            "Model",
            "n",
            "events",
            "status",
            "hr_per_sd",
            "ci_lower",
            "ci_upper",
            "p_value",
        ]
    ]


def build_wide(long_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort_label, cohort_df in long_results.groupby("cohort_label", sort=False):
        for exposure in EXPOSURE_ORDER:
            exp_df = cohort_df[cohort_df["Exposure"].eq(exposure)]
            if exp_df.empty:
                continue
            row: dict[str, object] = {
                "Cohort": cohort_label,
                "Exposure": exposure,
                "n": int(exp_df["n"].iloc[0]),
                "events": int(exp_df["events"].iloc[0]),
            }
            for model in MODEL_ORDER:
                model_df = exp_df[exp_df["Model"].eq(model)]
                row[model] = fmt_hr(model_df.iloc[0]) if not model_df.empty else "not estimable"
            rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(wide: pd.DataFrame) -> None:
    lines = [
        "# Table X. Associations of Breathing Stability and Comparator Sleep Metrics With All-Cause Mortality",
        "",
        "Cohort-specific Cox proportional hazards models were fit in MrOS, BIDMC, and MGH using one PSG per participant while retaining all available study types. Values are hazard ratios per 1-SD increase in the exposure with 95% confidence intervals and p values. Breathing stability was modeled as a 50-fold out-of-fold mortality score derived from BSI quantile features and 5-minute stable/unstable window features. Model 1 is unadjusted; Model 2 adjusts for age and sex; Model 3 adjusts for age, sex, and AHI, with AHI self-adjustment omitted for the AHI row; Model 4 adjusts the breathing stability score for age, sex, SS, AHI, HB, and ArI, and adjusts each comparator for age, sex, the breathing stability score, and the remaining comparator metrics. Sex was omitted from adjusted MrOS models because the cohort contains no sex variation.",
        "",
        "Abbreviations: AHI, apnea-hypopnea index; ArI, arousal index; BSI, breathing stability index; HB, hypoxic burden; PSG, polysomnogram; SS, self-similarity.",
        "",
    ]
    for cohort_label, cohort_df in wide.groupby("Cohort", sort=False):
        lines.extend([f"## {cohort_label}", "", markdown_table(cohort_df.drop(columns=["Cohort"])), ""])
    (OUTDIR / "manuscript_mortality_table_v1.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    long_results = pd.concat([load_bsi(), load_comparators()], ignore_index=True)
    long_results["Exposure"] = pd.Categorical(long_results["Exposure"], EXPOSURE_ORDER, ordered=True)
    long_results["Model"] = pd.Categorical(long_results["Model"], MODEL_ORDER, ordered=True)
    long_results["cohort_label"] = long_results["cohort_label"].replace({"MGH/S0001": "MGH"})
    long_results["cohort_label"] = pd.Categorical(long_results["cohort_label"], COHORT_ORDER, ordered=True)
    long_results = long_results.sort_values(["cohort_label", "Exposure", "Model"]).copy()
    wide = build_wide(long_results)
    wide.to_csv(OUTDIR / "manuscript_mortality_table_v1.csv", index=False)
    write_markdown(wide)
    print(f"Wrote {OUTDIR / 'manuscript_mortality_table_v1.md'}")
    print(f"Wrote {OUTDIR / 'manuscript_mortality_table_v1.csv'}")


if __name__ == "__main__":
    main()
