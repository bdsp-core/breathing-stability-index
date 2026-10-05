#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MASTER_CSV = Path("REVISION/master_analysis_table_primary_cohorts_v4_paper.csv")
OUTPUT_DIR = Path("REVISION/revision_tables_v2")
MORTALITY_SELECTED_CSV = Path("REVISION/mortality_v2/selection_one_row_per_subject.csv")
COHORT_ORDER = ["S0001", "I0002", "mros", "mgh-cog", "Overall"]
BSI_BAND_ORDER = ["<0.5", "0.5-1.5", ">1.5"]
BSI_EQUAL_TERTILE_ORDER = ["T1 lowest BSI", "T2 middle BSI", "T3 highest BSI"]

PHYSIOLOGY_READY_COLUMNS = [
    "bsi_median_sleep",
    "ss_percent_main",
    "ahi",
    "hypoxic_burden",
    "arousal_index",
    "desaturation_index",
    "tst_min",
    "n1_pct",
    "n2_pct",
    "n3_pct",
    "rem_pct",
]

CROSS_SECTIONAL_DX_SPECS = [
    ("dx-cs_t21", "Trisomy 21 flag"),
    ("dx-cs_epilepsy", "Epilepsy flag"),
    ("dx-cs-depression", "Depression flag"),
    ("dx-cs-dementia", "Dementia flag"),
]

MORTALITY_COVARIATE_SPECS = [
    ("mort_cov_hypertension", "Hypertension"),
    ("mort_cov_diabetesii", "Diabetes"),
    ("mort_cov_myocardial_infarction", "Myocardial infarction"),
    ("mort_cov_atrial_fibrillation", "Atrial fibrillation"),
]

RAW_COLUMNS = [
    "cohort",
    "sid",
    "fileid",
    "age",
    "sex",
    "bmi",
    "ahi",
    "rdi",
    "oai",
    "cai",
    "hypoxic_burden",
    "arousal_index",
    "desaturation_index",
    "odi3",
    "odi4",
    "desaturation_burden",
    "tst_min",
    "sleep_efficiency",
    "n1_pct",
    "n2_pct",
    "n3_pct",
    "rem_pct",
    "bsi_median_sleep",
    "bsi_quantile_75_sleep",
    "bsi_quantile_90_sleep",
    "bsi_instability_burden_gt_1p5_sleep",
    "ss_percent_main",
    "paper_psg_type",
    "paper_primary_psg_eligible",
    "posture_available",
    "paper_posture_usable",
    "supine_pct_tst",
    "position_tst_within_5min",
    "phenotype_osa_csa",
    "followup_days",
    "vital_status",
    "cog_fluid",
    "cog_crystallized",
    "cog_total",
    "dx-cs_t21",
    "dx-cs_epilepsy",
    "dx-cs-depression",
    "dx-cs-dementia",
]

TABLE_SPECS = [
    ("Cohort", "PSG rows, n", "n_rows", None),
    ("Cohort", "Unique subjects, n", "n_subjects", None),
    ("Study type", "Strict primary eligible, n (%)", "true_pct", "paper_primary_psg_eligible_bool"),
    ("Study type", "Diagnostic paper PSG type, n (%)", "eq_pct", ("paper_psg_type", "diagnostic")),
    ("Data availability", "Mortality follow-up available, n (%)", "true_pct", "has_mortality_followup"),
    ("Data availability", "Any cognition score available, n (%)", "true_pct", "has_cognition_any"),
    ("Data availability", "Posture usable, n (%)", "true_pct", "paper_posture_usable_bool"),
    ("Disease/comorbidity", "Cardiometabolic covariates assessed for strict mortality model, n (%)", "true_pct", "mort_covariates_any"),
    ("Disease/comorbidity", "Hypertension, present/assessed n/N (%)", "present_assessed", "mort_cov_hypertension"),
    ("Disease/comorbidity", "Diabetes, present/assessed n/N (%)", "present_assessed", "mort_cov_diabetesii"),
    ("Disease/comorbidity", "Myocardial infarction, present/assessed n/N (%)", "present_assessed", "mort_cov_myocardial_infarction"),
    ("Disease/comorbidity", "Atrial fibrillation, present/assessed n/N (%)", "present_assessed", "mort_cov_atrial_fibrillation"),
    ("Disease/comorbidity", "Depression flag (cross-sectional), present/assessed n/N (%)", "present_assessed", "dx-cs-depression"),
    ("Demographics", "Age, y", "mean_sd", "age"),
    ("Demographics", "Male sex, n (%)", "eq_pct", ("sex", 1)),
    ("Demographics", "BMI, kg/m^2", "median_iqr", "bmi"),
    ("Sleep and respiratory features", "AHI, events/h", "median_iqr", "ahi"),
    ("Sleep and respiratory features", "Hypoxic burden, %min/h", "median_iqr", "hypoxic_burden"),
    ("Sleep and respiratory features", "Arousal index, events/h", "median_iqr", "arousal_index"),
    ("Sleep and respiratory features", "Desaturation index, events/h", "median_iqr", "desaturation_index"),
    ("Sleep and respiratory features", "Total sleep time, min", "mean_sd", "tst_min"),
    ("Sleep architecture", "N1, %TST", "mean_sd", "n1_pct"),
    ("Sleep architecture", "N2, %TST", "mean_sd", "n2_pct"),
    ("Sleep architecture", "N3, %TST", "mean_sd", "n3_pct"),
    ("Sleep architecture", "REM, %TST", "mean_sd", "rem_pct"),
    ("Primary exposures", "BSI median sleep", "median_iqr", "bsi_median_sleep"),
    ("Primary exposures", "BSI quantile 75 sleep", "median_iqr", "bsi_quantile_75_sleep"),
    ("Primary exposures", "BSI instability burden >1.5 sleep", "median_iqr", "bsi_instability_burden_gt_1p5_sleep"),
    ("Primary exposures", "SS percent main", "median_iqr", "ss_percent_main"),
    ("Posture", "Supine %TST among rows with posture", "median_iqr", "supine_pct_tst"),
]

BAND_SPECS = [
    ("Cohort", "PSG rows, n", "n_rows", None),
    ("Cohort", "Unique subjects, n", "n_subjects", None),
    ("Demographics", "Age, y", "mean_sd", "age"),
    ("Demographics", "Male sex, n (%)", "eq_pct", ("sex", 1)),
    ("Demographics", "BMI, kg/m^2", "median_iqr", "bmi"),
    ("Sleep and respiratory features", "AHI, events/h", "median_iqr", "ahi"),
    ("Sleep and respiratory features", "Hypoxic burden, %min/h", "median_iqr", "hypoxic_burden"),
    ("Sleep and respiratory features", "Arousal index, events/h", "median_iqr", "arousal_index"),
    ("Sleep and respiratory features", "Desaturation index, events/h", "median_iqr", "desaturation_index"),
    ("Primary exposures", "BSI median sleep", "median_iqr", "bsi_median_sleep"),
    ("Primary exposures", "BSI quantile 75 sleep", "median_iqr", "bsi_quantile_75_sleep"),
    ("Primary exposures", "BSI quantile 90 sleep", "median_iqr", "bsi_quantile_90_sleep"),
    ("Primary exposures", "BSI instability burden >1.5 sleep", "median_iqr", "bsi_instability_burden_gt_1p5_sleep"),
    ("Primary exposures", "SS percent main", "median_iqr", "ss_percent_main"),
    ("Disease/comorbidity", "Cardiometabolic covariates assessed for strict mortality model, n (%)", "true_pct", "mort_covariates_any"),
    ("Disease/comorbidity", "Hypertension, present/assessed n/N (%)", "present_assessed", "mort_cov_hypertension"),
    ("Disease/comorbidity", "Diabetes, present/assessed n/N (%)", "present_assessed", "mort_cov_diabetesii"),
    ("Disease/comorbidity", "Myocardial infarction, present/assessed n/N (%)", "present_assessed", "mort_cov_myocardial_infarction"),
    ("Disease/comorbidity", "Atrial fibrillation, present/assessed n/N (%)", "present_assessed", "mort_cov_atrial_fibrillation"),
    ("Disease/comorbidity", "Depression flag (cross-sectional), present/assessed n/N (%)", "present_assessed", "dx-cs-depression"),
    ("Phenotype", "OSA phenotype, n (%)", "eq_pct", ("phenotype_osa_csa", "osa")),
    ("Phenotype", "CSA phenotype, n (%)", "eq_pct", ("phenotype_osa_csa", "csa")),
    ("Phenotype", "Mixed phenotype, n (%)", "eq_pct", ("phenotype_osa_csa", "mixed")),
    ("Posture", "Posture usable, n (%)", "true_pct", "paper_posture_usable_bool"),
]

AVAILABILITY_ITEMS = [
    ("age", "Age", "nonmissing"),
    ("sex", "Sex", "nonmissing"),
    ("bmi", "BMI", "nonmissing"),
    ("paper_psg_type", "Paper PSG type", "nonmissing"),
    ("paper_primary_psg_eligible_bool", "Strict primary eligible", "true"),
    ("bsi_median_sleep", "BSI median sleep", "nonmissing"),
    ("bsi_quantile_75_sleep", "BSI quantile 75 sleep", "nonmissing"),
    ("bsi_quantile_90_sleep", "BSI quantile 90 sleep", "nonmissing"),
    ("bsi_instability_burden_gt_1p5_sleep", "BSI instability burden >1.5 sleep", "nonmissing"),
    ("ss_percent_main", "SS percent main", "nonmissing"),
    ("ahi", "AHI", "nonmissing"),
    ("hypoxic_burden", "Hypoxic burden", "nonmissing"),
    ("arousal_index", "Arousal index", "nonmissing"),
    ("desaturation_index", "Desaturation index", "nonmissing"),
    ("odi3", "ODI3", "nonmissing"),
    ("odi4", "ODI4", "nonmissing"),
    ("desaturation_burden", "Desaturation burden", "nonmissing"),
    ("followup_days", "Mortality follow-up days", "nonmissing"),
    ("vital_status", "Vital status", "nonmissing"),
    ("has_cognition_any", "Any cognition score", "true"),
    ("dx-cs-depression", "Depression flag assessed", "nonmissing"),
    ("posture_available_bool", "Posture available", "true"),
    ("paper_posture_usable_bool", "Paper posture usable", "true"),
    ("supine_pct_tst", "Supine %TST", "nonmissing"),
    ("physiology_ready", "Physiology ready", "true"),
    ("mort_cov_hypertension", "Hypertension covariate assessed", "nonmissing"),
    ("mort_cov_diabetesii", "Diabetes covariate assessed", "nonmissing"),
    ("mort_cov_myocardial_infarction", "Myocardial infarction covariate assessed", "nonmissing"),
    ("mort_cov_atrial_fibrillation", "Atrial fibrillation covariate assessed", "nonmissing"),
]


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
    mapped = series.astype(str).str.strip().str.lower().map(
        {"m": 1.0, "male": 1.0, "1": 1.0, "1.0": 1.0, "f": 0.0, "female": 0.0, "0": 0.0, "0.0": 0.0}
    )
    return numeric.combine_first(mapped)


def load_master(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in RAW_COLUMNS if col in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["sex"] = safe_sex(df["sex"])
    df["paper_psg_type"] = df["paper_psg_type"].astype("string").str.strip().str.lower()
    df["paper_primary_psg_eligible_bool"] = safe_bool(df["paper_primary_psg_eligible"])
    df["posture_available_bool"] = safe_bool(df["posture_available"])
    df["paper_posture_usable_bool"] = safe_bool(df["paper_posture_usable"])
    df = attach_mortality_covariates(df)
    df["has_mortality_followup"] = df["followup_days"].notna() & df["vital_status"].notna()
    df["has_cognition_any"] = df[["cog_fluid", "cog_crystallized", "cog_total"]].notna().any(axis=1)
    df["physiology_ready"] = df[PHYSIOLOGY_READY_COLUMNS].notna().all(axis=1)
    df["analysis_subject_id"] = np.where(df["cohort"].astype(str).eq("mgh-cog"), df["fileid"], df["sid"])
    df["bsi_band"] = pd.cut(
        safe_numeric(df["bsi_median_sleep"]),
        bins=[-np.inf, 0.5, 1.5, np.inf],
        right=False,
        labels=BSI_BAND_ORDER,
    )
    return df


def mean_sd(series: pd.Series) -> str:
    s = safe_numeric(series).dropna()
    if s.empty:
        return "NA"
    return f"{s.mean():.1f} ({s.std():.1f})"


def median_iqr(series: pd.Series) -> str:
    s = safe_numeric(series).dropna()
    if s.empty:
        return "NA"
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.2f} [{q1:.2f}, {q3:.2f}]"


def present_assessed(series: pd.Series) -> str:
    values = safe_numeric(series)
    assessed = int(values.notna().sum())
    if assessed == 0:
        return "NA"
    present = int(values.eq(1).sum())
    return f"{present:,}/{assessed:,} ({100.0 * present / assessed:.1f}%)"


def count_pct(series: pd.Series, value: object = True) -> str:
    denom = len(series)
    if denom == 0:
        return "0 (NA)"
    if value is True:
        n = int(series.fillna(False).astype(bool).sum())
    else:
        n = int(series.eq(value).sum())
    return f"{n:,} ({100.0 * n / denom:.1f}%)"


def attach_mortality_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_cols = [
        "cov_hypertension",
        "cov_diabetesii",
        "cov_myocardial_infarction",
        "cov_atrial_fibrillation",
    ]
    renamed = {col: f"mort_{col}" for col in source_cols}
    for col in renamed.values():
        out[col] = np.nan
    if not MORTALITY_SELECTED_CSV.exists():
        out["mort_covariates_any"] = False
        return out

    selected = pd.read_csv(
        MORTALITY_SELECTED_CSV,
        usecols=["selection_mode", "fileid", *source_cols],
        low_memory=False,
    )
    selected = selected[selected["selection_mode"].eq("strict_primary")].drop_duplicates("fileid")
    selected = selected.rename(columns=renamed)
    selected = selected[["fileid", *renamed.values()]]
    out = out.drop(columns=[col for col in renamed.values() if col in out.columns])
    out = out.merge(selected, on="fileid", how="left")
    for col in renamed.values():
        values = safe_numeric(out[col])
        binary = values.isin([0, 1])
        out[col] = values.where(binary, np.nan)
    out["mort_covariates_any"] = out[list(renamed.values())].notna().any(axis=1)
    return out


def summarize(subset: pd.DataFrame, kind: str, arg: object) -> str:
    if kind == "n_rows":
        return f"{len(subset):,}"
    if kind == "n_subjects":
        return f"{subset[['cohort', 'analysis_subject_id']].dropna().drop_duplicates().shape[0]:,}"
    if kind == "mean_sd":
        return mean_sd(subset[str(arg)])
    if kind == "median_iqr":
        return median_iqr(subset[str(arg)])
    if kind == "true_pct":
        return count_pct(subset[str(arg)], True)
    if kind == "eq_pct":
        col, value = arg  # type: ignore[misc]
        return count_pct(subset[str(col)], value)
    if kind == "present_assessed":
        return present_assessed(subset[str(arg)])
    return "NA"


def build_grouped_table(df: pd.DataFrame, group_col: str | None, groups: list[str], specs: list[tuple[str, str, str, object]]) -> pd.DataFrame:
    rows = []
    for section, label, kind, arg in specs:
        row = {"section": section, "label": label}
        for group in groups:
            subset = df if group == "Overall" or group_col is None else df[df[group_col].astype(str).eq(group)]
            row[group] = summarize(subset, kind, arg)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_equal_psg_bsi_tertiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["bsi_equal_psg_tertile"] = pd.NA
    eligible = out[out["bsi_median_sleep"].notna()].copy()
    eligible["_source_order"] = np.arange(len(eligible))
    eligible = eligible.sort_values(
        ["bsi_median_sleep", "cohort", "fileid", "_source_order"],
        kind="mergesort",
    )

    groups = np.array_split(np.arange(len(eligible)), len(BSI_EQUAL_TERTILE_ORDER))
    cutpoint_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for label, positions in zip(BSI_EQUAL_TERTILE_ORDER, groups):
        idx = eligible.index[positions]
        out.loc[idx, "bsi_equal_psg_tertile"] = label
        values = safe_numeric(out.loc[idx, "bsi_median_sleep"])
        group_rows.append(
            {
                "row_type": "group",
                "label": label,
                "n_psg_rows": int(len(idx)),
                "min_bsi_median_sleep": values.min(),
                "max_bsi_median_sleep": values.max(),
                "median_bsi_median_sleep": values.median(),
            }
        )

    for cut_number, (lower, upper) in enumerate(zip(BSI_EQUAL_TERTILE_ORDER[:-1], BSI_EQUAL_TERTILE_ORDER[1:]), start=1):
        lower_values = safe_numeric(out.loc[out["bsi_equal_psg_tertile"].eq(lower), "bsi_median_sleep"])
        upper_values = safe_numeric(out.loc[out["bsi_equal_psg_tertile"].eq(upper), "bsi_median_sleep"])
        cut_value = lower_values.max()
        next_value = upper_values.min()
        cutpoint_rows.append(
            {
                "row_type": "cutpoint",
                "label": f"cutpoint_{cut_number}",
                "lower_group": lower,
                "upper_group": upper,
                "cutpoint_bsi_median_sleep": cut_value,
                "next_group_min_bsi_median_sleep": next_value,
                "ties_split_at_cutpoint": bool(pd.notna(cut_value) and pd.notna(next_value) and cut_value == next_value),
                "n_rows_at_cutpoint_value": int(safe_numeric(out["bsi_median_sleep"]).eq(cut_value).sum()),
            }
        )

    return out, pd.DataFrame([*group_rows, *cutpoint_rows])


def build_availability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label, mode in AVAILABILITY_ITEMS:
        row = {"column": col, "label": label, "mode": mode}
        for cohort in COHORT_ORDER:
            subset = df if cohort == "Overall" else df[df["cohort"].astype(str).eq(cohort)]
            denom = len(subset)
            if col not in subset:
                n = 0
            elif mode == "true":
                n = int(subset[col].fillna(False).astype(bool).sum())
            else:
                n = int(subset[col].notna().sum())
            row[f"{cohort}_n"] = n
            row[f"{cohort}_pct"] = round(100.0 * n / denom, 1) if denom else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_disease_endpoint_inventory(df: pd.DataFrame, table1_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col, label in CROSS_SECTIONAL_DX_SPECS:
        values_all = safe_numeric(df[col])
        values_table1 = safe_numeric(table1_df[col])
        rows.append(
            {
                "variable": col,
                "label": label,
                "analysis_role": "Cross-sectional diagnosis flag",
                "source_population": "v4 paper master",
                "overall_assessed_n": int(values_all.notna().sum()),
                "overall_present_n": int(values_all.eq(1).sum()),
                "table1_assessed_n": int(values_table1.notna().sum()),
                "table1_present_n": int(values_table1.eq(1).sum()),
                "timing": "No diagnosis date column in v4; do not use as incident/prospective endpoint.",
                "display_rule": "Report as present/assessed n/N (%), not assessed-only counts.",
            }
        )
    for col, label in MORTALITY_COVARIATE_SPECS:
        values_all = safe_numeric(df[col])
        values_table1 = safe_numeric(table1_df[col])
        rows.append(
            {
                "variable": col,
                "label": label,
                "analysis_role": "Cardiometabolic covariate for strict mortality adjustment",
                "source_population": "Strict primary one-row-per-subject mortality selection",
                "overall_assessed_n": int(values_all.notna().sum()),
                "overall_present_n": int(values_all.eq(1).sum()),
                "table1_assessed_n": int(values_table1.notna().sum()),
                "table1_present_n": int(values_table1.eq(1).sum()),
                "timing": "Used as baseline/cross-sectional adjustment covariate; not analyzed as an incident disease endpoint.",
                "display_rule": "Report binary-coded values as present/assessed n/N (%); nonbinary score fields and unavailable cells are shown as NA.",
            }
        )
    return pd.DataFrame(rows)


def write_table_notes(outdir: Path, table1_df: pd.DataFrame) -> None:
    lines = [
        "# Revision Table Notes v2",
        "",
        "## Recommended Assembly",
        "- To directly satisfy the Associate Editor's request for a BSI-stratified Table 1, use `table1_bsi_band_stratified_v2.csv` as the manuscript Table 1.",
        "- Use `tableS1_cohort_overview_v2.csv` as a supplement cohort overview table.",
        "- `table1_bsi_equal_psg_tertile_stratified_v2.csv` is an optional equal-count descriptive alternative, not the recommended main Table 1.",
        "- `table1_main_cohort_overview_v2.csv` and `table2_bsi_band_stratified_v2.csv` are retained for backwards-compatible file references.",
        "",
        "## Table 1 Source Population",
        f"- Table 1 is restricted to physiology-ready PSG rows: `{len(table1_df):,}` PSG rows.",
        "- Physiology-ready means BSI, SS, AHI, hypoxic burden, arousal index, desaturation index, total sleep time, and sleep-stage percentages are all available.",
        "- Table 1 columns are BSI bands, not case/control groups; there is no separate control group in this table.",
        "- Because some participants have more than one PSG, unique-subject counts within BSI bands should not be summed across bands.",
        "- For MGH-COG, each PSG row is treated as a unique participant because the raw subject code is not unique.",
        "- Equal-PSG tertiles use rank-based groups on `bsi_median_sleep`; tied BSI values at the tertile boundaries are split deterministically to keep group sizes equal.",
        "",
        "## Disease And Comorbidity Rows",
        "- Disease/comorbidity cells are shown as `present/assessed n/N (%)`; they are not assessed-only counts.",
        "- Cardiometabolic covariates are from the strict primary one-row-per-subject mortality selection and are used as adjustment covariates, not incident disease endpoints.",
        "- Only binary-coded cardiometabolic fields are shown as present/assessed disease counts; nonbinary disease-score fields are not tabulated as diagnosis prevalence.",
        "- Cross-sectional diagnosis flags in the v4 master have no diagnosis dates and should not be interpreted as incident/prospective outcomes.",
        "- `NA` means that the covariate or diagnosis flag is unavailable for that cohort/subset.",
        "",
        "## Mortality Follow-up",
        "- Mortality follow-up availability means both `followup_days` and `vital_status` are nonmissing.",
        "- Not all physiology-ready PSG rows have mortality follow-up; mortality analyses use the explicit one-row-per-subject selection in `REVISION/mortality_v2/selection_summary.csv`.",
    ]
    (outdir / "table_notes_v2.md").write_text("\n".join(lines) + "\n")


def write_summary(df: pd.DataFrame, outdir: Path) -> None:
    strict = df["paper_primary_psg_eligible_bool"].eq(True)
    physiology_ready = df["physiology_ready"].eq(True)
    table1_df = df[physiology_ready].copy()
    lines = [
        "# Revision Tables v2 Summary",
        "",
        f"Source master: `{MASTER_CSV}`",
        f"Rows: `{len(df):,}`",
        f"Unique subjects/cohort-subjects: `{df[['cohort', 'analysis_subject_id']].dropna().drop_duplicates().shape[0]:,}`",
        f"Strict primary eligible PSG rows: `{int(strict.sum()):,}`",
        f"Table 1 source: physiology-ready PSG rows with BSI, SS, respiratory metrics, and sleep architecture available: `{len(table1_df):,}`",
        f"Table 1 unique subjects/cohort-subjects: `{table1_df[['cohort', 'analysis_subject_id']].dropna().drop_duplicates().shape[0]:,}`",
        f"Paper posture usable rows: `{int(df['paper_posture_usable_bool'].eq(True).sum()):,}`",
        "",
        "Outputs:",
        "- `table1_bsi_band_stratified_v2.csv`",
        "- `table1_bsi_equal_psg_tertile_stratified_v2.csv`",
        "- `table1_bsi_equal_psg_tertile_cutpoints_v2.csv`",
        "- `table1_main_cohort_overview_v2.csv`",
        "- `table2_bsi_band_stratified_v2.csv`",
        "- `tableS1_cohort_overview_v2.csv`",
        "- `tableS2_availability_missingness_v2.csv`",
        "- `tableS3_disease_endpoint_inventory_v2.csv`",
        "- `table_notes_v2.md`",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_master(MASTER_CSV)
    table1_df = df[df["physiology_ready"].eq(True)].copy()
    table1_df, tertile_cutpoints = assign_equal_psg_bsi_tertiles(table1_df)
    main_table = build_grouped_table(table1_df, "cohort", COHORT_ORDER, TABLE_SPECS)
    band_table = build_grouped_table(
        table1_df[table1_df["bsi_band"].notna()].copy(),
        "bsi_band",
        BSI_BAND_ORDER,
        BAND_SPECS,
    )
    tertile_table = build_grouped_table(
        table1_df[table1_df["bsi_equal_psg_tertile"].notna()].copy(),
        "bsi_equal_psg_tertile",
        BSI_EQUAL_TERTILE_ORDER,
        BAND_SPECS,
    )
    availability = build_availability(df)
    disease_inventory = build_disease_endpoint_inventory(df, table1_df)
    main_table.to_csv(OUTPUT_DIR / "table1_main_cohort_overview_v2.csv", index=False)
    main_table.to_csv(OUTPUT_DIR / "tableS1_cohort_overview_v2.csv", index=False)
    band_table.to_csv(OUTPUT_DIR / "table1_bsi_band_stratified_v2.csv", index=False)
    band_table.to_csv(OUTPUT_DIR / "table2_bsi_band_stratified_v2.csv", index=False)
    tertile_table.to_csv(OUTPUT_DIR / "table1_bsi_equal_psg_tertile_stratified_v2.csv", index=False)
    tertile_cutpoints.to_csv(OUTPUT_DIR / "table1_bsi_equal_psg_tertile_cutpoints_v2.csv", index=False)
    availability.to_csv(OUTPUT_DIR / "tableS2_availability_missingness_v2.csv", index=False)
    disease_inventory.to_csv(OUTPUT_DIR / "tableS3_disease_endpoint_inventory_v2.csv", index=False)
    write_table_notes(OUTPUT_DIR, table1_df)
    write_summary(df, OUTPUT_DIR)
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
