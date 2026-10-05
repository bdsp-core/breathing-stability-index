#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
DEFAULT_MORTALITY = REPO_ROOT / "REVISION" / "mortality_v2"
DEFAULT_OUTDIR = REPO_ROOT / "REVISION" / "sensitivity_raw_bsi_v1"

MIN_STAGE_MINUTES = 15.0

CANDIDATES = {
    "bsi_median_sleep": {
        "label": "Median BSI sleep",
        "sleep_col": "bsi_median_sleep",
        "stage_cols": {
            "N1": "bsi_robust_mean_w2_ov0p9_n1_median_stability",
            "N2": "bsi_robust_mean_w2_ov0p9_n2_median_stability",
            "N3": "bsi_robust_mean_w2_ov0p9_n3_median_stability",
            "NREM": "bsi_robust_mean_w2_ov0p9_nrem_median_stability",
            "REM": "bsi_robust_mean_w2_ov0p9_rem_median_stability",
        },
        "primary_role": "physiologic_anchor",
    },
    "bsi_quantile_75_sleep": {
        "label": "BSI quantile 75 sleep",
        "sleep_col": "bsi_quantile_75_sleep",
        "stage_cols": {
            "N1": "bsi_robust_mean_w2_ov0p9_n1_quantile_75",
            "N2": "bsi_robust_mean_w2_ov0p9_n2_quantile_75",
            "N3": "bsi_robust_mean_w2_ov0p9_n3_quantile_75",
            "NREM": "bsi_robust_mean_w2_ov0p9_nrem_quantile_75",
            "REM": "bsi_robust_mean_w2_ov0p9_rem_quantile_75",
        },
        "primary_role": "raw_candidate",
    },
    "bsi_instability_burden_gt_1p5_sleep": {
        "label": "Instability burden >1.5 sleep",
        "sleep_col": "bsi_instability_burden_gt_1p5_sleep",
        "stage_cols": {},
        "primary_role": "raw_candidate",
    },
    "bsi_quantile_90_sleep": {
        "label": "BSI quantile 90 sleep",
        "sleep_col": "bsi_quantile_90_sleep",
        "stage_cols": {
            "N1": "bsi_robust_mean_w2_ov0p9_n1_quantile_90",
            "N2": "bsi_robust_mean_w2_ov0p9_n2_quantile_90",
            "N3": "bsi_robust_mean_w2_ov0p9_n3_quantile_90",
            "NREM": "bsi_robust_mean_w2_ov0p9_nrem_quantile_90",
            "REM": "bsi_robust_mean_w2_ov0p9_rem_quantile_90",
        },
        "primary_role": "deemphasized_candidate",
    },
}

COMPARATORS = [
    "ss_percent_main",
    "ahi",
    "hypoxic_burden",
    "arousal_index",
    "desaturation_index",
    "odi3",
    "odi4",
    "desaturation_burden",
]

STAGE_MIN_COLS = {
    "N1": "n1_min",
    "N2": "n2_min",
    "N3": "n3_min",
    "REM": "rem_min",
}
NREM_MIN_COMPONENTS = ["n1_min", "n2_min", "n3_min"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw BSI candidate sensitivity checks.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--mortality-dir", type=Path, default=DEFAULT_MORTALITY)
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


def candidate_required_columns() -> list[str]:
    cols = [
        "fileid",
        "sid",
        "cohort",
        "paper_primary_psg_eligible",
        "tst_min",
        *STAGE_MIN_COLS.values(),
        *NREM_MIN_COMPONENTS,
        *COMPARATORS,
    ]
    for spec in CANDIDATES.values():
        cols.append(spec["sleep_col"])
        cols.extend(spec["stage_cols"].values())
    for prefix in ["n1", "n2", "n3", "nrem", "rem"]:
        cols.append(f"bsi_robust_mean_w2_ov0p9_{prefix}_n_5min_windows_stable")
        cols.append(f"bsi_robust_mean_w2_ov0p9_{prefix}_n_5min_windows_unstable")
    return list(dict.fromkeys(cols))


def read_csv_available(path: Path, requested_cols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in requested_cols if col in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in requested_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_master(path: Path) -> pd.DataFrame:
    df = read_csv_available(path, candidate_required_columns())
    df["cohort"] = df["cohort"].astype(str)
    df["fileid"] = df["fileid"].astype(str)
    df["sid"] = df["sid"].astype(str)
    df["paper_primary_psg_eligible_bool"] = safe_bool(df["paper_primary_psg_eligible"])
    for col in df.columns:
        if col not in {"fileid", "sid", "cohort", "paper_primary_psg_eligible", "paper_primary_psg_eligible_bool"}:
            df[col] = safe_numeric(df[col])
    add_instability_burden_stage_cols(df)
    return df


def add_instability_burden_stage_cols(df: pd.DataFrame) -> None:
    stage_cols: dict[str, str] = {}
    for stage, prefix in [("N1", "n1"), ("N2", "n2"), ("N3", "n3"), ("NREM", "nrem"), ("REM", "rem")]:
        stable = f"bsi_robust_mean_w2_ov0p9_{prefix}_n_5min_windows_stable"
        unstable = f"bsi_robust_mean_w2_ov0p9_{prefix}_n_5min_windows_unstable"
        out_col = f"bsi_instability_burden_gt_1p5_{prefix}"
        denom = df[stable] + df[unstable]
        df[out_col] = np.where(denom > 0, df[unstable] / denom, np.nan)
        stage_cols[stage] = out_col
    CANDIDATES["bsi_instability_burden_gt_1p5_sleep"]["stage_cols"] = stage_cols


def build_missingness(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    strict = df["paper_primary_psg_eligible_bool"].eq(True)
    for candidate, spec in CANDIDATES.items():
        col = spec["sleep_col"]
        rows.append(
            {
                "candidate": candidate,
                "label": spec["label"],
                "role": spec["primary_role"],
                "n_total": int(len(df)),
                "n_nonmissing_total": int(df[col].notna().sum()),
                "pct_nonmissing_total": 100.0 * df[col].notna().mean(),
                "n_strict_primary": int(strict.sum()),
                "n_nonmissing_strict_primary": int(df.loc[strict, col].notna().sum()),
                "pct_nonmissing_strict_primary": 100.0 * df.loc[strict, col].notna().mean(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "candidate_missingness.csv", index=False)
    return out


def iqr_text(series: pd.Series) -> str:
    s = safe_numeric(series).dropna()
    if s.empty:
        return "NA"
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.3f} [{q1:.3f}, {q3:.3f}]"


def rem_nrem_mask(df: pd.DataFrame, rem_col: str, nrem_col: str) -> pd.Series:
    nrem_minutes = df[NREM_MIN_COMPONENTS].fillna(0).sum(axis=1)
    return df[rem_col].notna() & df[nrem_col].notna() & df["rem_min"].ge(MIN_STAGE_MINUTES) & nrem_minutes.ge(MIN_STAGE_MINUTES)


def n1_n2_n3_mask(df: pd.DataFrame, n1_col: str, n2_col: str, n3_col: str) -> pd.Series:
    return (
        df[n1_col].notna()
        & df[n2_col].notna()
        & df[n3_col].notna()
        & df["n1_min"].ge(MIN_STAGE_MINUTES)
        & df["n2_min"].ge(MIN_STAGE_MINUTES)
        & df["n3_min"].ge(MIN_STAGE_MINUTES)
    )


def build_stage_face_validity(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for candidate, spec in CANDIDATES.items():
        stage_cols = spec["stage_cols"]
        rem_col = stage_cols["REM"]
        nrem_col = stage_cols["NREM"]
        n1_col = stage_cols["N1"]
        n2_col = stage_cols["N2"]
        n3_col = stage_cols["N3"]
        rem_mask = rem_nrem_mask(df, rem_col, nrem_col)
        nrem_diff = df.loc[rem_mask, rem_col] - df.loc[rem_mask, nrem_col]
        n123_mask = n1_n2_n3_mask(df, n1_col, n2_col, n3_col)
        n1n2 = df.loc[n123_mask, n1_col] - df.loc[n123_mask, n2_col]
        n2n3 = df.loc[n123_mask, n2_col] - df.loc[n123_mask, n3_col]
        positive_cohorts = 0
        cohort_diffs = []
        for cohort, grp in df.groupby("cohort"):
            c_mask = rem_nrem_mask(grp, rem_col, nrem_col)
            diff = grp.loc[c_mask, rem_col] - grp.loc[c_mask, nrem_col]
            med = float(diff.median()) if len(diff) else np.nan
            cohort_diffs.append(f"{cohort}:{med:.3f}" if pd.notna(med) else f"{cohort}:NA")
            if pd.notna(med) and med > 0:
                positive_cohorts += 1
        recommend_drop = bool(pd.notna(nrem_diff.median()) and nrem_diff.median() <= 0)
        rows.append(
            {
                "candidate": candidate,
                "label": spec["label"],
                "n_pair_rem_nrem": int(rem_mask.sum()),
                "rem_iqr": iqr_text(df.loc[rem_mask, rem_col]),
                "nrem_iqr": iqr_text(df.loc[rem_mask, nrem_col]),
                "median_rem_minus_nrem": float(nrem_diff.median()) if len(nrem_diff) else np.nan,
                "pct_rem_gt_nrem": float((nrem_diff > 0).mean() * 100) if len(nrem_diff) else np.nan,
                "n_pair_n1_n2_n3": int(n123_mask.sum()),
                "median_n1_minus_n2": float(n1n2.median()) if len(n1n2) else np.nan,
                "median_n2_minus_n3": float(n2n3.median()) if len(n2n3) else np.nan,
                "n_cohorts_positive_rem_rebound": int(positive_cohorts),
                "cohort_rem_minus_nrem_medians": "; ".join(cohort_diffs),
                "recommend_drop_before_outcomes": recommend_drop,
                "screening_note": "Fails REM>NREM face-validity" if recommend_drop else "",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "candidate_stage_face_validity.csv", index=False)
    return out


def pearson_summary(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), n


def spearman_summary(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def build_comparator_correlations(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    cohorts = ["overall"] + sorted(df["cohort"].dropna().unique().tolist())
    for cohort in cohorts:
        sub = df if cohort == "overall" else df[df["cohort"].eq(cohort)]
        for candidate, spec in CANDIDATES.items():
            x = safe_numeric(sub[spec["sleep_col"]])
            for comparator in COMPARATORS:
                y = safe_numeric(sub[comparator])
                rho, p_s, n_s = spearman_summary(x, y)
                r, p_p, n_p = pearson_summary(x, y)
                rows.append(
                    {
                        "cohort": cohort,
                        "candidate": candidate,
                        "label": spec["label"],
                        "comparator": comparator,
                        "n_spearman": n_s,
                        "spearman_rho": rho,
                        "spearman_p": p_s,
                        "n_pearson": n_p,
                        "pearson_r": r,
                        "pearson_p": p_p,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "candidate_comparator_correlations.csv", index=False)
    return out


def build_mortality_summary(mortality_dir: Path, outdir: Path) -> pd.DataFrame:
    cox_path = mortality_dir / "cox_model_results.csv"
    if not cox_path.exists():
        out = pd.DataFrame()
        out.to_csv(outdir / "candidate_mortality_summary.csv", index=False)
        return out
    cox = pd.read_csv(cox_path)
    candidate_names = list(CANDIDATES.keys())
    keep_models = [
        "m1_age_sex_bmi",
        "m2_age_sex_bmi_cardiometabolic",
        "m3_plus_ahi",
        "m4_plus_hypoxic_burden",
        "m5_plus_arousal_index",
        "m6_plus_ss",
    ]
    out = cox[
        cox["selection_mode"].eq("strict_primary")
        & cox["analysis_exposure"].isin(candidate_names)
        & cox["model_name"].isin(keep_models)
    ].copy()
    out.to_csv(outdir / "candidate_mortality_summary.csv", index=False)

    direction_rows = []
    for candidate in candidate_names:
        cohort_rows = out[
            out["analysis_exposure"].eq(candidate)
            & out["model_name"].eq("m1_age_sex_bmi")
            & out["analysis_group"].isin(["I0002", "S0001", "mros"])
            & out["status"].eq("ok")
        ].copy()
        dirs = []
        for _, row in cohort_rows.iterrows():
            if row["hr_per_sd"] > 1:
                dirs.append("positive")
            elif row["hr_per_sd"] < 1:
                dirs.append("negative")
            else:
                dirs.append("null")
        non_null = [d for d in dirs if d != "null"]
        same = bool(non_null and len(set(non_null)) == 1)
        direction_rows.append(
            {
                "candidate": candidate,
                "n_cohort_m1_fits": int(len(cohort_rows)),
                "cohort_m1_directions": "; ".join(f"{r['analysis_group']}:{r['hr_per_sd']:.3f}" for _, r in cohort_rows.iterrows()),
                "same_direction_across_available_cohorts": same,
            }
        )
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(outdir / "candidate_mortality_direction_by_cohort.csv", index=False)
    return out


def first_row(df: pd.DataFrame, **filters: object) -> pd.Series | None:
    sub = df.copy()
    for col, value in filters.items():
        sub = sub[sub[col].eq(value)]
    if sub.empty:
        return None
    return sub.iloc[0]


def fmt_hr(row: pd.Series | None) -> str:
    if row is None or row.empty or row.get("status") != "ok":
        return "not estimable"
    return f"{row['hr_per_sd']:.2f} ({row['ci_lower']:.2f}-{row['ci_upper']:.2f}), p={row['p_value']:.3g}, n={int(row['n']):,}"


def write_recommendation(
    outdir: Path,
    missingness: pd.DataFrame,
    face: pd.DataFrame,
    corr: pd.DataFrame,
    mortality: pd.DataFrame,
) -> pd.DataFrame:
    direction_path = outdir / "candidate_mortality_direction_by_cohort.csv"
    directions = pd.read_csv(direction_path) if direction_path.exists() else pd.DataFrame()
    rows = []
    for candidate, spec in CANDIDATES.items():
        miss = first_row(missingness, candidate=candidate)
        fv = first_row(face, candidate=candidate)
        direction = first_row(directions, candidate=candidate) if not directions.empty else None
        m2 = first_row(
            mortality,
            analysis_group="overall_stratified",
            analysis_exposure=candidate,
            model_name="m2_age_sex_bmi_cardiometabolic",
        )
        m4 = first_row(
            mortality,
            analysis_group="overall_stratified",
            analysis_exposure=candidate,
            model_name="m4_plus_hypoxic_burden",
        )
        drop = bool(fv is not None and fv.get("recommend_drop_before_outcomes", False))
        same_direction = bool(direction is not None and direction.get("same_direction_across_available_cohorts", False))
        if candidate == "bsi_median_sleep":
            recommendation = "primary_raw_bsi"
            rationale = "Physiologic anchor; good missingness; preserves REM>NREM; no raw candidate clearly outperforms it in adjusted mortality."
        elif drop:
            recommendation = "drop_from_primary"
            rationale = "Fails REM>NREM face-validity, so keep only as a sensitivity/de-emphasized candidate."
        else:
            recommendation = "supplement_candidate"
            rationale = "Face-valid raw candidate, but not strong enough to replace median BSI as the main exposure."
        rows.append(
            {
                "candidate": candidate,
                "label": spec["label"],
                "recommendation": recommendation,
                "rationale": rationale,
                "pct_nonmissing_strict_primary": float(miss["pct_nonmissing_strict_primary"]) if miss is not None else np.nan,
                "median_rem_minus_nrem": float(fv["median_rem_minus_nrem"]) if fv is not None else np.nan,
                "n_cohorts_positive_rem_rebound": int(fv["n_cohorts_positive_rem_rebound"]) if fv is not None else 0,
                "same_direction_across_available_mortality_cohorts": same_direction,
                "m2_overall_hr": fmt_hr(m2),
                "m4_plus_hypoxic_burden_overall_hr": fmt_hr(m4),
            }
        )
    rec = pd.DataFrame(rows)
    rec.to_csv(outdir / "candidate_recommendation.csv", index=False)

    ss_corr = corr[(corr["cohort"].eq("overall")) & (corr["comparator"].eq("ss_percent_main"))]
    ahi_corr = corr[(corr["cohort"].eq("overall")) & (corr["comparator"].eq("ahi"))]
    desat_corr = corr[(corr["cohort"].eq("overall")) & (corr["comparator"].eq("desaturation_index"))]
    med_ss = first_row(ss_corr, candidate="bsi_median_sleep")
    med_ahi = first_row(ahi_corr, candidate="bsi_median_sleep")
    med_desat = first_row(desat_corr, candidate="bsi_median_sleep")
    median_rec = rec[rec["candidate"].eq("bsi_median_sleep")].iloc[0]
    q90_rec = rec[rec["candidate"].eq("bsi_quantile_90_sleep")].iloc[0]

    lines = [
        "# Raw BSI Candidate Sensitivity v1",
        "",
        "## Recommendation",
        "- Keep `bsi_median_sleep` as the primary raw BSI exposure and physiologic anchor.",
        "- Keep `bsi_quantile_75_sleep` and `bsi_instability_burden_gt_1p5_sleep` as supplemental raw-candidate sensitivities.",
        "- Drop `bsi_quantile_90_sleep` from primary consideration because it fails the REM>NREM face-validity rule.",
        "",
        "## Evidence",
        f"- `bsi_median_sleep` strict-primary nonmissingness: `{median_rec['pct_nonmissing_strict_primary']:.1f}%`.",
        f"- `bsi_median_sleep` REM-NREM median difference: `{median_rec['median_rem_minus_nrem']:.3f}`.",
        f"- `bsi_median_sleep` adjusted mortality model 2: `{median_rec['m2_overall_hr']}`.",
        f"- `bsi_quantile_90_sleep` REM-NREM median difference: `{q90_rec['median_rem_minus_nrem']:.3f}`.",
        "",
        "## Comparator relationships for median BSI",
        f"- SS Spearman rho: `{med_ss['spearman_rho']:.3f}` (n={int(med_ss['n_spearman']):,})." if med_ss is not None else "- SS correlation not estimable.",
        f"- AHI Spearman rho: `{med_ahi['spearman_rho']:.3f}` (n={int(med_ahi['n_spearman']):,})." if med_ahi is not None else "- AHI correlation not estimable.",
        f"- Desaturation index Spearman rho: `{med_desat['spearman_rho']:.3f}` (n={int(med_desat['n_spearman']):,})." if med_desat is not None else "- Desaturation-index correlation not estimable.",
        "",
        "## Reviewer-facing implication",
        "- The revision should not introduce an outcome-trained BSI composite as the main exposure.",
        "- The main manuscript can remain centered on median raw BSI while showing that the conclusion is not dependent on a single arbitrary summary.",
    ]
    (outdir / "candidate_recommendation.md").write_text("\n".join(lines) + "\n")
    return rec


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_master(args.master_csv)
    missingness = build_missingness(df, args.output_dir)
    face = build_stage_face_validity(df, args.output_dir)
    corr = build_comparator_correlations(df, args.output_dir)
    mortality = build_mortality_summary(args.mortality_dir, args.output_dir)
    rec = write_recommendation(args.output_dir, missingness, face, corr, mortality)
    pd.DataFrame(
        [
            {
                "rows_in_master": int(len(df)),
                "n_candidates": int(len(CANDIDATES)),
                "recommended_primary": "bsi_median_sleep",
                "n_candidate_recommendation_rows": int(len(rec)),
            }
        ]
    ).to_csv(args.output_dir / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
