#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
DEFAULT_OUTDIR = REPO_ROOT / "REVISION" / "study_type_sensitivity_v1"

METRICS = {
    "bsi_median_sleep": "BSI median sleep",
    "bsi_quantile_75_sleep": "BSI q75 sleep",
    "bsi_instability_burden_gt_1p5_sleep": "BSI instability burden >1.5 sleep",
    "ss_percent_main": "Self-similarity percent",
    "ahi": "AHI",
    "hypoxic_burden": "Hypoxic burden",
}

PAIR_COMPARISONS = {
    "pap_titration_vs_strict": {
        "strict_label": "strict_diagnostic",
        "comparator_label": "pap_titration",
        "comparator_type": "pap_titration",
        "difference": "pap_titration_minus_strict_diagnostic",
    },
    "split_night_vs_strict": {
        "strict_label": "strict_diagnostic",
        "comparator_label": "split_night",
        "comparator_type": "split_night",
        "difference": "split_night_minus_strict_diagnostic",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R2-4 study-type repeat/split/PAP sensitivity analysis.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def safe_bool(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.map({"true": True, "1": True, "1.0": True, "yes": True, "false": False, "0": False, "0.0": False, "no": False})


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def required_columns() -> list[str]:
    return [
        "cohort",
        "sid",
        "fileid",
        "visitno",
        "psg_date",
        "primary_subject_psg_rank",
        "study_type",
        "psg_type",
        "paper_psg_type",
        "paper_primary_psg_eligible",
        *METRICS.keys(),
    ]


def read_master(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in required_columns() if col in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in required_columns():
        if col not in df.columns:
            df[col] = np.nan

    for col in ["cohort", "sid", "fileid", "visitno", "study_type", "psg_type", "paper_psg_type"]:
        df[col] = df[col].fillna("").astype(str)
    df["paper_psg_type"] = df["paper_psg_type"].str.strip().str.lower()
    df["paper_primary_psg_eligible_bool"] = safe_bool(df["paper_primary_psg_eligible"])
    df["psg_date_parsed"] = pd.to_datetime(df["psg_date"], errors="coerce")
    df["primary_subject_psg_rank_num"] = safe_numeric(df["primary_subject_psg_rank"])
    for metric in METRICS:
        df[metric] = safe_numeric(df[metric])
    return df


def sorted_subject_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["cohort", "sid", "psg_date_parsed", "primary_subject_psg_rank_num", "fileid"],
        na_position="last",
        kind="mergesort",
    )


def build_subject_inventory(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    type_values = sorted([x for x in df["paper_psg_type"].dropna().unique() if str(x)])
    ordered = sorted_subject_frame(df).copy()
    ordered["strict_int"] = ordered["paper_primary_psg_eligible_bool"].eq(True).astype(int)

    base = (
        ordered.groupby(["cohort", "sid"], sort=False, dropna=False)
        .agg(n_psgs=("fileid", "size"), n_strict_diagnostic=("strict_int", "sum"))
        .reset_index()
    )
    type_counts = (
        ordered.groupby(["cohort", "sid", "paper_psg_type"], sort=False, dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    rename = {psg_type: f"n_{str(psg_type).replace('-', '_')}" for psg_type in type_values}
    type_counts = type_counts.rename(columns=rename)
    out = base.merge(type_counts, on=["cohort", "sid"], how="left")
    for psg_type in type_values:
        col = f"n_{str(psg_type).replace('-', '_')}"
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)
    out["has_strict_diagnostic"] = out["n_strict_diagnostic"].gt(0)
    out["has_pap_titration"] = out.get("n_pap_titration", pd.Series(0, index=out.index)).gt(0)
    out["has_split_night"] = out.get("n_split_night", pd.Series(0, index=out.index)).gt(0)
    out["has_strict_and_pap_titration"] = out["has_strict_diagnostic"] & out["has_pap_titration"]
    out["has_strict_and_split_night"] = out["has_strict_diagnostic"] & out["has_split_night"]
    out["has_repeat_strict_diagnostic"] = out["n_strict_diagnostic"].ge(2)
    out.to_csv(outdir / "study_type_inventory_by_subject.csv", index=False)
    return out


def build_cohort_inventory(df: pd.DataFrame, subject_inventory: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    type_values = sorted([x for x in df["paper_psg_type"].dropna().unique() if str(x)])
    rows = []
    for cohort, grp in df.groupby("cohort", dropna=False):
        subjects = subject_inventory[subject_inventory["cohort"].eq(cohort)]
        row: dict[str, object] = {
            "cohort": cohort,
            "n_psg_rows": int(len(grp)),
            "n_subjects": int(len(subjects)),
            "n_strict_diagnostic_rows": int(grp["paper_primary_psg_eligible_bool"].eq(True).sum()),
            "n_strict_diagnostic_subjects": int(subjects["has_strict_diagnostic"].sum()),
            "n_subjects_strict_and_pap_titration": int(subjects["has_strict_and_pap_titration"].sum()),
            "n_subjects_strict_and_split_night": int(subjects["has_strict_and_split_night"].sum()),
            "n_subjects_repeat_strict_diagnostic": int(subjects["has_repeat_strict_diagnostic"].sum()),
        }
        for psg_type in type_values:
            safe_name = str(psg_type).replace("-", "_")
            row[f"n_{safe_name}_rows"] = int(grp["paper_psg_type"].eq(psg_type).sum())
            row[f"n_{safe_name}_subjects"] = int(subjects[f"n_{safe_name}"].gt(0).sum()) if f"n_{safe_name}" in subjects else 0
        rows.append(row)

    all_row: dict[str, object] = {
        "cohort": "ALL",
        "n_psg_rows": int(len(df)),
        "n_subjects": int(len(subject_inventory)),
        "n_strict_diagnostic_rows": int(df["paper_primary_psg_eligible_bool"].eq(True).sum()),
        "n_strict_diagnostic_subjects": int(subject_inventory["has_strict_diagnostic"].sum()),
        "n_subjects_strict_and_pap_titration": int(subject_inventory["has_strict_and_pap_titration"].sum()),
        "n_subjects_strict_and_split_night": int(subject_inventory["has_strict_and_split_night"].sum()),
        "n_subjects_repeat_strict_diagnostic": int(subject_inventory["has_repeat_strict_diagnostic"].sum()),
    }
    for psg_type in type_values:
        safe_name = str(psg_type).replace("-", "_")
        all_row[f"n_{safe_name}_rows"] = int(df["paper_psg_type"].eq(psg_type).sum())
        all_row[f"n_{safe_name}_subjects"] = int(subject_inventory[f"n_{safe_name}"].gt(0).sum())
    rows.append(all_row)

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "study_type_inventory_by_cohort.csv", index=False)
    return out


def first_row(grp: pd.DataFrame) -> pd.Series:
    return grp.iloc[0]


def selected_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = sorted_subject_frame(df)
    strict_first = ordered[ordered["paper_primary_psg_eligible_bool"].eq(True)].groupby(["cohort", "sid"], sort=False, dropna=False).head(1)
    for comparison, spec in PAIR_COMPARISONS.items():
        comp_first = ordered[ordered["paper_psg_type"].eq(spec["comparator_type"])].groupby(["cohort", "sid"], sort=False, dropna=False).head(1)
        if comp_first.empty:
            continue
        merged = strict_first.merge(comp_first, on=["cohort", "sid"], suffixes=("_strict", "_comparator"), how="inner")
        for _, pair in merged.iterrows():
            base = {
                "comparison": comparison,
                "cohort": pair["cohort"],
                "sid": pair["sid"],
                "strict_fileid": pair["fileid_strict"],
                "strict_psg_date": pair["psg_date_strict"],
                "strict_paper_psg_type": pair["paper_psg_type_strict"],
                "comparator_fileid": pair["fileid_comparator"],
                "comparator_psg_date": pair["psg_date_comparator"],
                "comparator_paper_psg_type": pair["paper_psg_type_comparator"],
                "difference_direction": spec["difference"],
            }
            for metric in METRICS:
                strict_value = pair[f"{metric}_strict"]
                comp_value = pair[f"{metric}_comparator"]
                base[f"{metric}_strict"] = strict_value
                base[f"{metric}_comparator"] = comp_value
                base[f"{metric}_diff"] = comp_value - strict_value if pd.notna(strict_value) and pd.notna(comp_value) else np.nan
            rows.append(base)
    return pd.DataFrame(rows)


def wilcoxon_from_diff(diff: pd.Series) -> tuple[float, float]:
    values = safe_numeric(diff).dropna()
    if values.empty:
        return np.nan, np.nan
    nonzero = values[values.ne(0)]
    if nonzero.empty:
        return 0.0, 1.0
    stat, p_value = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p_value)


def correlation_pair(x: pd.Series, y: pd.Series, method: str) -> tuple[int, float, float]:
    pair = pd.DataFrame({"x": safe_numeric(x), "y": safe_numeric(y)}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return int(len(pair)), np.nan, np.nan
    if method == "spearman":
        stat = stats.spearmanr(pair["x"], pair["y"])
    elif method == "pearson":
        stat = stats.pearsonr(pair["x"], pair["y"])
    else:
        raise ValueError(method)
    return int(len(pair)), float(stat.statistic), float(stat.pvalue)


def summarize_pair_tests(pairs: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for comparison, spec in PAIR_COMPARISONS.items():
        sub = pairs[pairs["comparison"].eq(comparison)]
        for metric, label in METRICS.items():
            diff = sub[f"{metric}_diff"]
            paired = diff.dropna()
            stat, p_value = wilcoxon_from_diff(diff)
            n_corr, rho, rho_p = correlation_pair(sub[f"{metric}_strict"], sub[f"{metric}_comparator"], "spearman")
            _, pearson_r, pearson_p = correlation_pair(sub[f"{metric}_strict"], sub[f"{metric}_comparator"], "pearson")
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "metric_label": label,
                    "difference_direction": spec["difference"],
                    "n_subject_pairs_available": int(len(sub)),
                    "n_metric_pairs_nonmissing": int(len(paired)),
                    "strict_median": float(sub[f"{metric}_strict"].median()) if sub[f"{metric}_strict"].notna().any() else np.nan,
                    "comparator_median": float(sub[f"{metric}_comparator"].median()) if sub[f"{metric}_comparator"].notna().any() else np.nan,
                    "median_paired_difference": float(paired.median()) if len(paired) else np.nan,
                    "median_abs_paired_difference": float(paired.abs().median()) if len(paired) else np.nan,
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": p_value,
                    "spearman_n": n_corr,
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "paired_study_type_comparisons.csv", index=False)
    return out


def selected_repeat_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strict_df = sorted_subject_frame(df[df["paper_primary_psg_eligible_bool"].eq(True)]).copy()
    strict_df["_strict_order"] = strict_df.groupby(["cohort", "sid"], sort=False, dropna=False).cumcount()
    strict_counts = strict_df.groupby(["cohort", "sid"], sort=False, dropna=False).size().to_dict()
    first_rows = strict_df[strict_df["_strict_order"].eq(0)]
    second_rows = strict_df[strict_df["_strict_order"].eq(1)]
    merged = first_rows.merge(second_rows, on=["cohort", "sid"], suffixes=("_first", "_second"), how="inner")
    for _, pair in merged.iterrows():
        key = (pair["cohort"], pair["sid"])
        row = {
            "cohort": pair["cohort"],
            "sid": pair["sid"],
            "n_strict_diagnostic_psgs": int(strict_counts[key]),
            "first_fileid": pair["fileid_first"],
            "first_psg_date": pair["psg_date_first"],
            "second_fileid": pair["fileid_second"],
            "second_psg_date": pair["psg_date_second"],
            "difference_direction": "second_strict_diagnostic_minus_first_strict_diagnostic",
        }
        for metric in METRICS:
            first_value = pair[f"{metric}_first"]
            second_value = pair[f"{metric}_second"]
            row[f"{metric}_first"] = first_value
            row[f"{metric}_second"] = second_value
            row[f"{metric}_diff"] = second_value - first_value if pd.notna(first_value) and pd.notna(second_value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_repeat_tests(repeats: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for metric, label in METRICS.items():
        if repeats.empty:
            diff = pd.Series(dtype=float)
            first = pd.Series(dtype=float)
            second = pd.Series(dtype=float)
        else:
            diff = repeats[f"{metric}_diff"]
            first = repeats[f"{metric}_first"]
            second = repeats[f"{metric}_second"]
        paired = diff.dropna()
        stat, p_value = wilcoxon_from_diff(diff)
        n_corr, rho, rho_p = correlation_pair(first, second, "spearman")
        _, pearson_r, pearson_p = correlation_pair(first, second, "pearson")
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "difference_direction": "second_strict_diagnostic_minus_first_strict_diagnostic",
                "n_subject_pairs_available": int(len(repeats)),
                "n_metric_pairs_nonmissing": int(len(paired)),
                "first_median": float(first.median()) if first.notna().any() else np.nan,
                "second_median": float(second.median()) if second.notna().any() else np.nan,
                "median_paired_difference": float(paired.median()) if len(paired) else np.nan,
                "median_abs_paired_difference": float(paired.abs().median()) if len(paired) else np.nan,
                "wilcoxon_stat": stat,
                "wilcoxon_p": p_value,
                "spearman_n": n_corr,
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "repeat_strict_diagnostic_stability.csv", index=False)
    return out


def fmt_int(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(value):,}"


def fmt_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def markdown_pair_table(summary: pd.DataFrame, comparison: str) -> list[str]:
    rows = summary[summary["comparison"].eq(comparison)].copy()
    lines = [
        "| Metric | N complete pairs | Median paired difference | Wilcoxon p |",
        "|---|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['metric']} | {fmt_int(row['n_metric_pairs_nonmissing'])} | "
            f"{fmt_float(row['median_paired_difference'])} | {fmt_float(row['wilcoxon_p'])} |"
        )
    return lines


def markdown_repeat_table(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Metric | N complete pairs | Median paired difference | Median absolute difference | Spearman rho | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['metric']} | {fmt_int(row['n_metric_pairs_nonmissing'])} | "
            f"{fmt_float(row['median_paired_difference'])} | {fmt_float(row['median_abs_paired_difference'])} | "
            f"{fmt_float(row['spearman_rho'])} | {fmt_float(row['wilcoxon_p'])} |"
        )
    return lines


def write_markdown_note(
    df: pd.DataFrame,
    cohort_inventory: pd.DataFrame,
    subject_inventory: pd.DataFrame,
    pair_summary: pd.DataFrame,
    repeat_summary: pd.DataFrame,
    outdir: Path,
) -> None:
    all_inv = cohort_inventory[cohort_inventory["cohort"].eq("ALL")].iloc[0]
    diagnostic_mismatch = int((df["paper_primary_psg_eligible_bool"].eq(True) & ~df["paper_psg_type"].eq("diagnostic")).sum())
    pap_bsi = pair_summary[(pair_summary["comparison"].eq("pap_titration_vs_strict")) & (pair_summary["metric"].eq("bsi_median_sleep"))].iloc[0]
    split_bsi = pair_summary[(pair_summary["comparison"].eq("split_night_vs_strict")) & (pair_summary["metric"].eq("bsi_median_sleep"))].iloc[0]
    repeat_bsi = repeat_summary[repeat_summary["metric"].eq("bsi_median_sleep")].iloc[0]

    lines = [
        "# R2-4 Study-Type Sensitivity Note",
        "",
        "Source: `REVISION/master_analysis_table_primary_cohorts_v4_paper.csv`.",
        "",
        "## Inventory",
        f"- PSG rows: `{fmt_int(len(df))}` across `{fmt_int(len(subject_inventory))}` cohort-subject records.",
        f"- Strict diagnostic rows (`paper_primary_psg_eligible == True`): `{fmt_int(all_inv['n_strict_diagnostic_rows'])}` across `{fmt_int(all_inv['n_strict_diagnostic_subjects'])}` subjects.",
        f"- PAP/titration rows: `{fmt_int(all_inv.get('n_pap_titration_rows', np.nan))}` across `{fmt_int(all_inv.get('n_pap_titration_subjects', np.nan))}` subjects.",
        f"- Split-night rows: `{fmt_int(all_inv.get('n_split_night_rows', np.nan))}` across `{fmt_int(all_inv.get('n_split_night_subjects', np.nan))}` subjects.",
        f"- Subjects with paired strict diagnostic and PAP/titration PSGs: `{fmt_int(all_inv['n_subjects_strict_and_pap_titration'])}`.",
        f"- Subjects with paired strict diagnostic and split-night PSGs: `{fmt_int(all_inv['n_subjects_strict_and_split_night'])}`.",
        f"- Subjects with repeat strict diagnostic PSGs: `{fmt_int(all_inv['n_subjects_repeat_strict_diagnostic'])}`.",
        f"- Strict-eligibility rows not labeled `diagnostic`: `{fmt_int(diagnostic_mismatch)}`.",
        "",
        "## Paired Sensitivity",
        "Within each cohort-subject, the earliest strict diagnostic PSG was paired with the earliest comparator PSG by parsed PSG date, primary subject PSG rank, then fileid. Differences are comparator minus strict diagnostic.",
        f"- PAP/titration vs strict diagnostic: `{fmt_int(pap_bsi['n_subject_pairs_available'])}` subject pairs; for `bsi_median_sleep`, `{fmt_int(pap_bsi['n_metric_pairs_nonmissing'])}` metric-complete pairs had median paired difference `{fmt_float(pap_bsi['median_paired_difference'])}` and Wilcoxon p `{fmt_float(pap_bsi['wilcoxon_p'])}`.",
        f"- Split-night vs strict diagnostic: `{fmt_int(split_bsi['n_subject_pairs_available'])}` subject pairs; for `bsi_median_sleep`, `{fmt_int(split_bsi['n_metric_pairs_nonmissing'])}` metric-complete pairs had median paired difference `{fmt_float(split_bsi['median_paired_difference'])}` and Wilcoxon p `{fmt_float(split_bsi['wilcoxon_p'])}`.",
        "",
        "PAP/titration minus strict diagnostic:",
        "",
        *markdown_pair_table(pair_summary, "pap_titration_vs_strict"),
        "",
        "Split-night minus strict diagnostic:",
        "",
        *markdown_pair_table(pair_summary, "split_night_vs_strict"),
        "",
        "## Repeat Strict Diagnostic Stability",
        "Among subjects with multiple strict diagnostic PSGs, the earliest two strict diagnostic PSGs were compared as second minus first.",
        f"- Repeat strict diagnostic pairs: `{fmt_int(repeat_bsi['n_subject_pairs_available'])}` subjects; for `bsi_median_sleep`, `{fmt_int(repeat_bsi['n_metric_pairs_nonmissing'])}` metric-complete pairs had median paired difference `{fmt_float(repeat_bsi['median_paired_difference'])}`, median absolute difference `{fmt_float(repeat_bsi['median_abs_paired_difference'])}`, Spearman rho `{fmt_float(repeat_bsi['spearman_rho'])}`, and Wilcoxon p `{fmt_float(repeat_bsi['wilcoxon_p'])}`.",
        "",
        *markdown_repeat_table(repeat_summary),
        "",
        "## Reviewer-Facing Interpretation",
        "The strict primary cohort is cleanly diagnostic by the harmonized paper PSG type, and paired non-primary study types are present but are a minority of subjects. PAP/titration comparisons estimate treatment-study sensitivity and should not be interpreted as pure measurement repeatability. Split-night comparisons are also procedural sensitivity checks because the night includes both diagnostic and treatment portions. Repeat strict diagnostic pairs are the most direct stability check; their full metric-specific medians, absolute differences, correlations, and Wilcoxon tests are provided for transparent reviewer response.",
        "",
        "## Output Files",
        "- `study_type_inventory_by_cohort.csv`",
        "- `study_type_inventory_by_subject.csv`",
        "- `paired_study_type_pair_rows.csv`",
        "- `paired_study_type_comparisons.csv`",
        "- `repeat_strict_diagnostic_pair_rows.csv`",
        "- `repeat_strict_diagnostic_stability.csv`",
        "- `study_type_sensitivity_note.md`",
        "",
    ]
    (outdir / "study_type_sensitivity_note.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = read_master(args.master_csv)
    print(f"Loaded {len(master):,} PSG rows", flush=True)
    subject_inventory = build_subject_inventory(master, args.output_dir)
    cohort_inventory = build_cohort_inventory(master, subject_inventory, args.output_dir)
    print("Wrote study-type inventory", flush=True)

    pairs = selected_pair_rows(master)
    pairs.to_csv(args.output_dir / "paired_study_type_pair_rows.csv", index=False)
    pair_summary = summarize_pair_tests(pairs, args.output_dir)
    print(f"Wrote paired comparison tables for {len(pairs):,} selected pairs", flush=True)

    repeats = selected_repeat_rows(master)
    repeats.to_csv(args.output_dir / "repeat_strict_diagnostic_pair_rows.csv", index=False)
    repeat_summary = summarize_repeat_tests(repeats, args.output_dir)
    print(f"Wrote repeat stability tables for {len(repeats):,} selected pairs", flush=True)

    write_markdown_note(master, cohort_inventory, subject_inventory, pair_summary, repeat_summary, args.output_dir)

    print(f"Wrote study-type sensitivity outputs to {args.output_dir}")
    print(f"Paired strict/PAP subjects: {int(cohort_inventory.loc[cohort_inventory['cohort'].eq('ALL'), 'n_subjects_strict_and_pap_titration'].iloc[0])}")
    print(f"Paired strict/split subjects: {int(cohort_inventory.loc[cohort_inventory['cohort'].eq('ALL'), 'n_subjects_strict_and_split_night'].iloc[0])}")
    print(f"Repeat strict diagnostic subjects: {int(cohort_inventory.loc[cohort_inventory['cohort'].eq('ALL'), 'n_subjects_repeat_strict_diagnostic'].iloc[0])}")


if __name__ == "__main__":
    main()
