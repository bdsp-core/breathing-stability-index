#!/usr/bin/env python3
"""
Build the breathing-stability revision master analysis table.

This script starts from the broad `master_morpheus.csv` scaffold and adds a
fixed set of analysis-oriented columns. Optional joins can pull in already
processed BSI, SS, respiratory, macrostructure, timing, and outcome tables.

The goal is to keep one row per PSG while allowing missing data to remain NA
until each upstream pipeline is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION_DIR = Path(__file__).resolve().parent
revision_dir_text = str(REVISION_DIR)
if revision_dir_text not in sys.path:
    sys.path.insert(0, revision_dir_text)

from sleep_general_revision_features import load_sleep_general_revision_table

DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_morpheus.csv"
DEFAULT_OUTPUT = REPO_ROOT / "REVISION" / "master_analysis_table_v0.csv"
DEFAULT_AUDIT = REPO_ROOT / "REVISION" / "master_analysis_table_v0_audit.md"


PLACEHOLDER_COLUMNS = [
    "study_type",
    "site",
    "file_path",
    "psg_date",
    "signal_qc_pass",
    "signal_qc_note",
    "analyzable_duration_min",
    "analyzable_duration_n1_min",
    "analyzable_duration_n2_min",
    "analyzable_duration_n3_min",
    "analyzable_duration_rem_min",
    "primary_analysis_flag",
    "primary_analysis_reason",
    "primary_subject_psg_rank",
    "is_repeat_visit",
    "is_treatment_study",
    "cpap_present",
    "ahi",
    "rdi",
    "oai",
    "cai",
    "mai",
    "hyi",
    "rerai",
    "hypoxic_burden",
    "arousal_index",
    "desaturation_index",
    "odi3",
    "odi4",
    "event_duration_burden",
    "desaturation_burden",
    "tst_min",
    "sleep_efficiency",
    "waso_min",
    "n1_min",
    "n2_min",
    "n3_min",
    "rem_min",
    "n1_pct",
    "n2_pct",
    "n3_pct",
    "rem_pct",
    "supine_tst_min",
    "supine_pct_tst",
    "posture_available",
    "phenotype_osa_csa",
    "has_bsi_robust_mean",
    "has_bsi_summary_mean",
    "has_ss",
    "has_respiratory_metrics",
    "has_macrostructure",
    "bsi_primary_belt_mode",
    "bsi_primary_belt_channels_used",
    "bsi_primary_n_belts_used",
    "bsi_median_sleep",
    "bsi_quantile_25_sleep",
    "bsi_quantile_75_sleep",
    "bsi_quantile_90_sleep",
    "bsi_instability_burden_gt_1p5_sleep",
    "bsi_median_n1",
    "bsi_median_n2",
    "bsi_median_n3",
    "bsi_median_rem",
    "ss_main_channel",
    "ss_signal_duration_h",
    "ss_percent_main",
    "ss_percent_any",
    "death_date",
    "censor_date",
    "followup_days",
    "vital_status",
    "cognition_date",
    "cognition_days_from_psg",
]


CORE_AVAILABILITY_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "s3_path",
    "study_type",
    "ahi",
    "hypoxic_burden",
    "desaturation_index",
    "odi3",
    "odi4",
    "tst_min",
    "bsi_median_sleep",
    "ss_percent_main",
    "death_date",
    "followup_days",
]


RESPIRATORY_RENAME_MAP = {
    "f_ahi": "ahi",
    "f_rdi": "rdi",
    "f_oai": "oai",
    "f_cai": "cai",
    "f_mai": "mai",
    "f_hyi": "hyi",
    "f_rerai": "rerai",
    "f_hypoxia_burden": "hypoxic_burden",
    "hb_sleep": "hypoxic_burden",
}


MACRO_RENAME_MAP = {
    "total_sleep_time_min": "tst_min",
    "tst": "tst_min",
    "sleep_efficiency_pct": "sleep_efficiency",
    "sleep_efficiency": "sleep_efficiency",
    "waso": "waso_min",
    "waso_min": "waso_min",
    "n1_minutes": "n1_min",
    "n2_minutes": "n2_min",
    "n3_minutes": "n3_min",
    "rem_minutes": "rem_min",
    "n1_percent": "n1_pct",
    "n2_percent": "n2_pct",
    "n3_percent": "n3_pct",
    "rem_percent": "rem_pct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the revision master analysis table.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--keep-cohorts",
        type=str,
        default=None,
        help="Comma-separated cohort names to retain. If omitted, keep all cohorts.",
    )
    parser.add_argument(
        "--bsi-run-root",
        type=Path,
        action="append",
        default=[],
        help="Optional breathing-analysis run root(s) containing shard BSI outputs.",
    )
    parser.add_argument(
        "--ss-run-root",
        type=Path,
        action="append",
        default=[],
        help="Optional breathing-analysis run root(s) containing shard SS outputs.",
    )
    parser.add_argument("--respiratory-csv", type=Path, default=None)
    parser.add_argument("--macrostructure-csv", type=Path, default=None)
    parser.add_argument("--sleep-general-csv", type=Path, default=None)
    parser.add_argument("--timing-csv", type=Path, default=None)
    parser.add_argument("--outcomes-csv", type=Path, default=None)
    parser.add_argument(
        "--audit-title",
        default=None,
        help="Optional markdown heading for the audit file. Defaults to a title derived from the output CSV name.",
    )
    return parser.parse_args()


def normalize_fileid(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return Path(text).stem


def sanitize_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def format_float_tag(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return sanitize_name(str(value))
    text = f"{number:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def add_placeholder_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in PLACEHOLDER_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def build_base_master(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv, low_memory=False)
    if "fileid" not in df.columns:
        raise ValueError(f"{master_csv} is missing required column 'fileid'")
    df["fileid"] = df["fileid"].map(normalize_fileid)
    if "recording_type" in df.columns and "study_type" not in df.columns:
        df["study_type"] = df["recording_type"]
    if "cohort" in df.columns and "site" not in df.columns:
        df["site"] = df["cohort"]
    if "visitno" in df.columns:
        df["is_repeat_visit"] = pd.to_numeric(df["visitno"], errors="coerce").gt(1)
    if "recording_type" in df.columns:
        recording_text = df["recording_type"].astype(str).str.lower()
        df["is_treatment_study"] = recording_text.str.contains("cpap|pap|titration")
    return add_placeholder_columns(df)


def filter_cohorts(df: pd.DataFrame, keep_cohorts: str | None) -> tuple[pd.DataFrame, list[str]]:
    if keep_cohorts is None:
        return df, []
    cohorts = [item.strip() for item in keep_cohorts.split(",") if item.strip()]
    if not cohorts:
        return df, []
    filtered = df[df["cohort"].isin(cohorts)].copy()
    return filtered, cohorts


def concat_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_bsi_long(run_root: Path) -> pd.DataFrame:
    paths = sorted(run_root.glob("shards/shard_*/bsi/*/stability_index/summary_stability.csv"))
    df = concat_csvs(paths)
    if df.empty:
        return df
    if "file" not in df.columns:
        raise ValueError(f"BSI summaries under {run_root} do not contain a 'file' column.")
    df["fileid"] = df["file"].map(normalize_fileid)
    dedupe_cols = [c for c in ["fileid", "window_length", "overlap", "belt_mode"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")
    return df


def pivot_bsi_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["fileid"])
    records: dict[str, dict[str, object]] = {}
    for _, row in df_long.iterrows():
        fileid = normalize_fileid(row.get("fileid") or row.get("file"))
        if fileid is None:
            continue
        belt_mode = sanitize_name(str(row.get("belt_mode", "unknown")))
        window_tag = format_float_tag(row.get("window_length"))
        overlap_tag = format_float_tag(row.get("overlap"))
        prefix = f"bsi_{belt_mode}_w{window_tag}_ov{overlap_tag}"
        record = records.setdefault(fileid, {"fileid": fileid})
        for column, value in row.items():
            if column in {"file", "fileid", "window_length", "overlap", "belt_mode"}:
                continue
            record[f"{prefix}_{sanitize_name(column)}"] = value
    wide = pd.DataFrame(records.values())
    if wide.empty:
        return pd.DataFrame(columns=["fileid"])

    canonical_map = {
        "bsi_robust_mean_w2_ov0p9_sleep_median_stability": "bsi_median_sleep",
        "bsi_robust_mean_w2_ov0p9_sleep_quantile_25": "bsi_quantile_25_sleep",
        "bsi_robust_mean_w2_ov0p9_sleep_quantile_75": "bsi_quantile_75_sleep",
        "bsi_robust_mean_w2_ov0p9_sleep_quantile_90": "bsi_quantile_90_sleep",
        "bsi_robust_mean_w2_ov0p9_n1_median_stability": "bsi_median_n1",
        "bsi_robust_mean_w2_ov0p9_n2_median_stability": "bsi_median_n2",
        "bsi_robust_mean_w2_ov0p9_n3_median_stability": "bsi_median_n3",
        "bsi_robust_mean_w2_ov0p9_rem_median_stability": "bsi_median_rem",
        "bsi_robust_mean_w2_ov0p9_belt_channels_used": "bsi_primary_belt_channels_used",
        "bsi_robust_mean_w2_ov0p9_n_belts_used": "bsi_primary_n_belts_used",
    }
    for source, target in canonical_map.items():
        if source in wide.columns:
            wide[target] = wide[source]
    stable_col = "bsi_robust_mean_w2_ov0p9_sleep_n_5min_windows_stable"
    unstable_col = "bsi_robust_mean_w2_ov0p9_sleep_n_5min_windows_unstable"
    if stable_col in wide.columns and unstable_col in wide.columns:
        denom = pd.to_numeric(wide[stable_col], errors="coerce") + pd.to_numeric(wide[unstable_col], errors="coerce")
        burden = pd.to_numeric(wide[unstable_col], errors="coerce") / denom
        burden[denom <= 0] = np.nan
        wide["bsi_instability_burden_gt_1p5_sleep"] = burden
    if "bsi_robust_mean_w2_ov0p9_belt_channels_used" in wide.columns:
        wide["bsi_primary_belt_mode"] = "robust_mean"
    wide["has_bsi_robust_mean"] = wide.filter(regex=r"^bsi_robust_mean_").notna().any(axis=1)
    wide["has_bsi_summary_mean"] = wide.filter(regex=r"^bsi_summary_mean_").notna().any(axis=1)
    return wide


def collect_ss_wide(run_root: Path) -> pd.DataFrame:
    paths = sorted(run_root.glob("shards/shard_*/self/results_self_similarity.csv"))
    df = concat_csvs(paths)
    if df.empty:
        return pd.DataFrame(columns=["fileid"])
    if "fileid" not in df.columns:
        if "file" not in df.columns:
            raise ValueError(f"SS summaries under {run_root} do not contain 'fileid' or 'file'.")
        df["fileid"] = df["file"].map(normalize_fileid)
    else:
        df["fileid"] = df["fileid"].map(normalize_fileid)
    df = df.drop_duplicates(subset=["fileid"], keep="first")
    rename_map = {}
    for column in df.columns:
        if column == "fileid":
            continue
        clean = sanitize_name(column)
        rename_map[column] = clean if clean.startswith("ss_") else f"ss_{clean}"
    df = df.rename(columns=rename_map)
    df["has_ss"] = df.filter(regex=r"^ss_").notna().any(axis=1)
    return df


def prepare_join_table(
    path: Path | None,
    rename_map: dict[str, str] | None = None,
    keep_prefix: str | None = None,
) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["fileid"])
    df = pd.read_csv(path, low_memory=False)
    if "fileid" not in df.columns:
        candidate = "file" if "file" in df.columns else None
        if candidate is None:
            raise ValueError(f"{path} must contain 'fileid' or 'file'.")
        df["fileid"] = df[candidate].map(normalize_fileid)
    else:
        df["fileid"] = df["fileid"].map(normalize_fileid)
    if rename_map:
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if keep_prefix:
        rename_cols = {}
        for column in df.columns:
            if column == "fileid":
                continue
            clean = sanitize_name(column)
            rename_cols[column] = clean if clean.startswith(keep_prefix) else f"{keep_prefix}_{clean}"
        df = df.rename(columns=rename_cols)
    df = df.drop_duplicates(subset=["fileid"], keep="first")
    return df


def join_table(master: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return master
    if "fileid" not in table.columns:
        raise ValueError("Join table must contain a 'fileid' column.")
    overlap = [c for c in table.columns if c != "fileid" and c in master.columns]
    renamed_overlap = {column: f"__join_{column}" for column in overlap}
    if renamed_overlap:
        table = table.rename(columns=renamed_overlap)
    merged = master.merge(table, on="fileid", how="left")
    for column, joined_column in renamed_overlap.items():
        merged[column] = merged[joined_column].combine_first(merged[column])
        merged = merged.drop(columns=[joined_column])
    return merged


def finalize_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "bsi_median_sleep" in df.columns:
        df["has_bsi_robust_mean"] = df["bsi_median_sleep"].notna()
    if "ss_percent_main" in df.columns:
        df["has_ss"] = df["ss_percent_main"].notna()
    respiratory_cols = [c for c in ["ahi", "rdi", "oai", "cai", "hypoxic_burden"] if c in df.columns]
    if respiratory_cols:
        df["has_respiratory_metrics"] = df[respiratory_cols].notna().any(axis=1)
    macro_cols = [c for c in ["tst_min", "sleep_efficiency", "n1_pct", "rem_pct"] if c in df.columns]
    if macro_cols:
        df["has_macrostructure"] = df[macro_cols].notna().any(axis=1)
    return df


def default_audit_title(output_csv: Path) -> str:
    words = output_csv.stem.replace("-", "_").split("_")
    pretty = " ".join(word.upper() if word.lower() == "v1" else word.capitalize() for word in words if word)
    return f"{pretty} Audit" if pretty else "Master Analysis Table Audit"


def write_audit(
    df: pd.DataFrame,
    audit_md: Path,
    sources: list[str],
    retained_cohorts: list[str],
    *,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Rows: `{len(df):,}`",
        f"- Unique subjects (`sid`): `{df['sid'].nunique(dropna=True):,}`" if "sid" in df.columns else "- Unique subjects (`sid`): `n/a`",
        f"- Unique PSGs (`fileid`): `{df['fileid'].nunique(dropna=True):,}`",
        "",
        "## Sources",
        "",
    ]
    if sources:
        lines.extend([f"- `{source}`" for source in sources])
    else:
        lines.append("- Base master only; no optional feature joins were available during this build.")
    if retained_cohorts:
        lines.extend(
            [
                "",
                "## Retained Cohorts",
                "",
                *(f"- `{cohort}`" for cohort in retained_cohorts),
            ]
        )
    lines.extend(
        [
            "",
            "## Core Availability",
            "",
            "| Column | Non-missing |",
            "|---|---:|",
        ]
    )
    for column in CORE_AVAILABILITY_COLUMNS:
        if column in df.columns:
            lines.append(f"| `{column}` | `{int(df[column].notna().sum()):,}` |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This table is intentionally one-row-per-PSG and preserves `NA` for data not yet joined.",
            "- BSI and SS columns are designed to accept partial joins while upstream processing is still running.",
            "- The primary-analysis subject/study restriction is not yet finalized in this scaffold.",
            "",
        ]
    )
    audit_md.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    master = build_base_master(args.master_csv)
    master, retained_cohorts = filter_cohorts(master, args.keep_cohorts)
    sources = [str(args.master_csv)]

    for run_root in args.bsi_run_root:
        if run_root.exists():
            master = join_table(master, pivot_bsi_wide(collect_bsi_long(run_root)))
            sources.append(str(run_root))

    for run_root in args.ss_run_root:
        if run_root.exists():
            master = join_table(master, collect_ss_wide(run_root))
            sources.append(str(run_root))

    if args.sleep_general_csv and args.sleep_general_csv.exists():
        master = join_table(master, load_sleep_general_revision_table(args.sleep_general_csv))
        sources.append(str(args.sleep_general_csv))
    if args.respiratory_csv and args.respiratory_csv.exists():
        master = join_table(master, prepare_join_table(args.respiratory_csv, RESPIRATORY_RENAME_MAP))
        sources.append(str(args.respiratory_csv))
    if args.macrostructure_csv and args.macrostructure_csv.exists():
        master = join_table(master, prepare_join_table(args.macrostructure_csv, MACRO_RENAME_MAP))
        sources.append(str(args.macrostructure_csv))
    if args.timing_csv and args.timing_csv.exists():
        master = join_table(master, prepare_join_table(args.timing_csv, keep_prefix="timing"))
        sources.append(str(args.timing_csv))
    if args.outcomes_csv and args.outcomes_csv.exists():
        master = join_table(master, prepare_join_table(args.outcomes_csv, keep_prefix="outcome"))
        sources.append(str(args.outcomes_csv))

    master = finalize_flags(master)

    base_cols = list(pd.read_csv(args.master_csv, nrows=0).columns)
    ordered_cols = []
    for column in base_cols + PLACEHOLDER_COLUMNS:
        if column in master.columns and column not in ordered_cols:
            ordered_cols.append(column)
    for column in master.columns:
        if column not in ordered_cols:
            ordered_cols.append(column)
    master = master[ordered_cols]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output_csv, index=False)
    write_audit(
        master,
        args.audit_md,
        sources,
        retained_cohorts,
        title=args.audit_title or default_audit_title(args.output_csv),
    )

    print(f"Wrote {len(master):,} rows to {args.output_csv}")
    print(f"Wrote audit to {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
