#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from build_master_analysis_table import collect_bsi_long, collect_ss_wide, join_table, pivot_bsi_wide


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v2.csv"
DEFAULT_OUTPUT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v3.csv"
DEFAULT_AUDIT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v3_audit.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge BSI and SS outputs into the unified primary revision master.")
    p.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--bsi-run-root", type=Path, action="append", default=[])
    p.add_argument("--ss-run-root", type=Path, action="append", default=[])
    p.add_argument("--bsi-summary-csv", type=Path, action="append", default=[])
    p.add_argument("--ss-summary-csv", type=Path, action="append", default=[])
    return p.parse_args()


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


def collect_bsi_from_summary_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame(columns=["fileid"])
    df = pd.concat(frames, ignore_index=True)
    if "fileid" not in df.columns:
        if "file" not in df.columns:
            raise ValueError("BSI summary csv must contain `fileid` or `file`.")
        df["fileid"] = df["file"].map(normalize_fileid)
    else:
        df["fileid"] = df["fileid"].map(normalize_fileid)
    dedupe_cols = [c for c in ["fileid", "window_length", "overlap", "belt_mode"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")
    return pivot_bsi_wide(df)


def collect_ss_from_summary_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame(columns=["fileid"])
    df = pd.concat(frames, ignore_index=True)
    if "fileid" not in df.columns:
        if "file" not in df.columns:
            raise ValueError("SS summary csv must contain `fileid` or `file`.")
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


def write_audit(
    df: pd.DataFrame,
    path: Path,
    bsi_roots: list[Path],
    ss_roots: list[Path],
    bsi_csvs: list[Path],
    ss_csvs: list[Path],
) -> None:
    lines = [
        "# Primary Cohort Master v3 Audit",
        "",
        f"- Rows: `{len(df):,}`",
        f"- Unique PSGs: `{df['fileid'].nunique(dropna=True):,}`",
        "",
        "## Sources",
        "",
    ]
    for root in bsi_roots:
        lines.append(f"- BSI: `{root}`")
    for root in ss_roots:
        lines.append(f"- SS: `{root}`")
    for csv_path in bsi_csvs:
        lines.append(f"- BSI summary csv: `{csv_path}`")
    for csv_path in ss_csvs:
        lines.append(f"- SS summary csv: `{csv_path}`")

    lines.extend(
        [
            "",
            "## Exposure Coverage",
            "",
            "| cohort | rows | with_bsi | with_ss | with_both |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if "has_bsi_robust_mean" not in df.columns:
        df["has_bsi_robust_mean"] = False
    if "has_ss" not in df.columns:
        df["has_ss"] = False
    grouped = (
        df.groupby("cohort")
        .agg(
            rows=("fileid", "size"),
            with_bsi=("has_bsi_robust_mean", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            with_ss=("has_ss", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        )
        .reset_index()
    )
    grouped["with_both"] = (
        df.assign(_both=df["has_bsi_robust_mean"].fillna(False).astype(bool) & df["has_ss"].fillna(False).astype(bool))
        .groupby("cohort")["_both"]
        .sum()
        .reindex(grouped["cohort"])
        .to_numpy()
    )
    for _, row in grouped.iterrows():
        lines.append(
            f"| `{row['cohort']}` | `{int(row['rows']):,}` | `{int(row['with_bsi']):,}` | `{int(row['with_ss']):,}` | `{int(row['with_both']):,}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    master = pd.read_csv(args.master_csv, low_memory=False)

    for run_root in args.bsi_run_root:
        if run_root.exists():
            master = join_table(master, pivot_bsi_wide(collect_bsi_long(run_root)))
    for run_root in args.ss_run_root:
        if run_root.exists():
            master = join_table(master, collect_ss_wide(run_root))
    if args.bsi_summary_csv:
        master = join_table(master, collect_bsi_from_summary_csvs(args.bsi_summary_csv))
    if args.ss_summary_csv:
        master = join_table(master, collect_ss_from_summary_csvs(args.ss_summary_csv))

    master.to_csv(args.output_csv, index=False)
    write_audit(
        master,
        args.audit_md,
        args.bsi_run_root,
        args.ss_run_root,
        args.bsi_summary_csv,
        args.ss_summary_csv,
    )
    print(f"Wrote {len(master):,} rows to {args.output_csv}")
    print(f"Wrote audit to {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
