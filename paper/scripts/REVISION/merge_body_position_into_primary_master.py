#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v2_survival.csv"
DEFAULT_BODY_POSITION = REPO_ROOT / "REVISION" / "body_position_bdsp_features_v1.csv"
DEFAULT_OUTPUT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v3_body_position.csv"
DEFAULT_AUDIT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v3_body_position_audit.md"


BODY_POSITION_COLUMNS = [
    "posture_available",
    "supine_tst_min",
    "supine_pct_tst",
    "posture_summary",
    "posture_source",
    "posture_note",
    "tst_epochs_n",
    "position_coverage_pct",
    "supine_epochs_n",
    "left_epochs_n",
    "left_tst_min",
    "left_pct_tst",
    "right_epochs_n",
    "right_tst_min",
    "right_pct_tst",
    "prone_epochs_n",
    "prone_tst_min",
    "prone_pct_tst",
    "upright_epochs_n",
    "upright_tst_min",
    "upright_pct_tst",
    "unknown_epochs_n",
    "unknown_tst_min",
    "unknown_pct_tst",
]

USABLE_ONLY_COLUMNS = [
    "supine_tst_min",
    "supine_pct_tst",
    "posture_summary",
    "supine_epochs_n",
    "left_epochs_n",
    "left_tst_min",
    "left_pct_tst",
    "right_epochs_n",
    "right_tst_min",
    "right_pct_tst",
    "prone_epochs_n",
    "prone_tst_min",
    "prone_pct_tst",
    "upright_epochs_n",
    "upright_tst_min",
    "upright_pct_tst",
    "unknown_epochs_n",
    "unknown_tst_min",
    "unknown_pct_tst",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge BDSP body-position features into the primary revision master.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--body-position-csv", type=Path, default=DEFAULT_BODY_POSITION)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def merge_body_position(master: pd.DataFrame, body_position: pd.DataFrame) -> pd.DataFrame:
    if "fileid" not in master.columns:
        raise ValueError("Master table must contain `fileid`.")
    if "fileid" not in body_position.columns:
        raise ValueError("Body-position table must contain `fileid`.")
    if body_position["fileid"].duplicated().any():
        duplicated = body_position.loc[body_position["fileid"].duplicated(), "fileid"].head(5).tolist()
        raise ValueError(f"Body-position table contains duplicate fileids, examples: {duplicated}")

    merge_cols = ["fileid"] + [column for column in BODY_POSITION_COLUMNS if column in body_position.columns]
    body = body_position[merge_cols].copy()

    overlap = [column for column in body.columns if column != "fileid" and column in master.columns]
    if overlap:
        master = master.drop(columns=overlap)
    merged = master.merge(body, on="fileid", how="left")

    if "posture_available" in merged.columns:
        available = merged["posture_available"].fillna(False).astype(bool)
        merged["posture_available"] = available
        for column in USABLE_ONLY_COLUMNS:
            if column in merged.columns:
                merged.loc[~available, column] = pd.NA
    return merged


def write_audit(master: pd.DataFrame, body_position: pd.DataFrame, merged: pd.DataFrame, path: Path) -> None:
    matched = int(merged["posture_source"].notna().sum()) if "posture_source" in merged.columns else 0
    body_not_in_master = int((~body_position["fileid"].isin(master["fileid"])).sum())

    lines = [
        "# Primary Cohort Master v3 Body-Position Audit",
        "",
        f"- Master rows: `{len(master):,}`",
        f"- Merged rows: `{len(merged):,}`",
        f"- Unique PSGs: `{merged['fileid'].nunique(dropna=True):,}`",
        f"- Body-position rows: `{len(body_position):,}`",
        f"- Body-position rows matched to master: `{matched:,}`",
        f"- Body-position rows not in master: `{body_not_in_master:,}`",
        "",
        "## Body-Position Coverage",
        "",
        "| cohort | rows | with_posture_row | posture_available | with_supine_tst | median_position_coverage_pct | median_supine_pct_tst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    grouped = merged.groupby("cohort", dropna=False)
    for cohort, group in grouped:
        with_posture_row = int(group["posture_source"].notna().sum()) if "posture_source" in group.columns else 0
        posture_available = (
            int(group["posture_available"].fillna(False).astype(bool).sum())
            if "posture_available" in group.columns
            else 0
        )
        with_supine_tst = int(pd.to_numeric(group.get("supine_tst_min"), errors="coerce").notna().sum())
        coverage = pd.to_numeric(group.get("position_coverage_pct"), errors="coerce")
        supine = pd.to_numeric(group.get("supine_pct_tst"), errors="coerce")
        median_coverage = coverage.median() if coverage.notna().any() else pd.NA
        median_supine = supine.median() if supine.notna().any() else pd.NA
        coverage_text = "" if pd.isna(median_coverage) else f"{median_coverage:.1f}"
        supine_text = "" if pd.isna(median_supine) else f"{median_supine:.1f}"
        lines.append(
            f"| `{cohort}` | `{len(group):,}` | `{with_posture_row:,}` | `{posture_available:,}` | "
            f"`{with_supine_tst:,}` | `{coverage_text}` | `{supine_text}` |"
        )

    if "posture_note" in merged.columns:
        lines.extend(["", "## Posture Notes", ""])
        for note, count in merged["posture_note"].dropna().value_counts().head(20).items():
            lines.append(f"- `{note}`: `{int(count):,}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    master = pd.read_csv(args.master_csv, low_memory=False)
    body_position = pd.read_csv(args.body_position_csv, low_memory=False)
    merged = merge_body_position(master, body_position)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit_md.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    write_audit(master, body_position, merged, args.audit_md)

    print(f"Wrote {len(merged):,} rows to {args.output_csv}")
    print(f"Wrote audit to {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
