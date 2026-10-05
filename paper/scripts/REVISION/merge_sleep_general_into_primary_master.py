#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v1.csv"
DEFAULT_SLEEP_GENERAL = REPO_ROOT / "REVISION" / "final_completed_20260507" / "sleep_general_primary_cohorts_features_v1.csv"
DEFAULT_OUTPUT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v2.csv"
DEFAULT_AUDIT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v2_audit.md"

SLEEP_GENERAL_TRACE_COLUMNS = [
    "status",
    "message",
    "macro_status",
    "macro_message",
    "oxygen_status",
    "oxygen_message",
]

SLEEP_GENERAL_METRIC_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge sleep_general macro/respiratory outputs into the primary revision master.")
    p.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--sleep-general-csv", type=Path, default=DEFAULT_SLEEP_GENERAL)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT)
    return p.parse_args()


def merge_sleep_general(master: pd.DataFrame, sleep_general: pd.DataFrame) -> pd.DataFrame:
    sleep_general = sleep_general.drop_duplicates(subset=["fileid"], keep="first").copy()

    trace = sleep_general[["fileid"] + [c for c in SLEEP_GENERAL_TRACE_COLUMNS if c in sleep_general.columns]].copy()
    trace_cols = [c for c in trace.columns if c != "fileid"]
    for col in trace_cols:
        if col in master.columns:
            master = master.drop(columns=[col])
    master = master.merge(trace, on="fileid", how="left")

    metrics = sleep_general[["fileid"] + [c for c in SLEEP_GENERAL_METRIC_COLUMNS if c in sleep_general.columns]].copy()
    metrics = metrics.set_index("fileid")
    master = master.set_index("fileid")
    for col in metrics.columns:
        if col not in master.columns:
            master[col] = pd.NA
        fill = master[col].isna() & metrics[col].reindex(master.index).notna()
        master.loc[fill, col] = metrics.loc[master.index[fill], col].values
    master = master.reset_index()

    respiratory_cols = [c for c in ["ahi", "rdi", "oai", "cai", "hypoxic_burden"] if c in master.columns]
    macro_cols = [c for c in ["tst_min", "sleep_efficiency", "n1_pct", "rem_pct"] if c in master.columns]
    if respiratory_cols:
        master["has_respiratory_metrics"] = master[respiratory_cols].notna().any(axis=1)
    if macro_cols:
        master["has_macrostructure"] = master[macro_cols].notna().any(axis=1)

    if "phenotype_osa_csa" not in master.columns:
        master["phenotype_osa_csa"] = pd.NA
    if "oai" in master.columns and "cai" in master.columns:
        oai = pd.to_numeric(master["oai"], errors="coerce")
        cai = pd.to_numeric(master["cai"], errors="coerce")
        total = oai + cai
        obstructive_fraction = oai / total
        valid = total > 0
        master.loc[valid & (obstructive_fraction > 0.75), "phenotype_osa_csa"] = "osa"
        master.loc[valid & (obstructive_fraction < 0.25), "phenotype_osa_csa"] = "csa"
        master.loc[valid & obstructive_fraction.between(0.25, 0.75, inclusive="both"), "phenotype_osa_csa"] = "mixed"
    return master


def write_audit(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Primary Cohort Master v2 Audit",
        "",
        f"- Rows: `{len(df):,}`",
        f"- Unique PSGs: `{df['fileid'].nunique(dropna=True):,}`",
        "",
        "## Sleep General Coverage",
        "",
        "| cohort | rows | with_ahi | with_tst | with_hypoxic_burden | macro_ok | oxygen_ok |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = (
        df.groupby("cohort")
        .agg(
            rows=("fileid", "size"),
            with_ahi=("ahi", lambda s: int(s.notna().sum())),
            with_tst=("tst_min", lambda s: int(s.notna().sum())),
            with_hb=("hypoxic_burden", lambda s: int(s.notna().sum())),
            macro_ok=("macro_status", lambda s: int((s == "ok").sum())) if "macro_status" in df.columns else ("fileid", lambda s: 0),
            oxygen_ok=("oxygen_status", lambda s: int((s == "ok").sum())) if "oxygen_status" in df.columns else ("fileid", lambda s: 0),
        )
        .reset_index()
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['cohort']}` | `{int(row['rows']):,}` | `{int(row['with_ahi']):,}` | `{int(row['with_tst']):,}` | "
            f"`{int(row['with_hb']):,}` | `{int(row['macro_ok']):,}` | `{int(row['oxygen_ok']):,}` |"
        )

    if "status" in df.columns:
        counts = df["status"].fillna("missing").value_counts(dropna=False)
        lines.extend(["", "## Status Counts", ""])
        for key, value in counts.items():
            lines.append(f"- `{key}`: `{int(value):,}`")

    if "phenotype_osa_csa" in df.columns:
        phenotype_counts = df["phenotype_osa_csa"].value_counts(dropna=True)
        lines.extend(["", "## OSA/CSA Phenotype Counts", ""])
        if len(phenotype_counts) == 0:
            lines.append("- No phenotype rows were derived.")
        else:
            for key, value in phenotype_counts.items():
                lines.append(f"- `{key}`: `{int(value):,}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    master = pd.read_csv(args.master_csv, low_memory=False)
    sleep_general = pd.read_csv(args.sleep_general_csv, low_memory=False)
    merged = merge_sleep_general(master, sleep_general)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    write_audit(merged, args.audit_md)
    print(f"Wrote {len(merged):,} rows to {args.output_csv}")
    print(f"Wrote audit to {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
