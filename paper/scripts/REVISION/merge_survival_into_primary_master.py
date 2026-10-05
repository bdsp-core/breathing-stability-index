#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v2.csv"
DEFAULT_OUTPUT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v2_survival.csv"
DEFAULT_AUDIT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v2_survival_audit.md"

DEFAULT_MGH_SURVIVAL = Path("/home/wolfgang/repos/sleep_cognition/survival_analysis/data_mgh_survival.csv")
DEFAULT_BIDMC_SURVIVAL = Path("/home/wolfgang/repos/sleep_cognition/survival_analysis/data_bidmc_survival.csv")
DEFAULT_MROS_SURVIVAL = Path("/home/wolfgang/repos/sleep_cognition/survival_analysis/data_mros_survival.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge original-submission survival tables into the primary revision master.")
    p.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--mgh-survival-csv", type=Path, default=DEFAULT_MGH_SURVIVAL)
    p.add_argument("--bidmc-survival-csv", type=Path, default=DEFAULT_BIDMC_SURVIVAL)
    p.add_argument("--mros-survival-csv", type=Path, default=DEFAULT_MROS_SURVIVAL)
    return p.parse_args()


def normalize_fileid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.h5$", "", regex=True)


def standardize_survival_frame(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["fileid"] = normalize_fileid(df["fileid"])
    out["survival_source_cohort"] = cohort

    if "time_fu" in df.columns:
        time_years = pd.to_numeric(df["time_fu"], errors="coerce")
    elif "time" in df.columns:
        time_years = pd.to_numeric(df["time"], errors="coerce") / 365.25
    else:
        time_years = pd.Series(pd.NA, index=df.index, dtype="float64")

    if "death" in df.columns:
        death = pd.to_numeric(df["death"], errors="coerce")
    elif "event_observed" in df.columns:
        death = pd.to_numeric(df["event_observed"], errors="coerce")
    else:
        death = pd.Series(pd.NA, index=df.index, dtype="float64")

    out["followup_days"] = (time_years * 365.25).round(1)
    out["vital_status"] = pd.NA
    out.loc[death == 1, "vital_status"] = "dead"
    out.loc[death == 0, "vital_status"] = "alive"

    out["death_date"] = pd.NA
    if "DateOfDeath" in df.columns:
        out.loc[death == 1, "death_date"] = df.loc[death == 1, "DateOfDeath"].astype(str)

    out["censor_date"] = pd.NA
    if "latest_record" in df.columns:
        out.loc[death == 0, "censor_date"] = df.loc[death == 0, "latest_record"].astype(str)

    return out.drop_duplicates(subset=["fileid"], keep="first")


def merge_survival(master: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for cohort, path in paths.items():
        df = pd.read_csv(path, low_memory=False)
        frames.append(standardize_survival_frame(df, cohort))
    survival = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["fileid"], keep="first")

    overlap = [c for c in survival.columns if c != "fileid" and c in master.columns]
    if overlap:
        master = master.drop(columns=overlap)
    return master.merge(survival, on="fileid", how="left")


def write_audit(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Primary Cohort Master v2 Survival Audit",
        "",
        f"- Rows: `{len(df):,}`",
        f"- Unique PSGs: `{df['fileid'].nunique(dropna=True):,}`",
        "",
        "## Survival Coverage",
        "",
        "| cohort | rows | with_followup_days | with_vital_status | with_death_date | with_censor_date |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary = (
        df.groupby("cohort")
        .agg(
            rows=("fileid", "size"),
            with_followup_days=("followup_days", lambda s: int(s.notna().sum())),
            with_vital_status=("vital_status", lambda s: int(s.notna().sum())),
            with_death_date=("death_date", lambda s: int(s.notna().sum())),
            with_censor_date=("censor_date", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['cohort']}` | `{int(row['rows']):,}` | `{int(row['with_followup_days']):,}` | "
            f"`{int(row['with_vital_status']):,}` | `{int(row['with_death_date']):,}` | `{int(row['with_censor_date']):,}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    master = pd.read_csv(args.master_csv, low_memory=False)
    merged = merge_survival(
        master,
        {
            "S0001": args.mgh_survival_csv,
            "I0002": args.bidmc_survival_csv,
            "mros": args.mros_survival_csv,
        },
    )
    merged.to_csv(args.output_csv, index=False)
    write_audit(merged, args.audit_md)
    print(f"Wrote {len(merged):,} rows to {args.output_csv}")
    print(f"Wrote audit to {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
