from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIDE = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v3.csv"
DEFAULT_STUDY = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v2_survival.csv"
DEFAULT_POSTURE = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v3_body_position.csv"
DEFAULT_OUT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
DEFAULT_AUDIT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper_audit.md"


POSITION_PCT_COLS = [
    "supine_pct_tst",
    "left_pct_tst",
    "right_pct_tst",
    "prone_pct_tst",
    "upright_pct_tst",
    "unknown_pct_tst",
]

POSITION_MIN_COLS = [
    "supine_tst_min",
    "left_tst_min",
    "right_tst_min",
    "prone_tst_min",
    "upright_tst_min",
    "unknown_tst_min",
]

POSITION_EPOCH_COLS = [
    "tst_epochs_n",
    "supine_epochs_n",
    "left_epochs_n",
    "right_epochs_n",
    "prone_epochs_n",
    "upright_epochs_n",
    "unknown_epochs_n",
]

POSITION_META_COLS = [
    "posture_available",
    "position_coverage_pct",
    "posture_source",
    "posture_note",
    "posture_summary",
]

STUDY_META_COLS = ["study_type", "psg_type", "cpap_present"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the final paper master table from wide exposures + study metadata + body position.")
    p.add_argument("--wide-master", type=Path, default=DEFAULT_WIDE)
    p.add_argument("--study-master", type=Path, default=DEFAULT_STUDY)
    p.add_argument("--posture-master", type=Path, default=DEFAULT_POSTURE)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    p.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT)
    return p.parse_args()


def _to_boolish(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip().str.lower()
    return out.map({"true": True, "false": False})


def derive_paper_psg_type(df: pd.DataFrame) -> pd.Series:
    direct = df["psg_type"].copy()
    direct = direct.where(direct.notna(), None)

    derived = pd.Series(pd.NA, index=df.index, dtype="object")
    derived.loc[direct.notna()] = direct.loc[direct.notna()].astype(str)

    study = df["study_type"].astype("string")
    cohort = df["cohort"].astype("string")

    mapping = {
        "diagnostic_psg": "diagnostic",
        "split_night": "split_night",
        "pap_titration": "pap_titration",
        "hsat": "hsat",
        "mslt": "mslt",
        "other_unclear": "other_unclear",
    }
    for src, dest in mapping.items():
        mask = derived.isna() & study.eq(src)
        derived.loc[mask] = dest

    mask_mros_sleep = derived.isna() & cohort.eq("mros") & study.eq("sleep")
    derived.loc[mask_mros_sleep] = "diagnostic"

    mask_unspecified_sleep = derived.isna() & study.eq("sleep")
    derived.loc[mask_unspecified_sleep] = "sleep_unspecified"
    return derived


def derive_primary_eligibility(psg_type: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=psg_type.index, dtype="object")
    out.loc[psg_type.eq("diagnostic")] = True
    out.loc[psg_type.isin(["split_night", "pap_titration", "hsat", "mslt", "other_unclear"])] = False
    return out


def build_paper_master(wide_path: Path, study_path: Path, posture_path: Path) -> pd.DataFrame:
    wide = pd.read_csv(wide_path, low_memory=False)
    study = pd.read_csv(study_path, low_memory=False)
    posture = pd.read_csv(posture_path, low_memory=False)

    if wide["fileid"].duplicated().any():
        raise ValueError("Wide master contains duplicate fileid rows")
    if study["fileid"].duplicated().any():
        raise ValueError("Study master contains duplicate fileid rows")
    if posture["fileid"].duplicated().any():
        raise ValueError("Posture master contains duplicate fileid rows")

    position_cols = [c for c in POSITION_META_COLS + POSITION_PCT_COLS + POSITION_MIN_COLS + POSITION_EPOCH_COLS if c in posture.columns]
    study_cols = [c for c in STUDY_META_COLS if c in study.columns]

    merge_cols = sorted(set(study_cols + position_cols))

    base = wide.drop(columns=[c for c in merge_cols if c in wide.columns]).copy()
    merge_df = study[["fileid"] + study_cols].merge(posture[["fileid"] + position_cols], on="fileid", how="outer")
    out = base.merge(merge_df, on="fileid", how="left", validate="one_to_one")

    for col in POSITION_PCT_COLS + POSITION_MIN_COLS + POSITION_EPOCH_COLS + ["tst_min", "position_coverage_pct"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "posture_available" in out.columns:
        out["posture_available"] = _to_boolish(out["posture_available"])

    out["paper_psg_type"] = derive_paper_psg_type(out)
    out["paper_primary_psg_eligible"] = derive_primary_eligibility(out["paper_psg_type"])

    posture_true = out["posture_available"].eq(True)

    if all(col in out.columns for col in POSITION_PCT_COLS):
        out["position_pct_sum"] = np.where(
            posture_true,
            out[POSITION_PCT_COLS].fillna(0).sum(axis=1),
            np.nan,
        )
    else:
        out["position_pct_sum"] = np.nan

    if all(col in out.columns for col in POSITION_MIN_COLS):
        out["position_tst_sum_min"] = np.where(
            posture_true,
            out[POSITION_MIN_COLS].fillna(0).sum(axis=1),
            np.nan,
        )
    else:
        out["position_tst_sum_min"] = np.nan

    out["position_tst_delta_min"] = np.where(
        posture_true,
        out["position_tst_sum_min"] - out["tst_min"],
        np.nan,
    )
    out["position_pct_sum_within_1pct"] = np.where(
        out["posture_available"].eq(True),
        out["position_pct_sum"].between(99.0, 101.0, inclusive="both"),
        pd.NA,
    )
    out["position_tst_within_5min"] = np.where(
        out["posture_available"].eq(True),
        out["position_tst_delta_min"].abs() <= 5,
        pd.NA,
    )
    out["position_tst_within_30min"] = np.where(
        out["posture_available"].eq(True),
        out["position_tst_delta_min"].abs() <= 30,
        pd.NA,
    )
    out["paper_posture_usable"] = np.where(
        out["posture_available"].eq(True) & out["position_tst_within_5min"].eq(True),
        True,
        np.where(out["posture_available"].eq(False), False, pd.NA),
    )

    return out


def write_audit(df: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Final Paper Master Audit")
    lines.append("")
    lines.append("Source files:")
    lines.append(f"- `v3 wide`: `{DEFAULT_WIDE}`")
    lines.append(f"- `v2 survival/study`: `{DEFAULT_STUDY}`")
    lines.append(f"- `v3 body position`: `{DEFAULT_POSTURE}`")
    lines.append("")
    lines.append(f"Rows: `{len(df):,}`")
    lines.append(f"Unique fileids: `{df['fileid'].nunique():,}`")
    lines.append(f"Columns: `{len(df.columns):,}`")
    lines.append("")
    lines.append("## Study / PSG Type")
    lines.append(f"- Raw `psg_type` non-missing: `{int(df['psg_type'].notna().sum()):,}`")
    lines.append(f"- `study_type` non-missing: `{int(df['study_type'].notna().sum()):,}`")
    lines.append(f"- Harmonized `paper_psg_type` non-missing: `{int(df['paper_psg_type'].notna().sum()):,}`")
    lines.append(f"- `paper_primary_psg_eligible == True`: `{int(df['paper_primary_psg_eligible'].eq(True).sum()):,}`")
    lines.append(f"- `paper_primary_psg_eligible == False`: `{int(df['paper_primary_psg_eligible'].eq(False).sum()):,}`")
    lines.append(f"- `paper_primary_psg_eligible == NA`: `{int(df['paper_primary_psg_eligible'].isna().sum()):,}`")
    lines.append("")
    lines.append("Harmonized `paper_psg_type` counts:")
    lines.append("```")
    lines.append(df["paper_psg_type"].fillna("NA").value_counts(dropna=False).to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Body Position")
    lines.append(f"- `posture_available == True`: `{int(df['posture_available'].eq(True).sum()):,}`")
    lines.append(f"- `posture_available == False`: `{int(df['posture_available'].eq(False).sum()):,}`")
    lines.append(f"- `position_pct_sum_within_1pct == True`: `{int(pd.Series(df['position_pct_sum_within_1pct']).eq(True).sum()):,}`")
    lines.append(f"- `position_tst_within_5min == True`: `{int(pd.Series(df['position_tst_within_5min']).eq(True).sum()):,}`")
    lines.append(f"- `position_tst_within_30min == True`: `{int(pd.Series(df['position_tst_within_30min']).eq(True).sum()):,}`")
    lines.append(f"- `paper_posture_usable == True`: `{int(pd.Series(df['paper_posture_usable']).eq(True).sum()):,}`")
    lines.append("")
    lines.append("Cohort-level body-position availability:")
    lines.append("```")
    lines.append(
        df.groupby("cohort").agg(
            n_rows=("fileid", "size"),
            posture_available=("posture_available", lambda s: int(pd.Series(s).eq(True).sum())),
            paper_posture_usable=("paper_posture_usable", lambda s: int(pd.Series(s).eq(True).sum())),
        ).to_string()
    )
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("- The wide `v3` table preserved the full BSI/SS exposure family but had a regressed `study_type` column (`sleep` for all rows); this audit restores `study_type` from the updated study/body-position tables.")
    lines.append("- Raw BDSP `psg_type` was not directly populated in the provided source tables; `paper_psg_type` is therefore harmonized conservatively from `study_type` plus the existing MrOS `psg_type` values.")
    lines.append("- Body-position percentages are internally coherent, but not all rows align tightly with `tst_min`; keep `paper_posture_usable` or `position_tst_within_5min` as the conservative analysis filter.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    df = build_paper_master(args.wide_master, args.study_master, args.posture_master)
    df.to_csv(args.output_csv, index=False)
    write_audit(df, args.audit_md)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
