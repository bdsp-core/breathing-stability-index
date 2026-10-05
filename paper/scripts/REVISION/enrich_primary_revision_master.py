#!/usr/bin/env python3
"""
Enrich the retained-cohort revision master table with BMI and provisional PSG type.

Inputs:
- REVISION/master_analysis_table_primary_cohorts_v0.csv
- data/dataset_BMI_primary.csv
- data/dataset_BMI_alternative.csv
- /home/wolfgang/repos/sleep_cognition/step0a_prepare_data_cohorts/mros_table_all.csv

Rules:
- For BDSP cohorts, use BMI from the primary BDSP table first.
- Only use the alternative BDSP BMI table to fill holes left by the primary table.
- For MrOS, use `hwbmi`, keyed by the subject token embedded in `fileid`.
- Add `psg_type` if missing and set MrOS rows to `diagnostic` for now.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v0.csv"
DEFAULT_OUTPUT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v1.csv"
DEFAULT_AUDIT = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v1_audit.md"
DEFAULT_BMI_PRIMARY = REPO_ROOT / "data" / "dataset_BMI_primary.csv"
DEFAULT_BMI_ALTERNATIVE = REPO_ROOT / "data" / "dataset_BMI_alternative.csv"
DEFAULT_MROS_TABLE = Path("/home/wolfgang/repos/sleep_cognition/step0a_prepare_data_cohorts/mros_table_all.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich the retained-cohort revision master with BMI and psg_type.")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--bmi-primary-csv", type=Path, default=DEFAULT_BMI_PRIMARY)
    p.add_argument("--bmi-alternative-csv", type=Path, default=DEFAULT_BMI_ALTERNATIVE)
    p.add_argument("--mros-table-csv", type=Path, default=DEFAULT_MROS_TABLE)
    return p.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_scalar(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        dec = Decimal(text)
    except InvalidOperation:
        return text

    if dec == dec.to_integral_value():
        return str(dec.quantize(Decimal("1")))

    normalized = format(dec.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def load_bdsp_bmi_maps(path: Path) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str], float]]:
    df = pd.read_csv(path, low_memory=False)
    required = {"siteid", "bdsppatientid", "sessionid", "bmi"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df[df["bmi"].notna()].copy()
    df["cohort_key"] = df["siteid"].map(clean_text)
    df["sid_key"] = df["bdsppatientid"].map(normalize_scalar)
    df["visit_key"] = df["sessionid"].map(normalize_scalar)

    exact_map: dict[tuple[str, str, str], float] = {}
    subject_candidates: dict[tuple[str, str], list[float]] = {}

    for _, row in df.iterrows():
        cohort_key = row["cohort_key"]
        sid_key = row["sid_key"]
        visit_key = row["visit_key"]
        bmi_value = float(row["bmi"])
        exact_map[(cohort_key, sid_key, visit_key)] = bmi_value
        subject_candidates.setdefault((cohort_key, sid_key), []).append(bmi_value)

    subject_map: dict[tuple[str, str], float] = {}
    for key, values in subject_candidates.items():
        unique = {round(v, 6) for v in values}
        if len(unique) == 1:
            subject_map[key] = values[0]

    return exact_map, subject_map


def fill_bdsp_bmi(master: pd.DataFrame, bmi_sources: list[tuple[str, Path]]) -> pd.DataFrame:
    result = pd.DataFrame(index=master.index)
    result["bmi_bdsp"] = pd.NA
    result["bmi_source"] = pd.NA

    sid_key = master["sid"].map(normalize_scalar)
    visit_key = pd.to_numeric(master["visitno"], errors="coerce").map(
        lambda x: "" if pd.isna(x) else normalize_scalar(x)
    )

    for label, path in bmi_sources:
        exact_map, subject_map = load_bdsp_bmi_maps(path)

        exact_values = [
            exact_map.get((cohort, sid, visit), pd.NA)
            for cohort, sid, visit in zip(master["cohort"], sid_key, visit_key)
        ]
        exact_values = pd.Series(exact_values, index=master.index)
        fill_exact = result["bmi_bdsp"].isna() & exact_values.notna()
        result.loc[fill_exact, "bmi_bdsp"] = exact_values.loc[fill_exact]
        result.loc[fill_exact, "bmi_source"] = f"bdsp_{label}_exact"

        subject_values = [
            subject_map.get((cohort, sid), pd.NA)
            for cohort, sid in zip(master["cohort"], sid_key)
        ]
        subject_values = pd.Series(subject_values, index=master.index)
        fill_subject = result["bmi_bdsp"].isna() & subject_values.notna()
        result.loc[fill_subject, "bmi_bdsp"] = subject_values.loc[fill_subject]
        result.loc[fill_subject, "bmi_source"] = f"bdsp_{label}_subject"

    result["bmi_bdsp"] = pd.to_numeric(result["bmi_bdsp"], errors="coerce")
    return result


def load_mros_bmi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["sid", "hwbmi"], low_memory=False)
    df["mros_id_from_fileid"] = df["sid"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["mros_id_from_fileid"])
    return df[["mros_id_from_fileid", "hwbmi"]].rename(columns={"hwbmi": "bmi_mros"})


def extract_mros_fileid_token(fileid: object) -> object:
    if pd.isna(fileid):
        return pd.NA
    token = pd.Series([str(fileid)]).str.extract(r"mros-cycle\d+_([a-z]{2}\d+)")[0].iloc[0]
    if pd.isna(token):
        return pd.NA
    return str(token).lower()


def enrich_master(master: pd.DataFrame, bmi_primary: Path, bmi_alternative: Path, mros_table: Path) -> pd.DataFrame:
    bdsp = fill_bdsp_bmi(master, [("primary", bmi_primary), ("alternative", bmi_alternative)])
    master["bmi_bdsp"] = bdsp["bmi_bdsp"]
    master["bmi_source"] = bdsp["bmi_source"]

    master["mros_id_from_fileid"] = master["fileid"].map(extract_mros_fileid_token)
    mros_bmi = load_mros_bmi(mros_table)
    master = master.merge(mros_bmi, on="mros_id_from_fileid", how="left")

    master["bmi"] = master["bmi_bdsp"]
    fill_mros = master["cohort"].eq("mros") & master["bmi"].isna() & master["bmi_mros"].notna()
    master.loc[fill_mros, "bmi"] = master.loc[fill_mros, "bmi_mros"]
    master.loc[fill_mros, "bmi_source"] = "mros_hwbmi"

    if "psg_type" not in master.columns:
        insert_at = master.columns.get_loc("study_type") + 1 if "study_type" in master.columns else len(master.columns)
        master.insert(insert_at, "psg_type", pd.NA)
    master.loc[master["cohort"].eq("mros"), "psg_type"] = "diagnostic"

    return master


def write_audit(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Primary Cohort Master v1 Audit",
        "",
        f"- Rows: `{len(df):,}`",
        f"- Unique PSGs: `{df['fileid'].nunique(dropna=True):,}`",
        "",
        "## BMI Coverage",
        "",
        "| cohort | rows | bmi non-missing | psg_type non-missing |",
        "|---|---:|---:|---:|",
    ]
    summary = (
        df.groupby("cohort")
        .agg(
            rows=("fileid", "size"),
            bmi_non_missing=("bmi", lambda s: int(s.notna().sum())),
            psg_type_non_missing=("psg_type", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['cohort']}` | `{int(row['rows']):,}` | `{int(row['bmi_non_missing']):,}` | `{int(row['psg_type_non_missing']):,}` |"
        )

    source_counts = df["bmi_source"].value_counts(dropna=True)
    lines.extend(["", "## BMI Source Counts", ""])
    if len(source_counts) == 0:
        lines.append("- No BMI rows were populated.")
    else:
        for source, count in source_counts.items():
            lines.append(f"- `{source}`: `{int(count):,}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- BDSP BMI uses primary first, then alternative only as fill for remaining holes.",
            "- MrOS BMI uses `hwbmi` keyed by the subject token extracted from `fileid`.",
            "- `psg_type` is currently set to `diagnostic` for all `mros` rows only.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    master = pd.read_csv(args.input_csv, low_memory=False)
    master = enrich_master(master, args.bmi_primary_csv, args.bmi_alternative_csv, args.mros_table_csv)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output_csv, index=False)
    write_audit(master, args.audit_md)

    print(f"Wrote {len(master):,} rows to {args.output_csv}")
    print(f"Wrote audit to {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
