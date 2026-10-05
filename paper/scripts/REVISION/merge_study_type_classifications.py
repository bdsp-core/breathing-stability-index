#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PSG study-type classifier output into the revision master study_type column.")
    parser.add_argument("--master", default="master_analysis_table_primary_cohorts_v2_survival.csv")
    parser.add_argument("--classifications", default="study_type_classifications_S0001_I0002.csv")
    parser.add_argument("--backup", default="master_analysis_table_primary_cohorts_v2_survival.pre_study_type_classifier_backup.csv")
    parser.add_argument("--audit", default="study_type_classifications_S0001_I0002_merge_audit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master_path = Path(args.master)
    classifications_path = Path(args.classifications)
    backup_path = Path(args.backup)
    audit_path = Path(args.audit)

    master = pd.read_csv(master_path, dtype=str, low_memory=False).fillna("")
    classifications = pd.read_csv(classifications_path, dtype=str, low_memory=False).fillna("")

    required_master = {"cohort", "fileid", "study_type"}
    required_classifications = {"cohort", "fileid", "predicted_study_type"}
    missing_master = required_master - set(master.columns)
    missing_classifications = required_classifications - set(classifications.columns)
    if missing_master:
        raise ValueError(f"Master missing required columns: {sorted(missing_master)}")
    if missing_classifications:
        raise ValueError(f"Classifications missing required columns: {sorted(missing_classifications)}")
    if classifications.duplicated(["cohort", "fileid"]).any():
        duplicated = classifications[classifications.duplicated(["cohort", "fileid"], keep=False)][["cohort", "fileid"]]
        raise ValueError(f"Classification file has duplicated cohort/fileid keys, examples: {duplicated.head(10).to_dict('records')}")

    if not backup_path.exists():
        shutil.copy2(master_path, backup_path)

    lookup = classifications.set_index(["cohort", "fileid"])["predicted_study_type"]
    key_index = pd.MultiIndex.from_frame(master[["cohort", "fileid"]])
    predicted = pd.Series(key_index.map(lookup), index=master.index).fillna("")

    before = master["study_type"].copy()
    update_mask = predicted.ne("")
    master.loc[update_mask, "study_type"] = predicted.loc[update_mask]
    master.to_csv(master_path, index=False)

    audit_rows: list[dict[str, object]] = []
    for cohort, cohort_master in master.groupby("cohort", dropna=False):
        cohort_mask = master["cohort"].eq(cohort)
        cohort_update_mask = update_mask & cohort_mask
        audit_rows.append(
            {
                "cohort": cohort,
                "master_rows": int(cohort_mask.sum()),
                "rows_updated": int(cohort_update_mask.sum()),
                "rows_left_unchanged": int((cohort_mask & ~update_mask).sum()),
                "study_type_before_counts": before.loc[cohort_mask].value_counts(dropna=False).to_dict(),
                "study_type_after_counts": master.loc[cohort_mask, "study_type"].value_counts(dropna=False).to_dict(),
            }
        )
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    print(f"Updated {int(update_mask.sum())} master rows in {master_path}")
    print(f"Backup: {backup_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
