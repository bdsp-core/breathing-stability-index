#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
SELECTION = REPO_ROOT / "REVISION" / "mortality_v2" / "selection_one_row_per_subject.csv"
OUTDIR_MORTALITY = REPO_ROOT / "REVISION" / "mortality_v2"
OUTDIR_TABLES = REPO_ROOT / "REVISION" / "revision_tables_v2"

COHORT_ORDER = ["I0002", "S0001", "mros", "mgh-cog"]
COHORT_LABELS = {
    "I0002": "BIDMC (I0002)",
    "S0001": "MGH/S0001",
    "mros": "MrOS",
    "mgh-cog": "MGH-COG",
}
TIMING_SOURCE = {
    "I0002": "Exact death/censor dates plus follow-up duration",
    "S0001": "Follow-up duration and event flag; exact calendar dates unavailable",
    "mros": "Follow-up duration and event flag; exact calendar dates unavailable",
    "mgh-cog": "No mortality follow-up source in the retained analysis table",
    "Total": "",
}


def pct(n: int | float, d: int | float) -> float:
    if d == 0:
        return float("nan")
    return 100 * float(n) / float(d)


def fmt_int(x: int | float | None) -> str:
    if pd.isna(x):
        return "NA"
    return f"{int(round(float(x))):,}"


def fmt_pct(n: int | float, d: int | float) -> str:
    if d == 0 or pd.isna(n) or pd.isna(d):
        return "NA"
    return f"{int(n):,}/{int(d):,} ({pct(n, d):.1f}%)"


def fmt_followup(median: float, q1: float, q3: float) -> str:
    if pd.isna(median):
        return "NA"
    return f"{median:.1f} ({q1:.1f}-{q3:.1f})"


def summarize_master(master: pd.DataFrame) -> dict[str, dict[str, float]]:
    master = master.copy()
    master["sid_str"] = master["sid"].astype(str)
    master["mortality_followup_available"] = master["followup_days"].notna() & master["vital_status"].notna()
    master["event"] = master["vital_status"].astype("string").str.lower().eq("dead")

    rows: dict[str, dict[str, float]] = {}
    for cohort in COHORT_ORDER:
        sub = master[master["cohort"].eq(cohort)]
        ready = sub[sub["mortality_followup_available"]]
        rows[cohort] = {
            "master_psgs": len(sub),
            "master_subjects": sub["sid_str"].nunique(),
            "followup_subjects": ready["sid_str"].nunique(),
            "followup_events": int(ready.groupby("sid_str")["event"].max().sum()) if len(ready) else 0,
        }
    rows["Total"] = {
        "master_psgs": len(master),
        "master_subjects": master["sid_str"].nunique(),
        "followup_subjects": master.loc[master["mortality_followup_available"], "sid_str"].nunique(),
        "followup_events": int(master[master["mortality_followup_available"]].groupby("sid_str")["event"].max().sum()),
    }
    return rows


def summarize_selection(selection: pd.DataFrame) -> dict[str, dict[str, float]]:
    strict = selection[selection["selection_mode"].eq("strict_primary")].copy()
    strict["followup_years"] = strict["followup_days"] / 365.25

    rows: dict[str, dict[str, float]] = {}
    for cohort in COHORT_ORDER:
        sub = strict[strict["cohort"].eq(cohort)]
        rows[cohort] = {
            "primary_subjects": len(sub),
            "primary_events": int(sub["event"].sum()) if len(sub) else 0,
            "followup_median_years": sub["followup_years"].median() if len(sub) else float("nan"),
            "followup_q1_years": sub["followup_years"].quantile(0.25) if len(sub) else float("nan"),
            "followup_q3_years": sub["followup_years"].quantile(0.75) if len(sub) else float("nan"),
        }
    rows["Total"] = {
        "primary_subjects": len(strict),
        "primary_events": int(strict["event"].sum()),
        "followup_median_years": strict["followup_years"].median(),
        "followup_q1_years": strict["followup_years"].quantile(0.25),
        "followup_q3_years": strict["followup_years"].quantile(0.75),
    }
    return rows


def build_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(
        MASTER,
        usecols=["sid", "fileid", "cohort", "followup_days", "vital_status"],
        low_memory=False,
    )
    selection = pd.read_csv(
        SELECTION,
        usecols=["cohort", "selection_mode", "followup_days", "event"],
        low_memory=False,
    )

    master_rows = summarize_master(master)
    selected_rows = summarize_selection(selection)

    records = []
    for cohort in [*COHORT_ORDER, "Total"]:
        merged = {
            "cohort": COHORT_LABELS.get(cohort, cohort),
            "master_psgs": master_rows[cohort]["master_psgs"],
            "master_subjects": master_rows[cohort]["master_subjects"],
            "followup_subjects": master_rows[cohort]["followup_subjects"],
            "followup_subjects_percent": pct(
                master_rows[cohort]["followup_subjects"],
                master_rows[cohort]["master_subjects"],
            ),
            "followup_events": master_rows[cohort]["followup_events"],
            "followup_event_percent": pct(
                master_rows[cohort]["followup_events"],
                master_rows[cohort]["followup_subjects"],
            ),
            "primary_mortality_subjects": selected_rows[cohort]["primary_subjects"],
            "primary_mortality_events": selected_rows[cohort]["primary_events"],
            "primary_mortality_event_percent": pct(
                selected_rows[cohort]["primary_events"],
                selected_rows[cohort]["primary_subjects"],
            ),
            "primary_followup_median_years": selected_rows[cohort]["followup_median_years"],
            "primary_followup_q1_years": selected_rows[cohort]["followup_q1_years"],
            "primary_followup_q3_years": selected_rows[cohort]["followup_q3_years"],
            "timing_source": TIMING_SOURCE.get(cohort, ""),
        }
        records.append(merged)

    numeric = pd.DataFrame(records)

    display = pd.DataFrame(
        {
            "Cohort": numeric["cohort"],
            "PSG records in retained analysis table, n": numeric["master_psgs"].map(fmt_int),
            "Unique subjects in retained analysis table, n": numeric["master_subjects"].map(fmt_int),
            "Mortality follow-up available, n/N (%)": [
                fmt_pct(n, d) for n, d in zip(numeric["followup_subjects"], numeric["master_subjects"])
            ],
            "Deaths among follow-up subjects, n/N (%)": [
                fmt_pct(n, d) for n, d in zip(numeric["followup_events"], numeric["followup_subjects"])
            ],
            "Primary mortality analysis subjects, n": numeric["primary_mortality_subjects"].map(fmt_int),
            "Deaths in primary analysis, n/N (%)": [
                fmt_pct(n, d)
                for n, d in zip(numeric["primary_mortality_events"], numeric["primary_mortality_subjects"])
            ],
            "Primary follow-up, median (IQR), years": [
                fmt_followup(med, q1, q3)
                for med, q1, q3 in zip(
                    numeric["primary_followup_median_years"],
                    numeric["primary_followup_q1_years"],
                    numeric["primary_followup_q3_years"],
                )
            ],
            "Timing information available": numeric["timing_source"],
        }
    )
    return numeric, display


def write_markdown(display: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Table Sx. Mortality Follow-up Availability By Cohort",
        "",
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    lines.extend(
        [
            "",
            "Note. Mortality follow-up availability requires nonmissing follow-up time and vital status. "
            "The primary mortality analysis is the strict one-row-per-subject selection with diagnostic/untreated PSG eligibility and required core exposures/covariates available. "
            "Exact calendar death/censor dates are available for BIDMC; MGH/S0001 and MrOS contribute valid follow-up durations and event indicators.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUTDIR_MORTALITY.mkdir(parents=True, exist_ok=True)
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    numeric, display = build_table()

    targets = [
        OUTDIR_MORTALITY / "table_mortality_followup_by_cohort.csv",
        OUTDIR_TABLES / "tableS_mortality_followup_by_cohort_v1.csv",
    ]
    display_targets = [
        OUTDIR_MORTALITY / "table_mortality_followup_by_cohort_display.csv",
        OUTDIR_TABLES / "tableS_mortality_followup_by_cohort_v1_display.csv",
    ]
    markdown_targets = [
        OUTDIR_MORTALITY / "table_mortality_followup_by_cohort.md",
        OUTDIR_TABLES / "tableS_mortality_followup_by_cohort_v1.md",
    ]

    for path in targets:
        numeric.to_csv(path, index=False)
    for path in display_targets:
        display.to_csv(path, index=False)
    for path in markdown_targets:
        write_markdown(display, path)

    print(f"Wrote {targets[0]}")
    print(f"Wrote {markdown_targets[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
