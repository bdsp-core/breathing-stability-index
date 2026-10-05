from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRACE_COLUMNS = [
    "fileid",
    "cohort",
    "status",
    "message",
    "macro_status",
    "macro_message",
    "oxygen_status",
    "oxygen_message",
]

REVISION_SLEEP_GENERAL_METRIC_COLUMNS = [
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

REVISION_SLEEP_GENERAL_COLUMNS = TRACE_COLUMNS + REVISION_SLEEP_GENERAL_METRIC_COLUMNS


def normalize_fileid(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return Path(text).stem


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _scaled(series: pd.Series, factor: float, ndigits: int = 1) -> pd.Series:
    values = _numeric(series) * float(factor)
    return values.round(ndigits)


def _raw_fraction_to_percent(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return values
    if finite.max() <= 1.5:
        return values.mul(100.0).round(1)
    return values.round(1)


def transform_sleep_general_revision_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())
    if "fileid" in df.columns:
        out["fileid"] = df["fileid"].map(normalize_fileid)
    elif "file" in df.columns:
        out["fileid"] = df["file"].map(normalize_fileid)
    else:
        raise ValueError("sleep_general table must contain `fileid` or `file`")

    out["cohort"] = df["cohort"] if "cohort" in df.columns else pd.NA
    out["status"] = df["status"] if "status" in df.columns else pd.NA
    out["message"] = df["message"] if "message" in df.columns else pd.NA
    out["macro_status"] = df["macro_status"] if "macro_status" in df.columns else pd.NA
    out["macro_message"] = df["macro_message"] if "macro_message" in df.columns else pd.NA
    out["oxygen_status"] = df["oxygen_status"] if "oxygen_status" in df.columns else pd.NA
    out["oxygen_message"] = df["oxygen_message"] if "oxygen_message" in df.columns else pd.NA

    direct_map = {
        "ahi": "ahi",
        "rdi": "rdi",
        "oai": "oai",
        "cai": "cai",
        "mai": "mai",
        "hyi": "hyi",
        "rerai": "rerai",
        "arousal_index": "arousal_index",
        "hypoxic_burden": "hypoxia_burden",
        "odi3": "odi3",
        "odi4": "odi4",
        "desaturation_burden": "t90_desaturation",
    }
    for out_col, in_col in direct_map.items():
        if out_col in df.columns:
            out[out_col] = _numeric(df[out_col])
        elif in_col in df.columns:
            out[out_col] = _numeric(df[in_col])
        else:
            out[out_col] = np.nan

    # Keep the legacy manuscript-facing name for backward compatibility while
    # also exposing both ODI thresholds explicitly.
    out["desaturation_index"] = _numeric(df["desaturation_index"]) if "desaturation_index" in df.columns else out["odi3"]
    out["event_duration_burden"] = _numeric(df["event_duration_burden"]) if "event_duration_burden" in df.columns else np.nan

    if "hours_sleep" in df.columns:
        hours_sleep = _numeric(df["hours_sleep"])
    else:
        hours_sleep = pd.Series(np.nan, index=df.index)
    out["tst_min"] = _numeric(df["tst_min"]) if "tst_min" in df.columns else _scaled(hours_sleep, 60.0)
    out["sleep_efficiency"] = (
        _raw_fraction_to_percent(df["sleep_efficiency"]) if "sleep_efficiency" in df.columns else np.nan
    )
    out["waso_min"] = _numeric(df["waso_min"]) if "waso_min" in df.columns else _numeric(df["waso"]) if "waso" in df.columns else np.nan

    stage_pct_map = {
        "n1_pct": "perc_n1",
        "n2_pct": "perc_n2",
        "n3_pct": "perc_n3",
        "rem_pct": "perc_r",
    }
    for out_col, in_col in stage_pct_map.items():
        if out_col in df.columns:
            out[out_col] = _raw_fraction_to_percent(df[out_col])
        elif in_col in df.columns:
            out[out_col] = _scaled(df[in_col], 100.0)
        else:
            out[out_col] = np.nan

    for out_col, pct_col in [("n1_min", "perc_n1"), ("n2_min", "perc_n2"), ("n3_min", "perc_n3"), ("rem_min", "perc_r")]:
        if out_col in df.columns:
            out[out_col] = _numeric(df[out_col])
        elif pct_col in df.columns:
            out[out_col] = (_numeric(df[pct_col]) * hours_sleep * 60.0).round(1)
        else:
            out[out_col] = np.nan

    out = out.drop_duplicates(subset=["fileid"], keep="first")
    for col in REVISION_SLEEP_GENERAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[REVISION_SLEEP_GENERAL_COLUMNS]


def load_sleep_general_revision_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return transform_sleep_general_revision_table(df)


def write_sleep_general_revision_audit(
    df: pd.DataFrame,
    audit_md: Path,
    *,
    source: Path | str,
    title: str = "Sleep General Revision Features Audit",
) -> None:
    status_counts = (
        df["status"].fillna("missing").value_counts(dropna=False).sort_index()
        if "status" in df.columns
        else pd.Series(dtype=int)
    )
    lines = [
        f"# {title}",
        "",
        f"- Rows: `{len(df):,}`",
        f"- Unique PSGs (`fileid`): `{df['fileid'].nunique(dropna=True):,}`",
        f"- Source: `{source}`",
        "",
        "## Status Counts",
        "",
        "| status | rows |",
        "|---|---:|",
    ]
    if len(status_counts) == 0:
        lines.append("| `missing` | `0` |")
    else:
        for status, count in status_counts.items():
            lines.append(f"| `{status}` | `{int(count):,}` |")

    lines.extend(
        [
            "",
            "## Metric Availability",
            "",
            "| Column | Non-missing |",
            "|---|---:|",
        ]
    )
    for column in REVISION_SLEEP_GENERAL_METRIC_COLUMNS:
        lines.append(f"| `{column}` | `{int(df[column].notna().sum()):,}` |")

    if "cohort" in df.columns:
        lines.extend(
            [
                "",
                "## By Cohort",
                "",
                "| cohort | rows | ok | with_ahi | with_tst |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for cohort, cohort_df in df.groupby("cohort", dropna=False):
            cohort_label = cohort if pd.notna(cohort) else "missing"
            ok_rows = int((cohort_df["status"] == "ok").sum()) if "status" in cohort_df.columns else 0
            lines.append(
                f"| `{cohort_label}` | `{len(cohort_df):,}` | `{ok_rows:,}` | "
                f"`{int(cohort_df['ahi'].notna().sum()):,}` | `{int(cohort_df['tst_min'].notna().sum()):,}` |"
            )

    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
