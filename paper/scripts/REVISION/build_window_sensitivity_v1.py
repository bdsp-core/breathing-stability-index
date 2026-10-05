#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


OUTDIR = Path("REVISION/window_sensitivity_v1")
ANALYSIS_DIRS = {
    1.0: Path("analysis_cog_dx_1_90"),
    2.0: Path("analysis_cog_dx_2_90"),
    5.0: Path("analysis_cog_dx_5_90"),
}
SAMPLE_SUMMARY = Path("stability_index_samples/summary_stability.csv")


def read_bootstrap() -> pd.DataFrame:
    pieces = []
    for window, path in ANALYSIS_DIRS.items():
        csv = path / "results_predictor_set_performance_bootstrap_stability_variants.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["window_length_min"] = window
        df["source_file"] = str(csv)
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def summarize_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    key_sets = [
        "stability_global",
        "stability_quantiles",
        "stability_main_nrem_rem",
        "stability_hist_5",
        "stability_hist_12",
    ]
    key_tasks = ["cog_fluid", "cog_crystallized", "cog_total"]
    sub = df[df["predictor_set"].isin(key_sets) & df["task"].isin(key_tasks)].copy()
    return sub.sort_values(["task", "predictor_set", "window_length_min"])


def build_sample_correlations() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SAMPLE_SUMMARY.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(SAMPLE_SUMMARY)
    wide = df.pivot_table(index="file", columns="window_length", values=["median_stability", "stability_auc"])
    rows = []
    for metric in ["median_stability", "stability_auc"]:
        metric_wide = wide[metric].dropna()
        for a, b in [(1.0, 2.0), (2.0, 5.0), (1.0, 5.0)]:
            paired = metric_wide[[a, b]].dropna()
            if len(paired) >= 3:
                rho, p = stats.spearmanr(paired[a], paired[b])
                pearson, pearson_p = stats.pearsonr(paired[a], paired[b])
                diff = paired[b] - paired[a]
            else:
                rho = p = pearson = pearson_p = np.nan
                diff = pd.Series(dtype=float)
            rows.append(
                {
                    "metric": metric,
                    "window_a_min": a,
                    "window_b_min": b,
                    "n_paired_files": int(len(paired)),
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "spearman_p": float(p) if pd.notna(p) else np.nan,
                    "pearson_r": float(pearson) if pd.notna(pearson) else np.nan,
                    "pearson_p": float(pearson_p) if pd.notna(pearson_p) else np.nan,
                    "median_b_minus_a": float(diff.median()) if len(diff) else np.nan,
                    "median_abs_b_minus_a": float(diff.abs().median()) if len(diff) else np.nan,
                }
            )
    long = df.copy()
    return pd.DataFrame(rows), long


def write_note(bootstrap: pd.DataFrame, sample_corr: pd.DataFrame) -> None:
    lines = [
        "# Window-Length Sensitivity v1",
        "",
        "## Scope",
        "- This bundle summarizes pre-existing 1, 2, and 5 minute window artifacts.",
        "- It is not a new full-cohort v4 rerun and should be cited as supportive sensitivity evidence only.",
        "- All located artifacts use 90% overlap; no non-90% overlap sweep was found.",
        "",
        "## Located artifacts",
    ]
    for window, path in ANALYSIS_DIRS.items():
        lines.append(f"- `{window:g}` min cognition/disease analysis directory: `{path}`")
    lines.append(f"- Sample trace summary: `{SAMPLE_SUMMARY}`")

    if not sample_corr.empty:
        median_12 = sample_corr[
            sample_corr["metric"].eq("median_stability")
            & sample_corr["window_a_min"].eq(1.0)
            & sample_corr["window_b_min"].eq(2.0)
        ].iloc[0]
        median_25 = sample_corr[
            sample_corr["metric"].eq("median_stability")
            & sample_corr["window_a_min"].eq(2.0)
            & sample_corr["window_b_min"].eq(5.0)
        ].iloc[0]
        lines.extend(
            [
                "",
                "## Sample-level stability",
                f"- Sample files with 1/2/5 min summaries: `{int(median_12['n_paired_files'])}`.",
                f"- Median-stability Spearman 1 vs 2 min: `{median_12['spearman_rho']:.3f}`.",
                f"- Median-stability Spearman 2 vs 5 min: `{median_25['spearman_rho']:.3f}`.",
            ]
        )

    if not bootstrap.empty:
        rows = bootstrap[
            bootstrap["task"].eq("cog_total")
            & bootstrap["predictor_set"].eq("stability_quantiles")
            & bootstrap["metric"].eq("correlation")
        ].sort_values("window_length_min")
        if not rows.empty:
            parts = [f"{r['window_length_min']:g} min={r['ci_median']:.3f}" for _, r in rows.iterrows()]
            lines.extend(
                [
                    "",
                    "## Legacy outcome-model sensitivity",
                    "- Pre-existing cognition/disease bootstrap outputs are available for the same 90% overlap at 1, 2, and 5 min.",
                    f"- Example `cog_total` / `stability_quantiles` median bootstrap correlation: `{'; '.join(parts)}`.",
                ]
            )

    lines.extend(
        [
            "",
            "## Reviewer-facing interpretation",
            "- We can say prior 1/2/5 min artifacts exist and show broadly comparable behavior in sample-level summaries.",
            "- We should not claim a current v4 full-cohort window-size rerun unless new O2 outputs are generated.",
            "- No empirical non-90% overlap sensitivity or higher/native sampling-rate sensitivity was found.",
        ]
    )
    (OUTDIR / "window_sensitivity_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bootstrap = read_bootstrap()
    bootstrap_summary = summarize_bootstrap(bootstrap)
    sample_corr, sample_long = build_sample_correlations()
    bootstrap.to_csv(OUTDIR / "legacy_window_bootstrap_all.csv", index=False)
    bootstrap_summary.to_csv(OUTDIR / "legacy_window_bootstrap_key_tasks.csv", index=False)
    sample_corr.to_csv(OUTDIR / "sample_window_correlations.csv", index=False)
    sample_long.to_csv(OUTDIR / "sample_window_summary_long.csv", index=False)
    write_note(bootstrap_summary, sample_corr)
    pd.DataFrame(
        [
            {
                "n_bootstrap_rows": int(len(bootstrap)),
                "n_key_bootstrap_rows": int(len(bootstrap_summary)),
                "n_sample_rows": int(len(sample_long)),
                "n_sample_correlation_rows": int(len(sample_corr)),
            }
        ]
    ).to_csv(OUTDIR / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
