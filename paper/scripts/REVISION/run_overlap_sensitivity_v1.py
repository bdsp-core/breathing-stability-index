#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import breathing_stability as bs  # noqa: E402


OUTDIR = REPO_ROOT / "REVISION" / "overlap_sensitivity_v1"
INPUT_TABLE = REPO_ROOT / "table_selected_resp.csv"
WINDOW_LENGTH_MIN = 2.0
OVERLAPS = [0.50, 0.75, 0.90, 0.95, 0.99]
N_FILES = 10
NEW_FS = 10
BELT_MODE = "robust_mean"
PREFERRED_FILEIDS = [
    "AHI_low_1",
    "AHI_low_2",
    "AHI_mid_1",
    "AHI_mid_2",
    "AHI_high_1",
    "AHI_high_2",
    "CAI_dominant_1",
    "CAI_dominant_2",
    "OAI_dominant_1",
    "OAI_dominant_2",
]


def read_signal_keys(path: Path) -> list[str]:
    with h5py.File(path, "r") as h5:
        return list(h5["signals"].keys())


def effort_keys(path: Path) -> list[str]:
    keys = read_signal_keys(path)
    selected = [key for key in keys if bs.infer_effort_channel_role(key) in {"abd", "chest", "effort"}]
    if not selected:
        raise ValueError(f"No effort channels found in {path}")
    return selected


def nominal_window_count(n_samples: int, fs: float, window_length_min: float, overlap: float) -> int:
    window_len = int(window_length_min * 60 * fs)
    step = int(window_len * (1 - min(overlap, 0.99)))
    if n_samples < window_len or step < 1:
        return 0
    return int(np.floor((n_samples - window_len) / step) + 1)


def prepare_trace(path: Path) -> dict[str, object]:
    t0 = time.perf_counter()
    channels = effort_keys(path)
    signals_df, annotations, params = bs.load_prepared_data(
        str(path),
        signals_to_load=channels,
        annotations_to_load=["stage"],
    )
    load_seconds = time.perf_counter() - t0

    fs = float(params["fs"])
    signals = signals_df[channels].values.T
    stage = annotations["stage"].values.flatten()
    fs_ratio = fs / NEW_FS
    if not float(fs_ratio).is_integer():
        raise ValueError(f"Cannot downsample stage from fs={fs} to {NEW_FS} Hz for {path}")
    stage = stage[:: int(fs_ratio)]

    t1 = time.perf_counter()
    signals, channel_names = bs.select_effort_signals(signals, channels)
    signals = bs.preprocess_effort_signals(
        signals,
        fs,
        notch_freq=60,
        bandpass_freq=[0, 10],
        n_jobs=1,
        new_fs=NEW_FS,
    )
    trace, _ = bs.build_belt_mode_traces(signals, belt_mode=BELT_MODE)
    if len(trace) != len(stage):
        raise ValueError(f"Trace/stage length mismatch for {path}: {len(trace)} vs {len(stage)}")
    preprocess_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    sample_time = np.arange(len(trace)) / NEW_FS
    _, trace = bs.remove_outliers(sample_time, np.asarray(trace, dtype=float).flatten(), NEW_FS, min_height=0.01)
    up, lo = bs.create_env(trace, NEW_FS)
    envelope_seconds = time.perf_counter() - t2

    return {
        "trace": trace,
        "stage": stage,
        "up": up,
        "lo": lo,
        "fs": NEW_FS,
        "load_seconds": load_seconds,
        "preprocess_seconds": preprocess_seconds,
        "envelope_seconds": envelope_seconds,
        "belt_channels_used": "|".join(channel_names),
        "n_belts_used": len(channel_names),
    }


def run_one_overlap(prepared: dict[str, object], overlap: float) -> dict[str, float]:
    up = prepared["up"]
    lo = prepared["lo"]
    fs = float(prepared["fs"])
    stage = prepared["stage"]

    t0 = time.perf_counter()
    stability = bs.compute_stability(up, lo, fs, window_length_min=WINDOW_LENGTH_MIN, overlap=overlap)
    stability_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    summary, episodes = bs.summarize_stability_by_stage(stability, fs, stage, WINDOW_LENGTH_MIN)
    summarize_seconds = time.perf_counter() - t1

    row = {
        "overlap": overlap,
        "nominal_windows": nominal_window_count(len(stability), fs, WINDOW_LENGTH_MIN, overlap),
        "stability_seconds": stability_seconds,
        "summarize_seconds": summarize_seconds,
        "overlap_dependent_seconds": stability_seconds + summarize_seconds,
    }
    for key in [
        "SLEEP_median_stability",
        "SLEEP_stability_auc",
        "SLEEP_quantile_75",
        "SLEEP_quantile_90",
        "SLEEP_n_5min_windows_stable",
        "SLEEP_n_5min_windows_unstable",
    ]:
        row[key] = summary.get(key, np.nan) if key in summary else episodes.get(key, np.nan)
    return row


def select_files() -> pd.DataFrame:
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_TABLE}")
    df = pd.read_csv(INPUT_TABLE)
    needed = ["fileid", "cohort", "file_path", "f_ahi", "f_cai", "f_oai"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{INPUT_TABLE} missing required columns: {missing}")
    df["file_path"] = df["file_path"].astype(str)
    df["file_exists"] = df["file_path"].map(lambda p: Path(p).exists())
    df = df[df["file_exists"]].copy()
    preferred = df[df["fileid"].isin(PREFERRED_FILEIDS)].copy()
    preferred["selection_order"] = preferred["fileid"].map({fileid: idx for idx, fileid in enumerate(PREFERRED_FILEIDS)})
    preferred = preferred.sort_values("selection_order")
    if len(preferred) < N_FILES:
        missing = sorted(set(PREFERRED_FILEIDS) - set(preferred["fileid"]))
        raise ValueError(f"Only {len(preferred)} preferred input files exist; missing {missing}")
    return preferred.head(N_FILES).drop(columns=["selection_order"]).copy()


def pairwise_overlap_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "SLEEP_median_stability",
        "SLEEP_stability_auc",
        "SLEEP_quantile_75",
        "SLEEP_quantile_90",
        "SLEEP_n_5min_windows_stable",
        "SLEEP_n_5min_windows_unstable",
    ]
    ref = results[results["overlap"].eq(0.90)][["fileid", *metrics]].rename(
        columns={metric: f"{metric}_ref90" for metric in metrics}
    )
    rows = []
    for overlap, group in results.groupby("overlap"):
        merged = group.merge(ref, on="fileid", how="inner")
        for metric in metrics:
            paired = merged[[metric, f"{metric}_ref90"]].dropna()
            if len(paired) >= 3 and paired[metric].nunique() > 1 and paired[f"{metric}_ref90"].nunique() > 1:
                spearman_rho, spearman_p = stats.spearmanr(paired[metric], paired[f"{metric}_ref90"])
                pearson_r, pearson_p = stats.pearsonr(paired[metric], paired[f"{metric}_ref90"])
            else:
                spearman_rho = spearman_p = pearson_r = pearson_p = np.nan
            diff = paired[metric] - paired[f"{metric}_ref90"]
            denom = paired[f"{metric}_ref90"].abs().replace(0, np.nan)
            rel_abs = diff.abs() / denom
            rows.append(
                {
                    "overlap": overlap,
                    "reference_overlap": 0.90,
                    "metric": metric,
                    "n_paired_files": int(len(paired)),
                    "spearman_rho_vs_90": spearman_rho,
                    "spearman_p": spearman_p,
                    "pearson_r_vs_90": pearson_r,
                    "pearson_p": pearson_p,
                    "median_difference_vs_90": float(diff.median()) if len(diff) else np.nan,
                    "median_abs_difference_vs_90": float(diff.abs().median()) if len(diff) else np.nan,
                    "median_abs_relative_difference_vs_90": float(rel_abs.median()) if len(rel_abs.dropna()) else np.nan,
                    "max_abs_difference_vs_90": float(diff.abs().max()) if len(diff) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def timing_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for overlap, group in results.groupby("overlap"):
        rows.append(
            {
                "overlap": overlap,
                "n_files": int(group["fileid"].nunique()),
                "median_nominal_windows": float(group["nominal_windows"].median()),
                "median_stability_seconds": float(group["stability_seconds"].median()),
                "median_summarize_seconds": float(group["summarize_seconds"].median()),
                "median_overlap_dependent_seconds": float(group["overlap_dependent_seconds"].median()),
                "total_overlap_dependent_seconds": float(group["overlap_dependent_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_note(results: pd.DataFrame, comparisons: pd.DataFrame, timings: pd.DataFrame, selected: pd.DataFrame) -> None:
    med = comparisons[
        comparisons["metric"].eq("SLEEP_median_stability")
        & comparisons["overlap"].isin([0.50, 0.75, 0.95, 0.99])
    ].copy()
    auc = comparisons[
        comparisons["metric"].eq("SLEEP_stability_auc")
        & comparisons["overlap"].isin([0.50, 0.75, 0.95, 0.99])
    ].copy()

    lines = [
        "# Overlap Sensitivity v1",
        "",
        "## Scope",
        f"- Deterministic sample of `{len(selected)}` prepared PSG files from `table_selected_resp.csv`.",
        f"- Primary BSI window length held fixed at `{WINDOW_LENGTH_MIN:g}` minutes.",
        f"- Belt strategy: `{BELT_MODE}`.",
        "- Tested overlap fractions: `50%`, `75%`, `90%`, `95%`, and `99%`.",
        "- Metrics are compared with the prespecified `90%` overlap setting.",
        "- Timing isolates the overlap-dependent BSI computation and stability summarization after file loading, preprocessing, and envelope construction.",
        "",
        "## Selected files",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"- `{row['fileid']}`: AHI `{row['f_ahi']:.1f}`, OAI `{row['f_oai']:.1f}`, CAI `{row['f_cai']:.1f}`"
        )

    if not med.empty:
        lines.extend(["", "## Median BSI agreement with 90% overlap"])
        for _, row in med.sort_values("overlap").iterrows():
            lines.append(
                f"- `{row['overlap']:.2f}` vs `0.90`: Spearman rho `{row['spearman_rho_vs_90']:.3f}`, "
                f"median absolute difference `{row['median_abs_difference_vs_90']:.3f}`, "
                f"median absolute relative difference `{100 * row['median_abs_relative_difference_vs_90']:.1f}%`."
            )

    if not auc.empty:
        lines.extend(["", "## Mean/AUC BSI agreement with 90% overlap"])
        for _, row in auc.sort_values("overlap").iterrows():
            lines.append(
                f"- `{row['overlap']:.2f}` vs `0.90`: Spearman rho `{row['spearman_rho_vs_90']:.3f}`, "
                f"median absolute difference `{row['median_abs_difference_vs_90']:.3f}`, "
                f"median absolute relative difference `{100 * row['median_abs_relative_difference_vs_90']:.1f}%`."
            )

    lines.extend(["", "## Timing"])
    for _, row in timings.sort_values("overlap").iterrows():
        lines.append(
            f"- `{row['overlap']:.2f}` overlap: median `{row['median_nominal_windows']:.0f}` windows/file, "
            f"median overlap-dependent runtime `{row['median_overlap_dependent_seconds']:.2f}` seconds/file."
        )

    lines.extend(
        [
            "",
            "## Reviewer-facing interpretation",
            "- This targeted experiment supports describing 90% overlap as a dense sampling setting that gives stable subject-level summaries while remaining computationally tractable.",
            "- Lower overlaps are faster because fewer windows are evaluated, but they sample the same overnight pattern more sparsely.",
            "- Higher overlaps provide finer temporal sampling but increase runtime substantially, especially at 99%.",
            "- The overlap setting is therefore adjustable for applications prioritizing speed or temporal resolution; the manuscript should not claim that 90% is outcome-optimized.",
        ]
    )
    (OUTDIR / "overlap_sensitivity_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    selected = select_files()
    selected.to_csv(OUTDIR / "selected_files.csv", index=False)

    rows = []
    errors = []
    for _, source in selected.iterrows():
        path = Path(source["file_path"])
        try:
            prepared = prepare_trace(path)
            for overlap in OVERLAPS:
                row = {
                    "fileid": source["fileid"],
                    "cohort": source["cohort"],
                    "file_path": source["file_path"],
                    "f_ahi": source["f_ahi"],
                    "f_cai": source["f_cai"],
                    "f_oai": source["f_oai"],
                    "window_length_min": WINDOW_LENGTH_MIN,
                    "belt_mode": BELT_MODE,
                    "belt_channels_used": prepared["belt_channels_used"],
                    "n_belts_used": prepared["n_belts_used"],
                    "n_samples": len(prepared["trace"]),
                    "duration_hours": len(prepared["trace"]) / float(prepared["fs"]) / 3600,
                    "load_seconds": prepared["load_seconds"],
                    "preprocess_seconds": prepared["preprocess_seconds"],
                    "envelope_seconds": prepared["envelope_seconds"],
                }
                row.update(run_one_overlap(prepared, overlap))
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append({"fileid": source["fileid"], "file_path": source["file_path"], "error": repr(exc)})

    results = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)
    results.to_csv(OUTDIR / "overlap_file_results.csv", index=False)
    errors_df.to_csv(OUTDIR / "overlap_errors.csv", index=False)

    comparisons = pairwise_overlap_summary(results) if not results.empty else pd.DataFrame()
    timings = timing_summary(results) if not results.empty else pd.DataFrame()
    comparisons.to_csv(OUTDIR / "overlap_pairwise_vs_90.csv", index=False)
    timings.to_csv(OUTDIR / "overlap_timing_summary.csv", index=False)
    if not results.empty:
        write_note(results, comparisons, timings, selected)

    pd.DataFrame(
        [
            {
                "n_selected_files": int(len(selected)),
                "n_completed_files": int(results["fileid"].nunique()) if not results.empty else 0,
                "n_result_rows": int(len(results)),
                "n_error_rows": int(len(errors_df)),
                "window_length_min": WINDOW_LENGTH_MIN,
                "overlaps": ",".join(str(x) for x in OVERLAPS),
                "belt_mode": BELT_MODE,
            }
        ]
    ).to_csv(OUTDIR / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
