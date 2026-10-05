#!/usr/bin/env python3
"""Identify and plot paradoxical breathing examples for the BSI revision.

The scan is intentionally conservative:

- ABD/CHEST polarity is aligned per recording using the median whole-recording
  belt correlation, so a globally inverted sensor is not mislabeled paradoxical.
- Candidate windows require sustained negative belt phase relationship.
- Airflow amplitude is used as supporting evidence when present.
- The default selection keeps diagnostic PSGs only and caps examples per
  recording, producing visually independent examples from multiple subjects.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, hilbert, resample_poly, sosfiltfilt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from breathing_stability import compute_stability, create_env, remove_outliers


MASTER = REPO / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
OUTDIR = REPO / "REVISION" / "paradox_breathing_examples_v1"
NEW_FS = 10
SCAN_WINDOW_SEC = 30
SCAN_STEP_SEC = 5
PLOT_WINDOW_SEC = 300
LOCAL_BSI_WINDOW_SEC = 120
BSI_GRADIENT_CMAP = "coolwarm"
BSI_GRADIENT_VMAX = 3.0
BSI_MIN_YMAX = 3.0
DEFAULT_OVERVIEW_EXAMPLE_IDS = [1, 2, 3, 5, 6, 7, 14]
DEFAULT_ANNOTATION_CACHE_DIRS = [
    Path("/media/cdac_sleep/WolfgangGanglberger/psg_annotation_harmonization/cache/annotation_csvs"),
    Path("/d/cdac_sleep/WolfgangGanglberger/psg_annotation_harmonization/cache/annotation_csvs"),
]
DEFAULT_EXCLUDED_SELECTION_FILEIDS = []  # Author-specific recording exclusions omitted from the public release.
EVENT_COLORS = {
    "obstructive_apnea": "#0b3d91",
    "central_apnea": "#2ca25f",
    "hypopnea": "#56cfe1",
}
DEFAULT_EDF_DIRS = [
    Path("/home/wolfgang/repos/sleep-clinic-tools/input_files"),
    Path("/home/wolfgang/repos/bdsp-sleep-data"),
]


@dataclass
class RecordingSignals:
    fileid: str
    path: Path
    fs: float
    channels: dict[str, str]
    abd: np.ndarray
    chest: np.ndarray
    flow: np.ndarray | None
    abd_resp: np.ndarray
    chest_resp: np.ndarray
    flow_resp: np.ndarray | None
    orientation_median_corr: float
    chest_sign: int
    meas_date: object


def fileid_from_path(path: Path) -> str:
    name = path.name
    for suffix in [
        "_task-psg_eeg.edf",
        "_task-PSG_eeg.edf",
        ".edf",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def infer_role(channel: str) -> str | None:
    name = channel.lower()
    if "abd" in name or "abdom" in name:
        return "abd"
    if "chest" in name or "thor" in name or "tho" in name:
        return "chest"
    return None


def choose_channels(raw: mne.io.BaseRaw) -> dict[str, str]:
    channels: dict[str, str] = {}
    for channel in raw.info["ch_names"]:
        role = infer_role(channel)
        if role and role not in channels:
            channels[role] = channel

    flow_priorities = [
        ["ptaf", "npt"],
        ["airflow", "air flow", "new air"],
        ["therm"],
        ["c-flow", "cflow", "c flow"],
        ["flow"],
    ]
    for priority_group in flow_priorities:
        for token in priority_group:
            for channel in raw.info["ch_names"]:
                lower = channel.lower()
                if token in lower and not any(bad in lower for bad in ["pressure", "xflow", "volume", "leak"]):
                    channels["flow"] = channel
                    return channels
    return channels


def robust_z(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    lo, hi = np.nanpercentile(signal, [2, 98])
    clipped = np.clip(signal, lo, hi)
    median = np.nanmedian(clipped)
    iqr = np.nanpercentile(clipped, 75) - np.nanpercentile(clipped, 25)
    if not np.isfinite(iqr) or iqr < 1e-12:
        iqr = np.nanstd(clipped) + 1e-12
    return (signal - median) / iqr


def resample_to_10hz(signal: np.ndarray, fs: float) -> np.ndarray:
    fs_int = int(round(fs))
    gcd = math.gcd(fs_int, NEW_FS)
    return resample_poly(signal, NEW_FS // gcd, fs_int // gcd).astype(float)


def respiratory_band(signal: np.ndarray, fs: float = NEW_FS) -> np.ndarray:
    sos = butter(3, [0.05, 0.8], btype="bandpass", fs=fs, output="sos")
    centered = np.nan_to_num(signal - np.nanmedian(signal))
    return sosfiltfilt(sos, centered)


def load_recording(path: Path) -> RecordingSignals | None:
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    channels = choose_channels(raw)
    if "abd" not in channels or "chest" not in channels:
        return None

    picks = [channels["abd"], channels["chest"]]
    if "flow" in channels:
        picks.append(channels["flow"])

    data = raw.get_data(picks=picks, verbose=False)
    resampled = np.vstack([resample_to_10hz(channel_data, raw.info["sfreq"]) for channel_data in data])
    n = min(len(x) for x in resampled)
    resampled = resampled[:, :n]

    abd = robust_z(detrend(resampled[0]))
    chest = robust_z(detrend(resampled[1]))
    flow = robust_z(detrend(resampled[2])) if resampled.shape[0] > 2 else None

    abd_resp = respiratory_band(abd)
    chest_resp = respiratory_band(chest)
    flow_resp = respiratory_band(flow) if flow is not None else None

    correlations = []
    win = 60 * NEW_FS
    step = 30 * NEW_FS
    for start in range(0, n - win, step):
        a = abd_resp[start : start + win]
        b = chest_resp[start : start + win]
        if np.nanstd(a) > 1e-6 and np.nanstd(b) > 1e-6:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    orientation_median_corr = float(np.nanmedian(correlations)) if correlations else 1.0
    chest_sign = -1 if orientation_median_corr < 0 else 1
    if chest_sign == -1:
        chest = -chest
        chest_resp = -chest_resp

    return RecordingSignals(
        fileid=fileid_from_path(path),
        path=path,
        fs=NEW_FS,
        channels=channels,
        abd=abd,
        chest=chest,
        flow=flow,
        abd_resp=abd_resp,
        chest_resp=chest_resp,
        flow_resp=flow_resp,
        orientation_median_corr=orientation_median_corr,
        chest_sign=chest_sign,
        meas_date=raw.info["meas_date"],
    )


def scan_recording(signals: RecordingSignals) -> list[dict[str, float | str]]:
    n = len(signals.abd_resp)
    analytic_abd = hilbert(signals.abd_resp)
    analytic_chest = hilbert(signals.chest_resp)
    cos_phase_diff = np.cos(np.angle(analytic_abd) - np.angle(analytic_chest))
    effort_amp = np.minimum(np.abs(analytic_abd), np.abs(analytic_chest))
    effort_ref = np.nanpercentile(effort_amp, np.arange(1, 100))

    if signals.flow_resp is not None:
        flow_amp = np.abs(hilbert(signals.flow_resp))
        flow_ref = np.nanpercentile(flow_amp, np.arange(1, 100))
    else:
        flow_amp = None
        flow_ref = None

    win = SCAN_WINDOW_SEC * NEW_FS
    step = SCAN_STEP_SEC * NEW_FS
    rows = []
    for start in range(0, n - win, step):
        end = start + win
        abd = signals.abd_resp[start:end]
        chest = signals.chest_resp[start:end]
        if np.nanstd(abd) < 1e-6 or np.nanstd(chest) < 1e-6:
            continue

        corr = float(np.corrcoef(abd, chest)[0, 1])
        anti_phase_fraction = float(np.mean(cos_phase_diff[start:end] < -0.5))
        effort_percentile = float(np.searchsorted(effort_ref, np.nanmedian(effort_amp[start:end])) + 1)
        if flow_amp is not None and flow_ref is not None:
            flow_percentile = float(np.searchsorted(flow_ref, np.nanmedian(flow_amp[start:end])) + 1)
        else:
            flow_percentile = np.nan

        if (
            corr < -0.45
            and anti_phase_fraction > 0.45
            and effort_percentile > 25
            and (np.isnan(flow_percentile) or flow_percentile < 70)
        ):
            score = (
                (-corr) * 2
                + anti_phase_fraction
                + effort_percentile / 100
                + (0 if np.isnan(flow_percentile) else (100 - flow_percentile) / 120)
            )
            rows.append(
                {
                    "fileid": signals.fileid,
                    "source_path": str(signals.path),
                    "abd_channel": signals.channels["abd"],
                    "chest_channel": signals.channels["chest"],
                    "flow_channel": signals.channels.get("flow", ""),
                    "window_start_sec": start / NEW_FS,
                    "window_end_sec": end / NEW_FS,
                    "window_center_sec": (start + end) / (2 * NEW_FS),
                    "belt_corr": corr,
                    "anti_phase_fraction": anti_phase_fraction,
                    "effort_amp_percentile": effort_percentile,
                    "flow_amp_percentile": flow_percentile,
                    "candidate_score": score,
                    "orientation_median_corr": signals.orientation_median_corr,
                    "chest_sign_after_alignment": signals.chest_sign,
                }
            )

    rows = sorted(rows, key=lambda row: row["window_start_sec"])
    groups: list[list[dict[str, float | str]]] = []
    for row in rows:
        if not groups or row["window_start_sec"] > groups[-1][-1]["window_end_sec"] + 10:
            groups.append([row])
        else:
            groups[-1].append(row)

    grouped = []
    for group in groups:
        best = dict(max(group, key=lambda row: row["candidate_score"]))
        best["segment_start_sec"] = min(row["window_start_sec"] for row in group)
        best["segment_end_sec"] = max(row["window_end_sec"] for row in group)
        best["segment_duration_sec"] = best["segment_end_sec"] - best["segment_start_sec"]
        best["n_candidate_windows"] = len(group)
        grouped.append(best)

    return sorted(grouped, key=lambda row: row["candidate_score"], reverse=True)


def compute_local_bsi(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=float)
    t = np.arange(len(trace)) / NEW_FS
    try:
        _, cleaned = remove_outliers(t, trace, NEW_FS, min_height=0.01)
        up, lo = create_env(np.asarray(cleaned), NEW_FS)
        if up is None or lo is None:
            return np.nan
        stability = compute_stability(
            np.asarray(up, dtype=float),
            np.asarray(lo, dtype=float),
            NEW_FS,
            window_length_min=LOCAL_BSI_WINDOW_SEC / 60,
            overlap=0.90,
        )
        return float(np.nanmedian(stability))
    except Exception:
        return np.nan


def compute_bsi_trace(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=float)
    t = np.arange(len(trace)) / NEW_FS
    try:
        _, cleaned = remove_outliers(t, trace, NEW_FS, min_height=0.01)
        up, lo = create_env(np.asarray(cleaned), NEW_FS)
        if up is None or lo is None:
            return np.full(len(trace), np.nan)
        return compute_stability(
            np.asarray(up, dtype=float),
            np.asarray(lo, dtype=float),
            NEW_FS,
            window_length_min=2.0,
            overlap=0.90,
        )
    except Exception:
        return np.full(len(trace), np.nan)


def annotation_path_for_edf(path: Path) -> Path | None:
    fileid = fileid_from_path(path)
    candidates = []
    if path.name.endswith("_eeg.edf"):
        candidates.append(path.with_name(path.name.replace("_eeg.edf", "_annotations.csv")))
    candidates.append(path.with_suffix(".csv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    cache_matches = [
        candidate
        for candidate in cached_annotation_paths()
        if fileid in candidate.name and "sleep_annotations" not in candidate.name and "sleepannotations" not in candidate.name
    ]
    priority_tokens = [
        "events_annotations",
        "eventannotations",
        "_events",
        "_annotations",
    ]
    for token in priority_tokens:
        for candidate in cache_matches:
            if token in candidate.name:
                return candidate
    if cache_matches:
        return cache_matches[0]
    return None


@lru_cache(maxsize=1)
def cached_annotation_paths() -> tuple[Path, ...]:
    paths = []
    for cache_dir in DEFAULT_ANNOTATION_CACHE_DIRS:
        if cache_dir.exists():
            paths.extend(sorted(cache_dir.glob("*.csv")))
    return tuple(paths)


def classify_event(event: str) -> str | None:
    normalized = event.lower().replace("_", "").replace("-", "").replace(" ", "")
    if "obstructiveapnea" in normalized:
        return "obstructive_apnea"
    if "centralapnea" in normalized:
        return "central_apnea"
    if "hypopnea" in normalized:
        return "hypopnea"
    return None


def time_string_to_seconds(value: object) -> float | None:
    try:
        parts = str(value).split(":")
        if len(parts) != 3:
            return None
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return None


def numeric_duration(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        duration = float(value)
        if not np.isfinite(duration):
            return None
        return duration
    except Exception:
        return None


def recording_start_clock_seconds(signals: RecordingSignals) -> float:
    meas_date = signals.meas_date
    if meas_date is None:
        return 0.0
    meas_date = meas_date.replace(tzinfo=None)
    return meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second + meas_date.microsecond / 1e6


def load_marked_events(signals: RecordingSignals) -> pd.DataFrame:
    annotation_path = annotation_path_for_edf(signals.path)
    columns = ["event_type", "event_label", "start_sec", "end_sec", "duration_sec", "annotation_path"]
    if annotation_path is None:
        return pd.DataFrame(columns=columns)

    annotations = pd.read_csv(annotation_path)
    start_clock = recording_start_clock_seconds(signals)
    rows = []
    for _, row in annotations.iterrows():
        event_label = str(row.get("event", row.get("Description", "")))
        event_type = classify_event(event_label)
        duration = numeric_duration(row.get("duration", row.get("Length")))
        if event_type is None or duration is None or duration <= 0:
            continue
        record_sec = time_string_to_seconds(row.get("Record Time"))
        if record_sec is not None:
            rel_sec = record_sec
        else:
            clock_sec = time_string_to_seconds(row.get("time", row.get("Time")))
            if clock_sec is None:
                continue
            rel_sec = clock_sec - start_clock
            if rel_sec < 0:
                rel_sec += 24 * 3600
        rows.append(
            {
                "event_type": event_type,
                "event_label": event_label,
                "start_sec": rel_sec,
                "end_sec": rel_sec + duration,
                "duration_sec": duration,
                "annotation_path": str(annotation_path),
            }
        )
    event_df = pd.DataFrame(rows, columns=columns)
    if not event_df.empty:
        event_df["_rounded_start"] = event_df["start_sec"].round(1)
        event_df["_rounded_end"] = event_df["end_sec"].round(1)
        event_df = event_df.drop_duplicates(["event_type", "_rounded_start", "_rounded_end"])
        event_df = event_df.drop(columns=["_rounded_start", "_rounded_end"]).reset_index(drop=True)
    return event_df


def overlapping_events(events: pd.DataFrame, start_sec: float, end_sec: float) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events[(events["end_sec"] >= start_sec) & (events["start_sec"] <= end_sec)].copy()


def centered_plot_bounds(
    segment_start_sec: float,
    segment_end_sec: float,
    recording_duration_sec: float | None = None,
) -> tuple[float, float]:
    center_sec = (segment_start_sec + segment_end_sec) / 2
    start_sec = center_sec - PLOT_WINDOW_SEC / 2
    end_sec = center_sec + PLOT_WINDOW_SEC / 2

    if recording_duration_sec is None:
        if start_sec < 0:
            end_sec -= start_sec
            start_sec = 0.0
        return start_sec, end_sec

    if recording_duration_sec <= PLOT_WINDOW_SEC:
        return 0.0, recording_duration_sec

    if start_sec < 0:
        start_sec = 0.0
        end_sec = PLOT_WINDOW_SEC
    elif end_sec > recording_duration_sec:
        end_sec = recording_duration_sec
        start_sec = recording_duration_sec - PLOT_WINDOW_SEC

    return start_sec, end_sec


def add_event_context(candidate_df: pd.DataFrame, events_by_fileid: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, row in candidate_df.iterrows():
        row = row.to_dict()
        events = events_by_fileid.get(row["fileid"], pd.DataFrame())
        row["has_event_annotations"] = not events.empty

        plot_start_sec, plot_end_sec = centered_plot_bounds(
            float(row["segment_start_sec"]),
            float(row["segment_end_sec"]),
        )
        plot_events = overlapping_events(events, plot_start_sec, plot_end_sec)
        segment_events = overlapping_events(events, float(row["segment_start_sec"]), float(row["segment_end_sec"]))

        row["n_marked_events_in_plot"] = int(len(plot_events))
        row["n_marked_events_in_segment"] = int(len(segment_events))
        row["marked_event_types_in_plot"] = "|".join(sorted(plot_events["event_type"].unique())) if len(plot_events) else ""
        row["marked_event_types_in_segment"] = "|".join(sorted(segment_events["event_type"].unique())) if len(segment_events) else ""
        row["event_rank"] = (
            4 * int("obstructive_apnea" in row["marked_event_types_in_plot"])
            + 3 * int("central_apnea" in row["marked_event_types_in_plot"])
            + 2 * int("hypopnea" in row["marked_event_types_in_plot"])
            + int(row["n_marked_events_in_plot"] > 0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_local_bsi_and_plot_bounds(selected: pd.DataFrame, recordings: dict[str, RecordingSignals]) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        signals = recordings[row["fileid"]]
        plot_start_sec, plot_end_sec = centered_plot_bounds(
            float(row["segment_start_sec"]),
            float(row["segment_end_sec"]),
            len(signals.abd) / NEW_FS,
        )
        center = float(row["window_center_sec"])
        bsi_half = LOCAL_BSI_WINDOW_SEC / 2
        bsi_start_sec = max(0.0, center - bsi_half)
        bsi_end_sec = min(len(signals.abd) / NEW_FS, center + bsi_half)
        start_sec = bsi_start_sec
        end_sec = bsi_end_sec
        start = int(round(start_sec * NEW_FS))
        end = int(round(end_sec * NEW_FS))

        abd = robust_z(signals.abd[start:end])
        chest = robust_z(signals.chest[start:end])
        combined = robust_z((abd + chest) / 2)

        row = row.to_dict()
        row["plot_start_sec"] = plot_start_sec
        row["plot_end_sec"] = plot_end_sec
        row["plot_duration_sec"] = plot_end_sec - plot_start_sec
        row["bsi_window_start_sec"] = bsi_start_sec
        row["bsi_window_end_sec"] = bsi_end_sec
        row["bsi_abd_only_2min"] = compute_local_bsi(abd)
        row["bsi_chest_only_2min"] = compute_local_bsi(chest)
        row["bsi_current_mean_2min"] = compute_local_bsi(combined)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_examples(examples: pd.DataFrame, recordings: dict[str, RecordingSignals], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for old_panel in outdir.glob("paradox_example_*.png"):
        old_panel.unlink()
    for old_panel in outdir.glob("paradox_example_*.pdf"):
        old_panel.unlink()

    colors = {
        "flow": "#222222",
        "abd": "#1f77b4",
        "chest": "#d95f02",
        "combined": "#2ca02c",
    }
    bsi_cache: dict[str, np.ndarray] = {}

    fig = plt.figure(figsize=(13, 3.05 * len(examples)))
    outer_gs = fig.add_gridspec(nrows=len(examples), ncols=1, hspace=0.42)

    for idx, (_, row) in enumerate(examples.iterrows(), start=1):
        signals = recordings[row["fileid"]]
        plot_id = int(row["example_id"]) if "example_id" in row and pd.notna(row["example_id"]) else idx
        inner_gs = outer_gs[idx - 1, 0].subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[2.45, 1.0],
            hspace=0.04,
        )
        ax_signal = fig.add_subplot(inner_gs[0, 0])
        ax_bsi = fig.add_subplot(inner_gs[1, 0], sharex=ax_signal)
        _plot_one_example(
            row,
            signals,
            ax_signal,
            ax_bsi,
            colors,
            bsi_cache,
            plot_id,
            show_xlabel=idx == len(examples),
        )

        one_fig = plt.figure(figsize=(13, 3.8))
        one_gs = one_fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.45, 1.0], hspace=0.04)
        one_ax = one_fig.add_subplot(one_gs[0, 0])
        one_bsi_ax = one_fig.add_subplot(one_gs[1, 0], sharex=one_ax)
        _plot_one_example(
            row,
            signals,
            one_ax,
            one_bsi_ax,
            colors,
            bsi_cache,
            plot_id,
            show_xlabel=True,
        )
        one_fig.tight_layout()
        one_fig.savefig(outdir / f"paradox_example_{plot_id:02d}.png", dpi=180, bbox_inches="tight")
        one_fig.savefig(outdir / f"paradox_example_{plot_id:02d}.pdf", bbox_inches="tight")
        plt.close(one_fig)

    fig.tight_layout()
    fig.savefig(outdir / "paradox_breathing_examples_overview.png", dpi=180, bbox_inches="tight")
    fig.savefig(outdir / "paradox_breathing_examples_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def scale_for_plot(signal: np.ndarray) -> np.ndarray:
    scaled = robust_z(signal)
    denom = np.nanpercentile(np.abs(scaled), 98)
    if not np.isfinite(denom) or denom < 1e-9:
        denom = 1.0
    return scaled / denom


def draw_event_spans(
    ax: plt.Axes,
    events: pd.DataFrame,
    start_sec: float,
    end_sec: float,
    origin_sec: float,
) -> None:
    visible = overlapping_events(events, start_sec, end_sec)
    for _, event in visible.iterrows():
        event_start = max(float(event["start_sec"]), start_sec) - origin_sec
        event_end = min(float(event["end_sec"]), end_sec) - origin_sec
        ax.axvspan(
            event_start,
            event_end,
            color=EVENT_COLORS[event["event_type"]],
            alpha=0.30,
            lw=0,
            zorder=2,
        )


def get_mean_bsi_trace(signals: RecordingSignals, bsi_cache: dict[str, np.ndarray]) -> np.ndarray:
    if signals.fileid not in bsi_cache:
        mean_trace = robust_z((signals.abd + signals.chest) / 2)
        bsi_cache[signals.fileid] = compute_bsi_trace(mean_trace)
    return bsi_cache[signals.fileid]


def bsi_axis_ymax(bsi_segment: np.ndarray) -> float:
    finite = np.asarray(bsi_segment, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return BSI_MIN_YMAX
    p99 = float(np.nanpercentile(finite, 99))
    if not np.isfinite(p99):
        return BSI_MIN_YMAX
    return max(BSI_MIN_YMAX, p99)


def draw_bsi_gradient_background(ax: plt.Axes, duration_sec: float, ymax: float) -> None:
    gradient = np.linspace(0, ymax, 512)[:, None]
    ax.imshow(
        gradient,
        aspect="auto",
        origin="lower",
        extent=[0, duration_sec, 0, ymax],
        cmap=plt.get_cmap(BSI_GRADIENT_CMAP),
        norm=Normalize(vmin=0, vmax=BSI_GRADIENT_VMAX, clip=True),
        interpolation="bilinear",
        alpha=0.55,
        zorder=0,
    )


def _plot_one_example(
    row: pd.Series,
    signals: RecordingSignals,
    ax: plt.Axes,
    ax_bsi: plt.Axes,
    colors: dict[str, str],
    bsi_cache: dict[str, np.ndarray],
    idx: int,
    show_xlabel: bool = True,
) -> None:
    start = int(round(float(row["plot_start_sec"]) * NEW_FS))
    end = int(round(float(row["plot_end_sec"]) * NEW_FS))
    t = np.arange(end - start) / NEW_FS
    plot_start_sec = float(row["plot_start_sec"])
    plot_end_sec = float(row["plot_end_sec"])
    segment_start = float(row["segment_start_sec"]) - plot_start_sec
    segment_end = float(row["segment_end_sec"]) - plot_start_sec

    abd = scale_for_plot(signals.abd_resp[start:end])
    chest = scale_for_plot(signals.chest_resp[start:end])
    combined = scale_for_plot((signals.abd_resp[start:end] + signals.chest_resp[start:end]) / 2)
    offsets = {"flow": 4.5, "abd": 1.5, "chest": -1.5, "combined": -4.5}

    if signals.flow is not None:
        flow_source = signals.flow_resp if signals.flow_resp is not None else signals.flow
        flow = scale_for_plot(flow_source[start:end])
        ax.plot(t, flow + offsets["flow"], lw=0.8, color=colors["flow"], zorder=3)

    ax.axvspan(segment_start, segment_end, color="#f2c94c", alpha=0.18, lw=0, zorder=1)
    ax.plot(t, abd + offsets["abd"], lw=0.9, color=colors["abd"], zorder=3)
    ax.plot(t, chest + offsets["chest"], lw=0.9, color=colors["chest"], zorder=3)
    ax.plot(t, combined + offsets["combined"], lw=0.9, color=colors["combined"], zorder=3)

    ax.set_yticks([offsets["flow"], offsets["abd"], offsets["chest"], offsets["combined"]])
    ax.set_yticklabels([f"Flow ({row['flow_channel']})", "Abd", "Chest", "Mean"])
    ax.set_xlim(0, plot_end_sec - plot_start_sec)
    ax.set_ylim(-6.3, 5.8)
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.tick_params(labelbottom=False)
    title = (
        f"{idx}. {row['fileid']} | "
        f"corr={row['belt_corr']:.2f}, anti-phase={row['anti_phase_fraction']:.2f}, "
        f"flow pct={row['flow_amp_percentile']:.0f} | "
        f"BSI Abd={row['bsi_abd_only_2min']:.2f}, "
        f"Chest={row['bsi_chest_only_2min']:.2f}, "
        f"Mean={row['bsi_current_mean_2min']:.2f}"
    )
    ax.set_title(title, loc="left", fontsize=9)
    if idx == 1:
        legend_handles = [
            Patch(facecolor="#f2c94c", alpha=0.18, label="Detected anti-phase span"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    bsi_trace = get_mean_bsi_trace(signals, bsi_cache)
    bsi_segment = bsi_trace[start:end]
    bsi_ymax = bsi_axis_ymax(bsi_segment)
    draw_bsi_gradient_background(ax_bsi, plot_end_sec - plot_start_sec, bsi_ymax)
    ax_bsi.axvspan(segment_start, segment_end, color="#f2c94c", alpha=0.18, lw=0, zorder=1)
    ax_bsi.plot(t, bsi_segment, color="black", lw=1.8, zorder=4)
    ax_bsi.set_ylim(0, bsi_ymax)
    ax_bsi.set_ylabel("BSI", fontsize=8)
    ax_bsi.set_xlabel("Time (s)" if show_xlabel else "")
    ax_bsi.grid(axis="both", color="#ffffff", lw=0.5, alpha=0.55)
    for spine in ["top", "right"]:
        ax_bsi.spines[spine].set_visible(False)


def build_summary(examples: pd.DataFrame, outdir: Path) -> None:
    display_cols = [
        "example_id",
        "fileid",
        "paper_psg_type",
        "ahi",
        "oai",
        "cai",
        "segment_start_sec",
        "segment_end_sec",
        "plot_duration_sec",
        "belt_corr",
        "anti_phase_fraction",
        "flow_amp_percentile",
        "bsi_abd_only_2min",
        "bsi_chest_only_2min",
        "bsi_current_mean_2min",
        "n_marked_events_in_plot",
        "marked_event_types_in_plot",
    ]
    rounded = examples[display_cols].copy()
    for col in rounded.select_dtypes(include=[float]).columns:
        rounded[col] = rounded[col].round(3)

    lines = [
        "# Paradoxical Breathing Examples",
        "",
        "Selection criteria: 30 s windows with post-alignment ABD/CHEST correlation < -0.45, anti-phase fraction > 0.45, effort amplitude above the 25th percentile, and airflow amplitude below the 70th percentile when airflow was available.",
        "",
        f"The pale yellow shading marks the algorithm-selected anti-phase/paradox-breathing span. Each panel shows a {PLOT_WINDOW_SEC}-second window centered on that span, except at recording edges. Marked apnea/hypopnea annotations are summarized in the table but are not overlaid on the figure. The BSI columns are local 2-minute values centered on the selected window. `bsi_current_mean_2min` uses the mean of polarity-aligned, robust-normalized ABD and chest belts. The figure plots respiratory-band effort traces to make the phase relationship visible and includes the continuous mean-belt BSI trace below each panel. BSI subplot limits use a minimum upper limit of {BSI_MIN_YMAX:g} and otherwise scale to the 99th percentile of the displayed BSI trace. The BSI panel background uses the full-night BSI `{BSI_GRADIENT_CMAP}` gradient, with 0 mapped to blue and {BSI_GRADIENT_VMAX:g} or higher mapped to red; the BSI trace is shown as a black line.",
        "",
        f"Selected examples come from {examples['fileid'].nunique()} recordings.",
        "",
        dataframe_to_markdown(rounded),
        "",
        "Outputs:",
        "- `candidate_segments.csv`: all grouped candidates from scanned local EDFs.",
        f"- `examples.csv`: selected {len(examples)} examples.",
        f"- `paradox_breathing_examples_overview.png/pdf`: {len(examples)}-panel figure.",
        "- `paradox_example_XX.png/pdf`: individual panels.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines))


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def discover_local_edfs(edf_dirs: list[Path]) -> list[Path]:
    paths = []
    for edf_dir in edf_dirs:
        if edf_dir.exists():
            paths.extend(sorted(edf_dir.glob("*.edf")))
    return paths


def parse_example_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--include-nondiagnostic", action="store_true")
    parser.add_argument("--n-examples", type=int, default=15)
    parser.add_argument("--max-per-recording", type=int, default=3)
    parser.add_argument("--min-recordings", type=int, default=5)
    parser.add_argument(
        "--overview-example-ids",
        type=parse_example_ids,
        default=list(DEFAULT_OVERVIEW_EXAMPLE_IDS),
        help="comma-separated original example IDs to keep in examples.csv and the overview figure",
    )
    parser.add_argument(
        "--exclude-fileid",
        action="append",
        default=list(DEFAULT_EXCLUDED_SELECTION_FILEIDS),
        help="fileid to exclude from selected examples; can be passed multiple times",
    )
    parser.add_argument(
        "--allow-unannotated-selection",
        action="store_true",
        help="allow examples from recordings without local event annotation CSVs",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.outdir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master, low_memory=False)
    master = master.drop_duplicates("fileid")
    metadata_cols = ["fileid", "cohort", "sid", "age", "sex", "ahi", "oai", "cai", "hyi", "paper_psg_type"]
    metadata = master[[c for c in metadata_cols if c in master.columns]].copy()

    candidate_paths = discover_local_edfs(DEFAULT_EDF_DIRS)
    candidate_paths = [
        path
        for path in candidate_paths
        if fileid_from_path(path) in set(metadata["fileid"].astype(str))
    ]

    recordings: dict[str, RecordingSignals] = {}
    all_candidates = []
    for path in candidate_paths:
        signals = load_recording(path)
        if signals is None:
            continue
        recordings[signals.fileid] = signals
        candidates = scan_recording(signals)
        all_candidates.extend(candidates)
        print(f"{signals.fileid}: {len(candidates)} candidate segments")

    candidate_df = pd.DataFrame(all_candidates)
    if candidate_df.empty:
        raise SystemExit("No candidate paradoxical breathing segments found.")

    candidate_df = candidate_df.merge(metadata, on="fileid", how="left")
    events_by_fileid = {fileid: load_marked_events(signals) for fileid, signals in recordings.items()}
    candidate_df = add_event_context(candidate_df, events_by_fileid)
    candidate_df.sort_values("candidate_score", ascending=False).to_csv(
        args.outdir / "candidate_segments.csv", index=False
    )

    required_recordings = min(args.min_recordings, args.n_examples)

    selection_pool = candidate_df.copy()
    if args.exclude_fileid:
        selection_pool = selection_pool[~selection_pool["fileid"].isin(args.exclude_fileid)].copy()

    if not args.include_nondiagnostic and "paper_psg_type" in selection_pool.columns:
        diagnostic_pool = selection_pool[selection_pool["paper_psg_type"] == "diagnostic"].copy()
        if len(diagnostic_pool) >= args.n_examples and diagnostic_pool["fileid"].nunique() >= required_recordings:
            selection_pool = diagnostic_pool

    if not args.allow_unannotated_selection:
        annotated_pool = selection_pool[selection_pool["has_event_annotations"]].copy()
        if len(annotated_pool) >= args.n_examples and annotated_pool["fileid"].nunique() >= required_recordings:
            selection_pool = annotated_pool

    selected_rows = []
    counts: dict[str, int] = {}
    ranked_pool = selection_pool.sort_values(
        ["event_rank", "candidate_score"],
        ascending=[False, False],
    )

    for fileid in ranked_pool["fileid"].drop_duplicates():
        if len(selected_rows) >= min(args.min_recordings, args.n_examples):
            break
        row = ranked_pool[ranked_pool["fileid"] == fileid].iloc[0]
        selected_rows.append(row)
        counts[fileid] = counts.get(fileid, 0) + 1

    selected_keys = {
        (row["fileid"], float(row["segment_start_sec"]), float(row["segment_end_sec"]))
        for row in selected_rows
    }
    for _, row in ranked_pool.iterrows():
        key = (row["fileid"], float(row["segment_start_sec"]), float(row["segment_end_sec"]))
        if key in selected_keys:
            continue
        fileid = row["fileid"]
        if counts.get(fileid, 0) >= args.max_per_recording:
            continue
        selected_rows.append(row)
        selected_keys.add(key)
        counts[fileid] = counts.get(fileid, 0) + 1
        if len(selected_rows) == args.n_examples:
            break

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    n_selected_recordings = selected["fileid"].nunique()
    if n_selected_recordings < required_recordings:
        raise SystemExit(
            f"Only found {n_selected_recordings} recordings for selection; "
            f"{required_recordings} required. Consider --allow-unannotated-selection "
            "or adding more local EDFs."
        )
    selected.insert(0, "example_id", np.arange(1, len(selected) + 1))
    selected = add_local_bsi_and_plot_bounds(selected, recordings)
    selected.to_csv(args.outdir / "examples_all.csv", index=False)

    if args.overview_example_ids:
        requested_ids = set(args.overview_example_ids)
        selected = selected[selected["example_id"].isin(requested_ids)].copy()
        missing_ids = sorted(requested_ids - set(selected["example_id"]))
        if missing_ids:
            raise SystemExit(f"Requested overview example IDs were not selected: {missing_ids}")

    selected.to_csv(args.outdir / "examples.csv", index=False)

    plot_examples(selected, recordings, args.outdir)
    build_summary(selected, args.outdir)
    print(f"Wrote {len(selected)} examples to {args.outdir}")


if __name__ == "__main__":
    main()
