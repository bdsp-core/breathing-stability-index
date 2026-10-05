"""Side-effect-free array API using the revision's numerical kernels."""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np

from . import _kernels as kernels


@dataclass(frozen=True)
class Config:
    """Published primary settings; alternate window lengths are sensitivities."""

    window_minutes: float = 2.0
    overlap: float = 0.9
    processing_fs_hz: int = 10
    lowpass_hz: float = 10.0
    belt_mode: str = "robust_mean"

    def __post_init__(self):
        if not np.isfinite(self.window_minutes) or self.window_minutes <= 0:
            raise ValueError("window_minutes must be positive and finite")
        if not np.isfinite(self.overlap) or not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if self.processing_fs_hz != 10:
            raise ValueError("The published BSI scale requires processing_fs_hz=10")
        if not np.isfinite(self.lowpass_hz) or self.lowpass_hz <= 0:
            raise ValueError("lowpass_hz must be positive and finite")
        if self.belt_mode not in ("robust_mean", "raw_mean"):
            raise ValueError("belt_mode must be robust_mean or raw_mean")


@dataclass
class BSIResult:
    file_id: str
    fs_hz: int
    time_seconds: np.ndarray
    bsi: np.ndarray
    effort: np.ndarray
    upper_envelope: np.ndarray
    lower_envelope: np.ndarray
    stages: np.ndarray | None
    summary: dict[str, float]
    paper_features: dict[str, float]
    config: Config
    channels_used: tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    def summary_row(self) -> dict:
        """CSV-ready row, including historical summary names for paper scripts."""
        return {
            "file": self.file_id,
            "window_length": self.config.window_minutes,
            "overlap": self.config.overlap,
            "belt_mode": self.config.belt_mode,
            "belt_channels_used": "|".join(self.channels_used),
            "n_belts_used": len(self.channels_used),
            "processing_fs_hz": self.fs_hz,
            **self.summary,
            **self.paper_features,
        }


def _encode_stages(values, encoding):
    if encoding not in ("aasm", "legacy"):
        raise ValueError("stage_encoding must be aasm or legacy")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("sleep_stages must be a nonempty 1-D array")
    if raw.dtype.kind in "OUS":
        labels = {"W": 0, "WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4,
                  "R": 4, "UNKNOWN": -1, "UNSCORED": -1}
        try:
            return np.array([labels[str(v).strip().upper()] for v in raw], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Unknown sleep stage label: {exc.args[0]}") from exc
    numeric = np.asarray(raw, dtype=float)
    mapping = ({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, -1: -1} if encoding == "aasm"
               else {1: 3, 2: 2, 3: 1, 4: 4, 5: 0, 0: -1, -1: -1})
    out = []
    for value in numeric:
        if not np.isfinite(value):
            out.append(-1)
        elif value not in mapping:
            raise ValueError(f"Invalid {encoding} stage code: {value}")
        else:
            out.append(mapping[value])
    return np.asarray(out, dtype=int)


def _align_stages(values, *, encoding, epoch_seconds, n_input, fs_hz, n_output):
    encoded = _encode_stages(values, encoding)
    times = np.arange(n_output) / 10.0
    if epoch_seconds is None:
        if len(encoded) != n_input:
            raise ValueError("Sample-aligned sleep_stages must match the input signal length")
        idx = np.floor(times * fs_hz + 1e-8).astype(int)
        # FFT resampling can round the output length; clamp only the final edge.
        idx = np.minimum(idx, n_input - 1)
    else:
        if not np.isfinite(epoch_seconds) or epoch_seconds <= 0:
            raise ValueError("stage_epoch_seconds must be positive, or None for sample-aligned stages")
        duration = n_input / fs_hz
        if not (len(encoded) - 1) * epoch_seconds < duration <= len(encoded) * epoch_seconds + 1 / fs_hz:
            raise ValueError("Sleep stage epochs must cover the recording with at most one partial final epoch")
        idx = np.floor(times / epoch_seconds).astype(int)
        idx = np.minimum(idx, len(encoded) - 1)
    return encoded[idx]


def _summary(bsi, stages, cfg):
    masks = {"ALL": np.ones(len(bsi), dtype=bool)}
    if stages is not None:
        masks.update({"SLEEP": np.isin(stages, [1, 2, 3, 4]), "N1": stages == 1,
                      "N2": stages == 2, "N3": stages == 3, "NREM": np.isin(stages, [1, 2, 3]),
                      "REM": stages == 4, "WAKE": stages == 0})
    summary = {}
    for name, mask in masks.items():
        values = bsi[mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            metrics = kernels.stability_summary_features(values, cfg.processing_fs_hz)
        # Preserve the historical (misnamed) n_5min fields: the revision passed
        # the BSI window duration, which is 2 minutes for the primary pipeline.
        metrics.update(kernels.stability_episodes_features(values, cfg.processing_fs_hz,
                                                          window_length_min=cfg.window_minutes))
        metrics["stable_time_fraction"] = float(np.mean(values < 0.5)) if values.size else np.nan
        metrics["unstable_time_fraction"] = float(np.mean(values > 1.5)) if values.size else np.nan
        metrics["duration_minutes"] = float(values.size / cfg.processing_fs_hz / 60)
        summary.update({f"{name}_{key}": float(value) for key, value in metrics.items()})
    features = {}
    if cfg.belt_mode == "robust_mean" and cfg.window_minutes == 2 and cfg.overlap == 0.9:
        for name in masks:
            suffix = name.lower()
            features[f"bsi_median_{suffix}"] = summary[f"{name}_median_stability"]
            features[f"bsi_mean_{suffix}"] = summary[f"{name}_stability_auc"] / cfg.processing_fs_hz
            for q in [25, 75, 90]:
                features[f"bsi_quantile_{q}_{suffix}"] = summary[f"{name}_quantile_{q}"]
        if "SLEEP" in masks:
            stable = summary["SLEEP_n_5min_windows_stable"]
            unstable = summary["SLEEP_n_5min_windows_unstable"]
            features["bsi_instability_burden_gt_1p5_sleep"] = unstable / (stable + unstable) if stable + unstable > 0 else np.nan
    return summary, features


def compute_bsi(abdomen=None, thorax=None, *, fs_hz: float, sleep_stages=None,
                stage_epoch_seconds: float | None = 30.0, stage_encoding: str = "aasm",
                file_id: str = "recording", cfg: Config | None = None) -> BSIResult:
    """Compute BSI without writing files.

    Supply one or both 1-D effort belts in consistent arbitrary units. Stages
    default to 30-s epochs; numeric AASM codes are W=0,N1=1,N2=2,N3=3,REM=4.
    Use stage_epoch_seconds=None for one stage per input sample. Without stages,
    only whole-recording summaries are returned, never a sleep-only exposure.
    """
    cfg = cfg or Config()
    if not np.isfinite(fs_hz) or fs_hz <= 2 * cfg.lowpass_hz:
        raise ValueError("fs_hz must exceed twice lowpass_hz (20 Hz with the published filter)")
    traces, labels = [], []
    for label, values in (("abd", abdomen), ("chest", thorax)):
        if values is None:
            continue
        trace = np.asarray(values, dtype=float)
        if trace.ndim != 1 or trace.size == 0 or not np.all(np.isfinite(trace)):
            raise ValueError(f"{label} must be a nonempty, finite 1-D signal")
        if np.ptp(trace) == 0:
            raise ValueError(f"{label} has no amplitude variation; supply only usable belts")
        traces.append(trace)
        labels.append(label)
    if not traces:
        raise ValueError("Supply abdomen and/or thorax effort")
    if any(len(t) != len(traces[0]) for t in traces):
        raise ValueError("Effort belts must have the same length and sampling rate")
    if len(traces[0]) / fs_hz < cfg.window_minutes * 60:
        raise ValueError("Recording is shorter than one BSI window")
    processed = kernels.preprocess_effort_signals(np.vstack(traces), fs_hz, notch_freq=None,
                                                 bandpass_freq=[0, cfg.lowpass_hz],
                                                 new_fs=cfg.processing_fs_hz)
    for trace in processed:
        if np.subtract(*np.percentile(np.clip(trace, *np.percentile(trace, [2, 98])), [75, 25])) <= 0:
            raise ValueError("An effort belt has zero robust scale after preprocessing")
    with np.errstate(divide="ignore", invalid="ignore"):
        effort, _ = kernels.build_belt_mode_traces(processed, cfg.belt_mode)
    if not np.all(np.isfinite(effort)):
        raise ValueError("Combined belts have zero robust scale; inspect belt polarity/cancellation")
    stages = None if sleep_stages is None else _align_stages(
        sleep_stages, encoding=stage_encoding, epoch_seconds=stage_epoch_seconds,
        n_input=len(traces[0]), fs_hz=fs_hz, n_output=len(effort))
    times = np.arange(len(effort)) / cfg.processing_fs_hz
    try:
        _, effort = kernels.remove_outliers(times, effort, cfg.processing_fs_hz, min_height=0.01)
        up, lo = kernels.create_env(effort, cfg.processing_fs_hz)
    except (IndexError, ValueError) as exc:
        raise ValueError("Could not extract respiratory envelopes from this signal") from exc
    if up is None or lo is None or not np.all(np.isfinite(up)) or not np.all(np.isfinite(lo)):
        raise ValueError("Could not extract finite respiratory envelopes; inspect effort signal quality")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        bsi = kernels.compute_stability(up, lo, cfg.processing_fs_hz, cfg.window_minutes, cfg.overlap)
    if not np.all(np.isfinite(bsi)):
        raise ValueError("BSI computation produced nonfinite values")
    summary, features = _summary(bsi, stages, cfg)
    return BSIResult(str(file_id), cfg.processing_fs_hz, times, bsi, effort, up, lo,
                     stages, summary, features, cfg, tuple(labels))
