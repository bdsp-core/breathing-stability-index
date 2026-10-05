"""Reproducible file/manifest CLI; failed recordings produce a nonzero exit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from .api import Config
from .io import compute_bsi_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute the published respiratory effort Breathing Stability Index")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="One EDF/H5 file")
    inputs.add_argument("--manifest-csv", type=Path, help="CSV with filepath and optional file_id")
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("--window-minutes", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.9)
    parser.add_argument("--belt-mode", choices=["robust_mean", "raw_mean"], default="robust_mean")
    parser.add_argument("--abdomen-channel")
    parser.add_argument("--thorax-channel")
    parser.add_argument("--h5-stage-encoding", choices=["legacy", "aasm"], default="legacy")
    parser.add_argument("--save-timeseries", action="store_true", help="Save BSI/envelopes/effort to compressed NPZ")
    parser.add_argument("--plot", action="store_true", help="Save an overview PNG; requires the plot extra")
    return parser.parse_args(argv)


def _records(args):
    if args.input is not None:
        records = [(args.input, args.input.stem)]
    else:
        manifest = pd.read_csv(args.manifest_csv, dtype=str, keep_default_na=False)
        if "filepath" not in manifest or len(manifest) == 0:
            raise ValueError("Manifest must contain filepath and at least one recording")
        records = []
        for _, row in manifest.iterrows():
            if not row["filepath"].strip():
                raise ValueError("Manifest contains an empty filepath")
            path = Path(row["filepath"])
            if not path.is_absolute():
                path = args.manifest_csv.resolve().parent / path
            records.append((path, row.get("file_id", "") or path.stem))
    ids = [file_id for _, file_id in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Recording file_id values must be unique to avoid overwriting outputs")
    for file_id in ids:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", file_id):
            raise ValueError("file_id must start with a letter/digit and contain only letters, digits, '.', '_' or '-'")
    return records


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 5))
    minutes = result.time_seconds / 60
    step = max(1, len(minutes) // 15000)
    axes[0].plot(minutes[::step], result.effort[::step], color="0.65", linewidth=0.4)
    axes[0].plot(minutes[::step], result.upper_envelope[::step], color="#355c7d", linewidth=0.8)
    axes[0].plot(minutes[::step], result.lower_envelope[::step], color="#355c7d", linewidth=0.8)
    axes[0].set_ylabel("Normalized effort")
    axes[0].set_title(result.file_id)
    axes[1].plot(minutes, result.bsi, color="#a44949", linewidth=0.8)
    axes[1].axhline(0.5, color="0.5", linestyle=":", linewidth=0.7)
    axes[1].axhline(1.5, color="0.5", linestyle=":", linewidth=0.7)
    axes[1].set_ylabel("BSI (higher = less stable)")
    axes[1].set_xlabel("Time (minutes)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    try:
        cfg = Config(window_minutes=args.window_minutes, overlap=args.overlap, belt_mode=args.belt_mode)
        records = _records(args)
        if args.plot:
            import matplotlib  # Check the requested extra before processing.
        args.outdir.mkdir(parents=True, exist_ok=True)
        # One invocation owns its output directory: reject existing summaries.
        if (args.outdir / "summary_stability.csv").exists() or (args.outdir / "errors.csv").exists():
            raise ValueError("Output directory already contains a run; choose a new --outdir")
        rows, errors = [], []
        for path, file_id in records:
            try:
                result = compute_bsi_file(path, file_id=file_id, cfg=cfg,
                                          abdomen_channel=args.abdomen_channel, thorax_channel=args.thorax_channel,
                                          h5_stage_encoding=args.h5_stage_encoding)
                row = {"filepath": str(path), **result.summary_row()}
                if args.save_timeseries:
                    (args.outdir / "timeseries").mkdir(exist_ok=True)
                    arrays = {key: getattr(result, key) for key in
                              ["time_seconds", "bsi", "effort", "upper_envelope", "lower_envelope"]}
                    if result.stages is not None:
                        arrays["stages"] = result.stages
                    np.savez_compressed(args.outdir / "timeseries" / f"{file_id}.npz", **arrays)
                if args.plot:
                    (args.outdir / "figures").mkdir(exist_ok=True)
                    _plot(result, args.outdir / "figures" / f"{file_id}.png")
                (args.outdir / "json").mkdir(exist_ok=True)
                payload = {"summary": row, "config": asdict(cfg), "metadata": result.metadata}
                (args.outdir / "json" / f"{file_id}.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
                rows.append(row)
            except Exception as exc:
                errors.append({"filepath": str(path), "file_id": file_id,
                               "error": f"{type(exc).__name__}: {exc}"})
                print(f"Failed {file_id}: {exc}", file=sys.stderr)
        if rows:
            pd.DataFrame(rows).to_csv(args.outdir / "summary_stability.csv", index=False)
        if errors:
            pd.DataFrame(errors).to_csv(args.outdir / "errors.csv", index=False)
        print(f"Completed {len(rows)}/{len(records)} recordings. Output: {args.outdir}")
        return 1 if errors else 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
