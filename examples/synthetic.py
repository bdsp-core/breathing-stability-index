"""Deterministic synthetic effort signals; no participant data."""
from __future__ import annotations
import argparse
from pathlib import Path
import h5py
import numpy as np


def synthetic_effort(duration_seconds=720, fs_hz=100, seed=42):
    rng = np.random.default_rng(seed)
    time = np.arange(int(duration_seconds * fs_hz)) / fs_hz
    amplitude = np.where(time < duration_seconds / 2, 1.0, 1.0 + 0.8 * np.sin(2 * np.pi * time / 65))
    abdomen = amplitude * np.sin(2 * np.pi * 0.25 * time) + 0.02 * rng.standard_normal(time.size)
    thorax = 2.5 * amplitude * np.sin(2 * np.pi * 0.25 * time + 0.12) + 0.02 * rng.standard_normal(time.size)
    # Legacy BDSP encoding, deliberately including each sleep stage and wake.
    epochs = np.resize(np.array([5, 3, 2, 1, 4, 2]), int(np.ceil(duration_seconds / 30)))
    stages = np.repeat(epochs, int(30 * fs_hz))[:time.size]
    return abdomen, thorax, stages


def write_h5(path, duration_seconds=720, fs_hz=100):
    abdomen, thorax, stages = synthetic_effort(duration_seconds, fs_hz)
    with h5py.File(path, "w") as handle:
        handle.attrs["sampling_rate"] = fs_hz
        handle.attrs["unit_voltage"] = "arbitrary"
        handle.create_dataset("signals/abd", data=abdomen[:, None])
        handle.create_dataset("signals/chest", data=thorax[:, None])
        handle.create_dataset("annotations/stage", data=stages[:, None])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/synthetic.h5"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_h5(args.output)
    print(args.output)
