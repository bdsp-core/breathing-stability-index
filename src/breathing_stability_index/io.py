"""File adapters. Read only effort channels and explicitly supplied stages."""
from __future__ import annotations

from pathlib import Path

import h5py
import mne
import numpy as np

from .api import Config, compute_bsi


def _role(name):
    label = str(name).lower()
    if "abd" in label:
        return "abd"
    if any(token in label for token in ("chest", "thorax", "thoracic", "effort tho")):
        return "chest"
    if "effort" in label:
        return "effort"
    return None


def _channels(names, abdomen_channel, thorax_channel):
    selected = {}
    for role, explicit in (("abd", abdomen_channel), ("chest", thorax_channel)):
        if explicit:
            if explicit not in names:
                raise ValueError(f"Requested channel not found: {explicit}")
            selected[role] = explicit
        else:
            matches = [name for name in names if _role(name) == role]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous {role} channels: {matches}; select a channel explicitly")
            if matches:
                selected[role] = matches[0]
    if not selected:
        generic = [name for name in names if _role(name) == "effort"]
        if len(generic) == 1:
            selected["abd"] = generic[0]
        else:
            raise ValueError("No unambiguous respiratory effort belt found; select channels explicitly")
    if len(set(selected.values())) != len(selected):
        raise ValueError("Abdomen and thorax must refer to different channels")
    return selected


def _signal(dataset):
    values = np.asarray(dataset, dtype=float)
    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    if values.ndim != 1:
        raise ValueError("HDF5 effort and stage datasets must be 1-D or (N,1)")
    return values


def compute_bsi_file(filepath, *, file_id=None, cfg: Config | None = None,
                     abdomen_channel=None, thorax_channel=None, sleep_stages=None,
                     stage_epoch_seconds=30.0, stage_encoding="aasm",
                     h5_stage_encoding="legacy"):
    """Read EDF or prepared HDF5; HDF5 stages use the original BDSP coding.

    HDF5 requires root sampling_rate and effort datasets in signals/. Optional
    annotations/stage must have one label per input sample. Explicit sleep_stages
    override embedded stages, using stage_epoch_seconds and stage_encoding.
    """
    path = Path(filepath)
    selected = {}
    embedded_stages = False
    if path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path, "r") as handle:
            if "signals" not in handle or "sampling_rate" not in handle.attrs:
                raise ValueError("HDF5 requires signals/ and root sampling_rate")
            selected = _channels(list(handle["signals"]), abdomen_channel, thorax_channel)
            signals = {role: _signal(handle["signals"][name]) for role, name in selected.items()}
            fs = float(handle.attrs["sampling_rate"])
            if sleep_stages is None and "annotations/stage" in handle:
                sleep_stages = _signal(handle["annotations/stage"])
                stage_epoch_seconds = None
                stage_encoding = h5_stage_encoding
                embedded_stages = True
    elif path.suffix.lower() == ".edf":
        with mne.io.read_raw_edf(path, preload=False, stim_channel=None, verbose="ERROR") as raw:
            selected = _channels(raw.ch_names, abdomen_channel, thorax_channel)
            signals = {role: raw.get_data(picks=[name])[0] for role, name in selected.items()}
            fs = float(raw.info["sfreq"])
    else:
        raise ValueError("Supported file extensions are .edf, .h5, and .hdf5")
    result = compute_bsi(signals.get("abd"), signals.get("chest"), fs_hz=fs,
                         sleep_stages=sleep_stages, stage_epoch_seconds=stage_epoch_seconds,
                         stage_encoding=stage_encoding, file_id=file_id or path.stem, cfg=cfg)
    result.metadata.update({"filepath": str(path), "input_fs_hz": fs,
                            "input_channels": selected, "embedded_stages": embedded_stages,
                            "stage_encoding": stage_encoding if sleep_stages is not None else None})
    return result
