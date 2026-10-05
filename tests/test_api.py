from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from breathing_stability_index import Config, compute_bsi, compute_bsi_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from synthetic import synthetic_effort, write_h5


@pytest.fixture(scope="module")
def legacy():
    spec = importlib.util.spec_from_file_location("_archived_bsi", ROOT / "paper/scripts/breathing_stability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recording():
    return synthetic_effort()


def test_array_numerical_and_feature_parity_with_revision(legacy, recording):
    abdomen, thorax, stage = recording
    result = compute_bsi(abdomen, thorax, fs_hz=100, sleep_stages=stage,
                         stage_epoch_seconds=None, stage_encoding="legacy")
    signals = legacy.preprocess_effort_signals(np.vstack([abdomen, thorax]), 100)
    trace, _ = legacy.build_belt_mode_traces(signals, "robust_mean")
    _, trace = legacy.remove_outliers(np.arange(len(trace)) / 10, trace, 10, min_height=0.01)
    up, lo = legacy.create_env(trace, 10)
    reference = legacy.compute_stability(up, lo, 10, window_length_min=2, overlap=0.9)
    np.testing.assert_allclose(result.bsi, reference, rtol=0, atol=0)
    reference_summary, reference_episodes = legacy.summarize_stability_by_stage(reference, 10, stage[::10], 2)
    for key, value in {**reference_summary, **reference_episodes}.items():
        np.testing.assert_allclose(result.summary[key], value, rtol=0, atol=0, equal_nan=True)
    assert result.paper_features["bsi_median_sleep"] == reference_summary["SLEEP_median_stability"]
    assert result.summary["SLEEP_duration_minutes"] == 10
    assert result.summary["ALL_duration_minutes"] == 12


def test_stage_encodings_agree_and_no_stage_does_not_claim_sleep(recording):
    abdomen, _, legacy_stage = recording
    epochs = legacy_stage[::3000]
    labels = np.array([{5: "W", 3: "N1", 2: "N2", 1: "N3", 4: "REM"}[s] for s in epochs])
    result = compute_bsi(abdomen, fs_hz=100, sleep_stages=labels)
    assert np.mean(result.stages == 3) == pytest.approx(np.mean(legacy_stage == 1))
    no_stages = compute_bsi(abdomen, fs_hz=100)
    assert "bsi_median_sleep" not in no_stages.paper_features
    np.testing.assert_allclose(no_stages.bsi, result.bsi)


def test_hdf5_adapter_reads_stages_correctly(tmp_path, recording):
    path = tmp_path / "synthetic.h5"
    write_h5(path)
    result = compute_bsi_file(path)
    abdomen, thorax, stages = recording
    direct = compute_bsi(abdomen, thorax, fs_hz=100, sleep_stages=stages,
                         stage_epoch_seconds=None, stage_encoding="legacy")
    np.testing.assert_allclose(result.bsi, direct.bsi, rtol=0, atol=0)
    assert result.paper_features == direct.paper_features


def test_edf_adapter(tmp_path, recording):
    import mne
    abdomen, thorax, _ = recording
    info = mne.create_info(["Abdomen", "Chest"], 100, ch_types=["misc", "misc"])
    raw = mne.io.RawArray(np.vstack([abdomen, thorax]), info, verbose=False)
    path = tmp_path / "synthetic.edf"
    mne.export.export_raw(path, raw, fmt="edf", physical_range=(-5, 5), verbose=False)
    result = compute_bsi_file(path)
    # Compare against the actual quantized EDF samples, rather than the original.
    with mne.io.read_raw_edf(path, verbose=False) as reread:
        samples = reread.get_data()
    direct = compute_bsi(samples[0], samples[1], fs_hz=100)
    np.testing.assert_allclose(result.bsi, direct.bsi, rtol=0, atol=0)
    assert result.stages is None


def test_bad_data_and_unsupported_stage_coding(recording):
    abdomen, _, _ = recording
    with pytest.raises(ValueError, match="no amplitude"):
        compute_bsi(np.ones(12000), fs_hz=100)
    with pytest.raises(ValueError, match="finite"):
        compute_bsi(np.full(12000, np.nan), fs_hz=100)
    with pytest.raises(ValueError, match="shorter"):
        compute_bsi(abdomen[:100], fs_hz=100)
    with pytest.raises(ValueError, match="stage code"):
        compute_bsi(abdomen, fs_hz=100, sleep_stages=np.full(24, 5))


def test_published_defaults_and_invalid_overlap():
    cfg = Config()
    assert (cfg.window_minutes, cfg.overlap, cfg.belt_mode, cfg.processing_fs_hz) == (2, 0.9, "robust_mean", 10)
    with pytest.raises(ValueError, match="overlap"):
        Config(overlap=1)
