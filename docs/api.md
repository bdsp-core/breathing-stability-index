# Inputs and outputs

## Arrays

`compute_bsi(abdomen=None, thorax=None, *, fs_hz, sleep_stages=None,
stage_epoch_seconds=30.0, stage_encoding="aasm", file_id="recording", cfg=None)`
accepts finite one-dimensional effort arrays. Both belts must have the same
length and sampling rate; either belt can be omitted. Arbitrary units are
acceptable because the primary pipeline normalizes each belt separately.

Sleep stages must cover the recording, allowing one partial final epoch.
For one stage per input sample, pass `stage_epoch_seconds=None`. Sleep stages
are optional; no sleep-specific fields are returned without them.

`Config()` defaults: `window_minutes=2`, `overlap=0.9`,
`processing_fs_hz=10`, `lowpass_hz=10`, `belt_mode="robust_mean"`.
`raw_mean` and alternate window/overlap settings are available as explicitly
recorded sensitivities. For an input rate at or below 20 Hz, a lower low-pass
cutoff can be selected explicitly; this changes the primary preprocessing.

## Files

`compute_bsi_file(filepath, ...)` supports:

| Format | Required fields | Stages |
| --- | --- | --- |
| Prepared HDF5 | Root attribute `sampling_rate`; one or two effort datasets in `signals/` | Optional `annotations/stage`, one label per input sample, legacy coding by default |
| EDF | Recognizable abdomen/thorax effort channels, or explicit channel names | Supply stages through the Python API; EDF annotations are not interpreted as a hypnogram |

HDF5 datasets can be `(N,)`, `(N,1)`, or `(1,N)`. Other signals are not loaded.
Effort-channel labels containing `abd`, `chest`, `thorax`, `thoracic`, or
`Effort THO` are recognized. One generic effort channel can be used if no named
belts are available. Use `abdomen_channel` and `thorax_channel` to select labels
explicitly when automatic matching is ambiguous.

External `sleep_stages` override embedded HDF5 stages. Use
`h5_stage_encoding="aasm"` for files with public AASM-coded embedded stages.

## Result

`BSIResult` contains:

- `bsi`, `time_seconds`: the sample-aligned BSI series at 10 Hz.
- `effort`, `upper_envelope`, `lower_envelope`: intermediate processed arrays.
- `stages`: aligned public numeric stage codes, or `None`.
- `summary`: whole-recording and available stage-specific historical features,
  plus actual stable/unstable time fractions and duration.
- `paper_features`: canonical primary aliases such as `bsi_median_sleep`.
- `config`, `channels_used`, `metadata`: processing and file provenance.
- `summary_row()`: a flat row compatible with the revision's BSI-summary merger.

Empty stage groups have undefined numeric features (`NaN` in arrays/CSV, `null`
in JSON). The CLI's JSON is strict JSON; no NaN literals are emitted.

CLI NPZ output stores arrays without pickle: `time_seconds`, `bsi`, `effort`,
`upper_envelope`, `lower_envelope`, and optional `stages`.

See [the scientific contract](scientific-contract.md) before interpreting the
legacy episode-count, AUC, or instability-burden columns.
