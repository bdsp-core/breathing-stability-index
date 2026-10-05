# Breathing Stability Index

Compute the **Breathing Stability Index (BSI)** from abdominal and thoracic
respiratory effort signals. BSI quantifies effort-envelope instability:
**higher BSI means less stable breathing**.

This repository provides an installable Python API and command-line tool,
alongside the analysis code for:

> Ganglberger W, Sun H, Quinn TM, Westover MB, Thomas RJ.
> Dynamic instability of breathing during sleep predicts cognition, disease,
> and mortality. *SLEEP*. 2026; zsag212.
> [doi:10.1093/sleep/zsag212](https://doi.org/10.1093/sleep/zsag212)

[Published paper PDF](paper/Ganglberger_2026_SLEEP_Breathing_Stability_Index.pdf)
 · [Paper analysis guide](paper/README.md)
 · [Scientific definitions](docs/scientific-contract.md)

## Installation

Python 3.10 or newer is required; Python 3.11 is the tested environment.
Use a project environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For plots, install `pip install -e '.[plot]'`. For the paper scripts and tests,
install `pip install -e '.[paper,plot,dev]'`. The optional
[tested constraints](requirements-tested.txt) record the validation versions.

## Python API

```python
from breathing_stability_index import Config, compute_bsi

result = compute_bsi(
    abdomen,                  # 1-D effort signal in arbitrary units
    thorax,                   # optional; either usable belt can be supplied
    fs_hz=100,
    sleep_stages=stage_labels, # one label per 30-second epoch
    file_id="study-001",
    cfg=Config(),
)

median_sleep_bsi = result.paper_features["bsi_median_sleep"]
bsi_timeseries = result.bsi    # 10-Hz array
time_seconds = result.time_seconds
```

The default settings are the primary revision pipeline: individually robust
normalize the belts before averaging (`robust_mean`), 120-second windows,
90% overlap, and 10-Hz processing. This is an analytic signal-processing
measure and requires no model checkpoint.

Sleep stages may be strings (`W`, `N1`, `N2`, `N3`, `REM`, `UNKNOWN`) or numeric
AASM codes (`0`, `1`, `2`, `3`, `4`, `-1`). For sample-aligned stages, pass
`stage_epoch_seconds=None`. The original BDSP numeric coding is different;
pass `stage_encoding="legacy"` for N3=1, N2=2, N1=3, REM=4, wake=5.

Without sleep stages, results contain **whole-recording** summaries only.
Sleep-only BSI is never inferred from an unstaged recording. Input sampling
rate must exceed 20 Hz with the published 10-Hz low-pass filter.

File API:

```python
from breathing_stability_index import compute_bsi_file

result = compute_bsi_file("recording.h5")
# EDF is also supported; select ambiguous channel names explicitly:
result = compute_bsi_file("recording.edf", abdomen_channel="Abdomen", thorax_channel="Chest")
```

The array and file APIs return results without writing files. See
[inputs and outputs](docs/api.md) for formats and feature definitions.

## Command line

```bash
breathing-stability-index --input recording.h5 --outdir output/run-001
breathing-stability-index --manifest-csv manifest.csv --outdir output/batch-001 --save-timeseries
```

The manifest requires `filepath`; optional `file_id` values must be unique.
Relative paths resolve against the manifest directory. Each run writes
`summary_stability.csv` and per-recording JSON summaries. `--save-timeseries`
adds compressed NPZ arrays; `--plot` adds overview figures. Failed recordings
are listed in `errors.csv` and cause a nonzero exit. Use a new output directory
for each run.

## Try a synthetic recording

No participant data is bundled. From the installed source checkout:

```bash
python examples/synthetic.py --output output/synthetic.h5
breathing-stability-index --input output/synthetic.h5 --outdir output/demo --save-timeseries
python examples/array_api.py
```

For a plot as well, run `./run_sample.sh` inside your activated environment.

## Paper code and provenance

- `src/breathing_stability_index/`: public API, file readers, and CLI.
- `paper/scripts/`: preserved revision computation and analysis scripts.
- `paper/run_analysis.py`: portable runner using an external analysis workspace.
- `paper/provenance/`: source hashes, script inventory, and published-result checks.
- `examples/`: deterministic synthetic recording generator and API example.
- `tests/`: numerical parity, stage coding, file adapters, CLI, and provenance checks.

The paper's cohort-specific learned cognition/mortality scores are distinct from
the physiological BSI returned by this API. Their fitting code is supplied in
`paper/`; this package does not distribute a universal disease or survival model.

The preserved analysis inputs include access-controlled clinical and community
cohort data. See [the reproduction guide](paper/README.md) for the required
tables, commands, and current verification limits. Historical feature naming
and scaling details are documented in the scientific contract.

## License

**Noncommercial scientific research only. Commercial use is prohibited.**
The [Breathing Stability Index Noncommercial Research License](LICENSE) permits
independent, academic, nonprofit, and government scientific research. It grants
no permission for for-profit use, commercial R&D, product evaluation, paid
services, or commercial deployment, even when the code itself is free.
