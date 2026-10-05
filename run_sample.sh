#!/usr/bin/env bash
set -euo pipefail
task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_repo_root"
python -m pip install -e '.[plot]'
python examples/synthetic.py --output output/synthetic.h5
python -m breathing_stability_index --input output/synthetic.h5 --outdir output/sample-run --save-timeseries --plot
