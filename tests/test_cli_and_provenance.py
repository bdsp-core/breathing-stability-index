from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from breathing_stability_index.cli import main

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from synthetic import write_h5


def test_manifest_outputs_and_partial_failure(tmp_path):
    write_h5(tmp_path / "synthetic.h5")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,file_id\nsynthetic.h5,synthetic\nmissing.h5,missing\n")
    outdir = tmp_path / "run"
    code = main(["--manifest-csv", str(manifest), "--outdir", str(outdir), "--save-timeseries", "--plot"])
    assert code == 1
    summary = pd.read_csv(outdir / "summary_stability.csv")
    assert summary.loc[0, "belt_mode"] == "robust_mean"
    assert summary.loc[0, "window_length"] == 2
    assert summary.loc[0, "bsi_median_sleep"] > 0
    assert (outdir / "timeseries/synthetic.npz").is_file()
    assert (outdir / "figures/synthetic.png").is_file()
    payload = json.loads((outdir / "json/synthetic.json").read_text())
    assert payload["metadata"]["stage_encoding"] == "legacy"
    assert pd.read_csv(outdir / "errors.csv").loc[0, "file_id"] == "missing"
    assert main(["--manifest-csv", str(manifest), "--outdir", str(outdir)]) == 1


def test_duplicate_manifest_ids_rejected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,file_id\na.h5,same\nb.h5,same\n")
    outdir = tmp_path / "run"
    assert main(["--manifest-csv", str(manifest), "--outdir", str(outdir)]) == 1
    assert not outdir.exists()


def test_source_hashes():
    manifest = json.loads((ROOT / "paper/provenance/source_manifest.json").read_text())
    for record in manifest["files"]:
        assert hashlib.sha256((ROOT / record["release_path"]).read_bytes()).hexdigest() == record["sha256"]


def test_workspace_portability_and_no_code_overwrite(tmp_path):
    spec = importlib.util.spec_from_file_location("paper_runner", ROOT / "paper/run_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    workspace = module.prepare_workspace(tmp_path / "workspace")
    path = workspace / "REVISION/run_revision_mortality_v2.py"
    assert "/home/wolfgang/repos/" not in path.read_text()
    path.write_text("# local edits\n")
    with pytest.raises(ValueError, match="overwrite"):
        module.prepare_workspace(workspace)


def test_packaged_cli_help():
    completed = subprocess.run([sys.executable, "-m", "breathing_stability_index", "--help"],
                               check=True, capture_output=True, text=True)
    assert "--manifest-csv" in completed.stdout


def test_paper_mortality_path_on_synthetic_tables(tmp_path):
    from synthetic_paper_tables import write_tables
    master, selection = write_tables(tmp_path / "inputs")
    workspace = tmp_path / "workspace"
    subprocess.run([
        sys.executable, str(ROOT / "paper/run_analysis.py"), "--workspace", str(workspace),
        "--step", "mortality-score", "--include-mgh", "--",
        "--master-csv", str(master), "--selection-csv", str(selection),
        "--selection-mode", "all_study_type_sensitivity", "--n-splits", "3",
        "--feature-set", "original_style_quantile_windows",
    ], check=True, capture_output=True, text=True, timeout=60)
    output = workspace / "REVISION/mortality_bsi_score_cv_v1"
    results = pd.read_csv(output / "bsi_score_cv_results.csv")
    assert set(results.cohort) == {"I0002", "mros", "S0001"}
    assert len(results) == 12
    assert results.status.eq("ok").all()
    predictions = pd.read_csv(output / "bsi_score_oof_predictions.csv")
    assert predictions.bsi_score_oof.notna().all()
    assert predictions.groupby(["cohort", "model_name"]).fileid.nunique().eq(80).all()
