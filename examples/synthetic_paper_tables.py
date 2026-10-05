"""Synthetic tables for a bounded mortality-code demonstration, not study data."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def write_tables(outdir, n_per_cohort=80, seed=42):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for cohort in ["I0002", "mros", "S0001"]:
        for i in range(n_per_cohort):
            severity = rng.normal()
            survival = rng.exponential(1500 * np.exp(-0.4 * severity)) + 1
            censoring = rng.uniform(800, 3000)
            row = {"fileid": f"synthetic-{cohort}-{i:03d}", "sid": f"synthetic-{cohort}-{i:03d}",
                   "cohort": cohort, "age": rng.uniform(45, 85),
                   "sex": 1 if cohort == "mros" else rng.integers(0, 2),
                   "followup_days": min(survival, censoring),
                   "vital_status": "dead" if survival <= censoring else "alive"}
            for col in ["ss_percent_main", "ahi", "hypoxic_burden", "arousal_index"]:
                row[col] = float(np.exp(0.3 * severity + rng.normal()))
            for stage in ["sleep", "nrem", "rem"]:
                prefix = f"bsi_robust_mean_w2_ov0p9_{stage}_"
                row[prefix + "quantile_25"] = float(np.exp(0.2 * severity + rng.normal(scale=0.2)))
                row[prefix + "quantile_75"] = row[prefix + "quantile_25"] + rng.uniform(0.1, 1)
                row[prefix + "n_5min_windows_stable"] = int(rng.integers(1, 80))
                row[prefix + "n_5min_windows_unstable"] = int(rng.integers(1, 80))
            records.append(row)
    master = pd.DataFrame(records)
    master.to_csv(outdir / "synthetic_master.csv", index=False)
    selected = master[["fileid", "sid", "cohort"]].copy()
    selected["selection_mode"] = "all_study_type_sensitivity"
    selected.to_csv(outdir / "synthetic_selection.csv", index=False)
    return outdir / "synthetic_master.csv", outdir / "synthetic_selection.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    write_tables(args.outdir)
    print(f"Synthetic tables: {args.outdir}")
