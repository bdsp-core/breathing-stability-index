# Paper code and reproduction

This directory contains the published PDF and the selected revision code for
the SLEEP study. Scripts were captured from the author's local development
working tree on October 5, 2026; that tree included the final uncommitted
revision. Archived code is preserved verbatim except for the documented removal of two
recording identifiers from default paradox-example exclusions. SHA-256 hashes are in
[`provenance/source_manifest.json`](provenance/source_manifest.json).

## What is included

- `scripts/breathing_stability.py`: the revised full BSI computation, including
  raw-mean, robust-mean, and summary-mean belt sensitivity branches.
- `scripts/self_similarity.py`: the periodic-breathing comparator computation.
- `scripts/hypoxic_burden_desat.py` and `scripts/hypoxic_burden/`: historical
  desaturation/hypoxic-burden comparator code, retaining its source notices.
- `scripts/REVISION/`: master-table construction/joins, physiology, cognition,
  matched cross-sectional disease analyses, mortality, sensitivity analyses,
  and final cognition/mortality figure builders. Supporting imports are included.
- `run_analysis.py`: stages the code into an external workspace and runs named
  steps. Historical absolute input paths are remapped under the workspace's
  `inputs/`; adaptations and source/workspace hashes are logged there.

Participant-level recordings, master tables, linked outcomes, OOF predictions,
and original example-night data are not redistributed. The published PDF
contains the final figures; this release does not claim that every original
figure panel is regenerated from the bundled synthetic example.

## Data prerequisites

The paper's data-availability statement identifies:

- [BDSP Human Sleep Project](https://bdsp.io/content/hsp/): MGH and BIDMC PSGs
  and associated mortality follow-up, subject to its access conditions.
- [NSRR](https://sleepdata.org/): MrOS PSGs; additional outcomes and survival
  follow-up require the MrOS Coordinating Center's access process.
- The linked cognition/disease tables used by the authors: obtain authorized
  access independently; they are not silently synthesized or replaced here.

The final paper-master interface is
`REVISION/master_analysis_table_primary_cohorts_v4_paper.csv` in the external
workspace. It is assembled from the wide BSI/comparator table, study selection,
and posture joins. The complete upstream input inventory and script CLI flags
are in [`provenance/analysis_inventory.csv`](provenance/analysis_inventory.csv).
The source snapshot records author-specific lineage; it is not a claim that
BDSP/NSRR alone supplies every linked field needed to reconstruct this table.

Core inputs are identified by `fileid`, `sid`, and `cohort`, with cohorts
`S0001` (MGH), `I0002` (BIDMC), `mros`, and `mgh-cog`. Preserve the original
sex coding, outcome timing, study-type eligibility, one-PSG-per-subject
selection, and group-aware train/test separation.

## Prepare a workspace

Install the analysis extra, then stage a workspace outside the release repo:

```bash
pip install -e '.[paper]'
python paper/run_analysis.py --workspace /path/to/bsi-workspace
```

Place authorized inputs in that workspace. Paths based on the source author's
`/home/wolfgang/repos/` are mapped to `/path/to/bsi-workspace/inputs/`.
For example, the mortality disease-label default becomes
`inputs/sleep_cognition/disease_groups/bdsp_disease_cohorts_sex-age.csv`.
Pass explicit CLI paths when those inputs are elsewhere. The runner refuses to
overwrite edited workspace code. Original algorithms remain unchanged in Git. The paradox-example selector omits
two author-specific recording exclusion identifiers; supply them privately via
`--exclude-fileid` if reproducing the original selected examples.

All examples below use `/path/to/bsi-workspace`. Run each bounded step and
inspect its audit/results before scaling the cohort analysis.

## Compute and join BSI

The public CLI writes the historical long-format summary expected by the BSI
merger. Use primary defaults and explicitly check embedded stage coding:

```bash
breathing-stability-index --manifest-csv /path/to/authorized-recordings.csv \
  --outdir /path/to/bsi-workspace/bsi-primary --h5-stage-encoding legacy
```

From the prepared workspace, the preserved merger accepts these summaries:

```bash
python REVISION/merge_bsi_ss_into_primary_master.py \
  --master-csv REVISION/master_analysis_table_primary_cohorts_v2.csv \
  --bsi-summary-csv bsi-primary/summary_stability.csv \
  --ss-summary-csv /path/to/authorized-self-similarity-summary.csv
python REVISION/build_final_paper_master.py
```

This requires the v2 master and survival/study/posture inputs referenced by
those scripts. See the inventory for the individual upstream join builders.
Use the full archived computation with `--belt-mode robust_mean -w 2 -o 0.9`
when reproducing its combined BSI/self-similarity pipeline. Unlike the new API,
its defaults and file-writing behavior are historical.

## Main analyses

### Physiology and sleep stages

```bash
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step physiology -- \
  --master-csv REVISION/master_analysis_table_primary_cohorts_v4_paper.csv \
  --output-dir REVISION/physiology_v2
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step stage-ahi
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step tables
```

### Cognition and matched disease analyses

```bash
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step cognition-disease
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step cognition-figure
```

These steps require the v4 paper master and the linked `table_phi.csv` in the
workspace. Cognition/disease scores are trained and evaluated within the
preserved group folds; they are not shipped pretrained models. For another
input location, the runner supports Path-constant overrides:

```bash
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step cognition-disease \
  --set-path run_revision_cognition_disease_target_bsi_v1.MASTER=/data/master.csv \
  --set-path run_revision_cognition_disease_target_bsi_v1.TABLE_PHI=/data/table_phi.csv
```

### Mortality selection and transparent raw-BSI models

```bash
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step mortality-selection -- \
  --mgh-disease-csv /data/mgh-disease-labels.csv \
  --bidmc-survival-csv /data/bidmc-survival.csv --mros-table-csv /data/mros-table.csv
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step mortality-raw --include-mgh
```

The raw-BSI analysis uses strict-primary diagnostic/untreated selection.
It is distinct from the published main mortality **feature-score** analysis.

### Published mortality feature score

The final figure uses `original_style_quantile_windows`, **50 folds**, seed 42,
ridge alpha 0.01, and `all_study_type_sensitivity` selection. The source script's
default is 20 folds and strict-primary selection, so specify these settings:

```bash
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step mortality-score --include-mgh -- \
  --selection-mode all_study_type_sensitivity --n-splits 50 --random-state 42 \
  --ridge-alpha 0.01 --feature-set original_style_quantile_windows \
  --output-dir REVISION/mortality_bsi_score_cv_all_study_type_original_style_50fold_v1
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step mortality-comparators --include-mgh
python paper/run_analysis.py --workspace /path/to/bsi-workspace --step mortality-figure
```

`--include-mgh` explicitly adds S0001 to source-script cohort dictionaries that
originally listed only BIDMC and MrOS. It reproduces the cohort extension used
for the final MGH probe, without changing fitting, folds, or covariates. The
final figure builder already recognizes MGH and can read combined results or
the separate historical probe directories.

The saved source results agree with the main fully adjusted paper estimates:
MGH HR 1.376661 (CI 1.227681–1.543721), BIDMC HR 1.102371
(CI 1.021310–1.189866), and MrOS HR 1.015962 (CI 0.970556–1.063492).
[`published_results_check.json`](provenance/published_results_check.json)
records this aggregate check; it is not a fresh model rerun.

Named sensitivity steps include `rem-nrem`, `study-type`, `posture`, `osa-csa`,
and `raw-bsi-sensitivity`. Other preserved builders can be run directly from
the prepared workspace. For historical comparator raw-signal tools, install
the `comparators` extra as well; their raw-input paths and annotations must be
provided separately.

## Bounded synthetic analysis demonstration

To exercise the actual mortality fitting path without authorized cohort inputs:

```bash
python examples/synthetic_paper_tables.py --outdir /tmp/bsi-synthetic-inputs
python paper/run_analysis.py --workspace /tmp/bsi-synthetic-workspace --step mortality-score --include-mgh -- \
  --master-csv /tmp/bsi-synthetic-inputs/synthetic_master.csv \
  --selection-csv /tmp/bsi-synthetic-inputs/synthetic_selection.csv \
  --selection-mode all_study_type_sensitivity --n-splits 3 \
  --feature-set original_style_quantile_windows
```

This intentionally small, three-fold demonstration checks code execution across
three synthetic cohort labels. Its estimates have no empirical meaning and
do not reproduce the published study.

## Verification limits

The new API is checked against the preserved kernels with synthetic staged
recordings and a bounded real-recording segment. HDF5, EDF, manifest execution,
output serialization, and package installation are tested. Scientific feature
ambiguities are documented in [the contract](../docs/scientific-contract.md).
Full cohort matching/model fitting and exact published-figure reconstruction
have not been rerun for this release; they require the authorized source tables
and should be verified before making new reproduction claims.
