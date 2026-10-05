# Release verification

Validation performed October 5, 2026 with Python 3.11.14. Exact dependency
versions are recorded in `requirements-tested.txt`.

- 12 tests passed, covering numerical equality with the preserved revision
  code, stage coding/alignment, HDF5 and EDF readers, manifest failures,
  JSON/NPZ/plot output, source hashes, portable workspace creation, and the
  actual mortality fitting path on bounded synthetic tables.
- All 33 preserved revision modules imported successfully with the paper extra.
- A 600-second segment of an available real effort recording produced finite
  BSI and exactly matched the archived primary numerical pipeline (maximum
  absolute difference 0.0). The recording and its outputs are not bundled.
- A wheel and source distribution were built. The wheel was installed into a
  separate environment and exercised through the public array API and CLI.
- Existing aggregate mortality results round to the published fully adjusted
  MGH/BIDMC estimates. This is a source-result check, not a fresh cohort rerun.
- Release candidates contain no raw recording files or participant tables.
  Two recording exclusion identifiers were removed from the optional
  paradox-example selector; the manifest records original and released hashes.

Full cohort analyses and exact original figure reconstruction were not rerun.
Authorized linked source tables and example recordings remain prerequisites.
Historical naming/scaling differences are described in the scientific contract.
