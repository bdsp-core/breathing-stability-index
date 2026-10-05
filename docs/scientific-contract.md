# Scientific contract

## Authoritative sources

The supplied, published SLEEP PDF defines the primary method and reporting.
The local revised `breathing_stability.py` and selected `REVISION/` scripts
provide the executed computational definitions. They include uncommitted work
absent from the private development repo's remote. The source manifest records
file hashes and adaptations; archived scripts are unchanged copies except for
two recording identifiers removed from default paradox-example exclusions.

The public runtime extracts the BSI-related numerical functions verbatim into
`_kernels.py`. It wraps them with explicit defaults, stage alignment, input
validation, and structured results. Self-similarity/event detection is omitted
from runtime BSI computation; it does not affect the BSI calculation. Its
source remains in `paper/scripts/` for comparator reproduction.

## Primary signal processing

1. Supply one or two usable effort belts on a common time grid. Select abdomen
   and thorax explicitly when file labels are ambiguous.
2. Detrend, low-pass filter at 10 Hz, and use the original MNE FFT resampling
   implementation to obtain 10-Hz signals.
3. Apply the revision's robust normalization independently to each belt and
   average the normalized belts. Normalize the combined trace again, as in
   the revision's `robust_mean` branch.
4. Apply the original MAD outlier correction, extrema detection, cubic
   interpolation, and envelope safeguards.
5. For each 120-s window, divide the sum of mean absolute first differences
   of the upper and lower envelopes by their mean absolute amplitude plus
   `1e-4`, then multiply by 100. Advance windows by 12 s (90% overlap).
6. Assign each window value at its center, then forward-fill and backward-fill
   the trace, including the recording edges, as in the revision.

The numerical BSI is defined at 10 Hz. Changing the processing rate changes its
scale; the API fixes that rate. The published paper's continuous derivative
notation is implemented as the historical sample differences and factor 100.
The runtime deliberately preserves that numerical scale and its thresholds.

The source normalization helper clips amplitudes to estimate the median/IQR,
then normalizes the **original**, unclipped signal. It does not hard-clip the
returned signal. This detail and the second normalization are preserved for
agreement with the executed revision, rather than silently replacing them
with a different interpretation of the manuscript's prose.

## Stage coding

| Stage | Public numeric AASM | Original BDSP/legacy |
| --- | --- | --- |
| Wake | 0 | 5 |
| N1 | 1 | 3 |
| N2 | 2 | 2 |
| N3 | 3 | 1 |
| REM | 4 | 4 |
| Unknown | -1 | 0 or -1 |

Public string labels are independent of numeric encoding. HDF5 embedded stages
default to the legacy convention; the array API defaults to AASM. Unknown
samples contribute to whole-recording summaries but not sleep/stage summaries.
Stages are aligned at each 10-Hz output sample's timestamp, without interpolating
categorical labels. The historical full-recording BSI is computed before stage
selection; windows may cross stage boundaries.

## Preserved feature definitions

- `median_stability`: median after clipping BSI values at Q1−2×IQR and Q3+2×IQR.
- `stability_auc`: clipped mean multiplied by processing fs. Divide by 10 to
  obtain `bsi_mean_*`, as the revision did for the mean-BSI figure.
- Quantiles and histograms use the unclipped BSI. Histograms normalize over
  values falling inside their original finite bin range (0–8).
- `n_5min_windows_stable` and `n_5min_windows_unstable`: historical names.
  The primary script passed a **two-minute** block duration, counting complete
  blocks inside below-0.5 or above-1.5 episodes. Stage-specific calculations
  concatenate the selected samples and can join separated periods. These
  definitions are retained in the compatibility fields.
- `bsi_instability_burden_gt_1p5_sleep`: the paper master computed
  `unstable_episode_blocks / (stable_episode_blocks + unstable_episode_blocks)`.
  It is a 0–1 ratio, undefined when the denominator is zero. Intermediate BSI
  values are excluded from its denominator. Despite the historical name and
  published time-percentage wording, it is **not** the fraction of total sleep
  time above 1.5.
- `SLEEP_unstable_time_fraction`: the new API's actual fraction of sleep samples
  with BSI >1.5. This is separately named and does not replace the paper field.

Canonical `bsi_*` paper aliases are emitted only for the primary robust-mean,
two-minute, 90%-overlap settings. Nonprimary settings retain the explicit
historical summary columns and configuration metadata.

## Scope and limits

BSI measures respiratory effort-envelope instability, not calibrated ventilation.
The runtime preserves supplied belt polarity; it does not infer phase alignment.
Anti-phase belts can cancel and are not automatically corrected. Inspect signal
quality and exclude unusable belts explicitly. Nonfinite signals, zero robust
scale, insufficient duration, missing envelopes, and ambiguous channels produce
errors rather than fabricated scores.

The paper's learned multi-feature scores use cohort-specific fitting and held-out
predictions. They are not equivalent to median BSI and are not pretrained
inference heads in this package. Preserve matching, subject selection, feature
sets, folds, seeds, and covariate definitions when reproducing them.
