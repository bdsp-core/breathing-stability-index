from breathing_stability_index import compute_bsi
from synthetic import synthetic_effort

abdomen, thorax, stages = synthetic_effort()
result = compute_bsi(abdomen, thorax, fs_hz=100, sleep_stages=stages,
                     stage_epoch_seconds=None, stage_encoding="legacy", file_id="synthetic")
print("Median sleep BSI:", result.paper_features["bsi_median_sleep"])
print("Actual fraction of sleep above BSI 1.5:", result.summary["SLEEP_unstable_time_fraction"])
