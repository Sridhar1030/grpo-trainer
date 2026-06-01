# Baseline Summary: Local Path vs PVC Path Sync

## What Was Compared

- **localtmp**: sync-only from `/tmp/grpo-local-grpo-trained`
- **pvc**: sync-only from `/mnt/checkpoint/grpo-trained`
- Same checkpoint content used for both paths in each run.

## Key Results


| Metric                      | localtmp                          | pvc                               |
| --------------------------- | --------------------------------- | --------------------------------- |
| Checkpoint size             | 958M /tmp/grpo-local-grpo-trained | 958M /mnt/checkpoint/grpo-trained |
| Sync elapsed (s, mean)      | 0.480                             | 0.450                             |
| Post-eval latency (s, mean) | 0.238                             | 0.237                             |
| Post-eval accuracy (mean)   | 0.0000                            | 0.0000                            |


## Interpretation

- Sync performance is effectively equivalent between local path and PVC path in this run set.
- No observable quality delta in post-eval accuracy.
- This benchmark isolates **sync-path differences**, not full train+sync pipeline differences.

