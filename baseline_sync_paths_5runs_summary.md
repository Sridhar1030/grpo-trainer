# Sync Path Baseline (5 Runs)

- Namespace: `grpoxtrainer`
- vLLM pod: `grpo-vllm-rollout-7cf46f7665-sg9cw`
- Compared paths: `localtmp=/tmp/grpo-local-grpo-trained` vs `pvc=/mnt/checkpoint/grpo-trained`
- Same checkpoint used for both paths per run (sync-only benchmark).

## Checkpoint Size

- PVC: `958M	/mnt/checkpoint/grpo-trained`
- Local tmp copy: `958M	/tmp/grpo-local-grpo-trained`

## Results (Mean ± Std)

| Metric | localtmp | pvc |
|---|---:|---:|
| Sync elapsed (s) | 0.478 ± 0.033 | 0.458 ± 0.007 |
| Post-eval latency (s) | 0.215 ± 0.009 | 0.218 ± 0.009 |
| Post-eval accuracy | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

## What Each Metric Means

- **localtmp**: checkpoint was read from local disk path inside the vLLM pod (`/tmp/grpo-local-grpo-trained`).
- **pvc**: checkpoint was read from shared RWX PVC path (`/mnt/checkpoint/grpo-trained`).
- **Mean ± Std**: average across 5 runs, plus run-to-run variation (standard deviation). Lower std = more stable.
- **Checkpoint Size**: total bytes of checkpoint directory used during sync; should match between both paths for fair comparison.
- **Sync elapsed (s)**: time for `--sync-only` weight transfer step (load checkpoint + NCCL transfer + vLLM weight update).
- **Post-eval latency (s)**: average response latency after sync on the fixed 5-question deterministic eval set.
- **Post-eval accuracy**: accuracy on that same 5-question eval after sync; used here as a basic quality sanity check.

## Raw Per-Run Sync Times (s)

- localtmp: [0.45, 0.45, 0.54, 0.47, 0.48]
- pvc: [0.46, 0.47, 0.45, 0.45, 0.46]

## Interpretation

- Sync-path performance is effectively equivalent between localtmp and PVC in this environment.
- No quality difference observed on the 5-question deterministic eval subset (both remain 0.0).
- This is a sync-only baseline (no retraining in-loop); use full pipeline runs separately for train+sync SLOs.
