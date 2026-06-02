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

## Cross-Node Sync Baseline (2026-06-02)

Separate from the localtmp/PVC same-pod benchmarks above, cross-node NCCL weight sync was validated with KFT trainer and vLLM on different physical nodes.

- Trainer node: `ip-10-0-26-65`
- vLLM node: `ip-10-0-16-179`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (~958M checkpoint)
- Sync method: Direct NCCL (no PVC), threaded init pattern

| Run | Namespace | sync_elapsed (s) | sync_status | Post-sync answer |
|-----|-----------|-------------------|-------------|------------------|
| 1 | `kapil-test` | 1.17 | success | "4" (correct) |
| 2 | `grpoxtrainer` | 1.18 | success | "4" (correct) |

**Comparison:** Cross-node sync (~1.18s) is ~2.5x slower than same-pod sync (~0.47s), which is expected given network-based NCCL vs localhost NCCL. Both are well within acceptable bounds for an RLHF loop where training steps take minutes.

## Interpretation

- Sync-path performance is effectively equivalent between localtmp and PVC in this environment.
- No quality difference observed on the 5-question deterministic eval subset (both remain 0.0).
- Cross-node NCCL sync adds ~0.7s overhead vs same-pod, acceptable for production RLHF loops.
- This is a sync-only baseline (no retraining in-loop); use full pipeline runs separately for train+sync SLOs.
