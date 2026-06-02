# GRPO + Kubeflow Trainer v2

TRL GRPO fine-tuning on Kubeflow Trainer v2, with cross-node vLLM weight sync validation via NCCL.

## Current status (2026-06-02)

- **Phase 1 (KFT + TRL GRPO):** validated. Multi-node multi-GPU (2x2) TrainJobs complete successfully.
- **Phase 2 (deployed vLLM + single-pod sync):** validated. 3-GPU same-pod train+sync works.
- **Phase 3 (cross-node KFT <-> vLLM sync):** validated. KFT trainer pod and vLLM serving pod on **different nodes**, live NCCL weight sync without PVC, no deadlock. Two consecutive successful runs across `kapil-test` and `grpoxtrainer` namespaces.
- **Production decision:** Go forward with scope guard -- delivery track ready, hardening track (network policy, recovery automation, version pinning) required before production.

### Cross-node validation results (2026-06-02)

| Run | Namespace | Trainer node | vLLM node | sync_status | sync_elapsed |
|-----|-----------|-------------|-----------|-------------|-------------|
| 1 | `kapil-test` | `ip-10-0-26-65` | `ip-10-0-16-179` | success | 1.17s |
| 2 | `grpoxtrainer` | `ip-10-0-26-65` | `ip-10-0-16-179` | success | 1.18s |

All vLLM lifecycle endpoints returned 200: `init` -> `pause` -> `start_weight_update` -> `update_weights` -> `finish_weight_update` -> `resume`.

## What's here

| Path | What |
|------|------|
| `notebooks/grpo_kft_v2.ipynb` | Phase 1 baseline: multi-GPU KFT TrainJob with TRL GRPO |
| `notebooks/grpo_vllm_training_cross_node.ipynb` | Phase 2 single-pod (3-GPU) + Phase 3 cross-node validation |
| `notebooks/grpo_vllm_training.ipynb` | Phase 2 notebook for rollout + sync workflow |
| `notebooks/grpo-vllm-rollout-example.ipynb` | Serving rollout validation |
| `notebooks/prometheus_grafana_vllm.ipynb` | Prometheus + Grafana monitoring for vLLM |
| `scripts/grpo_vllm_train_sync.py` | Train + NCCL sync driver script |
| `scripts/grpo_kft_train_fn.py` | KFT training function (saves to PVC) |
| `scripts/post_train_sync.py` | Post-train PVC->vLLM sync orchestration |
| `scripts/benchmark_sync_paths.py` | localtmp vs PVC sync-path benchmark |
| `scripts/weight_sync_test.py` | Focused weight-sync harness |
| `vllm-weight-sync-limitations.md` | Limitations, deadlock analysis, architecture options |
| `baseline_sync_paths_5runs_summary.md` | Sync-path baseline (5-run localtmp vs PVC comparison) |
| `spike-findings.md` | Phase 1 compatibility findings and constraints |
| `runs_log/gpu-baseline-trl-grpo.md` | GPU baseline for TRL GRPO without vLLM |

## Architecture

```
Option A: Cross-node live sync (Phase 3 -- validated in PoC)

  +---------------------+         +-------------------------+
  |  KFT TrainJob        |  NCCL  |  vLLM Deployment         |
  |  (node A, 1 GPU)     |<------>|  (node B, 1 GPU)         |
  |  Threaded init/update |        |  --weight-transfer-config |
  +---------------------+         +-------------------------+
       NetworkPolicy: vLLM -> trainer ingress for NCCL ports

Option B: Multi-node KFT + PVC + localhost sync (Phase 2 -- validated)

  +-----------------------+              +--------------------------+
  |  KFT TrainJob (2x2)   |  checkpoint  |  vLLM Pod (2 GPUs)       |
  |  Node 0: 2 GPUs \     |----> PVC --->|  GPU 0: inference        |
  |  Node 1: 2 GPUs / DDP |              |  GPU 1: load + NCCL sync |
  +-----------------------+              +--------------------------+
```

## Quick run path

1. **KFT GRPO baseline:** `notebooks/grpo_kft_v2.ipynb`
2. **Single-pod train + sync:** Top section of `notebooks/grpo_vllm_training_cross_node.ipynb`
3. **Cross-node validation:** Bottom section ("Cross-node validation") of `notebooks/grpo_vllm_training_cross_node.ipynb`
4. **Sync-path benchmark:** `scripts/benchmark_sync_paths.py`

## Production-quality gaps to close

- Deterministic NCCL connectivity model with least-privilege network policy (current PoC uses permissive ingress).
- Automated timeout/recovery for stuck async sync operations.
- Pinned compatible versions across vLLM, torch/CUDA/NCCL, TRL, transformers, accelerate.
- Hardened runtime image behavior for OpenShift random UID and cache/persistence paths.
- Combined validation: multi-node GRPO (2x2) + cross-node live sync to vLLM in one integrated flow.
