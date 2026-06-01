# GRPO + Kubeflow Trainer v2

TRL GRPO fine-tuning on Kubeflow Trainer v2, plus Phase 2 validation for deployed vLLM rollout generation and NCCL-based weight sync.

## Current status (2026-06-01)

- **Phase 1 (KFT + TRL GRPO):** validated.
- **Phase 2 (deployed vLLM + KFT updates):** validated in PoC form.
- **NCCL native sync:** works with correct trainer ingress/network policy; cross-node reliability depends on hardened networking and recovery logic.
- **Production decision:** move ahead conditionally, with hardening tasks (network policy model, timeout/recovery automation, pinned runtime matrix).

## What’s here

| Path | What |
|------|------|
| `notebooks/grpo_kft_v2.ipynb` | Phase 1 baseline: multi-GPU KFT TrainJob with TRL GRPO |
| `notebooks/grpo_vllm_training.ipynb` | Phase 2 notebook for rollout + sync workflow |
| `notebooks/grpo-vllm-rollout-example.ipynb` | Serving rollout validation |
| `scripts/grpo_vllm_train_sync.py` | Train + NCCL sync driver script |
| `scripts/weight_sync_test.py` | Focused weight-sync harness |
| `scripts/validate_cold_start_weight_sync.sh` | Cold-start sync validation helper |
| `spike-findings.md` | Phase 1 compatibility findings and constraints |
| `runs_log/gpu-baseline-trl-grpo.md` | GPU baseline for TRL GRPO without vLLM |
| `kft-vllm-grpo-questions-answered-2026-06-01.md` | Consolidated Q&A across Phase 1/2/3 |
| `rbac-trainer-access.yaml` | RBAC setup for Trainer API access from notebook SA |

## Key technical updates from Phase 2

- Split serving+trainer topology was exercised (different nodes).
- vLLM native sync lifecycle was validated (`init/start/update/finish/resume`).
- A major blocker was network-policy/data-channel openness for NCCL:
  - restrictive ingress caused init stalls at `initializing`,
  - corrected/widened ingress allowed init transition and successful sync completion.
- Temporary unblocker tactics ("hacks") were used for PoC speed and are documented in the decision brief; they must be replaced before production use.

## Quick run path

1. Run `notebooks/grpo_kft_v2.ipynb` for baseline KFT GRPO validation.
2. Run `notebooks/grpo_vllm_training.ipynb` for deployed serving + sync workflow.
3. Use `scripts/weight_sync_test.py` / `scripts/validate_cold_start_weight_sync.sh` for focused sync diagnostics.

## Production-quality gaps to close

- Deterministic NCCL connectivity model with least-privilege network policy.
- Automated timeout/recovery for stuck async sync operations.
- Pinned compatible versions across vLLM, torch/CUDA/NCCL, TRL, transformers, accelerate.
- Hardened runtime image behavior for OpenShift random UID and cache/persistence paths.
