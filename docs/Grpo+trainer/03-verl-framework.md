# veRL Framework — Deep Reference

> **Repo:** https://github.com/volcengine/verl (also https://github.com/verl-project/verl)  
> **Docs:** https://verl.readthedocs.io/en/latest/  
> **Paper:** HybridFlow (arXiv:2409.19256v2, EuroSys 2025)

---

## Overview

veRL ("Volcano Engine RL") is an open-source RL post-training library for LLMs from ByteDance Seed. It emphasizes modular plugging of **training backends** (FSDP, FSDP2, Megatron-LM) and **rollout backends** (vLLM, SGLang, HF, TensorRT-LLM).

### Architecture: HybridFlow

| Layer | Role |
|-------|------|
| **Control flow** | Single-process controller implements RL loop |
| **Computation flow** | Multi-process workers run FSDP/Megatron training, vLLM generation, etc. |

**Worker groups** (via Ray):
- **ActorRolloutRefWorker**: hosts actor (training), rollout (inference), and reference policy — colocated for fast weight sync or split across resource pools
- **Critic** and **reward model**: separate TrainingWorker groups when enabled

---

## GRPO Configuration

### Enabling GRPO

```yaml
algorithm:
  adv_estimator: grpo          # default is gae; change to grpo
  norm_adv_by_std_in_grpo: true  # false = DrGRPO-style
  use_kl_in_reward: false       # GRPO uses KL in actor loss instead
  kl_penalty: kl
  gamma: 1.0
  lam: 1.0

actor_rollout_ref:
  rollout:
    n: 8                        # group size G
  actor:
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    ppo_mini_batch_size: 256
    ppo_epochs: 1
    clip_ratio: 0.2
    loss_agg_mode: token-mean

data:
  train_batch_size: 1024        # prompt batch; responses = batch × n
```

### DrGRPO Variant

```yaml
algorithm:
  norm_adv_by_std_in_grpo: false
actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-sum-norm
    use_kl_loss: false
```

### Reference Script

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  ...
```

---

## Advantage Computation

`compute_grpo_outcome_advantage`:
1. Per-sequence score = sum of token-level rewards over response
2. Group by prompt id (`index`)
3. For each group: mean μ, std σ (with ε for stability)
4. Normalized: `(r - μ) / (σ + ε)` if `norm_adv_by_std_in_grpo`, else `r - μ`
5. Broadcast scalar across tokens × response_mask

---

## vLLM Integration

### Rollout Configuration

```yaml
actor_rollout_ref:
  rollout:
    name: vllm                    # or sglang, hf, trtllm
    gpu_memory_utilization: 0.5   # KV cache budget fraction
    tensor_model_parallel_size: 2
    data_parallel_size: 1
    max_num_batched_tokens: 8192
    max_num_seqs: 1024
    enforce_eager: false
    dtype: bfloat16
    enable_chunked_prefill: true
    enable_prefix_caching: true
```

### GPU Memory: FSDP vs vLLM

From the [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html):
- `gpu_memory_utilization ~0.5-0.7` balances throughput vs OOM when actor weights/optimizer occupy GPU
- CUDA graphs (`cudagraph_capture_sizes`) stay on GPU during actor update → can force OOM
- **Recommended: vLLM ≥ 0.8.2** with FSDP

### Fully Async Policy Trainer

veRL supports NCCL-based parameter sync between Trainer and Rollouter:
- Megatron/FSDP + vLLM in server mode
- Reports ~2.35-2.67x end-to-end speedup for Qwen2.5-7B @ 128 GPUs

---

## Built-in GSM8K Reward

```python
from verl.utils.reward_score.gsm8k import compute_score

r = compute_score(
    solution_str=model_output,
    ground_truth="42",
    method="strict",      # or "flexible"
    format_score=0.0,
    score=1.0,
)
```

- **strict**: regex on last 300 chars for `#### <number>`
- **flexible**: last plausible numeric token

---

## Standalone Mode (Non-Ray)

### Current State

- RL (PPO/GRPO) controller is **Ray-centric**: `main_ppo` + `RayPPOTrainer` + `RayWorkerGroup`
- SFT/DPO/RM trainers can run with `torchrun` directly (SPMD)
- RFC for TorchRPC as Ray alternative exists ([Issue #1221](https://github.com/verl-project/verl/issues/1221)) — emerging, not default

### Viability for KFT

| Approach | Fit |
|----------|-----|
| Ray cluster in K8s (KubeRay) backing `main_ppo` | Most aligned with upstream |
| Single-pod multi-GPU | Feasible; still typically Ray local |
| Pure `torchrun` RL loop | Not documented default for GRPO |

---

## FSDP Configuration

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp2             # or fsdp
    fsdp_config:
      param_offload: false
      optimizer_offload: false
      offload_policy: false     # FSDP2: param/grad/opt offload during train
      reshard_after_forward: true
      fsdp_size: -1
      dtype: bfloat16
```

### Memory Optimization Techniques

- Gradient checkpointing: `model.enable_gradient_checkpointing=True`
- Activation offload (FSDP): `enable_activation_offload=True`
- Sequence packing: `use_remove_padding=True`
- Dynamic batching: `use_dynamic_bsz=True`
- Entropy chunking/checkpointing for logit memory peaks
- Liger Kernel, Ulysses SP

---

## Limitations

| Area | Notes | Severity |
|------|-------|----------|
| Ray dependency | Operational complexity; harder in restricted K8s | High |
| Config surface | 200+ OmegaConf params; mis-set micro_batch causes silent inefficiency | High |
| Documentation drift | Legacy `fsdp_workers` removed; docs now center `engine_workers` | Medium |
| Standalone RL | Not first-class; TorchRPC is RFC-stage | Medium |
| vLLM + training colocation | Documented OOM footguns with CUDA graphs | Medium |

---

## Supported Algorithms

PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, DAPO, DrGRPO, and more via `verl-recipe` repo.

---

## References

- [veRL GitHub](https://github.com/volcengine/verl)
- [veRL Docs](https://verl.readthedocs.io/en/latest/)
- [GRPO docs](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [HybridFlow guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
- [Performance tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
- [HybridFlow paper](https://arxiv.org/abs/2409.19256v2)
