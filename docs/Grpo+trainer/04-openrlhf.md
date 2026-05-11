# OpenRLHF — Deep Reference

> **Repo:** https://github.com/OpenRLHF/OpenRLHF  
> **Docs:** https://openrlhf.readthedocs.io/en/latest/  
> **Paper:** arXiv:2405.11143

---

## Overview

OpenRLHF is an open-source (Apache-2.0) framework for RLHF and agentic post-training. Combines Ray (distributed scheduling), vLLM (high-throughput rollouts), DeepSpeed ZeRO (sharded training), and HF Transformers.

Designed for **70B+** models in distributed layouts.

---

## GRPO in OpenRLHF

GRPO is NOT a separate trainer class — it's a **choice of advantage estimator** in the shared PPO/Ray training stack:

| Algorithm | Flag | Notes |
|-----------|------|-------|
| PPO | *(default)* | Uses critic |
| REINFORCE++ | `reinforce` | PPO-style tricks without critic |
| REINFORCE++-baseline | `reinforce_baseline` | Mean baseline |
| RLOO | `rloo` | Multi-sample / per-token KL |
| **GRPO** | **`group_norm`** | Group normalization |
| **Dr. GRPO** | **`dr_grpo`** | Drops local `/std` normalization |

Enable with:
```bash
--algo.advantage.estimator group_norm
--rollout.n_samples_per_prompt 8  # must be > 1 for group_norm
--algo.kl.use_loss true
--algo.kl.estimator k3
```

---

## Architecture: Ray + vLLM + DeepSpeed

### Two Runtime Patterns

1. **Distributed mode**: Separate GPU pools per role (70B+ models)
2. **Hybrid Engine**: Colocate all roles on same GPUs, time-slice with sleep modes

### Worker Roles

- **PolicyModelActor**: trainable actor (DeepSpeed)
- **ReferenceModelActor**: frozen reference for KL
- **CriticModelActor**: value network (PPO only, not GRPO)
- **RewardModelActor** or `--reward.remote_url`: reward model or Python function
- **vLLM engines**: separate Ray workers for generation

### Colocation Flags

| Flag | Role |
|------|------|
| `--train.colocate_all` | vLLM + actor + ref + reward + critic on same GPUs |
| `--train.colocate_actor_ref` | Actor + reference share GPUs |
| `--train.colocate_critic_reward` | Critic + reward share GPUs |
| `--vllm.enable_sleep` | Time-slice memory between gen and training |
| `--vllm.sync_backend nccl` | Fast weight broadcast to vLLM |

**Constraints:**
- `--vllm.enable_sleep` incompatible with `--train.async_enable`
- NCCL "duplicate GPU" errors when actor rank 0 and vLLM worker map to same GPU

---

## On Kubernetes

### Official Path: Slurm + Ray

OpenRLHF's multi-node examples use Slurm + containers + `ray job submit`. No first-party Kubeflow Trainer manifest exists.

### KubeRay Approach

KubeRay is the standard way to run Ray on K8s. Ray documents a post-training walkthrough using verl on KubeRay which is architecturally analogous.

### KFT Integration Challenges

| Area | Issue |
|------|-------|
| **Ports** | Ray head GCS, dashboard (8265), object store, worker ports — conflicts with strict pod security |
| **GPU visibility** | Hybrid colocation + vLLM TP + ZeRO requires consistent `CUDA_VISIBLE_DEVICES`, `/dev/shm`, sometimes `hostNetwork` |
| **Gang scheduling** | Ray placement groups expect atomic GPU bundles; K8s scheduler may fragment |
| **Embedded Ray** | Single-node possible; multi-node requires cross-pod Ray (KubeRay) |

### Verdict for KFT

**Feasible but custom.** Requires Ray-on-Kubernetes (KubeRay) + container image + `ray job submit`. No out-of-the-box OpenRLHF + KFT recipe exists publicly.

---

## OpenRLHF vs TRL vs veRL

| Dimension | OpenRLHF | TRL | veRL |
|-----------|----------|-----|------|
| **Core stack** | Ray + vLLM + DeepSpeed | HF Transformers/PEFT; optional Ray | FSDP/Megatron + vLLM; Ray |
| **GRPO** | `group_norm` on Ray PPO path | GRPOTrainer + GRPOConfig | OmegaConf `adv_estimator=grpo` |
| **Scale** | 70B+ distributed | Faster to start; HF ecosystem | Flexible device mesh |
| **Complexity** | Very High | Lower | High |
| **KFT fit** | Requires KubeRay | Native CustomTrainer | Requires Ray or custom bridge |

### When OpenRLHF Fits

- vLLM-first rollout performance with documented hybrid colocation + NCCL weight sync
- Already running Ray clusters (Slurm, KubeRay, cloud)
- One codebase toggling PPO / REINFORCE++ / GRPO / RLOO

---

## Limitations for Phase 1

| Limitation | Severity |
|------------|----------|
| Ray as RL hub — cannot be optional | Critical |
| Resource accounting with fractional GPU actors breaks simple requests/limits | High |
| ZeRO all-reduces + vLLM TP + NCCL broadcast need stable process groups | High |
| KubeRay required for multi-node; single Trainer pod insufficient | High |
| Documentation centered on Slurm, not KFT CRDs | Medium |
| Feature set far exceeds Phase 1 needs | Medium |

**Recommendation:** For Phase 1, document OpenRLHF's limits only. Do not attempt full integration — the Ray bootstrapping inside KFT pods is fragile and unsupported.

---

## References

- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [Architecture](https://openrlhf.readthedocs.io/en/latest/architecture.html)
- [Hybrid Engine](https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html)
- [Agent training guide](https://openrlhf.readthedocs.io/en/latest/agent_training.html)
- [Multi-node training](https://openrlhf.readthedocs.io/en/latest/multi-node.html)
- [Paper (arXiv:2405.11143)](https://arxiv.org/abs/2405.11143)
