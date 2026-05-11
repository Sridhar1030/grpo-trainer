# GRPO on Kubeflow Trainer — Executive Summary

> **Jira Epic:** RHOAIENG-61094  
> **Date:** 2026-05-11  
> **Status:** Research complete. Ready for Phase 1 execution.  
> **Core research question:** Can Kubeflow Trainer serve as an alternative to Ray for GRPO fine-tuning?

---

## One-Paragraph Summary

GRPO (Group Relative Policy Optimization) is a critic-free RL algorithm from DeepSeekMath that generates G completions per prompt, z-scores their rewards within the group, and applies PPO-clip loss — no value network needed. It's the most-copied pattern in post-R1 reasoning RL. **The core objective of this PoC is to prove that Kubeflow Trainer can be the alternative to Ray for running GRPO fine-tuning.** Ray is already available and supports GRPO via veRL, OpenRLHF, and Ray Train+TRL — but both veRL and OpenRLHF require Ray as a mandatory dependency. On KFT without Ray, **TRL is the only viable GRPO framework**, using PyTorch DDP/FSDP with no Ray anywhere in the stack. Phase 1 validates the TRL GRPOTrainer on KFT via `CustomTrainer` with GSM8K. Phase 2 adds TRL's `AsyncGRPOTrainer` with a separate vLLM Deployment and NCCL weight sync (PR #31943, merged Feb 2026) — still no Ray. KFT has no built-in RL runtime (KEP-2839 is tracking TRL backend integration but is still in GSoC/proposal stage), so `CustomTrainer` is the path.

---

## What Each Document Covers

### [01 — GRPO Algorithm](01-grpo-algorithm.md)
- Introduced in DeepSeekMath (arXiv:2402.03300), applied at scale in DeepSeek-R1
- Core math: `A_i = (r_i - mean) / std` within each prompt's group of G completions
- PPO-clip loss with group-relative advantages + optional KL penalty
- **No critic network** → simpler and more memory-efficient than PPO
- GRPO vs PPO vs REINFORCE++ vs DPO comparison table
- Rollout generation is the main bottleneck in the RL loop

### [02 — TRL GRPOTrainer](02-trl-grpo-trainer.md)
- Full API reference: `GRPOConfig` with `num_generations`, `loss_type` (dapo/dr_grpo/grpo), `scale_rewards`, `beta` (KL)
- `beta=0` by default → no reference model loaded → saves ~50% GPU memory
- vLLM integration: `use_vllm=True` with `vllm_mode="server"` or `"colocate"`
- `AsyncGRPOTrainer` for fully decoupled training + generation via NCCL weight sync
- Batch divisibility rule: `(world_size × batch × grad_accum) % num_generations == 0`
- **Key limitation:** HF `.generate()` is 3-10x slower than vLLM; blocks training loop

### [03 — veRL Framework](03-verl-framework.md) *(reference only — requires Ray)*
- ByteDance's HybridFlow design: separate control flow (controller) from computation flow (workers)
- GRPO via `algorithm.adv_estimator=grpo` + `rollout.n=8`
- Native vLLM rollout integration with configurable `gpu_memory_utilization`
- FSDP/FSDP2 + Megatron-LM training backends
- **Ruled out:** Ray is mandatory for RL controller. Standalone mode is RFC-stage only ([#1221](https://github.com/verl-project/verl/issues/1221))

### [04 — OpenRLHF](04-openrlhf.md) *(reference only — requires Ray)*
- Ray + vLLM + DeepSpeed stack; designed for 70B+ models
- GRPO via `--algo.advantage.estimator group_norm` (shared PPO/Ray training stack)
- Hybrid Engine: colocate all roles + time-slice with sleep modes
- **Ruled out:** Ray is mandatory by design. No alternative execution path exists.

### [05 — Kubeflow Trainer v2](05-kubeflow-trainer-v2.md)
- Single `TrainJob` CRD + `TrainingRuntime` blueprints, backed by JobSet
- `CustomTrainer`: serialize Python function with cloudpickle, execute on distributed pods
- Built-in runtimes: torch-distributed, deepspeed, jax, mlx, xgboost, torchtune — **no RL runtime**
- PVCs/secrets via `RuntimePatch` → `PodSpecPatch` (verbose but flexible)
- Kueue integration for GPU quota and gang scheduling
- **Key limitation:** Single-role TrainJob can't natively co-locate trainer + vLLM; need raw JobSet or separate Deployment

### [06 — vLLM Weight Sync](06-vllm-weight-sync.md)
- PR #31943 merged Feb 5, 2026 — native NCCL weight transfer APIs
- Four-phase protocol: `init_weight_transfer_engine` → `start_weight_update` → `update_weights` → `finish_weight_update`
- `pause`/`resume` endpoints to quiesce generation during sync
- Requires `VLLM_SERVER_DEV_MODE=1` + `--weight-transfer-config '{"backend":"nccl"}'`
- Complete HTTP+NCCL code example included
- **Key for Phase 2:** This eliminates pod restarts between training steps

### [07 — GSM8K & Rewards](07-gsm8k-rewards.md)
- GSM8K: 7,473 train / 1,319 test math word problems with `#### N` answer format
- Rule-based correctness reward: extract number after `####`, compare to ground truth
- Format rewards, partial credit, multi-reward composition strategies
- Hyperparameter guidance: G=8-16, LR=1e-6 to 1e-5, beta=0-0.1, temp=0.1-0.7
- veRL built-in `compute_score()` function for GSM8K
- Alternative datasets: MATH, AIME, code generation, formal proofs

### [08 — Kubernetes RL Patterns](08-k8s-rl-patterns.md)
- No dedicated Kubernetes RL Working Group — relevant groups are WG Batch, JobSet, Kueue
- Two-role JobSet YAML example: trainer pods + vLLM inference pods
- NCCL on K8s: Socket (TCP) vs IB (InfiniBand/RoCE), key env vars, timeout troubleshooting
- GPU scheduling: MIG, time-slicing, disaggregated (separate pods) is most K8s-idiomatic
- Shared PVC patterns: RWX with NFS/CephFS/cloud managed storage
- KubeRay vs KFT: coexist in same cluster; Ray for Ray-native apps, KFT for K8s-native batch

### [09 — Phase 1 Guide](09-phase1-guide.md)
- Phase 1 = TRL GRPOTrainer + GSM8K + CustomTrainer on KFT (no vLLM, no async sync, **no Ray**)
- Complete training function and KFT submission code included
- Phase 2 = TRL AsyncGRPOTrainer + vLLM Deployment + NCCL weight sync (still no Ray)
- veRL and OpenRLHF documented as reference only (Ray dependency rules them out)

### [10 — Questions Answered](10-questions-answered.md)
- **Q1:** Deploy LLM alongside KFT → TrainJob + Deployment (Option A) or multi-role JobSet (Option B)
- **Q2:** vLLM async weight update → Yes, PR #31943 merged, three APIs documented
- **Q3:** TRL limitations → 6 items: slow rollouts, double memory, no async, batch constraints, NCCL timeouts, colocate OOM
- **Q4:** Single TrainJob for GRPO → Partially: single process (simple), colocated vLLM (PoC), or multi-role JobSet (production)
- **Q5-Q8:** Phase 1 specific: yes GRPO works on KFT, gaps are speed/memory/vLLM, four prerequisites for Phase 2, 4-week validation path

---

## Key Numbers

| Metric | Value |
|--------|-------|
| GSM8K train size | 7,473 problems |
| TRL default group size G | 8 completions per prompt |
| vLLM speedup over HF .generate() | ~3-10x (batch-dependent) |
| veRL async speedup (reported) | ~2.35-2.67x end-to-end |
| vLLM weight sync PR merge date | Feb 5, 2026 |
| TRL latest version | 1.4.0 |
| KFT built-in RL runtimes | 0 (CustomTrainer is the path) |

---

## Critical Decisions

1. **No Ray.** The objective is to prove Trainer as an alternative to Ray. veRL and OpenRLHF are ruled out (Ray mandatory). TRL is the sole GRPO framework.
2. **Phase 1:** TRL GRPOTrainer + CustomTrainer on KFT. Pure PyTorch DDP/FSDP distributed training.
3. **Phase 2:** TRL AsyncGRPOTrainer + FSDP2 + vLLM Deployment (separate K8s resource). NCCL weight sync via PR #31943. Still no Ray anywhere.
4. **KFT architecture:** TrainJob + separate vLLM Deployment for PoC; multi-role JobSet for production.
5. **Upstream alignment:** KEP-2839 (GSoC 2026) is adding TRL as a first-class KFT backend — our PoC validates the same direction ahead of upstream.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| TRL rollout speed too slow for meaningful training | High | Medium | Accept for Phase 1 baseline; add vLLM server in Phase 2 via AsyncGRPOTrainer |
| TRL AsyncGRPOTrainer + vLLM integration has rough edges | Medium | High | TRL is actively developed; fall back to manual vLLM HTTP calls + custom weight sync |
| NCCL timeouts during long generation | Medium | Medium | Increase NCCL_TIMEOUT; reduce max_completion_length |
| GPU memory insufficient for policy + reference | Medium | Medium | Use beta=0 (no reference model) or LoRA/PEFT |
| KFT CustomTrainer cloudpickle serialization fails for complex RL | Low | High | Use CustomTrainerContainer with stable image instead |
| TRL is the only option — single point of failure | Low | Medium | TRL is HuggingFace upstream with active development; KEP-2839 will add first-class KFT support |

---

## Upstream Tracking

| Item | Link | Status |
|------|------|--------|
| KFT RL/TRL backend | [KEP-2839](https://github.com/kubeflow/trainer/issues/2839) | Open, GSoC 2026 |
| KFT TRL integration PR | [PR #3263](https://github.com/kubeflow/trainer/pull/3263) | Closed (design redirected) |
| Ray LLM RL improvements | [RFC #54021](https://github.com/ray-project/ray/issues/54021) | Open, active |
| vLLM weight transfer | [PR #31943](https://github.com/vllm-project/vllm/pull/31943) | Merged |
| veRL TorchRPC (non-Ray) | [Issue #1221](https://github.com/verl-project/verl/issues/1221) | RFC stage |
