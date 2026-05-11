# GRPO Algorithm Support: Kubeflow Trainer vs Ray

> **Purpose:** Determine whether Kubeflow Trainer can serve as an alternative to Ray for GRPO fine-tuning.  
> **Decision:** Ray is already available. The question is whether Trainer can be the complementary/alternative path for the fine-tuning step of the Agentic Continual Learning PoC.

---

## TL;DR

**Neither KFT nor Ray "owns" GRPO as a built-in algorithm.** Both are orchestration layers. On Ray, three frameworks support GRPO (TRL, veRL, OpenRLHF). On KFT without Ray, **only TRL works** — veRL and OpenRLHF have Ray as a mandatory dependency. TRL's `GRPOTrainer` and `AsyncGRPOTrainer` use pure PyTorch DDP/FSDP with no Ray, making it the sole viable path for proving Trainer as an alternative.

**There IS enough discovery area for a comparison ticket** — scoped to validating whether KFT + TRL (no Ray) can match the RL capabilities that Ray provides.

---

## Kubeflow Trainer: Current State

### Built-in Runtimes (from `manifests/base/runtimes/`)

```
torch_distributed.yaml
deepspeed_distributed.yaml
mlx_distributed.yaml
jax_distributed.yaml
xgboost_distributed.yaml
torchtune/
```

**No RL, GRPO, PPO, RLHF, or reward-related runtime exists.**

### How GRPO Runs on KFT Today

`CustomTrainer` + TRL `GRPOTrainer` inside the training function. KFT provides:
- Pod scheduling, MASTER_ADDR/WORLD_SIZE injection, gang scheduling (Kueue)
- JobSet lifecycle management
- PVC mounts for checkpoints/cache

KFT does NOT provide:
- RL-specific worker roles (rollout server, reward model)
- Async training loop support
- vLLM sidecar or weight sync integration
- GRPO-specific metrics or configuration

### Upstream Roadmap

| Issue | What | Status |
|-------|------|--------|
| [KEP-2839](https://github.com/kubeflow/trainer/issues/2839) | Dynamic LLM Trainer Framework — TRL as pluggable backend for DPO/PPO/GRPO | Open, GSoC 2026, design phase |
| [PR #3263](https://github.com/kubeflow/trainer/pull/3263) | KEP doc proposing `TRLTrainer` with SFT/DPO/KTO/GRPO support | Closed (redirected to GSoC mentorship) |
| [Issue #3317](https://github.com/kubeflow/trainer/issues/3317) | Unsloth backend (related to #2839) | Open |
| [Issue #2752](https://github.com/kubeflow/trainer/issues/2752) | Future of LLM Trainer V2 | Open |

**Key quote from PR #3263:** "TorchTune does not support DPO/KTO/GRPO" — this is the explicit motivation for adding TRL backend support to KFT.

**Timeline estimate:** KEP-2839 is in proposal/GSoC stage. Don't expect a merged TRL/GRPO runtime before late 2026 at the earliest.

---

## Ray Ecosystem: Current State

### RLlib

**Does NOT support GRPO.** RLlib is designed for classical RL (Atari, robotics, action spaces). Published algorithms: PPO, APPO, DQN/Rainbow, SAC, IMPALA, DreamerV3, BC, MARWIL. No plans to add LLM post-training algorithms.

### Ray Train + TRL

**Official GRPO example exists:** [RL Post-Training using HF TRL with GRPO](https://docs.ray.io/en/latest/train/examples/transformers/transformer_reinforcement_learning/README.html)

Pattern: `TorchTrainer` wraps a training function that builds `GRPOTrainer`/`GRPOConfig`, with `RayTrainReportCallback` and `prepare_trainer` for distributed setup.

Caveats:
- TRL's default `accuracy_reward` timeouts break under Ray (use `parsing_timeout=0`)
- Shared checkpoint path needed (NFS/S3)
- vLLM colocate vs server mode trade-offs

### Ray + veRL

**First-class GRPO support** via `algorithm.adv_estimator=grpo`. veRL uses Ray as its orchestration backend.

KubeRay doc ([verl on KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html)) walks through PPO on GSM8K — not GRPO specifically, but it's a config/script swap on the same infrastructure.

### Ray + OpenRLHF

**GRPO via `--algo.advantage.estimator group_norm`** on the Ray PPO training stack. Ray-native by design.

No first-party KubeRay + OpenRLHF GRPO recipe published. SkyPilot documents OpenRLHF deployment (K8s-capable).

### Ray's LLM RL Strategy

[RFC #54021](https://github.com/ray-project/ray/issues/54021) ("Improving Ray for Post-Training / RL for LLM Projects") explicitly:
- Lists veRL, NeMo-RL, OpenRLHF, ROLL, AReaL, SkyRL as key projects
- Focuses on platform improvements (actors, GPU transfer, observability)
- Does NOT plan to absorb GRPO into Ray itself

**Ray's story:** "We're the distributed runtime. GRPO lives in TRL/veRL/OpenRLHF. We make those run better."

---

## Head-to-Head Comparison

| Dimension | Kubeflow Trainer | Ray (incl. KubeRay) |
|-----------|------------------|---------------------|
| **GRPO algorithm** | Via CustomTrainer + TRL/veRL in container | Via Ray Train + TRL, or veRL/OpenRLHF natively |
| **Native RL runtime** | None (KEP-2839 in progress) | RLlib (classical RL only, not LLM GRPO) |
| **Multi-role RL jobs** | Custom TrainingRuntime + JobSet (you build it) | Actor model — natural fit for heterogeneous RL roles |
| **vLLM integration** | Manual (separate Deployment or colocated subprocess) | veRL/OpenRLHF have native vLLM rollout workers |
| **Async training loop** | Not supported natively | Ray actors naturally support async patterns |
| **Gang scheduling** | First-class via Kueue | KubeRay + Kueue (documented pattern) |
| **GPU quota/tenancy** | Kueue ClusterQueue/LocalQueue — tight K8s integration | KubeRay + Kueue or Ray autoscaler |
| **Ops model** | K8s-native CRDs, GitOps, standard kubectl | Ray cluster lifecycle + K8s (two control planes) |
| **Failure handling** | JobSet failurePolicy + Kueue retry | Ray fault tolerance + K8s pod restart |
| **Time to GRPO** | ~1 day (TRL + CustomTrainer) | ~1 day (Ray Train + TRL) or ~3 days (veRL) |
| **Production RL scale** | Needs custom engineering (multi-role JobSet) | veRL/OpenRLHF designed for this |

---

## Where Each Falls Short

### KFT Gaps for RL

1. **No async actor pattern** — RL needs concurrent rollout + training; KFT is batch-oriented
2. **Single-role TrainJob** — can't express trainer + vLLM + reward model in one resource
3. **No vLLM lifecycle management** — you manage inference servers yourself
4. **No RL-aware scheduling** — Kueue doesn't know about rollout vs training priority
5. **KEP-2839 is pre-alpha** — TRL integration is months away from landing

### Ray Gaps for RL on K8s

1. **Two schedulers** — Ray's internal scheduler can compete with K8s scheduler
2. **Ops complexity** — KubeRay adds a control plane; Ray head node, workers, autoscaler
3. **Resource accounting** — fractional GPU actors break simple requests/limits
4. **Networking** — Ray ports, NCCL, pod-to-pod — more surfaces to debug
5. **K8s policy compliance** — Ray clusters can sprawl outside standard K8s governance

---

## Recommendation: Trainer as Alternative to Ray

### What the PoC Will Prove

The PoC validates **Option A** — KFT + TRL (no Ray) as a complete GRPO fine-tuning path:

```
Phase 1: KFT TrainJob + TRL GRPOTrainer + PyTorch DDP/FSDP
          → Proves basic GRPO training works on KFT

Phase 2: KFT TrainJob + TRL AsyncGRPOTrainer + vLLM Deployment + NCCL weight sync
          → Proves production-grade RL loop works on KFT without Ray
```

### What Trainer Provides That Ray Doesn't

- Kubernetes-native CRDs (TrainJob, TrainingRuntime) — GitOps friendly
- Kueue integration — GPU quota, gang scheduling, priority, preemption
- JobSet lifecycle — failure policies, multi-role orchestration (for Phase 2)
- No second control plane — no KubeRay operator, no Ray head node, no Ray ports

### What Trainer Gives Up vs Ray

- Only one GRPO framework (TRL) instead of three (TRL + veRL + OpenRLHF)
- No native actor model for heterogeneous RL roles — you build it with JobSet/Deployments
- No built-in async actor communication — TRL's AsyncGRPOTrainer handles this at the framework level
- Upstream RL support is months away (KEP-2839 GSoC 2026)

### Is a Comparison Ticket Still Warranted?

**Yes** — but reframed. The ticket should validate:

1. **Phase 1 baseline:** Does TRL GRPO on KFT produce correct training (reward improves on GSM8K)?
2. **Phase 2 async:** Does TRL AsyncGRPO + vLLM on KFT match the speed/quality that Ray-based stacks achieve?
3. **Operational comparison:** What does KFT give us (Kueue, CRDs, simplicity) vs what we lose (veRL/OpenRLHF, actor model)?
4. **Recommendation:** Is KFT + TRL sufficient for the Agentic Continual Learning PoC, or do we need Ray?

---

## Key Upstream Links

| Resource | Link |
|----------|------|
| KFT KEP-2839 (TRL backend) | https://github.com/kubeflow/trainer/issues/2839 |
| KFT PR #3263 (GRPO motivation) | https://github.com/kubeflow/trainer/pull/3263 |
| Ray RFC #54021 (LLM RL improvements) | https://github.com/ray-project/ray/issues/54021 |
| Ray Train TRL GRPO example | https://docs.ray.io/en/latest/train/examples/transformers/transformer_reinforcement_learning/README.html |
| KubeRay veRL PPO walkthrough | https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html |
| Anyscale LLM RL page | https://www.anyscale.com/llm-rl |
| Google Cloud RL on GKE | https://cloud.google.com/blog/products/compute/run-high-scale-rl-for-llms-on-gke |
| Red Hat ET agentic RFT | https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning |
