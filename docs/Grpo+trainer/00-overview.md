# GRPO on Kubeflow Trainer — Knowledge Base

> **Jira Epic:** RHOAIENG-61094  
> **Goal:** Implement GRPO (Group Relative Policy Optimization) using Kubeflow Trainer, orchestrate RL jobs including vLLM instances, and leverage vLLM's native weight sync (NCCL) for rollout generation.  
> **Dataset:** GSM8K (math reasoning, verifiable rewards)  
> **Strategic context:** Feeds the fine-tuning component of the Agentic Continual Learning PoC

---

## Knowledge Base Index

| Document | Contents |
|----------|----------|
| [01-grpo-algorithm.md](01-grpo-algorithm.md) | GRPO fundamentals — math, papers, comparison with PPO/DPO/REINFORCE++ |
| [02-trl-grpo-trainer.md](02-trl-grpo-trainer.md) | HuggingFace TRL GRPOTrainer — API, config, vLLM integration, limitations |
| [03-verl-framework.md](03-verl-framework.md) | veRL (Volcengine) — architecture, GRPO config, vLLM integration, standalone mode |
| [04-openrlhf.md](04-openrlhf.md) | OpenRLHF — GRPO support, Ray architecture, Kubernetes compatibility |
| [05-kubeflow-trainer-v2.md](05-kubeflow-trainer-v2.md) | KFT v2 — TrainJob, CustomTrainer, TrainingRuntime, JobSet, Kueue, Python SDK |
| [06-vllm-weight-sync.md](06-vllm-weight-sync.md) | vLLM PR #31943 — weight transfer APIs, NCCL backend, integration patterns |
| [07-gsm8k-rewards.md](07-gsm8k-rewards.md) | GSM8K dataset, reward function design, hyperparameter guidance |
| [08-k8s-rl-patterns.md](08-k8s-rl-patterns.md) | Kubernetes RL architecture — JobSet, NCCL, GPU scheduling, KubeRay vs KFT |
| [09-phase1-guide.md](09-phase1-guide.md) | Phase 1 PoC implementation guide — TRL baseline, veRL, OpenRLHF comparison |
| [10-questions-answered.md](10-questions-answered.md) | Direct answers to every Jira question with evidence |

---

## Quick Context

### What is GRPO?
Group Relative Policy Optimization — a critic-free RL algorithm introduced in DeepSeekMath (arXiv:2402.03300). For each prompt, generate G completions, score with rewards, normalize advantages within the group as `(r_i - mean) / std`, then apply PPO-clip loss. No value network needed → simpler and more memory-efficient than PPO.

### Why KFT?
Kubeflow Trainer v2 provides Kubernetes-native distributed training orchestration. It handles `MASTER_ADDR`, `WORLD_SIZE`, gang scheduling via Kueue, and integrates with JobSet for multi-role workloads. GRPO runs via `CustomTrainer` since there's no built-in RL runtime.

### Why vLLM weight sync matters
The RL loop's bottleneck is rollout generation. vLLM's PagedAttention is 3-10x faster than HF `.generate()`. PR #31943 (merged Feb 5, 2026) adds native NCCL weight transfer APIs so the trainer can push updated weights to vLLM without pod restarts — the key enabler for a production RL loop.

### Phase 1 scope
Prove a minimal end-to-end GRPO baseline on KFT before adding vLLM / async weight sync. Use TRL GRPOTrainer with GSM8K, identify limitations, and produce guidance for Phase 2.

---

## Key References

| Resource | Link |
|----------|------|
| DeepSeekMath paper (GRPO origin) | https://arxiv.org/abs/2402.03300 |
| DeepSeek-R1 paper (GRPO at scale) | https://arxiv.org/abs/2501.12948 |
| TRL GRPOTrainer docs | https://huggingface.co/docs/trl/main/en/grpo_trainer |
| TRL Async GRPO | https://huggingface.co/docs/trl/async_grpo_trainer |
| veRL GitHub | https://github.com/volcengine/verl |
| veRL GRPO docs | https://verl.readthedocs.io/en/latest/algo/grpo.html |
| OpenRLHF GitHub | https://github.com/OpenRLHF/OpenRLHF |
| Kubeflow Trainer docs | https://www.kubeflow.org/docs/components/trainer/ |
| KFT Python SDK | https://sdk.kubeflow.org/en/latest/train/index.html |
| vLLM weight transfer | https://docs.vllm.ai/en/latest/training/weight_transfer/ |
| vLLM PR #31943 | https://github.com/vllm-project/vllm/pull/31943 |
| GSM8K dataset | https://huggingface.co/datasets/openai/gsm8k |
| JobSet docs | https://jobset.sigs.k8s.io/docs/overview/ |
| Kueue TrainJob | https://kueue.sigs.k8s.io/docs/tasks/run/trainjobs/ |
| Agentic RFT cookbook (Red Hat) | https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning |
