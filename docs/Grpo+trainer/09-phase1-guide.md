# Phase 1 Implementation Guide

> **Goal:** Prove Kubeflow Trainer as an alternative to Ray for GRPO fine-tuning  
> **Dataset:** GSM8K  
> **Framework:** TRL (only viable option without Ray — veRL and OpenRLHF require Ray by design)  
> **Constraint:** No Ray anywhere in the stack

---

## Phase 1 Scope

### What Phase 1 IS

```
Python SDK
    │
    ▼
KFT TrainJob (CustomTrainer)
    │
    ▼
N × Worker Pods
    │  each runs the full GRPO loop:
    │  generate rollouts → score with rule-based reward → GRPO advantage → update policy
    │
    ▼
AllReduce (NCCL) gradient sync
    │
    ▼
Checkpoint saved to PVC
```

### What Phase 1 is NOT (deferred to Phase 2)

| Deferred Feature | Why |
|---|---|
| vLLM for rollout generation | Requires multi-role JobSet or separate Deployment |
| Async weight sync (vLLM PR #31943) | Depends on deployed vLLM instance |
| Learned reward model | Requires separate reward model server |
| Multi-node ZeRO-3 at scale | Phase 1 validates correctness, not scale |

---

## Prerequisites

### Cluster Requirements

```bash
kubectl get crd trainjobs.trainer.kubeflow.org
kubectl get clustertrainingruntime torch-distributed
kubectl get localqueue -n kubeflow
```

### Shared PVC

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grpo-poc-cache
  namespace: kubeflow
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 200Gi
  storageClassName: nfs
```

### HuggingFace Token Secret

```bash
kubectl create secret generic hf-credentials \
  --from-literal=HF_TOKEN=hf_xxxxxxxxxxxx \
  -n kubeflow
```

---

## Implementation 1: TRL (Baseline)

### Why TRL First

- Zero new infrastructure — works with KFT's CustomTrainer out of the box
- Establishes reward baseline and training speed benchmark
- Answers: does GRPO run on KFT at all?

### Training Function

```python
def grpo_trl_train():
    import os, re, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir="/cache/huggingface",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/cache/huggingface")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw = load_dataset("gsm8k", "main", split="train")

    def format_prompt(example):
        return {
            "prompt": f"Solve step by step.\n\nProblem: {example['question']}\n\nSolution:",
            "answer": example["answer"],
        }

    dataset = raw.map(format_prompt)

    def gsm8k_reward(completions, prompts, answer, **kwargs):
        rewards = []
        for completion, ref_answer in zip(completions, answer):
            ref_match = re.search(r'####\s*([\d,]+)', ref_answer)
            ref_num = ref_match.group(1).replace(',', '') if ref_match else None
            pred_match = re.search(r'####\s*([\d,]+)', completion)
            pred_num = pred_match.group(1).replace(',', '') if pred_match else None
            if ref_num and pred_num and ref_num == pred_num:
                rewards.append(1.0)
            elif pred_match:
                rewards.append(0.2)
            else:
                rewards.append(0.0)
        return rewards

    config = GRPOConfig(
        output_dir="/checkpoints/grpo-trl-gsm8k",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_completion_length=512,
        learning_rate=5e-7,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(
        model=model, args=config,
        reward_funcs=gsm8k_reward,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    if RANK == 0:
        trainer.save_model("/checkpoints/grpo-trl-gsm8k/final")
        tokenizer.save_pretrained("/checkpoints/grpo-trl-gsm8k/final")
```

### Submit to KFT

```python
from kubeflow.trainer import TrainerClient
from kubeflow.trainer.types import CustomTrainer

client = TrainerClient()
job = client.train(
    trainer=CustomTrainer(
        func=grpo_trl_train,
        num_nodes=2,
        resources_per_node={"cpu": "16", "memory": "64Gi", "nvidia.com/gpu": "2"},
        env={
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_DEBUG": "WARN",
        },
    ),
    name="grpo-trl-phase1",
    namespace="kubeflow",
)
```

**Note on PVC mounting:** Use `options=[RuntimePatch(...)]` as documented in `05-kubeflow-trainer-v2.md`.

### TRL Limitations Found

| Limitation | Severity |
|---|---|
| HF `.generate()` — no paged attention or batching | High |
| Policy + reference model both on GPU (when beta > 0) | High |
| `num_generations` constraint: `(G × batch) % world_size == 0` | Medium |
| No async rollouts — trainer blocks during generation | Medium |
| NCCL timeout risk during long generation | Medium |

---

## Why Not veRL or OpenRLHF?

Both were researched extensively (see [03-verl-framework.md](03-verl-framework.md) and [04-openrlhf.md](04-openrlhf.md)) but **ruled out** because the objective is Trainer as an alternative to Ray:

| Framework | GRPO support | Ray dependency | Verdict |
|-----------|-------------|----------------|---------|
| **TRL** | `GRPOTrainer` + `AsyncGRPOTrainer` | None — pure PyTorch DDP/FSDP | **Use this** |
| **veRL** | First-class `adv_estimator=grpo` | Mandatory for RL controller | Ruled out |
| **OpenRLHF** | `group_norm` advantage estimator | Mandatory by design | Ruled out |

veRL has an experimental TorchRPC RFC ([#1221](https://github.com/verl-project/verl/issues/1221)) for non-Ray support, but it's not production-ready. OpenRLHF has no alternative execution path at all.

---

## Recommended Execution Sequence

```
Phase 1 (Week 1-2): TRL GRPOTrainer on KFT
    KFT TrainJob (CustomTrainer)
        └── TRL GRPOTrainer + PyTorch DDP/FSDP
            └── HF .generate() for rollouts
            └── GSM8K rule-based rewards
    → Proves GRPO training loop works on KFT without Ray
    → Establishes reward baseline and training speed benchmark

Phase 2 (Week 3-4): TRL AsyncGRPOTrainer + vLLM
    KFT TrainJob (CustomTrainer)
        └── TRL AsyncGRPOTrainer + FSDP2
            └── Calls vLLM server for fast rollouts
            └── NCCL weight sync (PR #31943)
    +
    vLLM Deployment (separate K8s resource)
        └── Serves rollouts via OpenAI API
        └── Receives weight updates via NCCL
    → Proves full RL loop with fast rollouts, no Ray
    → Measures speedup vs Phase 1 baseline
```

---

## Success Criteria

### Phase 1: TRL on KFT

- [ ] TrainJob reaches `Complete` status
- [ ] Reward on GSM8K improves over training steps (even slightly)
- [ ] Checkpoint saved to PVC
- [ ] No NCCL timeout errors
- [ ] Documented: what KFT provides vs what you had to build yourself

### Phase 2: TRL AsyncGRPO + vLLM on KFT

- [ ] vLLM Deployment serves rollouts successfully
- [ ] NCCL weight sync from trainer to vLLM works
- [ ] Rollout tokens/sec measurably higher than Phase 1
- [ ] Same or better reward trajectory
- [ ] Full RL loop runs without pod restarts between training steps
- [ ] Documented: Trainer viability as Ray alternative for GRPO fine-tuning
