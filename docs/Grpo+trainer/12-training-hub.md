# Training Hub — Red Hat AI Innovation Team

> **Repo:** https://github.com/Red-Hat-AI-Innovation-Team/training_hub  
> **Branch:** `lora-grpo` (LoRA + GRPO feature)  
> **Latest Release:** v0.8.0 (Apr 25, 2026) — "Adapter-Based RLVR With Verl and OpenPipe ART"  
> **Intro Article:** https://developers.redhat.com/articles/2025/11/19/get-started-language-model-post-training-using-training-hub

---

## Overview

Training Hub is an **algorithm-focused Python interface** for LLM training, continual learning, and reinforcement learning — developed by the **Red Hat AI Innovation Team**. It wraps multiple backends behind clean one-function-call APIs.

```python
from training_hub import lora_grpo

result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./tool_call_traces.jsonl",
    ckpt_output_dir="./grpo_output",
    num_iterations=15,
    group_size=8,
    lora_r=32,
)
```

---

## Support Matrix

| Algorithm | Backend(s) | Status |
|-----------|-----------|--------|
| SFT | InstructLab-Training | Implemented |
| Continual Learning (OSFT) | RHAI Mini-Trainer, PEFT | Implemented |
| LoRA + SFT | Unsloth | Implemented |
| **LoRA + GRPO** | **ART (single-GPU), veRL (multi-GPU)** | **Implemented** |
| DPO | veRL | Planned |

---

## LoRA + GRPO: Two Backends

### Backend 1: ART (OpenPipe ART + Unsloth)

- **Single-GPU** only
- Co-located vLLM + Unsloth GRPO with **time-sharing**: vLLM generates rollouts → sleeps → Unsloth trains LoRA adapter → vLLM wakes for next rollout
- **No Ray** — pure single-process
- Fast iteration, low infrastructure complexity
- Best for: 4B–8B models on one GPU

```python
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    backend="art",           # single GPU, no Ray
    num_iterations=15,
    group_size=8,
    lora_r=32,
)
```

### Backend 2: veRL

- **Multi-GPU** distributed training
- FSDP for LoRA training + vLLM for rollout generation
- **Requires Ray** — veRL uses Ray for orchestration
- Scales to 70B+ models across multiple GPUs

```python
result = lora_grpo(
    model_path="Qwen/Qwen3-4B",
    data_path="./traces.jsonl",
    ckpt_output_dir="./output",
    backend="verl",          # multi-GPU, uses Ray
    n_gpus=4,
    num_iterations=3,
    group_size=4,
)
```

---

## Key Features

### LoRA-Only Training
Trains LoRA adapters, not full models. Memory-efficient, allows adapter composition.

### Multi-Turn Tool-Call Decomposition
Multi-turn traces are automatically broken into per-turn training samples:
- 5-turn conversation → 5 independent samples, each with ground-truth context prefix
- Reward: 1.0 = correct tool name + args, 0.5 = correct name, 0.0 = wrong

### Custom Reward Functions
```python
def my_reward(response, expected_name, expected_args):
    return 1.0  # your logic

result = lora_grpo(..., reward_fn=my_reward)
```

### Custom Rollout Functions
```python
async def my_rollout(model, task):
    client = model.openai_client()
    # ... your agentic loop ...
    trajectory = art.Trajectory(messages_and_choices=[...])
    trajectory.reward = compute_reward(...)
    return trajectory

result = lora_grpo(..., rollout_fn=my_rollout, tasks=my_task_list)
```

### Math Reward Support (veRL backend)
The veRL backend includes built-in math answer verification using `math_verify`:
- Extracts `\boxed{}` answers from model output
- Semantic comparison (handles LaTeX equivalence like `\frac{1}{2}` == `0.5`)
- Auto-detects dataset type from filename (GSM8K, MATH, AIME)

---

## Architecture Deep Dive

### ART Backend Internals

```
Single GPU
├── vLLM AsyncLLM engine (rollout generation)
│   └── PagedAttention, continuous batching
│   └── Sleeps during training phase
├── Unsloth LoRA trainer (GRPO gradient updates)
│   └── Wakes after vLLM sleeps
│   └── Memory-efficient LoRA training
└── Time-sharing: vLLM ↔ Unsloth alternate on same GPU
```

### veRL Backend Internals

```
Multi-GPU (via Ray)
├── Ray orchestrates worker placement
├── FSDP shards LoRA parameters across GPUs
├── vLLM generates rollouts in parallel
├── Data prepared as Parquet → verl expects this format
└── Reward function written to temp file, loaded by verl workers
```

The veRL backend:
1. Converts data to Parquet format (`prompt`, `reward_model`, `data_source` columns)
2. Writes a reward function to a temp Python file
3. Launches veRL via `subprocess` with Ray
4. veRL handles FSDP + vLLM + GRPO internally

---

## Relevance to KFT GRPO PoC

### Can We Use Training Hub?

**Partially — with important caveats.**

| Aspect | ART Backend | veRL Backend |
|--------|-------------|-------------|
| **Ray dependency** | No Ray | **Requires Ray** |
| **Multi-node** | No (single GPU only) | Yes (via Ray) |
| **KFT compatible** | Could run in CustomTrainer (single GPU) | Same Ray problem as raw veRL |
| **LoRA only** | Yes — no full fine-tuning | Yes — LoRA only |
| **Math reward** | Custom function | Built-in `math_verify` |
| **Useful for PoC?** | Limited — single GPU, LoRA only | Not without Ray |

### What's Useful

1. **Algorithm design patterns**: The `Algorithm` / `Backend` / `AlgorithmRegistry` abstraction is a clean pattern our `trainer-rl` framework sprint could adopt
2. **Reward function implementations**: `tool_call_reward`, `binary_reward`, and the math verification code are reusable
3. **Data loading utilities**: Multi-turn decomposition, HuggingFace dataset loading, Parquet conversion
4. **Proof of concept from our own org**: Red Hat team already built GRPO training — validates the approach

### What Doesn't Fit

1. **LoRA only**: Our PoC wants full GRPO (full model training), not just LoRA adapters
2. **ART is single-GPU**: Can't do distributed training on KFT
3. **veRL backend uses Ray**: Same limitation we've already documented
4. **No KFT integration**: Training Hub runs locally or via Ray, not via TrainJob/JobSet
5. **Tool-call focused**: Primary use case is tool-calling agents, not math reasoning (though math is supported in veRL backend)

### Potential Integration Path

If we build the `trainer-rl` framework, Training Hub could become a **user-facing API layer** on top:

```
training_hub.lora_grpo(backend="kft")    # NEW backend
    └── trainer-rl SDK
        └── KFT TrainJob / JobSet
            └── FSDP + vLLM + NCCL weight sync
```

This would add a `KFTLoRAGRPOBackend` to Training Hub's registry, giving users the same `lora_grpo()` API but running on Kubernetes instead of Ray.

---

## Code Architecture

```
src/training_hub/
├── __init__.py                    # Exports: sft, osft, lora_sft, lora_grpo
├── algorithms/
│   ├── __init__.py                # Algorithm, Backend, AlgorithmRegistry base classes
│   ├── sft.py                     # SFT with InstructLab-Training
│   ├── osft.py                    # Orthogonal Subspace Fine-Tuning
│   ├── lora.py                    # LoRA + SFT with Unsloth
│   ├── lora_grpo.py               # LoRA + GRPO with ART backend (~1500 lines)
│   ├── lora_grpo_verl.py          # LoRA + GRPO with veRL backend (~1100 lines)
│   └── rewards.py                 # tool_call_reward, binary_reward
├── profiling/
│   └── memory_estimator.py        # GPU memory estimation
└── visualization.py               # plot_loss
```

---

## Key Takeaways

1. **Red Hat already has GRPO training** — Training Hub proves the approach works at our org
2. **Two backends, same API** — clean abstraction pattern worth borrowing
3. **ART (no Ray) is single-GPU only** — can't meet our multi-node KFT requirement
4. **veRL (multi-GPU) requires Ray** — same blocker we've identified everywhere
5. **The gap is exactly what we're building**: a multi-GPU GRPO backend that runs on KFT without Ray
6. **Potential collaboration**: Our KFT work could become a new Training Hub backend

---

## References

- [Training Hub GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)
- [LoRA + GRPO docs](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/lora-grpo/docs/algorithms/lora_grpo.md)
- [OpenPipe ART](https://github.com/OpenPipe/ART)
- [veRL](https://github.com/volcengine/verl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Intro blog post](https://developers.redhat.com/articles/2025/11/19/get-started-language-model-post-training-using-training-hub)
