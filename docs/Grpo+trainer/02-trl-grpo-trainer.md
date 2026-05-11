# TRL GRPOTrainer — Deep Reference

> **Source of truth:** [`GRPOConfig`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py), [`GRPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)  
> **Docs:** https://huggingface.co/docs/trl/main/en/grpo_trainer  
> **PyPI:** `trl` (latest: 1.4.0 at time of research)

---

## Constructor Signature

```python
GRPOTrainer(
    model: str | PreTrainedModel | PeftModel,
    reward_funcs: RewardFunc | list[RewardFunc],
    args: GRPOConfig | None = None,
    train_dataset: Dataset | IterableDataset | None = None,
    eval_dataset: Dataset | IterableDataset | dict | None = None,
    processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
    reward_processing_classes: PreTrainedTokenizerBase | list | None = None,
    callbacks: list[TrainerCallback] | None = None,
    optimizers: tuple[Optimizer | None, LambdaLR | None] = (None, None),
    peft_config: PeftConfig | None = None,
    tools: list[Callable] | None = None,
    rollout_func: RolloutFunc | None = None,        # experimental
    environment_factory: EnvironmentFactory | None = None,  # experimental
)
```

### Key Parameters

- **`reward_funcs`**: `str` (HF model id), `PreTrainedModel` (num_labels=1), callable, or list mixing types. Callables may be sync or async; multiple async rewards run concurrently. May return `None` per sample for multi-task setups.
- **`processing_class`**: Tokenizer/processor. Must use `padding_side="left"`. If no `pad_token`, `eos_token` is used.

---

## GRPOConfig — Key Parameters

### Group Size / Batch Logic

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_generations` | 8 | Completions per prompt. Must be ≥ 2 |
| `num_generations_eval` | None | Override for eval; falls back to `num_generations` |

### Lengths

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_completion_length` | 256 | Max new tokens per completion |

Note: `max_prompt_length` is NOT a field on current `GRPOConfig`.

### Generation Controls

| Parameter | Notes |
|-----------|-------|
| `generation_batch_size` | Mutually exclusive with `steps_per_generation` |
| `steps_per_generation` | Defaults to `gradient_accumulation_steps` |
| `temperature`, `top_p`, `top_k`, `min_p` | Sampling params |
| `repetition_penalty` | Standard HF generation penalty |

### KL / Reference Policy

| Parameter | Default | Notes |
|-----------|---------|-------|
| `beta` | **0.0** | KL coefficient. `0` → no reference model loaded (saves memory) |
| `sync_ref_model` | False | TR-DPO-style reference syncing (only when beta > 0) |
| `ref_model_mixup_alpha` | 0.9 | Reference model sync mixing |
| `ref_model_sync_steps` | 64 | How often to sync reference |

### Loss Variants

| `loss_type` | Description |
|-------------|-------------|
| `"grpo"` | Standard GRPO |
| `"dr_grpo"` | Dr. GRPO (no std scaling) |
| `"dapo"` | **Default.** DAPO (addresses length bias in long CoT) |
| `"bnpo"`, `"cispo"`, `"sapo"`, `"luspo"`, `"vespo"` | Other variants |

### Reward Scaling

| `scale_rewards` | Notes |
|-----------------|-------|
| `"group"` | Default. Per-prompt group normalization |
| `"batch"` | Batch-level std |
| `"none"` / `False` | No scaling |

### vLLM Integration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `use_vllm` | False | Enable vLLM backend |
| `vllm_mode` | `"colocate"` | `"server"` (HTTP) or `"colocate"` (in-process) |
| `vllm_gpu_memory_utilization` | 0.3 | For colocate mode |
| `vllm_tensor_parallel_size` | 1 | TP for vLLM |
| `vllm_max_model_length` | None | Context window |
| `vllm_enable_sleep_mode` | False | Offload during optimizer steps |
| `vllm_server_base_url` | None | For server mode |
| `vllm_importance_sampling_correction` | True | IS correction for train/infer mismatch |
| `vllm_importance_sampling_mode` | `"sequence_mask"` | Correction strategy |

### Distributed Training Defaults

- `gradient_checkpointing=True`, `bf16=True`, `learning_rate=1e-6`, `logging_steps=10`
- `ddp_find_unused_parameters=False`
- `ds3_gather_for_generation=True` (ZeRO-3 gather; False is incompatible with vLLM)

---

## Batch Size Divisibility Rules

From `GRPOConfig` validation:

1. `(world_size × per_device_train_batch_size × gradient_accumulation_steps) % num_generations == 0`
2. `generation_batch_size % num_generations == 0`
3. Eval: `(per_device_eval_batch_size × world_size) % num_generations_eval == 0`

**Common misconception:** The constraint is NOT `(num_generations × batch_size) % world_size == 0`. The checks are about **global batch sizes being divisible by num_generations**.

---

## Architecture: Internal Training Loop

1. **Sample prompts** from dataloader (per process)
2. **Generate** G completions per prompt (HF .generate() or vLLM)
3. **Reward** each completion (sum/weighted across `reward_funcs`)
4. **Advantages** — group-relative (mean/std within prompt's group)
5. **Policy loss** — GRPO/DAPO/Dr-GRPO variant per `loss_type`
6. **KL penalty** — only if `beta != 0`
7. **Optimizer step** — with optional `num_iterations` inner repeats

### Generation Paths

1. `use_vllm=True` → vLLM generation (+ periodic weight sync)
2. Paged/continuous batching → `unwrapped_model.generate_batch()`
3. Default → pad prompts → `model.generate()` → strip prompt tokens

### Memory Layout

- **Policy**: always resident (possibly sharded via DeepSpeed/FSDP)
- **Reference**: only if `beta != 0`. **beta=0 → no reference model → big VRAM savings**
- **vLLM colocate**: shares GPUs — tune `vllm_gpu_memory_utilization`, consider `vllm_enable_sleep_mode`

---

## TRL + vLLM Integration

### Server Mode (Recommended for Multi-Node)

```bash
# Launch vLLM server separately
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B \
    --max-model-len 4096 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}'
```

**Warning:** Sharing GPUs between server and trainer can cause NCCL errors. Use dedicated GPUs.

### Colocate Mode

vLLM engine runs in the same process as training. Risk of OOM — use `vllm_enable_sleep_mode`.

### Async GRPO (TRL)

`AsyncGRPOTrainer` decouples rollout (vLLM server) from training. After every `weight_sync_steps`, weights go to vLLM via NCCL. Requires:
- `vllm>=0.17.1`, `transformers>=5.2.0`
- FSDP2 only for distributed

---

## Dataset Format

- **`"prompt"`** column is mandatory — string or chat messages
- Any other columns (e.g. `ground_truth`, `answer`) are **passed through to reward callables as kwargs**
- `remove_unused_columns=False` by default so extra columns survive

### GSM8K Pattern

```python
from datasets import load_dataset

ds = load_dataset("gsm8k", "main", split="train")

def to_prompt(example):
    return {
        "prompt": example["question"],
        "ground_truth": example["answer"],
    }

dataset = ds.map(to_prompt)
```

---

## Known Limitations

| Limitation | Severity |
|------------|----------|
| HF `.generate()` is 3-10x slower than vLLM for rollouts | High |
| `beta > 0` loads reference model → extra GPU memory | High |
| Colocated vLLM competes with training memory → OOM risk | High |
| Long generation can cause NCCL timeouts | Medium |
| `num_generations` divisibility constraints limit batch flexibility | Medium |
| `rollout_func`, `environment_factory` are experimental | Low |

---

## Minimal Usage

```python
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

---

## References

- [GRPOConfig source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)
- [GRPOTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
- [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [vLLM integration docs](https://huggingface.co/docs/trl/main/en/vllm_integration)
- [Async GRPO docs](https://huggingface.co/docs/trl/async_grpo_trainer)
- [GRPO + vLLM cookbook](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
