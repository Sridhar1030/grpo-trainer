# GSM8K Dataset & Reward Function Design

---

## GSM8K Dataset

### What Is It

**GSM8K** (Grade School Math 8K) — ~8.5k English grade-school math word problems requiring 2-8 step reasoning with elementary arithmetic. Introduced in Cobbe et al., 2021 ([arXiv:2110.14168](https://arxiv.org/abs/2110.14168)). MIT license.

### Structure

| Item | Detail |
|------|--------|
| **Dataset id** | `openai/gsm8k` |
| **Configs** | `main` (standard), `socratic` (sub-questions prepended) |
| **Columns** | `question` (str), `answer` (str) |
| **Splits** | `train` (7,473 rows), `test` (1,319 rows) |

### Answer Format

The `answer` field contains a **full natural-language solution** with calculator annotations:

```
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

The final answer is on the last line after `####`.

### Loading

```python
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")
train = ds["train"]   # 7473 rows
test = ds["test"]     # 1319 rows
```

### Why GSM8K Fits GRPO

- **Verifiable reward**: numeric equality check, no learned reward model needed
- **Clear extraction target**: `####` delimiter gives stable parsing contract
- **Standard benchmark**: widely reported GSM8K accuracy for before/after comparison
- **Group-relative signal**: binary correctness + per-prompt normalization works well

---

## Reward Function Design

### Baseline: Rule-Based Correctness

```python
import re
from typing import Optional

GSM8K_FINAL_RE = re.compile(r"####\s*([^\n]+)\s*$", re.MULTILINE)

def extract_gsm8k_answer(text: str) -> Optional[str]:
    m = GSM8K_FINAL_RE.search(text.rstrip())
    return m.group(1).strip() if m else None

def parse_number(s: str) -> Optional[float]:
    s = s.replace(",", "").strip()
    s = re.sub(r"^\$|\$$", "", s)
    s = s.rstrip(".")
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None

def correctness_reward(completion: str, gold_solution: str) -> float:
    gold = extract_gsm8k_answer(gold_solution)
    pred = extract_gsm8k_answer(completion)
    if gold is None or pred is None:
        return 0.0
    g, p = parse_number(gold), parse_number(pred)
    if g is None or p is None:
        return 0.0
    return 1.0 if abs(g - p) < 1e-6 else 0.0
```

### Partial Credit

| Strategy | Role | Risk |
|----------|------|------|
| Binary {0, 1} | Default for GRPO; clean semantics | Sparse signal on weak models |
| Small bonus for "any number after ####" | Reduces parse failure early | Reward hacking (nonsense + random number) |
| Proximity (within 10% of gold) | Shapes numerical closeness | Encourages wrong method |

### Format Rewards

```python
FORMAT_OK = re.compile(r"####\s*[-+]?[\d.,$/]+\s*$", re.MULTILINE | re.IGNORECASE)

def format_reward(text: str) -> float:
    return 1.0 if FORMAT_OK.search(text.rstrip()) else 0.0
```

Scale format components so they do NOT dominate correctness.

### Multi-Reward Composition

```
r = λ_fmt × r_fmt + λ_cor × r_cor + λ_aux × r_aux
```

- `r_cor`: binary/strict match (primary, weight ≫ others)
- `r_fmt`: `####` present/parseable
- `r_aux`: optional (length band, "≥N steps", calculator-tag usage)

TRL supports multiple reward callables; each logs `reward/{name}/mean`.

### veRL Built-in GSM8K Reward

```python
from verl.utils.reward_score.gsm8k import compute_score

r = compute_score(
    solution_str=model_output,
    ground_truth="42",
    method="strict",       # regex on last 300 chars for #### <number>
    format_score=0.0,
    score=1.0,
)
```

---

## TRL Dataset Format for GRPO

```python
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main", split="train")

def to_prompt(example):
    return {
        "prompt": example["question"],        # required column
        "ground_truth": example["answer"],     # passed to reward as kwarg
    }

dataset = ds.map(to_prompt)
```

For conversational models, build `prompt` as list of messages (system/user).

---

## Hyperparameter Guidance

### From Published GRPO Work

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Group size G | 8-16 | TRL default 8; DeepSeekMath uses 8 |
| Learning rate | 1e-6 to 1e-5 | Conservative; higher = collapse risk |
| KL beta | 0 to 0.1 | TRL defaults 0; re-enable if entropy blows up |
| Temperature | 0.1-0.7 | Lower for stable math RL; higher for exploration |
| max_completion_length | 512-1024 | GSM8K rarely needs >512 |

### Model Sizes

| Scale | Notes |
|-------|-------|
| ~0.5B-1.5B | Fast iteration, method development; may need SFT warmup |
| ~3B | Balanced for GRPO + LoRA + quantization demos |
| ~7B | Strong math results; GSM8K often saturated earlier than MATH |

### Example Config (Illustrative)

From a recent ablation study (Qwen2.5-0.5B-Instruct):
- LR: 5e-6, KL β: 0.1, G: 16
- Per-device batch: 16, grad accum: 4
- Context: 200 tokens, temperature: 0.1
- GSM8K test Pass@1: ~45-50%

---

## Alternative Datasets for GRPO

| Dataset | Verifiable Signal | Notes |
|---------|-------------------|-------|
| **MATH** (Hendrycks) | Extract `\boxed{...}`, symbolic check | Harder than GSM8K; community mirrors available |
| **AIME / competition** | Integer [0, 999], exact match | Often held-out eval |
| **Code generation** | Unit tests, stdin/stdout | APPS, CodeContests, MBPP |
| **Formal math** | Proof checkers (Lean/Coq) | Strongest verdict; hardest infra |
| **Multiple-choice** | Letter match | Cheap but easier to game |

---

## Reward Engineering Best Practices

### Avoiding Reward Hacking

- Keep correctness ≥80-95% of total return once policy can format outputs
- Don't pay for spurious patterns (repeating `####`, stuffing gold numbers, huge outputs)
- Audit rollouts: log truncation rate, EOS rate, format failure rate
- Use held-out verifiers: eval with strict correctness even if training has format rewards

### KL Penalty Tuning

- **Too low/zero**: faster exploration but collapse/reward hacking risk
- **Too high**: under-learning; advantages shrink toward 0
- **Practice**: start with small positive β for instruction-tuned models; decay if stalled

### Group Normalization Considerations

- When all rewards identical, std = 0 → mask zero-variance groups
- Consider `scale_rewards="batch"` in TRL if per-group std injects difficulty bias
- Watch `loss_type` variants for long CoT (DAPO, DrGRPO)

### Common Pitfalls

- Mismatched parsing between training and eval (different regex → false negatives)
- Length bias in loss: use documented `loss_type` choices
- Partial credit contradicting true objective
- Contaminated test labels in synthetic corpora
- Numeric tolerance: integers vs floats vs fractions — pick one canonical normalization

---

## References

- Cobbe et al., *Training Verifiers to Solve Math Word Problems*, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- [GSM8K on HuggingFace](https://huggingface.co/datasets/openai/gsm8k)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [HF Cookbook: Advanced GRPO multi-reward](https://huggingface.co/learn/cookbook/trl_grpo_reasoning_advanced_reward)
- [RL with Verifiable Rewards analysis](https://arxiv.org/pdf/2503.06639)
