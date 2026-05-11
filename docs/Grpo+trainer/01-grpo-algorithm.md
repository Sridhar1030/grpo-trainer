# GRPO Algorithm — Fundamentals

## Origins

| Paper | Role | Link |
|-------|------|------|
| **DeepSeekMath** (Shao et al., 2024) | Introduces GRPO as a critic-free PPO-style algorithm for math RL | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| **DeepSeek-R1** (DeepSeek-AI, 2025) | Applies GRPO at scale for pure RL reasoning (R1-Zero) with rule-based rewards | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |

GRPO's **algorithmic definition** comes from DeepSeekMath. DeepSeek-R1 is the high-visibility **production** use case that drove broad open-source adoption.

---

## Mathematical Formulation

### Standard PPO (what GRPO replaces)

PPO maintains a **policy** (actor) π_θ and a **value function** (critic) V_φ. Advantages are estimated via GAE, then a clipped surrogate limits policy updates:

```
L_PPO ∝ E[ Σ_t min(r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t) ]
```

where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`.

**Problem:** The critic adds parameters, optimizer state, activations, and backward passes comparable to training another full model.

### GRPO Core Idea: Group-Relative Advantage (No Critic)

For each prompt `q`, sample a **group** of `G` completions `{o_1, ..., o_G}` from π_θ_old. Score each with a scalar reward `r_i`.

**Group-relative advantages** by z-scoring within the group:

```
A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})
```

**Intuition:** Reinforce completions that beat the local group average, suppress those below it — an implicit comparative baseline without learning V_φ.

**Engineering caveat:** When all `r_i` are identical, `std = 0`. Implementations add epsilon stabilization, skip, or fallback.

### Full GRPO Objective (DeepSeek-R1, Eq. 1)

```
J_GRPO(θ) = E_{q~P(Q), {o_i}~π_old(·|q)} [
  (1/G) Σ_{i=1}^G (
    min(π_θ(o_i|q)/π_old(o_i|q) · A_i, clip(...) · A_i)
    - β · D_KL(π_θ || π_ref)
  )
]
```

With a low-variance KL surrogate:
```
D_KL(π_θ || π_ref) ≈ π_ref(o|q)/π_θ(o|q) - log(π_ref(o|q)/π_θ(o|q)) - 1
```

### Token-Level Variant (TRL Implementation)

TRL's `GRPOTrainer` uses a **token-level** surrogate with the same advantage `A_i` broadcast to all tokens in completion `i`:

```
L_GRPO(θ) = -(1/Σ|o_i|) Σ_i Σ_t [
  min(π_θ(o_{i,t}|q,o_{i,<t})/π_old(o_{i,t}|q,o_{i,<t}) · Â_{i,t}, clip(...) · Â_{i,t})
  - β · D_KL[π_θ || π_ref]
]
```

TRL defaults `beta = 0` (no KL penalty) because several R1-like works report KL is not essential in some RLVR setups. Treat KL as a **stability knob**.

---

## Why GRPO is Simpler Than PPO

1. **No critic network** → fewer trainable parameters and optimizer states
2. **Single scalar reward per completion** suffices for RLVR (math/code judges)
3. Inherits **PPO's clipping intuition** — teams can reuse PPO infrastructure
4. But does NOT remove the dominant cost of **sampling G long completions per prompt**

---

## GRPO vs PPO vs REINFORCE++ vs DPO

| Aspect | PPO | GRPO | REINFORCE++ | DPO |
|--------|-----|------|-------------|-----|
| **Regime** | On-policy RL | On-policy RL | On-policy RL (critic-free) | Offline preference optimization |
| **Baseline** | Critic + GAE | Group z-score | Global normalization | Implicit preference objective |
| **Data** | Rollouts from π_θ | Rollouts, group per prompt | Rollouts, improved normalization | Preference dataset (y_w, y_l) |
| **Reward** | RM scalar | Often verifiable correctness | RM / task rewards | Preference labels |
| **Memory** | Highest (actor+critic) | Lower (no critic) | Lower (no critic) | Lower per step (no online gen) |

### When to Use Which

- **DPO:** High-quality preference data, cheap stable fine-tuning, no online generation needed
- **PPO:** RM-driven RLHF, classical critic + broad tooling maturity
- **GRPO:** RLVR (math, code, formats), cheap verifiable rewards, PPO-like clipping without critic — **most copied pattern post-R1**
- **REINFORCE++:** Instability from local z-normalization, global normalization perspective

### Notable Related Work

- [REINFORCE++: Stabilizing Critic-Free Policy Optimization](https://arxiv.org/abs/2501.03262) — critiques prompt-local normalization (including GRPO)
- [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977) — GRPO implements implicit contrastive objective; 2-GRPO (only two rollouts) can be efficient
- [DAPO](https://huggingface.co/papers/2503.14476) — addresses loss aggregation issues in long-CoT GRPO training

---

## The RL Post-Training Loop

For each training step:
1. **Sample prompts** q from training distribution
2. **Roll out** G completions per prompt (bottleneck — vLLM helps here)
3. **Score** each completion: r_i = R(q, o_i)
4. **Advantage:** group-normalize {r_i} → {A_i}
5. **Policy update:** PPO-style clipped surrogate (+ optional KL)
6. **Repeat**

### Why Rollout Generation is the Bottleneck

- Autoregressive decoding at large T and G dominates wall-clock
- Long CoT pushes sequence length upward
- GPUs idle waiting for completion generation
- **This is why vLLM integration matters** — it directly addresses this bottleneck

### How Verifiable Rewards Simplify the Loop

- No RM forward pass in the critical path (lower latency, simpler debugging)
- Deterministic labels reduce disagreement noise
- Easier replay/testing: unit-test reward code independent of model weights
- DeepSeek-R1-Zero relies on accuracy + format rule rewards, avoiding neural reward models

---

## References

1. Shao et al., *DeepSeekMath*, 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
2. DeepSeek-AI, *DeepSeek-R1*, 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
3. Schulman et al., *PPO*, 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
4. Rafailov et al., *DPO*, 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
5. Hu et al., *REINFORCE++*, 2025. [arXiv:2501.03262](https://arxiv.org/abs/2501.03262)
6. TRL GRPO Trainer docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
