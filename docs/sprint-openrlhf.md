# Sprint Plan: OpenRLHF GRPO on KFT (Without Ray) — Contingency

> **Epic:** RHOAIENG-61094  
> **Status:** Contingency sprint — only pursue if both TRL and veRL sprints are insufficient  
> **Framework:** OpenRLHF (Ray + vLLM + DeepSpeed by design; this sprint explores removing Ray)  
> **Dataset:** GSM8K  
> **Constraint:** No Ray anywhere in the stack  
> **Risk Level:** VERY HIGH — OpenRLHF is the most Ray-coupled of the three frameworks

---

## Why This Sprint Exists

OpenRLHF is the most complete open-source RLHF framework: supports PPO, REINFORCE++, GRPO, RLOO, DPO in one codebase, designed for 70B+ models, has hybrid engine with vLLM colocation + NCCL weight sync, and battle-tested at scale. If both TRL and veRL paths are inadequate, OpenRLHF-without-Ray is the last resort.

**The core problem:** Every OpenRLHF component is a `@ray.remote` actor. The training loop, rollout engine, reward model, reference model — all Ray actors. Ray isn't a plugin; it's the skeleton.

**The scale of the challenge:** OpenRLHF is harder to de-Ray than veRL because:
- veRL has a single-controller design (one process drives Ray workers) → replace the controller
- OpenRLHF has a fully distributed Ray actor graph → every actor must be replaced

---

## Sprint Overview

| Phase | What | Duration | Outcome |
|-------|------|----------|---------|
| **Phase A** | Feasibility: audit Ray coupling depth | 1 week | Go/no-go (likely no-go) |
| **Phase B** | Fork: replace Ray actors with DeepSpeed + torch.distributed | 3 weeks | Working prototype (if feasible) |
| **Phase C** | KFT integration and comparison | 1 week | Performance data + final recommendation |

---

## Phase A: Feasibility Audit (Week 1)

### Goal
Determine if OpenRLHF can realistically be de-coupled from Ray for the GRPO path. This is expected to be harder than veRL.

### Stories

#### A.1 — Map the Ray actor graph for GRPO
**Points:** 5  
**Description:** Trace the full OpenRLHF GRPO execution path. Document every Ray actor, remote call, and data flow.

**Acceptance criteria:**
- [ ] Entry point traced: CLI → `GRPOTrainer` initialization
- [ ] All Ray actors cataloged with their roles:
  - `PolicyModelActor` (trainable actor, DeepSpeed)
  - `ReferenceModelActor` (frozen reference for KL)
  - `vLLM engines` (rollout generation)
  - `RewardModelActor` (or custom reward function)
- [ ] Data flow mapped: how prompts, rollouts, rewards, and gradients move between actors
- [ ] Count: files importing `ray`, total `ray.remote` decorators, `ray.get`/`ray.put` calls
- [ ] Identify: which parts are Ray scheduling vs actual compute

**Key source paths:**
- `openrlhf/trainer/` — GRPOTrainer, PPOTrainer
- `openrlhf/models/` — Actor, Critic wrapper classes
- `openrlhf/utils/` — distributed utilities
- CLI entry points and config parsing

---

#### A.2 — Assess DeepSpeed as the replacement orchestrator
**Points:** 3  
**Description:** OpenRLHF already uses DeepSpeed ZeRO for training. Evaluate whether DeepSpeed's process groups can replace Ray for the full RL loop.

**Deliverables:**
- [ ] Can DeepSpeed ZeRO handle both actor training AND reference model in one process group?
- [ ] Can vLLM run as a subprocess alongside DeepSpeed (no Ray)?
- [ ] How does OpenRLHF's hybrid engine colocation work? Can it function without Ray placement groups?
- [ ] What happens to `--train.colocate_all`, `--vllm.enable_sleep` without Ray?

---

#### A.3 — Compare effort: fork OpenRLHF vs build from scratch
**Points:** 3  
**Description:** OpenRLHF may be so Ray-coupled that writing a minimal GRPO trainer from scratch (DeepSpeed + vLLM + custom loop) is faster than stripping Ray.

**Deliverables:**
- [ ] Estimate: hours to fork OpenRLHF and remove Ray
- [ ] Estimate: hours to write a minimal GRPO loop using DeepSpeed + vLLM directly
- [ ] If "build from scratch" wins, this sprint pivots to that approach
- [ ] Document the GRPO algorithm steps that need implementation (advantage, clipping, KL penalty)

---

#### A.4 — Go/no-go decision
**Points:** 1  
**Description:** Three possible outcomes:

1. **GO (fork):** Ray surface area is manageable, proceed with Phase B as fork
2. **GO (from scratch):** Too hard to fork, but a minimal DeepSpeed+vLLM GRPO trainer is viable
3. **NO-GO:** Neither path is worth pursuing; TRL or veRL-fork is sufficient

**No-go criteria:**
- Estimated effort > 6 weeks
- DeepSpeed + vLLM colocation has unsolved GPU memory conflicts without Ray
- The algorithm itself is trivial to implement (just use TRL's instead)

---

### Phase A Definition of Done

- [ ] Complete audit documented
- [ ] Go/no-go with evidence
- [ ] If no-go: sprint closed, findings added to knowledge base

---

## Phase B: Fork or Build (Weeks 2–4)

> Only proceed if Phase A produced a GO decision.

### Path 1: Fork OpenRLHF (if Ray surface is manageable)

#### Architecture Target

```
OpenRLHF GRPO (forked, Ray-free)
    │
    ├── Training Controller (replaces Ray actor graph)
    │   └── Single process drives the RL loop
    │   └── torch.distributed for gradient sync
    │
    ├── Policy Model (DeepSpeed ZeRO-2/3)
    │   └── GRPO gradient updates
    │   └── AllReduce across training workers
    │
    ├── vLLM Engine (subprocess or separate pods)
    │   └── Rollout generation
    │   └── Weight sync via NCCL (PR #31943)
    │
    └── Reference Model (DeepSpeed, frozen)
        └── KL divergence computation
        └── param_offload to CPU
```

### Path 2: Minimal GRPO from scratch (if fork is impractical)

#### Architecture Target

```
Custom GRPO Trainer (new, minimal)
    │
    ├── DeepSpeed ZeRO-2 for policy training
    ├── vLLM for rollout generation (HTTP or subprocess)
    ├── Frozen reference model (DeepSpeed, offloaded)
    ├── GRPO advantage: (r - μ) / σ per group
    ├── PPO-clip loss with KL penalty
    └── NCCL weight sync to vLLM (PR #31943)
```

### Stories (apply to whichever path is chosen)

#### B.1 — Implement the training controller
**Points:** 8  
**Description:** The central RL loop that orchestrates: sample prompts → generate rollouts → compute rewards → compute advantages → update policy → sync weights.

**Acceptance criteria:**
- [ ] Single-process controller that drives the full loop
- [ ] No Ray imports
- [ ] Uses `torch.distributed` for multi-node communication
- [ ] Supports configurable: batch size, num_generations (G), learning rate, KL coefficient

---

#### B.2 — Wire DeepSpeed for policy + reference model
**Points:** 5  
**Description:** Initialize DeepSpeed for the trainable policy and frozen reference model.

**Acceptance criteria:**
- [ ] Policy model wrapped in DeepSpeed ZeRO-2 (or ZeRO-3 for larger models)
- [ ] Reference model loaded with `inference=True`, weights offloaded to CPU
- [ ] KL divergence computed between policy and reference log-probs
- [ ] Gradient accumulation works correctly with DeepSpeed

---

#### B.3 — Integrate vLLM for rollout generation
**Points:** 5  
**Description:** Use vLLM (as subprocess or HTTP server) for fast rollout generation. No Ray.

**Acceptance criteria:**
- [ ] vLLM engine starts without Ray (`from vllm import LLM` works standalone)
- [ ] Generates G completions per prompt with temperature sampling
- [ ] Returns token IDs and log-probs needed for advantage computation
- [ ] GPU memory partitioned: vLLM gets allocated portion, DeepSpeed gets the rest

---

#### B.4 — Implement GRPO advantage and loss
**Points:** 3  
**Description:** The core GRPO math: group-relative advantage estimation and PPO-clip loss.

**Acceptance criteria:**
- [ ] For each prompt group of G completions: compute `advantage = (r - mean(r)) / std(r)`
- [ ] PPO-clip loss: `min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)`
- [ ] KL penalty: `β * KL(π_θ || π_ref)` added to loss
- [ ] Token-level masking (only compute loss on completion tokens)
- [ ] Unit test: verify advantage computation matches reference implementation

---

#### B.5 — Single-node validation
**Points:** 3  
**Description:** Run on a single machine with 4 GPUs. GSM8K, rule-based reward.

**Acceptance criteria:**
- [ ] Full GRPO loop executes without errors
- [ ] Reward improves over training steps
- [ ] No Ray anywhere in the process tree
- [ ] Memory usage is stable (no leaks over multiple steps)

---

#### B.6 — Package and run on KFT
**Points:** 5  
**Description:** Containerize and submit to KFT.

**Acceptance criteria:**
- [ ] Dockerfile with DeepSpeed + vLLM + custom trainer
- [ ] TrainJob submitted via KFT SDK
- [ ] Multi-node works (2 nodes minimum)
- [ ] GRPO training completes on GSM8K
- [ ] Checkpoints saved to PVC

---

### Phase B Definition of Done

- [ ] GRPO trains on KFT without Ray
- [ ] vLLM provides fast rollouts
- [ ] DeepSpeed handles distributed training
- [ ] At least one GSM8K run completes with improving rewards

---

## Phase C: Compare and Recommend (Week 5)

### Stories

#### C.1 — Three-way comparison
**Points:** 3  
**Description:** Compare all paths that were actually executed.

**Deliverables:**

| Metric | TRL (sprint.md) | veRL-fork (sprint-verl.md) | OpenRLHF-fork / custom |
|--------|-----------------|---------------------------|------------------------|
| Rollout tokens/sec | | | |
| Training steps/hour | | | |
| Peak GPU memory | | | |
| Lines of custom code | | | |
| Maintenance burden | | | |
| KFT integration quality | | | |

---

#### C.2 — Final recommendation
**Points:** 2  
**Description:** Which path should the Agentic Continual Learning PoC use?

**Deliverables:**
- [ ] Ranked recommendation with justification
- [ ] If custom GRPO trainer: plan for testing, hardening, and documentation
- [ ] If fork: upstream contribution strategy or maintenance plan
- [ ] If TRL was sufficient all along: document why the other sprints weren't needed

---

## Sprint Capacity

| Story | Points | Phase | Depends on |
|-------|--------|-------|------------|
| A.1 Map Ray actor graph | 5 | A | — |
| A.2 DeepSpeed as replacement | 3 | A | A.1 |
| A.3 Fork vs build from scratch | 3 | A | A.1 |
| A.4 Go/no-go | 1 | A | A.1, A.2, A.3 |
| B.1 Training controller | 8 | B | A.4 (GO) |
| B.2 DeepSpeed policy + ref | 5 | B | B.1 |
| B.3 vLLM rollout integration | 5 | B | B.1 |
| B.4 GRPO advantage + loss | 3 | B | B.2, B.3 |
| B.5 Single-node validation | 3 | B | B.4 |
| B.6 Package and run on KFT | 5 | B | B.5 |
| C.1 Three-way comparison | 3 | C | B.6 |
| C.2 Final recommendation | 2 | C | C.1 |
| **Total** | **46** | | |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Ray is the skeleton, not a plugin — fork is a rewrite | Very High | Critical | Phase A kills this early; "build from scratch" is the fallback |
| DeepSpeed + vLLM GPU memory conflicts without Ray's placement groups | High | High | Profile memory early in B.3; use `--vllm.gpu_memory_utilization=0.4` |
| Building from scratch re-invents what TRL already provides | High | Medium | Phase A.3 catches this: if the algorithm is simple, just use TRL |
| Fork maintenance becomes a full-time job | High | High | Only pursue if veRL-fork and TRL are both proven inadequate |
| Multi-node DeepSpeed + vLLM untested without Ray | Medium | High | Validate single-node first (B.5); expand carefully |

---

## Honest Assessment

This sprint is the **highest risk and highest effort** of the three. It should only be pursued if:

1. TRL sprint proved that HF `.generate()` is unacceptable even for the PoC
2. veRL-fork sprint failed (Ray too deep in veRL as well)
3. The team specifically needs OpenRLHF's features (70B+ scale, hybrid engine, multi-algorithm support)

**Most likely outcome:** Phase A produces a no-go, and the recommendation is to either:
- Stick with TRL (simplest, proven on KFT)
- Use veRL-fork (best balance of features and feasibility)
- Build a minimal custom GRPO trainer using DeepSpeed + vLLM directly (if both forks fail)

---

## Key Dependencies

| Dependency | Version | Why |
|------------|---------|-----|
| OpenRLHF | latest main branch | Fork base (if fork path chosen) |
| DeepSpeed | ≥ 0.15 | ZeRO-2/3 for policy training |
| vLLM | ≥ 0.17.1 | Rollout engine + weight sync |
| PyTorch | ≥ 2.5 | FSDP2 / distributed |
| CUDA | ≥ 12.x | BF16, NCCL |
| kubeflow-trainer SDK | latest | CustomTrainerContainer |

---

## Exit Criteria

This sprint should be **abandoned** if:
- Phase A produces a no-go
- Phase B exceeds 4 weeks with no single-node prototype
- TRL or veRL-fork sprint already meets the PoC requirements
- Team concludes that building a custom GRPO trainer from scratch is simpler than modifying OpenRLHF
