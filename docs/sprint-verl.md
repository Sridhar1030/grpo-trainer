# Sprint Plan: veRL GRPO on KFT (Without Ray) — Contingency

> **Epic:** RHOAIENG-61094  
> **Status:** Contingency sprint — only pursue if TRL sprint hits a wall  
> **Framework:** veRL (requires Ray by design; this sprint explores removing it)  
> **Dataset:** GSM8K  
> **Constraint:** No Ray anywhere in the stack  
> **Risk Level:** HIGH — veRL's RL controller is Ray-native. This is a source-code-modification effort.

---

## Why This Sprint Exists

veRL has significant advantages over TRL: native vLLM rollouts, FSDP + vLLM GPU memory coexistence, built-in GSM8K rewards, async policy training with NCCL weight sync, and 200+ tuning knobs. If TRL proves too limited (slow rollouts, memory pressure, no async), forking veRL to strip Ray is the next option.

**The core problem:** veRL's GRPO controller (`main_ppo` → `RayPPOTrainer` → `RayWorkerGroup`) uses Ray actors for worker orchestration. SFT/DPO trainers work with `torchrun` alone, but the RL loop does not.

**The opportunity:** An RFC exists (veRL Issue #1221) proposing TorchRPC as a Ray alternative. This sprint evaluates whether we can build on that or write our own non-Ray controller.

---

## Sprint Overview

| Phase | What | Duration | Outcome |
|-------|------|----------|---------|
| **Phase A** | Feasibility: audit Ray surface area in veRL GRPO path | 1 week | Go/no-go on stripping Ray |
| **Phase B** | Fork and replace Ray with torch.distributed / TorchRPC | 2 weeks | Working veRL GRPO without Ray |
| **Phase C** | Run on KFT and compare to TRL sprint | 1 week | Performance comparison + recommendation |

---

## Phase A: Feasibility Audit (Week 1)

### Goal
Map every Ray touchpoint in the veRL GRPO code path. Determine if stripping Ray is weeks of work or months.

### Stories

#### A.1 — Trace the veRL GRPO execution path
**Points:** 5  
**Description:** Starting from `python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo`, trace every Ray call. Map the dependency graph.

**Acceptance criteria:**
- [ ] Full call graph from `main_ppo` → `RayPPOTrainer` → workers documented
- [ ] Every `ray.remote`, `ray.get`, `ray.put`, `@ray.remote`, Ray actor creation identified
- [ ] Count of Ray-dependent modules (files that import ray)
- [ ] Classify each Ray usage: scheduling, communication, state management, or resource allocation

**Key files to audit:**
- `verl/trainer/main_ppo.py` — entry point
- `verl/trainer/ppo/ray_trainer.py` — `RayPPOTrainer`
- `verl/workers/` — `RayWorkerGroup`, actor/rollout/ref workers
- `verl/single_controller/ray/` — Ray-specific controller logic

---

#### A.2 — Identify what Ray provides that torch.distributed doesn't
**Points:** 3  
**Description:** For each Ray usage, determine if `torch.distributed`, `TorchRPC`, or plain multiprocessing can replace it.

**Deliverables:**
- [ ] Table: Ray feature → replacement → effort estimate → risk
- [ ] Identify irreplaceable Ray features (if any)
- [ ] Estimate: is this a 2-week fork or a 2-month rewrite?

**Known replacements:**
| Ray feature | veRL usage | Replacement |
|-------------|-----------|-------------|
| `ray.remote` actors | Worker groups (actor, rollout, ref) | `torch.distributed` process groups or TorchRPC |
| `ray.get` / `ray.put` | Passing data between workers | NCCL collectives or shared memory |
| Placement groups | GPU allocation per role | K8s pod scheduling + `CUDA_VISIBLE_DEVICES` |
| Object store | Sharing rollout batches | Shared PVC or NCCL broadcast |
| Failure handling | Worker restart | KFT pod restart policy |

---

#### A.3 — Evaluate veRL TorchRPC RFC (Issue #1221)
**Points:** 2  
**Description:** Read the open RFC, check if any code exists, assess viability.

**Acceptance criteria:**
- [ ] RFC status documented (proposal only? PR exists? merged?)
- [ ] If code exists, does it cover the GRPO path?
- [ ] Can we build on it, or is it dead?
- [ ] Contact veRL maintainers if unclear (open a GitHub discussion)

---

#### A.4 — Go/no-go decision
**Points:** 1  
**Description:** Based on A.1–A.3, decide whether to proceed with Phase B.

**Go criteria:**
- Ray surface area is < 10 files to modify
- No irreplaceable Ray features for GRPO
- Estimated effort ≤ 2 weeks for a working prototype

**No-go criteria:**
- Ray is deeply woven into data flow (not just scheduling)
- Replacing Ray requires rewriting the advantage computation or vLLM integration
- Estimated effort > 4 weeks

---

### Phase A Definition of Done

- [ ] Complete audit of Ray in veRL GRPO path
- [ ] Go/no-go decision documented with evidence
- [ ] If no-go: close this sprint, document why, stick with TRL

---

## Phase B: Fork and Strip Ray (Weeks 2–3)

> Only proceed if Phase A produced a GO decision.

### Goal
Fork veRL. Replace Ray with torch.distributed / TorchRPC. Get veRL's GRPO running on pure PyTorch distributed — no Ray.

### Architecture Target

```
veRL GRPO (forked, Ray-free)
    │
    ├── Controller (replaces RayPPOTrainer)
    │   └── torch.distributed process groups
    │   └── TorchRPC for cross-role communication
    │
    ├── Actor Worker Group (FSDP training)
    │   └── torch.distributed.fsdp
    │
    ├── Rollout Worker Group (vLLM)
    │   └── vLLM engine (same process or subprocess)
    │   └── Weight sync via NCCL (reuse veRL's existing sync code)
    │
    └── Reference Policy (FSDP, param_offload=True)
```

### Stories

#### B.1 — Fork veRL and create Ray-free controller
**Points:** 8  
**Description:** Replace `RayPPOTrainer` with a `TorchDistributedPPOTrainer` that uses `torch.distributed` process groups for worker coordination.

**Acceptance criteria:**
- [ ] Fork created at `github.com/<org>/verl-kft` (or local branch)
- [ ] New controller file: `verl/trainer/ppo/torch_trainer.py`
- [ ] Workers launched via `torchrun` or KFT-injected env vars
- [ ] GRPO advantage computation works without Ray (it's pure PyTorch math — should be straightforward)
- [ ] Data batching between controller and workers uses NCCL collectives, not Ray object store

---

#### B.2 — Replace RayWorkerGroup with torch.distributed workers
**Points:** 8  
**Description:** Rewrite worker group to use `torch.distributed.new_group()` for role-based process groups instead of Ray actors.

**Acceptance criteria:**
- [ ] ActorWorker, RolloutWorker, RefWorker initialized via `torch.distributed`
- [ ] Workers communicate via NCCL (gradients, weights) and CPU tensors (rewards, prompts)
- [ ] vLLM rollout worker still functions (vLLM itself doesn't need Ray)
- [ ] All workers join the correct NCCL groups for their role

**Risk:** This is the hardest story. veRL's `RayWorkerGroup` handles heterogeneous resource allocation (different GPU counts per role). On KFT, this must be pre-planned at pod spec time.

---

#### B.3 — Validate on single-node multi-GPU
**Points:** 3  
**Description:** Run the forked veRL GRPO on a single machine with 4 GPUs before attempting KFT.

**Acceptance criteria:**
- [ ] `torchrun --nproc_per_node=4` launches all roles correctly
- [ ] vLLM uses GPUs 0–1, FSDP actor uses GPUs 2–3 (or colocated)
- [ ] One full GRPO step completes: prompt → rollout → reward → advantage → update
- [ ] No Ray imports anywhere in the execution path

---

#### B.4 — Run on KFT via CustomTrainerContainer
**Points:** 5  
**Description:** Package the forked veRL into a container and run on KFT.

**Acceptance criteria:**
- [ ] Dockerfile with forked veRL + vLLM + FSDP
- [ ] TrainJob submitted via KFT Python SDK
- [ ] Multi-node training works (2 nodes × 2 GPUs minimum)
- [ ] GRPO training loop completes on GSM8K
- [ ] Reward improves over training steps

---

### Phase B Definition of Done

- [ ] veRL GRPO runs on KFT without Ray
- [ ] vLLM rollouts work (fast generation)
- [ ] FSDP training works (distributed gradient updates)
- [ ] At least one full training run on GSM8K completes

---

## Phase C: Compare and Recommend (Week 4)

### Stories

#### C.1 — Performance comparison: veRL-fork vs TRL on KFT
**Points:** 3  
**Description:** Same GSM8K task, same hardware. Compare metrics.

**Deliverables:**
- [ ] Rollout tokens/sec: veRL (vLLM) vs TRL (HF generate)
- [ ] Training throughput: steps/hour
- [ ] Peak GPU memory per worker
- [ ] Time to first reward improvement
- [ ] Code complexity: lines modified in fork, maintainability assessment

---

#### C.2 — Upstream viability assessment
**Points:** 2  
**Description:** Can this fork be maintained long-term? Or will it drift from upstream veRL?

**Deliverables:**
- [ ] Number of lines changed from upstream
- [ ] Which veRL releases would break the fork
- [ ] Is contributing the non-Ray path upstream viable? (Check community interest)
- [ ] Maintenance cost estimate: hours/month to keep fork current

---

#### C.3 — Final recommendation
**Points:** 1  
**Description:** TRL vs veRL-fork: which path for production?

**Deliverables:**
- [ ] Decision matrix with weighted criteria
- [ ] Recommendation for Agentic Continual Learning PoC
- [ ] If veRL-fork wins: plan for upstream contribution or long-term maintenance

---

## Sprint Capacity

| Story | Points | Phase | Depends on |
|-------|--------|-------|------------|
| A.1 Trace GRPO execution path | 5 | A | — |
| A.2 Ray → torch.distributed mapping | 3 | A | A.1 |
| A.3 Evaluate TorchRPC RFC | 2 | A | — |
| A.4 Go/no-go decision | 1 | A | A.1, A.2, A.3 |
| B.1 Ray-free controller | 8 | B | A.4 (GO) |
| B.2 Replace RayWorkerGroup | 8 | B | B.1 |
| B.3 Single-node validation | 3 | B | B.2 |
| B.4 Run on KFT | 5 | B | B.3 |
| C.1 Performance comparison | 3 | C | B.4 |
| C.2 Upstream viability | 2 | C | B.4 |
| C.3 Final recommendation | 1 | C | C.1, C.2 |
| **Total** | **41** | | |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Ray is too deeply embedded; fork becomes a rewrite | High | Critical | Phase A audit catches this early; no-go kills sprint |
| vLLM integration breaks when Ray is removed | Medium | High | vLLM itself is Ray-free; only veRL's wrapper uses Ray |
| Fork diverges from upstream veRL within months | High | Medium | Track upstream releases; contribute non-Ray path if community interest exists |
| Performance doesn't justify the effort vs TRL | Medium | Medium | Phase C measures this; if TRL is close enough, don't fork |
| TorchRPC RFC is abandoned | Medium | Low | Build our own controller; TorchRPC was only one option |

---

## Key Dependencies

| Dependency | Version | Why |
|------------|---------|-----|
| veRL | latest main branch | Fork base |
| vLLM | ≥ 0.17.1 | Rollout engine + weight sync |
| PyTorch | ≥ 2.5 | FSDP2 + TorchRPC |
| CUDA | ≥ 12.x | BF16, NCCL |
| kubeflow-trainer SDK | latest | CustomTrainerContainer |

---

## Exit Criteria

This sprint should be **abandoned** if:
- Phase A produces a no-go (Ray too deep)
- Phase B takes > 3 weeks with no working single-node prototype
- TRL sprint already meets the PoC requirements (this sprint becomes unnecessary)
