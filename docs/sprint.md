# Sprint Plan: GRPO on Kubeflow Trainer PoC

> **Epic:** RHOAIENG-61094  
> **Objective:** Prove Kubeflow Trainer as an alternative to Ray for GRPO fine-tuning  
> **Framework:** TRL (only viable GRPO framework without Ray)  
> **Dataset:** GSM8K  
> **Constraint:** No Ray anywhere in the stack

---

## Sprint Overview

| Phase | What | Duration | Outcome |
|-------|------|----------|---------|
| **Phase 1** | TRL GRPO baseline on KFT | 2 weeks | GRPO training loop works, reward improves |
| **Phase 2** | Add vLLM for fast rollouts + async weight sync | 2 weeks | Production-grade RL loop, no Ray |

---

## Phase 1: TRL GRPO Baseline on KFT (Weeks 1–2)

### Goal
Prove a minimal end-to-end GRPO training loop runs successfully on Kubeflow Trainer using TRL GRPOTrainer with GSM8K.

### Stories

#### 1.1 — Cluster setup and prerequisites
**Points:** 2  
**Description:** Verify KFT v2 CRDs, torch-distributed runtime, Kueue queue, and GPU node pool. Create shared PVC and HuggingFace token secret.

**Acceptance criteria:**
- [ ] `kubectl get crd trainjobs.trainer.kubeflow.org` succeeds
- [ ] `kubectl get clustertrainingruntime torch-distributed` succeeds
- [ ] `grpo-poc-cache` PVC is bound (200Gi, RWX)
- [ ] `hf-credentials` secret exists in target namespace
- [ ] Kueue LocalQueue has sufficient GPU quota (min 2 nodes × 2 GPUs)

---

#### 1.2 — Build training container image
**Points:** 3  
**Description:** Create a container image with TRL, transformers, datasets, torch, and all dependencies pre-installed. Test locally before pushing.

**Acceptance criteria:**
- [ ] Dockerfile builds successfully with `trl>=1.4.0`, `transformers`, `datasets`, `torch` with CUDA
- [ ] Image pushed to accessible registry
- [ ] `python -c "from trl import GRPOTrainer; print('OK')"` passes inside the container
- [ ] Image size is reasonable (ideally < 15GB)

**Notes:** Using `CustomTrainerContainer` (pre-built image) instead of `CustomTrainer` (cloudpickle) avoids serialization issues with complex dependencies.

---

#### 1.3 — Implement TRL GRPO training function
**Points:** 3  
**Description:** Write the training function that runs TRL GRPOTrainer on GSM8K with a rule-based correctness reward. Must work in distributed mode (DDP/FSDP).

**Acceptance criteria:**
- [ ] Training function reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from environment
- [ ] GSM8K loaded and formatted with `prompt` + `answer` columns
- [ ] Reward function extracts `#### N` from completions and compares to ground truth
- [ ] GRPOConfig uses: `num_generations=8`, `max_completion_length=512`, `bf16=True`, `beta=0.0`
- [ ] Batch sizes satisfy divisibility rule: `(world_size × batch × grad_accum) % num_generations == 0`
- [ ] Model and tokenizer saved to PVC on rank 0

---

#### 1.4 — Submit TrainJob to KFT and validate
**Points:** 5  
**Description:** Submit the training job via KFT Python SDK or YAML. Monitor job to completion. Validate reward improvement.

**Acceptance criteria:**
- [ ] TrainJob submitted and pods scheduled via Kueue
- [ ] All worker pods start and join the distributed group
- [ ] Training logs show GRPO loop executing (generation → reward → advantage → update)
- [ ] Reward/mean metric improves over training steps (even slightly)
- [ ] No NCCL timeout errors in logs
- [ ] TrainJob reaches `Complete` status
- [ ] Checkpoint saved to PVC at expected path

---

#### 1.5 — Document Phase 1 findings
**Points:** 2  
**Description:** Write up what worked, what didn't, performance numbers, and KFT-specific observations.

**Deliverables:**
- [ ] Training speed: steps/sec, tokens/sec for rollout generation
- [ ] GPU utilization during generation vs training phases
- [ ] Memory usage: peak GPU memory per worker
- [ ] KFT observations: what Trainer provided vs what we had to build
- [ ] Blockers or issues encountered
- [ ] Recommendation: proceed to Phase 2 or pivot

---

### Phase 1 Definition of Done

- [ ] GRPO training on KFT produces improving rewards on GSM8K
- [ ] Checkpoint is usable (can load and generate from it)
- [ ] Findings documented with concrete numbers
- [ ] Go/no-go decision for Phase 2

---

## Phase 2: TRL AsyncGRPO + vLLM (Weeks 3–4)

### Goal
Add vLLM for fast rollout generation with NCCL weight sync. Prove the full production RL loop works on KFT without Ray.

### Architecture

```
KFT TrainJob (CustomTrainerContainer)
    └── TRL AsyncGRPOTrainer + FSDP2
        ├── Training: policy gradient updates
        ├── HTTP → vLLM for rollout generation
        └── NCCL → vLLM for weight sync (PR #31943)

vLLM Deployment (separate K8s resource)
    └── vLLM serve with --weight-transfer-config '{"backend":"nccl"}'
    └── VLLM_SERVER_DEV_MODE=1
    └── Receives weight updates via NCCL from trainer
```

### Stories

#### 2.1 — Deploy vLLM inference server
**Points:** 3  
**Description:** Deploy vLLM as a Kubernetes Deployment + Service with weight transfer enabled. Verify it serves completions.

**Acceptance criteria:**
- [ ] vLLM Deployment running with `--weight-transfer-config '{"backend":"nccl"}'`
- [ ] `VLLM_SERVER_DEV_MODE=1` environment variable set
- [ ] ClusterIP Service exposes vLLM on port 8000
- [ ] `curl http://vllm-svc:8000/v1/completions` returns a valid response
- [ ] `GET /get_world_size` endpoint responds correctly
- [ ] Model loaded matches training model (e.g. Llama-3.2-1B-Instruct)

---

#### 2.2 — Implement NCCL weight sync from trainer to vLLM
**Points:** 5  
**Description:** Wire the weight transfer protocol: trainer (rank 0) pushes updated weights to vLLM via NCCL after each training step.

**Acceptance criteria:**
- [ ] `POST /init_weight_transfer_engine` succeeds (NCCL group formed)
- [ ] `POST /pause` → `POST /start_weight_update` → NCCL broadcast → `POST /update_weights` → `POST /finish_weight_update` → `POST /resume` cycle works
- [ ] Weights in vLLM match trainer weights after sync (verify via generation quality)
- [ ] No NCCL errors during transfer
- [ ] Sync completes in < 30s for 1B parameter model

---

#### 2.3 — Run TRL AsyncGRPOTrainer with vLLM server
**Points:** 5  
**Description:** Switch from Phase 1's HF `.generate()` to TRL's AsyncGRPOTrainer with `vllm_mode="server"`. Validate end-to-end.

**Acceptance criteria:**
- [ ] AsyncGRPOTrainer connects to vLLM server successfully
- [ ] Rollouts generated by vLLM (not HF .generate())
- [ ] Weight sync happens automatically every `weight_sync_steps`
- [ ] Training loop runs without pod restarts between steps
- [ ] Reward trajectory matches or exceeds Phase 1 baseline
- [ ] Rollout tokens/sec measurably higher than Phase 1 (target: 3x+)

**Config notes:**
- `use_vllm=True`, `vllm_mode="server"`
- `vllm_server_base_url="http://vllm-svc:8000"`
- FSDP2 for distributed training (required by AsyncGRPOTrainer)
- Trainer GPUs and vLLM GPUs must be separate (NCCL conflict otherwise)

---

#### 2.4 — Measure and compare: KFT vs Ray baseline
**Points:** 3  
**Description:** Collect performance metrics and compare against what Ray-based stacks report. Document the trade-offs.

**Deliverables:**
- [ ] Rollout speed: tokens/sec (Phase 1 HF generate vs Phase 2 vLLM)
- [ ] Training throughput: samples/sec, steps/hour
- [ ] Weight sync latency: time per sync cycle
- [ ] GPU utilization: training vs generation vs idle
- [ ] Comparison table: what KFT provides vs what Ray provides for this workload
- [ ] Honest assessment: is Trainer viable as Ray alternative for GRPO fine-tuning?

---

#### 2.5 — Final write-up and recommendations
**Points:** 2  
**Description:** Produce the PoC conclusion document answering the Jira epic questions.

**Deliverables:**
- [ ] Answer: Can Trainer be the alternative to Ray for GRPO? (with evidence)
- [ ] Architecture recommendation for Agentic Continual Learning PoC
- [ ] List of KFT gaps that need filling (upstream contributions / custom runtime)
- [ ] Phase 3 recommendations if proceeding (multi-node scale, learned rewards, larger models)

---

### Phase 2 Definition of Done

- [ ] Full RL loop running: KFT training + vLLM rollouts + NCCL weight sync
- [ ] No Ray in the stack
- [ ] Performance numbers documented and compared
- [ ] Go/no-go recommendation for using Trainer in the continual learning PoC

---

## Sprint Capacity

| Story | Points | Phase | Depends on |
|-------|--------|-------|------------|
| 1.1 Cluster setup | 2 | 1 | — |
| 1.2 Container image | 3 | 1 | — |
| 1.3 Training function | 3 | 1 | 1.2 |
| 1.4 Submit and validate | 5 | 1 | 1.1, 1.3 |
| 1.5 Document findings | 2 | 1 | 1.4 |
| 2.1 vLLM deployment | 3 | 2 | 1.4 (Phase 1 done) |
| 2.2 NCCL weight sync | 5 | 2 | 2.1 |
| 2.3 AsyncGRPO + vLLM | 5 | 2 | 2.1, 2.2 |
| 2.4 Measure and compare | 3 | 2 | 2.3 |
| 2.5 Final write-up | 2 | 2 | 2.4 |
| **Total** | **33** | | |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| HF .generate() too slow for meaningful Phase 1 training | High | Medium | Accept — Phase 1 proves correctness, not speed. Phase 2 fixes this. |
| TRL AsyncGRPOTrainer + vLLM has bugs or undocumented requirements | Medium | High | Fall back to manual vLLM HTTP calls + custom NCCL weight sync code |
| NCCL networking between TrainJob pods and vLLM Deployment fails | Medium | High | Debug with `NCCL_DEBUG=INFO`; ensure pod-to-pod IP reachability; set `NCCL_SOCKET_IFNAME` |
| GPU memory OOM with 1B model + GRPO (8 generations) | Medium | Medium | Reduce `num_generations` to 4; use `beta=0` (no reference model); try LoRA |
| KFT cloudpickle serialization fails | Low | High | Use `CustomTrainerContainer` with pre-built image instead |
| vLLM weight sync requires `VLLM_SERVER_DEV_MODE` (not production-ready) | Low | Low | Acceptable for PoC; track vLLM releases for stable API |

---

## Out of Scope (for this sprint)

- veRL or OpenRLHF integration (requires Ray — ruled out)
- Learned reward model (rule-based GSM8K rewards are sufficient for PoC)
- Multi-node scale testing beyond 2 nodes × 2 GPUs
- Contributing upstream to KEP-2839 (track it, don't block on it)
- Production hardening (monitoring, alerting, auto-scaling)

---

## Key Dependencies

| Dependency | Version | Why |
|------------|---------|-----|
| TRL | ≥ 1.4.0 | GRPOTrainer + AsyncGRPOTrainer |
| transformers | ≥ 5.2.0 | Required by TRL for FSDP2 + async features |
| vLLM | ≥ 0.17.1 | Weight transfer APIs (PR #31943) |
| kubeflow-trainer SDK | latest | CustomTrainer / CustomTrainerContainer |
| PyTorch | ≥ 2.5 | FSDP2 support |
| CUDA | ≥ 12.x | BF16, NCCL |

---

## Definition of Done (Sprint)

- [ ] Phase 1 complete: GRPO trains on KFT, rewards improve
- [ ] Phase 2 complete: vLLM rollouts + weight sync work without Ray
- [ ] Performance documented with concrete numbers
- [ ] Architecture recommendation delivered for continual learning PoC
- [ ] All findings in knowledge base (`docs/`)
