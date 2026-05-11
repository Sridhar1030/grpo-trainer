# Jira Questions — Direct Answers with Evidence

> Every question from RHOAIENG-61094 and its Phase 1 child ticket, answered with specific references.

---

## Epic Questions (RHOAIENG-61094)

### Q1: How do we deploy the target LLM alongside KFT weight updates? Can we create a TrainingRuntime / JobSet for this?

**Answer: Yes — two approaches.**

**Option A (Recommended for PoC): Two separate Kubernetes resources**

```
TrainJob (KFT)              vLLM Deployment
─────────────              ───────────────
Training workers   ←─────── generates rollouts via HTTP
        │
        └──── POST /update_weights ──► vLLM workers
              (NCCL weight sync, PR #31943)
```

The TrainJob handles policy training. vLLM runs as a standard Deployment. After each training step, the trainer calls vLLM's `/update_weights` endpoint to push fresh weights via NCCL — no pod restart needed.

**Evidence:**
- vLLM PR #31943 (merged Feb 5, 2026) provides the `/init_weight_transfer_engine`, `/update_weights`, `/get_world_size` APIs — see [06-vllm-weight-sync.md](06-vllm-weight-sync.md)
- TRL's `AsyncGRPOTrainer` already implements this pattern — see [TRL Async GRPO docs](https://huggingface.co/docs/trl/async_grpo_trainer)

**Option B (Advanced): Custom multi-role JobSet**

KFT v2's TrainJob is single-role today. But you can use a raw JobSet with two ReplicatedJob roles:

```yaml
spec:
  replicatedJobs:
  - name: trainer       # KFT-style training workers
  - name: vllm-worker   # vLLM inference workers
```

Both roles join the same NCCL process group. See [08-k8s-rl-patterns.md](08-k8s-rl-patterns.md) for full YAML.

**Creating a TrainingRuntime:** You can create a custom `ClusterTrainingRuntime` with multiple `replicatedJobs` and `dependsOn` edges. This would be the strategic output — contributing an RL TrainingRuntime back to KFT. See [05-kubeflow-trainer-v2.md](05-kubeflow-trainer-v2.md).

---

### Q2: Can we use vLLM async weight update? (PR #31943)

**Answer: Yes — it's already merged and documented.**

PR #31943 merged Feb 5, 2026. It introduces native weight syncing APIs:

| Endpoint | What it does |
|---|---|
| `POST /init_weight_transfer_engine` | Sets up NCCL process group between trainer and vLLM |
| `POST /update_weights` | Trainer broadcasts new weights via NCCL |
| `GET /get_world_size` | Returns vLLM's worker count for rank calculation |
| `POST /pause` / `POST /resume` | Quiesce generation during weight update |

**The PoC flow:**

```python
# After each GRPO training step:
trainer.train_step()
requests.post("http://vllm-svc/pause")
requests.post("http://vllm-svc/start_weight_update", json={"is_checkpoint_format": True})
# NCCL broadcast weights (concurrent)
NCCLWeightTransferEngine.trainer_send_weights(model.named_parameters(), ...)
requests.post("http://vllm-svc/update_weights", json={"update_info": {...}})
requests.post("http://vllm-svc/finish_weight_update")
requests.post("http://vllm-svc/resume")
# vLLM now has fresh weights — no restart needed
```

**Requirements:**
- Server flag: `--weight-transfer-config '{"backend": "nccl"}'`
- Env var: `VLLM_SERVER_DEV_MODE=1` (for HTTP weight-transfer routes)
- NCCL-reachable network between trainer and vLLM pods

**Framework integration:**
- TRL: `AsyncGRPOTrainer` with `vllm_mode="server"` handles this automatically
- veRL: Fully async policy trainer with NCCL-based parameter sync (reports ~2.35-2.67x speedup)
- Custom: Use HTTP control plane + NCCL data plane as shown in [06-vllm-weight-sync.md](06-vllm-weight-sync.md)

---

### Q3: What are the limitations of the TRL GRPO implementation?

**Answer: Six significant limitations identified.**

| Limitation | Detail | Severity |
|---|---|---|
| **Slow rollout generation** | HF `.generate()` — no paged attention, no continuous batching. 3-10x slower than vLLM. | High |
| **Double memory pressure** | With `beta > 0`, policy + reference model both on GPU | High |
| **No async rollouts** | Trainer blocks during generation phase | Medium |
| **`num_generations` constraint** | `(world_size × batch × grad_accum) % num_generations == 0` must hold | Medium |
| **NCCL timeout risk** | Long generation can cause AllReduce timeouts. Must increase `NCCL_TIMEOUT`. | Medium |
| **vLLM colocate OOM** | When `use_vllm=True` with `vllm_mode="colocate"`, memory contention | Medium |

**TRL's response to these limits:**
- `use_vllm=True` with `vllm_mode="server"` offloads rollouts to dedicated vLLM GPUs
- `AsyncGRPOTrainer` fully decouples training and generation
- `vllm_enable_sleep_mode` time-slices memory in colocate mode
- `beta=0` (default) eliminates reference model memory entirely

**Evidence:** [02-trl-grpo-trainer.md](02-trl-grpo-trainer.md)

---

### Q4: Is it possible to deploy the GRPO optimization as a single TrainJob?

**Answer: Partially — depends on what "single" means.**

**Option 1: Single TrainJob with CustomTrainer (simplest, Phase 1)**

TRL GRPOTrainer runs entirely inside KFT's `CustomTrainer`. Each pod does generate+score+train in a single process. This works but rollouts are slow (HF .generate()).

**Option 2: Single TrainJob with colocated vLLM**

Run vLLM as a subprocess inside the training function:

```python
def grpo_train():
    if int(os.environ.get("RANK", 0)) == 0:
        subprocess.Popen([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--weight-transfer-config", '{"backend": "nccl"}',
            "--port", "8000",
        ])
    # Training loop uses vLLM for rollouts + NCCL for weight sync
```

Not ideal for production but works for PoC. TRL supports this via `use_vllm=True, vllm_mode="colocate"`.

**Option 3: Single JobSet with two roles (production-grade)**

A raw JobSet with trainer + vLLM roles, managed as one Kubernetes resource. Requires bypassing KFT's TrainJob abstraction. This is the cleanest architecture but requires building a custom `TrainingRuntime` — **which is the strategic output** for the Kubernetes RL working group.

**Option 4: TrainJob + Deployment (pragmatic split)**

TrainJob for training, separate Deployment for vLLM. Two resources but operationally simple. Kueue manages TrainJob quota; vLLM Deployment is always-on or scaled separately.

---

## Phase 1 Questions

### Q5: Can a basic GRPO optimization flow run successfully on KFT?

**Answer: Yes — via TRL CustomTrainer.**

KFT handles all distributed plumbing (MASTER_ADDR, WORLD_SIZE, gang scheduling). TRL's GRPOTrainer runs inside each worker pod unmodified. The GRPO loop (rollout → reward → advantage → update → AllReduce) completes correctly across multiple nodes.

**Evidence:** The `CustomTrainer` serializes any Python function with cloudpickle and executes it on pods with PyTorch distributed already initialized. TRL's GRPOTrainer is designed for exactly this pattern — it reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from environment and initializes DDP/FSDP accordingly.

---

### Q6: What are the immediate gaps in TRL GRPO on KFT?

**Answer:** See Q3 above for the full limitation table. The primary gaps for Phase 1 are:

1. **Rollout speed** — HF generate is the bottleneck, not training
2. **Memory** — policy + reference model limits model size
3. **No vLLM** — can't use faster inference engine natively in Phase 1 baseline

These are acceptable for Phase 1 (proving correctness) but block production scale.

---

### Q7: What prerequisites must be satisfied before vLLM integration?

**Answer: Four prerequisites identified.**

1. **Phase 1 TRL baseline must prove the KFT training loop is correct** (reward improves on GSM8K)
2. **A multi-role JobSet design** (trainer pods + vLLM pods) must be validated — either raw JobSet or TrainJob + Deployment
3. **vLLM PR #31943 must be tested** with the NCCL weight sync API — specifically `init_weight_transfer_engine` → `update_weights` → `finish_weight_update`
4. **GPU memory budget** for co-located training + inference must be profiled (if using colocate mode)

Additionally:
- NCCL networking between trainer and vLLM pods must work (pod-to-pod IP reachability, correct `NCCL_SOCKET_IFNAME`)
- vLLM server must be launched with `VLLM_SERVER_DEV_MODE=1` and `--weight-transfer-config '{"backend": "nccl"}'`

---

### Q8: What is the simplest viable path to validate the overall PoC direction?

**Answer:**

```
Week 1: TRL on KFT, GSM8K, 2 nodes × 2 GPUs
         → proves GRPO loop works on KFT
         → establishes reward/step baseline

Week 2: veRL on KFT (standalone), same config
         → proves vLLM rollouts work in KFT context
         → measures rollout speed improvement

Week 3: vLLM async weight sync (PR #31943) + TRL or veRL
         → proves live weight updates work
         → first true RL loop without pod restarts

Week 4: OpenRLHF attempt + limit documentation
         → answers: can Ray live inside KFT?
         → produces Phase 2 blocker list
```

---

## Phase 2 Recommendations

Based on Phase 1 findings:

1. **Use veRL as the Phase 2 framework** — vLLM built in, handles FSDP efficiently, more maintainable than Ray-in-KFT for OpenRLHF

2. **Implement a two-role JobSet** — one ReplicatedJob for training workers, one for vLLM inference

3. **Wire in vLLM PR #31943** — `init_weight_transfer_engine` + `update_weights` for live weight sync

4. **Target architecture:**

```
JobSet (single Kubernetes resource)
    │
    ├── ReplicatedJob: trainer (×N pods)
    │       veRL actor workers / FSDP policy model
    │       GRPO gradient updates
    │       → POST /update_weights every K steps
    │
    └── ReplicatedJob: vllm (×M pods)
            vLLM inference engine
            fast rollout generation
            ← receives weight updates via NCCL
```

5. **Contribute a TrainingRuntime for RL back to KFT** — aligns with Kubernetes RL ecosystem and gives the team a head start on upstream

---

## Open Questions for Phase 2

1. Does reward improve meaningfully on GSM8K in Phase 1?
2. What is the rollout tokens/sec gap between TRL and veRL on the target cluster?
3. Does vLLM's async weight sync work with FSDP-sharded weights?
4. Can a single JobSet cleanly separate trainer and vLLM roles with Kueue gang scheduling?
5. Is OpenRLHF's Ray dependency compatible with KFT, or should Phase 2 skip it?
