# vLLM Weight Sync — What Works, What Doesn't, and Why

> Covers vLLM v0.18.0+ (weight sync shipped Feb 2026).
> Based on vLLM source code, GitHub issues/PRs, and our spike testing.
> **Note:** Spike testing used a custom patched image (`vllm-async-init-poc`), not stock vLLM.

---

## How Weight Sync Works (30-second version)

vLLM exposes HTTP endpoints (control plane — request routing/lifecycle) + an NCCL data plane (bulk weight transfer over GPU interconnect) for updating model weights at runtime.

```
Trainer (GPU 1)                          vLLM Server (GPU 0)
     │                                        │
     ├── POST /init_weight_transfer_engine ──→ │  Both sides join NCCL group
     ├── NCCLWeightTransferEngine.trainer_init │
     │         (must happen simultaneously)    │
     ├── POST /pause ─────────────────────────→│  Stop serving requests
     ├── POST /start_weight_update ───────────→│
     ├── POST /update_weights ────────────────→│  Trainer sends via NCCL
     │   + trainer_send_weights (NCCL)         │  vLLM receives via NCCL
     ├── POST /finish_weight_update ──────────→│
     └── POST /resume ────────────────────────→│  Resume serving
```

> **Threading note:** Steps 1–2 (`/init_weight_transfer_engine` + `trainer_init()`) must execute concurrently from the trainer side using separate threads — the HTTP POST blocks until NCCL init completes on both sides.

> `**is_checkpoint_format` parameter:** When `True`, tells vLLM that incoming weights are in checkpoint/saved format (may need de-sharding or dtype conversion) rather than pre-sharded inference-ready tensors. Use when syncing directly from a training checkpoint.

---

## What Works


| Feature                          | Status                                  | Notes                                                                                                                                                      |
| -------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Single-node, single-GPU sync** | Works                                   | Tested: Trainer GPU 1 → vLLM GPU 0                                                                                                                         |
| **Single-node, TP > 1**          | Works                                   | Official example uses TP=2; trainer broadcasts full weights, each TP worker re-shards locally                                                              |
| **Packed tensor transfer**       | Works                                   | Reduces NCCL ops, uses double-buffering (two weight buffers alternate — one serves inference while the other receives updates); recommended for production |
| **Quantized (FP8) reload**       | Works                                   | Layerwise reloading with on-the-fly quantization                                                                                                           |
| **HTTP control plane**           | Works                                   | Requires `VLLM_SERVER_DEV_MODE=1`                                                                                                                          |
| **Sleep/wake for memory**        | Works (per vLLM docs; not tested by us) | `sleep(level=2)` frees GPU memory during weight update                                                                                                     |
| **IPC backend (same GPU)**       | Works (per vLLM docs; not tested by us) | Zero-copy via CUDA IPC handles; colocated trainer+inference                                                                                                |


---

## What Doesn't Work

### 1. Cross-Pod NCCL Init Deadlocks

**The problem:** `init_weight_transfer_engine` is synchronous. vLLM's HTTP handler calls NCCL group join and **blocks the request** until all ranks join. But the trainer is also waiting for its NCCL join to complete. When they're in separate pods, neither can proceed.

**Case A: Same Pod (localhost) — WORKS**

```
 Trainer Process (GPU 1)                   vLLM Process (GPU 0)
 ═══════════════════════                   ═════════════════════
          │                                         │
 Step 1:  │── HTTP POST /init_weight_transfer_engine →│
          │   (sent from background thread)         │
          │                                         │
          │                                Step 2:  │ receives HTTP request
          │                                         │ calls NCCL group join
          │                                         │ BLOCKS ⏳ (waiting for
          │                                         │ trainer to also join)
          │                                         │
 Step 3:  │ (main thread, runs in parallel)         │
          │ calls trainer_init()                    │
          │ NCCL group join ─── localhost ──────────→│
          │                                         │
          ▼                                         ▼
     Both NCCL ranks join at the same time = ✅ SUCCESS
```

**Case B: Separate Pods (network) — DEADLOCKS**

```
 Trainer Pod (Node A)                      vLLM Pod (Node B)
 ═══════════════════                       ═════════════════
          │                                         │
 Step 1:  │── HTTP POST /init_weight_transfer_engine →│
          │   (waits for HTTP response...)          │
          │   BLOCKS ⏳                              │
          │                                         │
          │                                Step 2:  │ receives HTTP request
          │                                         │ calls NCCL group join
          │                                         │ BLOCKS ⏳ (waiting for
          │                                         │ trainer to also join)
          │                                         │
          │   Can't run trainer_init() because      │
          │   HTTP POST hasn't returned yet!        │
          │                                         │
          ▼                                         ▼
     Trainer waits for HTTP response                vLLM waits for NCCL join
     HTTP response waits for NCCL join              NCCL join waits for trainer
                         ╔═══════════════╗
                         ║  ∞ DEADLOCK   ║
                         ╚═══════════════╝

     Trainer: "I'll join NCCL after I get the HTTP response"
     vLLM:    "I'll send the HTTP response after you join NCCL"
     Neither ever proceeds.
```

**Why:** On localhost, Thread 1 (HTTP POST) and Thread 2 (NCCL join) run concurrently in the trainer process. vLLM's handler blocks on NCCL, but the trainer's Thread 2 can still reach it via localhost. Across pods, the trainer's HTTP POST is stuck waiting for vLLM's response, and vLLM's NCCL join is stuck waiting for the trainer's NCCL join — circular dependency.

**Workaround:** Run trainer inside the vLLM pod on a different GPU. That's what we did.

**Status:** No fix exists or has been proposed. The RFC (#31848) lists "dynamic NCCL group create/destroy" as a TODO but no PR exists.

---

### 2. CUDA Graphs Break Weight Sync (Silent Stale Weights)

**The problem:** CUDA graphs capture tensor memory **addresses** at graph-build time. Weight sync may **replace** tensor objects (new allocations at different addresses) rather than writing into existing buffers. The captured graph still references the **old addresses** — your new weights are silently ignored.

**Why:** CUDA graph replay doesn't re-resolve tensor pointers; it replays the exact captured kernel arguments pointing to the original memory locations.

**Workaround:** Must use `--enforce-eager` (disables CUDA graphs entirely).

**Cost:** ~30–50% throughput loss (per vLLM GitHub discussions and community benchmarks; not measured by us).

**Fix proposed:** A `patch_weights` API that invalidates and rebuilds CUDA graphs after updates (Issue #40536). **Not implemented yet.**

*Reference: [Issue #40536](https://github.com/vllm-project/vllm/issues/40536)*

---

### 3. Pipeline Parallelism (PP > 1) Multi-Node is Blocked

**The problem:** V1 engine's `MultiprocessExecutor` only supports shared-memory IPC (single-node). Cross-node PP fails with `AssertionError: collective_rpc should not be called on follower node`.

**Why:** The executor was designed for single-machine multi-GPU. Multi-node PP would need Ray or a new distributed executor.

**Impact on weight sync:** You can't run PP > 1 inference across nodes, so weight sync across PP nodes is moot.

**Status:** Single-node PP works. Multi-node PP blocked by [Issue #41864](https://github.com/vllm-project/vllm/issues/41864).

---

### 4. NCCL Symmetric Memory Deadlocks (Subset GPU Visibility)

**The problem:** On multi-GPU hosts (e.g., 8×B200 — reported in GitHub Issue #38550, not our hardware), if only a subset of GPUs is visible via `CUDA_VISIBLE_DEVICES`, NCCL's symmetric memory rendezvous expects **all fabric-connected GPUs** to participate → deadlock.

**Workaround:**

```bash
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
```

*Reference: [Issue #38550](https://github.com/vllm-project/vllm/issues/38550)*

---

### 5. HTTP Endpoints Require Dev Mode + Insecure Serialization


| Env var                               | Why                                                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `VLLM_SERVER_DEV_MODE=1`              | Weight sync endpoints are gated behind dev mode                                                                |
| `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | IPC handles use pickle; disabled by default for security. **Only needed for IPC-over-HTTP backend, not NCCL.** |


`VLLM_SERVER_DEV_MODE` is required for all weight sync setups. `VLLM_ALLOW_INSECURE_SERIALIZATION` is only needed when using the IPC backend (which serializes CUDA IPC handles via HTTP). The RFC lists removing these requirements as a TODO.

**K8s note:** On Kubernetes with vGPU, `TCPStore` broken pipe errors can occur during NCCL group init (Issue #42389). This affects weight sync reliability in virtualized GPU environments.

---

### 6. Full Inference Downtime During Sync

**The problem:** Weight sync requires pausing all inference, transferring weights, then resuming. There is no online/rolling weight update — vLLM must stop serving while weights are being swapped.

**Why:** `update_weights` calls `receive_weights()` (blocking NCCL broadcast from rank 0, the trainer) on each worker, followed by an explicit `torch.accelerator.synchronize()` (PyTorch 2.4+ unified device sync API). The model state is inconsistent during transfer.

**Modes available (passed to `/pause`):**

- `mode="abort"` — immediately kills all in-flight requests **(this is the default if not specified)**
- `mode="wait"` — drains in-flight requests before pausing
- `mode="keep"` — freezes requests mid-generation, resumes after sync (tokens before swap use old weights, after use new)

**Timeout tuning:** NCCL operations use a configurable timeout (default 10 min via `NCCL_TIMEOUT` or `--nccl-timeout`). For large models over slower interconnects, increase this to avoid spurious failures.

---

### 7. No Incremental/Partial Weight Updates

The NCCL backend always transfers the **full model weights** — every parameter tensor. There is no mechanism for sending only changed layers or delta updates. For large models, this means significant transfer time and memory pressure even for small fine-tuning changes.

**Memory caveat:** The sync process must hold the full model weights in GPU memory on the trainer side during transfer.

**LoRA/adapter-only sync:** Not supported. Even if only adapter weights changed (e.g., after LoRA fine-tuning), the NCCL backend has no mechanism to send a partial parameter set. You must merge adapters into base weights and sync the full model.

---

### 8. No Retry/Resilience

A failed NCCL operation (timeout, network glitch, rank crash) will **hang or crash** the entire transfer. There is no built-in retry, checkpointing, or graceful degradation. The trainer is always rank 0 in the weight-sync NCCL group (vLLM workers join at `rank_offset=1`), so any trainer failure is unrecoverable without restarting the entire flow.

---

### 9. RDMA Backend is TODO

From `base.py`, the abstract `WeightTransferEngine` lists RDMA as a planned backend alongside NCCL and IPC. **It is not implemented.** Only `nccl` and `ipc` backends are registered in the factory.

---

### 10. Post-Training CUDA Context Pollution

**The problem:** After TRL `GRPOTrainer.train()`, the CUDA context has leftover NCCL state from DDP. Initializing a **new** NCCL group for weight sync in the same process fails with `RuntimeError: NCCL error: Duplicate GPU detected` (raised by PyTorch's NCCL process group wrapper, not NCCL itself).

**Workaround:** Save a checkpoint, then spawn a **fresh subprocess** that loads the checkpoint and does the NCCL sync in a clean CUDA context.

**This is what we do:** `grpo_vllm_train_sync.py --train-only` (torchrun) → save checkpoint → `grpo_vllm_train_sync.py --sync-only` (fresh process).

---

### 11. TRL Version Incompatibility with vLLM

TRL's `GRPOTrainer` with `use_vllm=True` only supports vLLM 0.12.0–0.18.0. Our setup uses vLLM 0.21.0 (for weight sync API improvements), which is outside TRL's supported range. This means TRL's built-in vLLM integration cannot be used directly — we manage the vLLM server and weight sync separately.

---

## Summary: Multi-Node Multi-GPU Specifically


| Scenario                                                                | Works?        | Why / Why Not                                           |
| ----------------------------------------------------------------------- | ------------- | ------------------------------------------------------- |
| **Multi-node DDP training** (KFT `num_nodes=2`)                         | Yes           | Standard PyTorch DDP via torchrun                       |
| **Multi-node vLLM inference** (TP across nodes)                         | Documented    | NCCL over InfiniBand; needs correct env vars            |
| **Multi-node PP inference**                                             | No            | V1 engine limitation (#41864)                           |
| **Cross-pod weight sync** (trainer pod → vLLM pod)                      | **Deadlocks** | Synchronous HTTP+NCCL init; designed for localhost only |
| **Same-pod weight sync** (trainer GPU 1 → vLLM GPU 0)                   | Yes           | Localhost NCCL; our working pattern                     |
| **Same-pod multi-GPU sync** (DDP on GPUs 1-2, sync from GPU 1 to GPU 0) | Yes           | Our current working setup (see Option 2 below)          |


---

## Our Workaround Architecture

```
┌──────────── Single vLLM Pod (3 GPUs) ────────────┐
│                                                    │
│  GPU 0: vLLM inference (--enforce-eager)           │
│  GPU 1: ┐ torchrun DDP GRPO training               │
│  GPU 2: ┘                                          │
│                                                    │
│  After training:                                   │
│  GPU 1: fresh process loads checkpoint             │
│         → NCCL weight sync to vLLM on GPU 0        │
│         (localhost, no cross-pod issue)             │
└────────────────────────────────────────────────────┘
```

**Why this works:** Trainer and vLLM share localhost. The threaded HTTP+NCCL pattern (POST in background thread, NCCL join in main thread) avoids the deadlock because both sides can reach each other instantly.

> **Note on GPU IDs:** The diagram shows physical GPU IDs. Each process uses `CUDA_VISIBLE_DEVICES` to remap — e.g., vLLM sees physical GPU 0 as its `cuda:0`, while the trainer sees physical GPUs 1-2 as `cuda:0` and `cuda:1`. NCCL uses UUIDs internally, so the physical/logical distinction doesn't affect sync correctness.

---

## Key GitHub References


| Issue/PR                                                    | What                                                      |
| ----------------------------------------------------------- | --------------------------------------------------------- |
| [#31848](https://github.com/vllm-project/vllm/issues/31848) | RFC: Native weight sync APIs (tracking issue, open TODOs) |
| [#31943](https://github.com/vllm-project/vllm/pull/31943)   | NCCL weight sync merged (v0.18.0)                         |
| [#40536](https://github.com/vllm-project/vllm/issues/40536) | CUDA graphs + weight sync incompatibility                 |
| [#41864](https://github.com/vllm-project/vllm/issues/41864) | PP > 1 multi-node blocked                                 |
| [#38550](https://github.com/vllm-project/vllm/issues/38550) | NCCL symmetric memory deadlock on subset GPUs             |
| [#42389](https://github.com/vllm-project/vllm/issues/42389) | TCPStore broken pipe on K8s with vGPU                     |


---

## How Can We Actually Do Multi-Node Multi-GPU?

Given the limitations above, here are the realistic paths forward:

### Option 1: Separate Training and Inference (What Works Today)

```
┌─────────────────────────────────┐     ┌─────────────────────────────┐
│  Training Nodes (KFT v2)        │     │  vLLM Inference Pod          │
│                                 │     │                             │
│  Node 0: 2 GPUs  ┐             │     │  GPU 0: vLLM serving        │
│  Node 1: 2 GPUs  ┘ DDP GRPO    │     │  (--enforce-eager)          │
│                                 │     │                             │
│  Saves checkpoint to shared     │     │  After training finishes:   │
│  storage (PVC / S3 / NFS)  ────────→  │  Restart vLLM with new      │
│                                 │     │  checkpoint path            │
└─────────────────────────────────┘     └─────────────────────────────┘
```

**How:** Train with KFT `num_nodes=2`, save checkpoint to a shared PVC, then restart the vLLM Deployment with the new `--model` path pointing to the checkpoint. No weight sync API needed.

**Pros:** Multi-node training works fully. No deadlock risk.
**Cons:** vLLM has downtime during restart. Not a live/hot weight update.

### Option 2: Same-Pod Colocation (What We Do Now)

```
┌──────────── Single Pod (3 GPUs) ────────────┐
│  GPU 0: vLLM inference                       │
│  GPU 1-2: torchrun DDP GRPO training         │
│  After training: NCCL weight sync on localhost│
└──────────────────────────────────────────────┘
```

**How:** All GPUs in one pod. Train on GPUs 1-2 with DDP, sync to vLLM on GPU 0 via NCCL localhost.

**Pros:** Live weight sync, no restart, no deadlock.
**Cons:** Architecturally limited to GPUs on a single node. In our cluster, that's 3 GPUs per node (cluster-specific limit, not a vLLM constraint).

### Option 3: Multi-Node Training + Same-Pod Sync (Best of Both)

```
  Phase 1: Train                           Phase 2: Sync

┌────────────────────────┐              ┌──────────── vLLM Pod ──────────┐
│  KFT v2 Training Job   │              │                                │
│  Node 0: 2 GPUs ┐      │  checkpoint  │  GPU 0: vLLM inference         │
│  Node 1: 2 GPUs ┘ DDP  │────→ PVC ───→│  GPU 1: load checkpoint        │
│                         │              │         NCCL sync to GPU 0     │
└─────────────────────────┘              │         (localhost, no deadlock)│
                                         └────────────────────────────────┘
```

**How:**

1. Run multi-node GRPO training via KFT v2 (`num_nodes=2, gpu=2`) — full DDP across 4 GPUs
2. Save checkpoint to a shared PVC
3. Inside the vLLM pod, run a sync script that loads the checkpoint on GPU 1 and uses the NCCL weight sync API to push weights to vLLM on GPU 0 (localhost)

**Pros:** Full multi-node training power + live weight sync to vLLM without restart.
**Cons:** Requires shared storage (PVC). vLLM pod needs an extra GPU for the sync process (provision by requesting `nvidia.com/gpu: 2` in the pod spec — one for inference, one for the sync sidecar). Two-step flow.

### Option 4: Hypothetical — vLLM Makes Init Non-Blocking (No One Has Proposed This)

If vLLM were to make `init_weight_transfer_engine` non-blocking (return immediately, complete NCCL join in the background), the cross-pod deadlock would go away and direct trainer-pod → vLLM-pod NCCL sync would become possible. **However, nobody has proposed or requested this — not in RFC #31848, not in any open issue.** The current synchronous design is intentional for their same-machine use case.

---

### Recommendation

**Option 3** is the most practical for production RLHF today:

- Multi-node training gives you scale (4+ GPUs across nodes)
- Localhost weight sync gives you live model updates without restarting vLLM
- The only extra requirement is a shared PVC between training pods and vLLM pod

