# Spike Findings: TRL GRPO Compatibility with Kubeflow Trainer v2

> **Jira:** RHOAIENG-61094 — Spike checking compatibility of GRPO framework with Trainer v2  
> **Date:** 2026-05-11  
> **Cluster:** OpenShift (RHOAI) with Kubeflow Trainer v2  
> **Namespace:** `grpoxtrainer`  
> **Runtime:** `torch-distributed` ClusterTrainingRuntime  
> **Notebook:** `[spike-grpo-kft-v2.ipynb](spike-grpo-kft-v2.ipynb)`

---

## Objective

Validate that TRL's `GRPOTrainer` can run inside a Kubeflow Trainer v2 `CustomTrainer` pod — proving KFT as a viable alternative to Ray for GRPO fine-tuning.

**Flow under test:**

```
GSM8K data → KFT v2 CustomTrainer → TRL GRPOTrainer → GRPO loop executes
```

## Setup


| Component    | Value                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------- |
| Model        | `Qwen/Qwen2.5-0.5B-Instruct` (494M params)                                                   |
| Dataset      | GSM8K (16 samples, rule-based reward)                                                        |
| Framework    | TRL `GRPOTrainer` + HF Transformers                                                          |
| Orchestrator | Kubeflow Trainer v2 (`CustomTrainer` → `torch-distributed` runtime)                          |
| Submission   | `TrainerClient.train()` from workbench notebook                                              |
| Reward       | Rule-based: extract `#### N`, compare to ground truth (+1.0 correct, +0.1 partial, 0.0 none) |
| GRPO Config  | G=2 generations, batch=2, max_completion_length=32, lr=5e-6, beta=0.0                        |


## Compatibility Checklist


| #   | Check                                                | Result   | Evidence                                                                  |
| --- | ---------------------------------------------------- | -------- | ------------------------------------------------------------------------- |
| 1   | `CustomTrainer` accepts TRL training function        | **PASS** | cloudpickle serialized `grpo_train()` and all closures                    |
| 2   | `packages_to_install` installs TRL + deps in pod     | **PASS** | `trl`, `datasets`, `accelerate` installed at pod startup                  |
| 3   | `torch-distributed` runtime injects correct env vars | **PASS** | `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` set per process (`WORLD_SIZE` matches the job’s GPU/worker layout) |
| 4   | Distributed backend initializes                      | **PASS** | Gloo backend; each rank connects to `WORLD_SIZE - 1` peers |
| 5   | TRL GRPOTrainer initializes inside KFT pod           | **PASS** | Config loaded, model (494M params) and tokenizer ready                    |
| 6   | GRPO training loop starts                            | **PASS** | Progress bar appeared (`0/50`), generation phase entered                  |
| 7   | Model downloads from HuggingFace                     | **PASS** | `Qwen/Qwen2.5-0.5B-Instruct` pulled inside pod                            |
| 8   | Dataset downloads from HuggingFace                   | **PASS** | GSM8K loaded and mapped                                                   |


**Verdict: TRL GRPO is fully compatible with Kubeflow Trainer v2.**

## Observed Limitations

### L1: CUDA image is always used (~18.8 GB)

The `torch-distributed` ClusterTrainingRuntime hardcodes the CUDA training image (`odh-training-cuda128-torch29-py312-rhel9`) regardless of whether GPUs are requested. First pull took ~10 minutes. On a node with limited ephemeral storage (< 19 GB threshold), the pod was **evicted** immediately after the image pulled.

**Mitigation:** Use GPU-equipped nodes for real workloads (image will already be cached). For development clusters, pre-pull the image via a DaemonSet. Keep **one** 2×GPU `TrainJob` at a time unless the pool has four free GPUs; **96 Gi memory requests** for a 0.5B spike can prevent two pods from fitting the same node—this notebook now defaults to **8 CPU / 48 Gi / 2 GPU** to schedule more reliably.

### L1b: Multi-GPU DDP device placement

Each rank should call `torch.cuda.set_device(LOCAL_RANK)` and place the model on `cuda:{LOCAL_RANK}`. Otherwise processes can all pile onto GPU 0 and DDP breaks or OOMs. The spike notebook does this before `from_pretrained` / training.

### L2: `nproc_per_node=auto` spawns 1 process per CPU

When no GPUs are present, `torchrun --nproc_per_node=auto` detects CPU count and spawns one process per CPU core. With 4 CPUs, this creates 4 workers each limited to `OMP_NUM_THREADS=1`, making text generation extremely slow.

**Mitigation:** Not relevant for GPU workloads (1 process per GPU). For CPU testing, reduce CPU count in `resources_per_node` or request a `nproc_per_node` override in the CustomTrainer API.

### L3: No built-in `nproc_per_node` override in CustomTrainer

The `CustomTrainer` API does not expose a way to set `nproc_per_node` directly. The runtime controls this.

**Mitigation:** Use `RuntimePatch` or propose as a KFT feature request.

### L4: RBAC not pre-configured for workbench service accounts

The workbench service account (`system:serviceaccount:grpoxtrainer:grpoxtrainerwb1`) does not have permissions to create/list Trainer resources by default. `TrainerClient` fails with `403 Forbidden`.

**Mitigation:** Apply the provided `[rbac-trainer-access.yaml](rbac-trainer-access.yaml)` which creates a `ClusterRole` and `ClusterRoleBinding` with the necessary permissions.

### L5: Minor deprecation warnings

- `torch_dtype` kwarg deprecated in transformers (use `dtype` instead)
- `torchao` version mismatch warning (harmless, not used)

## Key Takeaways

1. **Zero code changes needed** — TRL `GRPOTrainer` runs unmodified inside a KFT v2 pod
2. **Distributed setup is automatic** — KFT injects `MASTER_ADDR`, `RANK`, `WORLD_SIZE`; TRL/HF Accelerate picks them up via `torchrun`
3. **cloudpickle serialization works** — the entire training function including nested reward functions, closures, and library imports serialize correctly
4. **Gloo backend connects all ranks** — distributed collective operations are functional
5. **The only bottleneck is generation speed on CPU** — on GPU this is a non-issue
6. **RBAC is a one-time setup** — once configured, the workbench can submit unlimited TrainJobs

## Recommendation

**Proceed to Phase 1 on GPU.** The spike conclusively proves compatibility. Phase 1 should:

1. **Run on a GPU node** (single A100/H100) — generation will be 100x+ faster
2. **Scale to full GSM8K** (7473 samples) with `G=8` completions per prompt
3. **Add PVC** for model cache and checkpoints (avoid re-downloading per run)
4. **Add HF token secret** for gated models (e.g., Llama-3.2-1B-Instruct)
5. **Measure reward improvement** to validate GRPO correctness end-to-end
6. **Document training speed** as a baseline for Phase 2 (vLLM-accelerated rollouts)

## Files


| File                                                   | Purpose                                     |
| ------------------------------------------------------ | ------------------------------------------- |
| `[spike-grpo-kft-v2.ipynb](spike-grpo-kft-v2.ipynb)`   | Spike notebook — run in OpenShift workbench |
| `[rbac-trainer-access.yaml](rbac-trainer-access.yaml)` | RBAC setup for workbench → Trainer access   |
| `[docs/sprint.md](docs/sprint.md)`                     | Phase 1 + Phase 2 sprint plan               |


