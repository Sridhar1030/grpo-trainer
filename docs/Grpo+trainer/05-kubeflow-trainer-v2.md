# Kubeflow Trainer v2 — Deep Reference

> **Docs:** https://www.kubeflow.org/docs/components/trainer/  
> **SDK:** https://sdk.kubeflow.org/en/latest/train/index.html  
> **Repo:** https://github.com/kubeflow/trainer  
> **KEP:** https://github.com/kubeflow/trainer/tree/master/docs/proposals/2170-kubeflow-trainer-v2

---

## Architecture: v1 → v2

| Aspect | Training Operator v1 | Trainer v2 |
|--------|----------------------|------------|
| Job API | Separate CRDs (PyTorchJob, TFJob, …) | Single **TrainJob** + `runtimeRef` |
| Infra template | Embedded in each job's replica specs | **TrainingRuntime** / **ClusterTrainingRuntime** (blueprints) |
| Orchestration | Operator-built pods/services | **JobSet** (`jobset.x-k8s.io`) as workload primitive |
| Scheduling | Various integrations | **Kueue** integration for quota and admission |

---

## TrainJob CRD

`TrainJob` (`trainer.kubeflow.org/v1alpha1`) is what practitioners submit:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: pytorch-distributed
  namespace: default
  labels:
    kueue.x-k8s.io/queue-name: user-queue
spec:
  runtimeRef:
    apiGroup: trainer.kubeflow.org
    name: torch-distributed
    kind: ClusterTrainingRuntime
  trainer:
    numNodes: 2
    resourcesPerNode:
      requests:
        cpu: "4"
        memory: "8Gi"
        nvidia.com/gpu: "1"
```

---

## TrainingRuntime / ClusterTrainingRuntime

- **ClusterTrainingRuntime**: cluster-scoped, reusable from any namespace
- **TrainingRuntime**: namespace-scoped, TrainJob must be in same namespace
- Both embed `spec.template` that becomes the JobSet template

### Built-in Runtimes (manifests)

- `torch-distributed` (PyTorch + torchrun)
- `deepspeed_distributed`
- `mlx_distributed`
- `jax_distributed`
- `xgboost_distributed`
- `torchtune/` (LLM fine-tuning blueprints)

**No built-in RL/GRPO runtime exists.** Use `CustomTrainer` or build a custom runtime.

---

## CustomTrainer API

### Python SDK

```python
from kubeflow.trainer import TrainerClient
from kubeflow.trainer.types import CustomTrainer

def train():
    """All imports must live inside this function."""
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    # ... training logic ...
    dist.destroy_process_group()

client = TrainerClient()
job = client.train(
    runtime="torch-distributed",
    trainer=CustomTrainer(
        func=train,
        func_args={"lr": 1e-4},
        num_nodes=4,
        resources_per_node={"gpu": 1, "cpu": 4, "memory": "32Gi"},
        env={"MY_VAR": "value"},
        packages_to_install=["trl", "vllm"],
    ),
)
```

### Parameters

| Parameter | Meaning |
|-----------|---------|
| `func` | Training callable |
| `func_args` | Optional kwargs dict for `func` |
| `image` | Container image override |
| `packages_to_install` | Pip installs before running |
| `pip_index_urls` | Extra pip index URLs |
| `num_nodes` | Node count |
| `resources_per_node` | e.g. `{"gpu": 1, "cpu": 3, "memory": "16Gi"}` |
| `env` | `dict[str, str]` environment variables |

### Serialization

1. Serialize function with `cloudpickle`
2. Package into container payload
3. Create TrainJob
4. Execute on each training process

**Implication:** Complex RL stacks may prefer `CustomTrainerContainer` with a stable image.

### Distributed Environment

KFT uses `torchrun`. Practitioners should use:
- `dist.get_world_size()`, `dist.get_rank()`
- `os.environ["LOCAL_RANK"]`

---

## TrainerClient.train() Signature

```python
TrainerClient.train(
    runtime: str | Runtime | None = None,    # default: "torch-distributed"
    initializer: Initializer | None = None,
    trainer: CustomTrainer | CustomTrainerContainer | BuiltinTrainer | None = None,
    options: list | None = None,
) -> str  # returns job name
```

### Monitoring

```python
client.wait_for_job_status(job, timeout=86400)
tj = client.get_job(job)
for line in client.get_job_logs(job, step="node-0", follow=True):
    print(line, end="")
```

---

## PVCs, Volumes, Secrets (via options)

PVCs are attached via `RuntimePatch` → `PodSpecPatch`:

```python
from kubeflow.trainer.options import (
    RuntimePatch, TrainingRuntimeSpecPatch, JobSetTemplatePatch,
    JobSetSpecPatch, ReplicatedJobPatch, JobTemplatePatch,
    JobSpecPatch, PodTemplatePatch, PodSpecPatch, ContainerPatch,
)

pvc_patch = RuntimePatch(
    training_runtime_spec=TrainingRuntimeSpecPatch(
        template=JobSetTemplatePatch(
            spec=JobSetSpecPatch(
                replicated_jobs=[
                    ReplicatedJobPatch(
                        name="node",
                        template=JobTemplatePatch(
                            spec=JobSpecPatch(
                                template=PodTemplatePatch(
                                    spec=PodSpecPatch(
                                        volumes=[{
                                            "name": "cache",
                                            "persistentVolumeClaim": {"claimName": "grpo-cache"},
                                        }],
                                        containers=[ContainerPatch(
                                            name="node",
                                            volume_mounts=[{
                                                "name": "cache",
                                                "mountPath": "/cache",
                                            }],
                                        )],
                                    )
                                )
                            )
                        ),
                    )
                ]
            )
        )
    )
)

job = client.train(
    trainer=CustomTrainer(func=train, env={"HF_HOME": "/cache/hf"}),
    options=[pvc_patch],
)
```

### Kueue Queue Label

```python
from kubeflow.trainer.options import Labels
client.train(..., options=[Labels(labels={"kueue.x-k8s.io/queue-name": "gpu-queue"})])
```

---

## JobSet Integration

KFT controller composes a **JobSet** from runtime template + TrainJob spec + policies.

### Multi-Role JobSets

Runtimes can declare **multiple `replicatedJobs`** with `dependsOn` edges:
- `dataset-initializer` → `model-initializer` → `trainer`

**Caveat for GRPO:** This flexibility is oriented around **training pipeline roles**, not a first-class "async inference fleet + trainer" feature. A separate vLLM Deployment for rollouts is NOT automatically wired into the same TrainJob lifecycle.

---

## Kueue Integration

- Enable TrainJob in Kueue configuration (Trainer v2.0.0+)
- Select queue with label: `kueue.x-k8s.io/queue-name: <local-queue>`
- Default `spec.suspend: true` → Kueue unsuspends when admitted
- Gang scheduling via `podGroupPolicy` in ClusterTrainingRuntime

---

## Limitations for RL Workloads

| Limitation | Impact |
|------------|--------|
| Single-role TrainJob | Can't natively co-locate trainer + vLLM server in one TrainJob |
| Batch/checkpoint oriented | No managed async RL event loop, trajectory store, or sample-rate SLOs |
| No RL runtime | No built-in environment stepping API, rollout worker CRD, or GRPO implementation |
| cloudpickle serialization | Complex RL stacks may break; prefer container images |
| vLLM not accounted | GPU/memory for vLLM pods doesn't participate in TrainJob Kueue accounting |
| Networking | RL needs wide egress to rollout services; requires NetworkPolicy, TLS, tokens |

### Workarounds

- Use raw **JobSet** with two `ReplicatedJob` roles (trainer + vLLM)
- Deploy vLLM as separate **Deployment** + Service; trainer calls it via HTTP
- Use `CustomTrainerContainer` with stable image for complex dependencies

---

## References

- [Trainer overview](https://www.kubeflow.org/docs/components/trainer/overview/)
- [Runtimes guide](https://www.kubeflow.org/docs/components/trainer/operator-guides/runtime/)
- [Job template guide](https://www.kubeflow.org/docs/components/trainer/operator-guides/job-template/)
- [PyTorch user guide](https://www.kubeflow.org/docs/components/trainer/user-guides/pytorch/)
- [Custom training (SDK)](https://sdk.kubeflow.org/en/latest/train/custom-training.html)
- [Options reference](https://sdk.kubeflow.org/en/latest/train/options.html)
- [API reference](https://sdk.kubeflow.org/en/latest/train/api.html)
- [Kueue + TrainJob](https://kueue.sigs.k8s.io/docs/tasks/run/trainjobs/)
- [KEP-2170](https://github.com/kubeflow/trainer/blob/master/docs/proposals/2170-kubeflow-trainer-v2/README.md)
