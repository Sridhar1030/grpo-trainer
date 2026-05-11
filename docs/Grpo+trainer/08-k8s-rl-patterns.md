# Kubernetes RL Architecture Patterns

---

## Community Landscape

There is **no dedicated Kubernetes RL Working Group**. Relevant community surfaces:

| Group | Relevance |
|-------|-----------|
| **WG Batch** | Batch scheduling, queueing for distributed jobs |
| **JobSet** (SIG Apps) | Multi-job distributed ML/HPC API |
| **Kueue** (SIG Scheduling) | Quotas, queuing, admission |
| **WG AI Gateway** | Broader AI on K8s interoperability |

RL training is modeled as **multi-role batch**: rollout/inference + trainer + optional reference/critic + reward workers.

---

## JobSet for Multi-Role RL

### Two-Role JobSet: Trainer + vLLM

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: grpo-rl
  labels:
    kueue.x-k8s.io/queue-name: team-rl-queue
spec:
  network:
    subdomain: grpo-rl
    enableDNSHostnames: true
  coordinator:
    replicatedJob: trainer
    jobIndex: 0
    podIndex: 0
  failurePolicy:
    maxRestarts: 2
  replicatedJobs:
    - name: trainer
      replicas: 1
      template:
        spec:
          parallelism: 4
          completions: 4
          backoffLimit: 0
          template:
            spec:
              restartPolicy: OnFailure
              containers:
                - name: trainer
                  image: your-registry/grpo-trainer:latest
                  resources:
                    limits:
                      nvidia.com/gpu: 1
                  env:
                    - name: VLLM_SERVICE
                      value: "grpo-rl-vllm-0-0.grpo-rl"
                    - name: NCCL_SOCKET_IFNAME
                      value: eth0
    - name: vllm
      replicas: 1
      template:
        spec:
          parallelism: 2
          completions: 2
          backoffLimit: 0
          template:
            spec:
              restartPolicy: OnFailure
              containers:
                - name: vllm
                  image: vllm/vllm-openai:latest
                  resources:
                    limits:
                      nvidia.com/gpu: 1
```

### Gang Scheduling with Kueue

- Label JobSet: `kueue.x-k8s.io/queue-name: <local-queue>`
- Kueue admits when quotas allow all pods
- Requires Kueue v0.8.3+ / v0.9.0+ for coordinator support

---

## NCCL on Kubernetes

### Transport Options

| Environment | Transport | Notes |
|-------------|-----------|-------|
| Generic cloud K8s | **Socket** (TCP/IP) | Easiest; tune MTU, CNI, interface |
| GPU-optimized clusters | **IB** (InfiniBand/RoCE) | Higher bandwidth; requires hardware + CNI |

### Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `NCCL_DEBUG` | `INFO` or `WARN` for debugging |
| `NCCL_SOCKET_IFNAME` | Interface selection (often mandatory with multiple NICs) |
| `NCCL_NET` | Force `Socket` or `IB` |
| `NCCL_IB_HCA` | InfiniBand HCA selection |
| `NCCL_IB_GID_INDEX` | GID index for RoCE |
| `NCCL_IB_TIMEOUT` | IB timeout |
| `NCCL_IB_RETRY_CNT` | IB retry count |
| `NCCL_TIMEOUT` | Collective timeout |

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Connection refused | Wrong MASTER_ADDR, DNS, CNI | Fix service/subdomain; verify pod-to-pod connectivity |
| Hang at first collective | Interface mismatch (picked lo) | Set `NCCL_SOCKET_IFNAME` |
| Slow then timeout | Stragglers, oversubscribed GPUs | Reduce batch; topology-aware scheduling |
| IB errors on cloud | Wrong GID/TC | Set `NCCL_IB_GID_INDEX` |

---

## GPU Scheduling and Memory

### Resource Requests

Use `nvidia.com/gpu` in `limits`. Requests must match limits for GPUs.

### GPU Sharing Options

| Method | Notes |
|--------|-------|
| **MIG** | Hardware-isolated instances on supported GPUs |
| **Time-slicing** | Logical oversubscription; throughput not additive |
| **Separate pods** | Most K8s-idiomatic for training + inference |

For GRPO: **disaggregated** (trainer pods + inference pods on separate GPUs) is most Kubernetes-idiomatic.

---

## Storage for RL Training

### Shared PVC Pattern

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rl-shared
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 2Ti
  storageClassName: nfs-client
```

### Storage Backend Options

| Backend | Notes |
|---------|-------|
| **NFS** | Ubiquitous RWX; watch latency for many small files (HF cache) |
| **CephFS / Lustre** | Better at scale; common in HPC-style K8s |
| **Cloud RWX** (EFS, Filestore, Azure Files) | Managed; tune performance tier |

### HuggingFace Cache

- Set `HF_HOME` on shared PV for model downloads
- HF also publishes a CSI driver for mounting models

---

## Agentic Continual Learning

### The Continual Loop

```
Data/Product → Prompts → Rollout (vLLM) → GRPO Trainer → Deploy Best Checkpoint → Repeat
```

On Kubernetes: CronJob / CI / Argo drives new TrainJobs, plus KServe/vLLM for rollout.

### Related Projects

- [redhat-et/agentic-reasoning-reinforcement-fine-tuning](https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning) — hands-on cookbook for agentic reasoning + RFT using TRL + GRPO, positioned for OpenShift AI
- [NeMo-RL on GKE](https://gke-ai-labs.dev/docs/blueprints/nemo-rl-on-gke/) — Ray + shared storage + multi-role workers

---

## KubeRay vs Kubeflow Trainer

| Dimension | KubeRay | KFT |
|-----------|---------|-----|
| Programming model | Python-first task graph (@ray.remote) | CRD-first TrainJob; GitOps-friendly |
| RL ecosystem | RLlib; verl on KubeRay documented | Your container (TRL, etc.) + JobSet |
| Kueue | RayJob + Kueue patterns | First-class Kueue integration |
| **Best for** | Ray-native apps (OpenRLHF, veRL) | K8s-native batch (TRL, custom) |

**They can coexist** — different operators in the same cluster. Use KubeRay when the app is Ray-native; use KFT for K8s-native batch semantics.

### The Scheduling Tension

Both sit above a core issue: K8s default scheduler is pod-at-a-time, but distributed RL wants coordinated placement. Mitigations:
- Kueue for admission + workload-specific integrations
- Optional secondary schedulers / vendor batch schedulers

---

## Reference Architecture: GRPO on KFT

```
                    ┌──────────── Kueue ─────────────┐
                    │ ClusterQueue / LocalQueue       │
                    └────────────┬────────────────────┘
                                 │
                    JobSet (Train + Inference roles)
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
   ┌─────▼──────┐         ┌─────▼──────┐         ┌──────▼───────┐
   │ Replicated │         │ Replicated │         │ Object Store │
   │ Job:       │         │ Job:       │         │ (checkpoints)│
   │ trainers   │         │ vLLM       │         └──────────────┘
   └─────┬──────┘         └─────┬──────┘
         │                       │
         └───── RWX PVC ─────────┘
         (HF cache, checkpoints)
```

---

## References

- [Introducing JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset/)
- [JobSet concepts](https://jobset.sigs.k8s.io/docs/concepts)
- [Kueue: run JobSets](https://kueue.sigs.k8s.io/docs/tasks/run/jobsets/)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Schedule GPUs (K8s)](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
- [verl on KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html)
