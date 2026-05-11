# Sprint Plan: Build a Kubernetes-Native RL Framework for KFT

> **Epic:** RHOAIENG-61094  
> **Status:** Exploratory — pursue after TRL sprint validates the core GRPO loop  
> **Objective:** Build an open-source RL post-training framework that is Kubernetes/KFT-native, the way OpenRLHF is Ray-native  
> **Constraint:** No Ray. Pure Kubernetes primitives (JobSet, Kueue, NCCL, vLLM).  
> **Codename:** `trainer-rl` (working name)

---

## Why Build Our Own

| Existing option | Problem |
|-----------------|---------|
| **TRL** | Great GRPO algorithm impl, but no multi-role orchestration, slow HF rollouts, single TrainJob only |
| **veRL** | Best architecture (HybridFlow), but Ray is mandatory — forking is a maintenance burden |
| **OpenRLHF** | Most complete feature set, but Ray is the skeleton — can't realistically strip it |

**The gap:** No framework exists that is Kubernetes-native AND handles the full RL loop (rollout + training + weight sync + reward). TRL handles the algorithm but not the infrastructure. veRL/OpenRLHF handle the infrastructure but require Ray.

**The opportunity:** Build a thin orchestration layer on top of:
- **KFT/JobSet** for multi-role pod scheduling (trainer + vLLM + optional reward model)
- **vLLM** for fast rollout generation (already Kubernetes-friendly, no Ray needed)
- **PyTorch FSDP/DeepSpeed** for distributed training (already works on KFT)
- **vLLM PR #31943** for NCCL weight sync (already merged, HTTP + NCCL)
- **TRL's GRPO math** for the algorithm itself (can vendor or depend on)

This is NOT about reinventing GRPO. It's about building the **Kubernetes-native orchestration** that connects these pieces — the thing Ray provides for veRL/OpenRLHF, but on KFT instead.

---

## Sprint Overview

| Phase | What | Duration | Outcome |
|-------|------|----------|---------|
| **Phase 1** | Architecture design + core abstractions | 1 week | Design doc, API surface, project scaffold |
| **Phase 2** | RL loop engine: rollout → reward → advantage → train → sync | 2 weeks | Working single-node GRPO on GSM8K |
| **Phase 3** | KFT integration: JobSet runtime, multi-role, Python SDK | 2 weeks | `TrainJob` submits full RL job |
| **Phase 4** | Hardening, docs, and upstream proposal | 1 week | Publishable PoC + KEP draft |

**Total: 6 weeks**

---

## Phase 1: Architecture Design (Week 1)

### Goal
Design the framework's abstractions, API surface, and component boundaries before writing code.

### Target Architecture

```
                         trainer-rl Python SDK
                                │
                    ┌───────────▼───────────┐
                    │     RLJob (new CRD     │
                    │     or TrainJob +      │
                    │     custom runtime)    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │       JobSet           │
                    │  (Kueue gang-scheduled)│
                    └─────┬─────────┬───────┘
                          │         │
              ┌───────────▼──┐  ┌───▼───────────┐
              │ ReplicatedJob│  │ ReplicatedJob  │
              │ "trainer"    │  │ "rollout"      │
              │ N pods       │  │ M pods         │
              │ FSDP/DeepSpd │  │ vLLM serve     │
              │ GRPO update  │  │ weight-transfer│
              └──────┬───────┘  └───┬────────────┘
                     │              │
                     └── NCCL ──────┘  (weight sync)
                     └── HTTP ──────┘  (rollout requests)
                     └── RWX PVC ───┘  (checkpoints, HF cache)
```

### Stories

#### 1.1 — Define the framework's scope and non-goals
**Points:** 2  
**Description:** Write a design doc that clearly states what this framework IS and ISN'T.

**IS:**
- [ ] Kubernetes-native RL post-training orchestrator
- [ ] Multi-role job management: trainer + rollout (vLLM) + optional reward model
- [ ] GRPO as first algorithm, extensible to PPO/REINFORCE++/RLOO
- [ ] NCCL weight sync between trainer and vLLM (PR #31943)
- [ ] Python SDK for job submission
- [ ] Works on KFT (TrainJob/ClusterTrainingRuntime) or raw JobSet

**ISN'T:**
- [ ] A new RL algorithm library (use TRL/custom for the math)
- [ ] A replacement for Ray in general (only for RL post-training on K8s)
- [ ] A model serving framework (vLLM handles serving)
- [ ] An environment simulator (this is LLM RL, not game/robotics RL)

---

#### 1.2 — Design the core abstractions
**Points:** 5  
**Description:** Define the Python classes and interfaces.

**Core abstractions:**

```python
class RLConfig:
    """Top-level configuration for an RL training job."""
    algorithm: AlgorithmConfig      # GRPO, PPO, etc.
    trainer: TrainerConfig          # FSDP/DeepSpeed, LR, batch sizes
    rollout: RolloutConfig          # vLLM server settings
    reward: RewardConfig            # reward function or model endpoint
    weight_sync: WeightSyncConfig   # NCCL backend, sync frequency
    data: DataConfig                # dataset, prompt formatting
    infrastructure: InfraConfig     # num_nodes, GPUs, PVC, namespace

class AlgorithmConfig:
    name: str                       # "grpo", "ppo", "reinforce_pp"
    num_generations: int            # G: completions per prompt
    clip_ratio: float               # PPO clip epsilon
    kl_coef: float                  # beta for KL penalty
    normalize_advantage: bool       # z-score within group

class RolloutConfig:
    engine: str                     # "vllm" (future: "sglang", "trtllm")
    num_instances: int              # number of vLLM pods
    gpus_per_instance: int
    tensor_parallel_size: int
    temperature: float
    top_p: float
    max_tokens: int
    gpu_memory_utilization: float

class WeightSyncConfig:
    backend: str                    # "nccl" or "ipc"
    sync_every_n_steps: int         # how often to push weights
    pause_generation: bool          # pause vLLM during sync

class RewardConfig:
    type: str                       # "function", "model", "endpoint"
    function: Callable | None       # rule-based reward (GSM8K, code exec)
    endpoint: str | None            # HTTP URL for reward model server
```

**Acceptance criteria:**
- [ ] All abstractions defined with type hints and docstrings
- [ ] Config is serializable to YAML (for CRD-style usage)
- [ ] Config is constructible from Python (for SDK-style usage)
- [ ] Separation of concerns: algorithm logic vs infrastructure vs rollout

---

#### 1.3 — Design the multi-role JobSet template
**Points:** 3  
**Description:** Design the Kubernetes manifests that the framework generates.

**Deliverables:**
- [ ] JobSet YAML template with parameterized roles (trainer, rollout, optional reward-model)
- [ ] ClusterTrainingRuntime YAML for KFT integration
- [ ] Environment variable injection plan (MASTER_ADDR, RANK, VLLM_SERVICE, etc.)
- [ ] NCCL networking plan: how trainer pods discover vLLM pods for weight sync
- [ ] PVC mounting strategy: shared cache + checkpoints

---

### Phase 1 Definition of Done

- [ ] Design doc reviewed and agreed upon
- [ ] API surface defined (Python classes + YAML templates)
- [ ] Project scaffold created: `trainer_rl/`, `tests/`, `manifests/`, `examples/`

---

## Phase 2: RL Loop Engine (Weeks 2–3)

### Goal
Implement the core RL training loop that connects all the pieces. Get GRPO working on a single node first, then multi-node.

### Stories

#### 2.1 — Implement the RLEngine (main loop)
**Points:** 8  
**Description:** The central orchestrator that drives the RL loop.

```python
class RLEngine:
    def __init__(self, config: RLConfig): ...

    def run(self):
        self.setup_distributed()      # torch.distributed init
        self.load_model()             # policy + reference
        self.connect_rollout()        # connect to vLLM
        self.init_weight_sync()       # NCCL group with vLLM

        for epoch in range(self.config.trainer.epochs):
            for batch in self.dataloader:
                # 1. Get rollouts from vLLM
                completions = self.rollout_client.generate(batch.prompts)

                # 2. Score rewards
                rewards = self.reward_fn(completions, batch)

                # 3. Compute advantages (algorithm-specific)
                advantages = self.algorithm.compute_advantages(rewards)

                # 4. Compute log-probs for policy and reference
                policy_logprobs = self.forward_policy(completions)
                ref_logprobs = self.forward_reference(completions)

                # 5. Compute loss (PPO-clip + KL)
                loss = self.algorithm.compute_loss(
                    policy_logprobs, ref_logprobs, advantages
                )

                # 6. Backward + optimizer step (FSDP handles AllReduce)
                loss.backward()
                self.optimizer.step()

                # 7. Sync weights to vLLM
                if step % self.config.weight_sync.sync_every_n_steps == 0:
                    self.sync_weights_to_vllm()

            self.save_checkpoint()
```

**Acceptance criteria:**
- [ ] Full loop runs end-to-end on a single machine
- [ ] Each step is independently testable
- [ ] Logging: loss, reward mean, advantage stats, sync latency per step
- [ ] Graceful shutdown and checkpoint saving

---

#### 2.2 — Implement GRPO algorithm module
**Points:** 5  
**Description:** The algorithm-specific logic: advantage estimation and loss computation.

```python
class GRPOAlgorithm:
    def compute_advantages(self, rewards: Tensor, group_size: int) -> Tensor:
        """Z-score within each group of G completions."""
        groups = rewards.view(-1, group_size)
        mean = groups.mean(dim=1, keepdim=True)
        std = groups.std(dim=1, keepdim=True) + 1e-8
        return ((groups - mean) / std).view(-1)

    def compute_loss(self, policy_lp, ref_lp, advantages, clip_ratio, kl_coef):
        """PPO-clip loss + KL penalty."""
        ratio = (policy_lp - old_lp).exp()
        clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        kl = ref_lp / policy_lp.exp() - (ref_lp - policy_lp) - 1  # low-var KL
        kl_loss = kl_coef * kl.mean()

        return policy_loss + kl_loss
```

**Acceptance criteria:**
- [ ] Advantage computation matches reference (veRL/TRL) on synthetic data
- [ ] Loss computation matches reference on synthetic data
- [ ] Unit tests with known inputs/outputs
- [ ] Handles edge case: all rewards identical (std=0)
- [ ] Extensible: base class `Algorithm` with `GRPOAlgorithm` subclass

---

#### 2.3 — Implement vLLM rollout client
**Points:** 5  
**Description:** HTTP client that talks to vLLM's OpenAI-compatible API for rollout generation.

```python
class VLLMRolloutClient:
    def __init__(self, base_url: str, config: RolloutConfig): ...

    def generate(self, prompts: list[str]) -> list[Completion]:
        """Generate G completions per prompt via vLLM."""
        # Uses /v1/completions with n=G, temperature, top_p, max_tokens
        ...

    def get_world_size(self) -> int:
        """GET /get_world_size for NCCL group setup."""
        ...
```

**Acceptance criteria:**
- [ ] Generates completions with configurable temperature, top_p, max_tokens
- [ ] Returns token IDs and log-probs (needed for advantage computation)
- [ ] Handles batching efficiently (one request per prompt, or batched)
- [ ] Timeout and retry logic for vLLM server
- [ ] Works with vLLM's OpenAI-compatible API

---

#### 2.4 — Implement weight sync manager
**Points:** 5  
**Description:** Wraps vLLM PR #31943's weight transfer protocol.

```python
class WeightSyncManager:
    def __init__(self, config: WeightSyncConfig, vllm_url: str): ...

    def init_nccl_group(self, model): ...
    def sync_weights(self, model): ...
        # pause → start_weight_update → NCCL broadcast → finish → resume
```

**Acceptance criteria:**
- [ ] Full 4-phase protocol implemented (init, start, update, finish)
- [ ] NCCL broadcast works between trainer and vLLM
- [ ] Pause/resume generation during sync
- [ ] Handles FSDP-sharded models (gather params before broadcast)
- [ ] Sync latency logged per call

---

#### 2.5 — End-to-end validation: single-node GRPO on GSM8K
**Points:** 5  
**Description:** Run the full loop on one machine. Trainer + vLLM as subprocess.

**Acceptance criteria:**
- [ ] vLLM starts as subprocess with weight-transfer enabled
- [ ] GRPO loop runs: prompt → vLLM rollout → GSM8K reward → advantage → update → weight sync
- [ ] Reward improves over training steps
- [ ] Weight sync works (vLLM generates better completions after sync)
- [ ] No Ray anywhere in the process tree

---

### Phase 2 Definition of Done

- [ ] `trainer_rl` package runs GRPO on GSM8K end-to-end
- [ ] All components independently testable
- [ ] Single-node performance baselined

---

## Phase 3: KFT Integration (Weeks 4–5)

### Goal
Make the framework submit and manage RL jobs on Kubernetes via KFT.

### Stories

#### 3.1 — Create the ClusterTrainingRuntime for RL
**Points:** 5  
**Description:** A reusable KFT runtime that launches multi-role RL jobs.

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: rl-grpo
spec:
  template:
    spec:
      replicatedJobs:
        - name: trainer
          # FSDP/DeepSpeed training pods
          template:
            spec:
              parallelism: {{num_trainer_nodes}}
              completions: {{num_trainer_nodes}}
              template:
                spec:
                  containers:
                    - name: trainer
                      image: {{trainer_image}}
                      resources:
                        limits:
                          nvidia.com/gpu: {{gpus_per_trainer}}
        - name: rollout
          # vLLM inference pods
          template:
            spec:
              parallelism: {{num_rollout_nodes}}
              completions: {{num_rollout_nodes}}
              template:
                spec:
                  containers:
                    - name: vllm
                      image: {{vllm_image}}
                      resources:
                        limits:
                          nvidia.com/gpu: {{gpus_per_rollout}}
```

**Acceptance criteria:**
- [ ] ClusterTrainingRuntime YAML templated and parameterized
- [ ] Works with KFT's `TrainJob` via `runtimeRef`
- [ ] Kueue gang-schedules both roles together
- [ ] DNS hostnames work for inter-role communication
- [ ] Environment variables injected: MASTER_ADDR, RANK, VLLM_SERVICE

---

#### 3.2 — Build the Python SDK integration
**Points:** 5  
**Description:** Users submit RL jobs with a simple Python API.

```python
from trainer_rl import RLClient, GRPOConfig, RolloutConfig

client = RLClient()

job = client.train(
    model="meta-llama/Llama-3.2-1B-Instruct",
    algorithm=GRPOConfig(
        num_generations=8,
        kl_coef=0.001,
        clip_ratio=0.2,
    ),
    rollout=RolloutConfig(
        engine="vllm",
        num_instances=2,
        gpus_per_instance=1,
        temperature=0.7,
    ),
    reward_fn=gsm8k_reward,
    dataset="gsm8k",
    trainer_nodes=2,
    gpus_per_trainer_node=2,
    namespace="kubeflow",
)

client.wait(job)
client.logs(job, role="trainer")
```

**Acceptance criteria:**
- [ ] `RLClient.train()` creates a TrainJob or JobSet
- [ ] Config validated before submission
- [ ] `.wait()`, `.logs()`, `.status()` monitoring methods
- [ ] Can use existing ClusterTrainingRuntime or generate ad-hoc JobSet

---

#### 3.3 — Multi-node validation on KFT cluster
**Points:** 8  
**Description:** Run GRPO with 2 trainer nodes + 2 vLLM nodes on a real KFT cluster.

**Acceptance criteria:**
- [ ] All pods scheduled and start correctly
- [ ] Trainer pods form FSDP group (NCCL AllReduce works)
- [ ] Trainer pods connect to vLLM pods via HTTP (rollout generation)
- [ ] NCCL weight sync works across pods (trainer → vLLM)
- [ ] GRPO training completes on GSM8K with improving rewards
- [ ] Checkpoint saved to shared PVC

---

#### 3.4 — Add support for custom reward functions and reward model servers
**Points:** 3  
**Description:** Reward can be a Python function (GSM8K), an HTTP endpoint (reward model), or a container sidecar.

**Acceptance criteria:**
- [ ] Rule-based reward: Python function passed via config (GSM8K, code execution)
- [ ] Model reward: HTTP POST to an external reward model endpoint
- [ ] Sidecar reward: third ReplicatedJob role in JobSet running a reward model
- [ ] Reward type is pluggable via `RewardConfig`

---

### Phase 3 Definition of Done

- [ ] Full RL job runs on KFT: trainer + vLLM + reward
- [ ] Python SDK submits and monitors jobs
- [ ] Multi-node training validated with improving rewards
- [ ] ClusterTrainingRuntime is reusable

---

## Phase 4: Hardening and Upstream (Week 6)

### Stories

#### 4.1 — Error handling, retries, and observability
**Points:** 5  
**Description:** Make the framework production-grade.

**Acceptance criteria:**
- [ ] NCCL timeout detection and recovery (retry sync, not crash)
- [ ] vLLM server health check before starting training
- [ ] Structured logging with step number, reward stats, sync latency, GPU memory
- [ ] Metrics export (Prometheus-compatible) for reward curve, throughput, memory
- [ ] Graceful shutdown: save checkpoint on SIGTERM (K8s pod termination)

---

#### 4.2 — Documentation and examples
**Points:** 3  
**Description:** README, quickstart, API reference, and example notebooks.

**Deliverables:**
- [ ] README with architecture diagram and quickstart
- [ ] `examples/gsm8k_grpo.py` — minimal GRPO on GSM8K
- [ ] `examples/custom_reward.py` — bring-your-own reward function
- [ ] `examples/jobset.yaml` — raw JobSet without Python SDK
- [ ] API reference (auto-generated from docstrings)

---

#### 4.3 — Performance benchmark and comparison
**Points:** 3  
**Description:** Formal benchmark comparing trainer-rl vs TRL vs Ray-based stacks.

**Deliverables:**
- [ ] Benchmark script: fixed model, dataset, hardware config
- [ ] Metrics: rollout tokens/sec, training steps/hour, weight sync latency, peak GPU memory
- [ ] Comparison table: trainer-rl vs TRL-on-KFT vs veRL-on-Ray (literature numbers)
- [ ] Analysis: where does trainer-rl win, where does it lose, and why

---

#### 4.4 — KEP draft for KFT RL runtime
**Points:** 3  
**Description:** Write a Kubeflow Enhancement Proposal for an upstream RL training runtime.

**Deliverables:**
- [ ] KEP following KFT proposal format (see KEP-2170 as template)
- [ ] Problem statement: RL post-training needs multi-role orchestration
- [ ] Proposed solution: `rl-grpo` ClusterTrainingRuntime + SDK extensions
- [ ] Alignment with KEP-2839 (TRL backend integration, GSoC project)
- [ ] Alignment with Kubernetes RL community (WG Batch, JobSet)
- [ ] Discussion opened on kubeflow/trainer repo

---

### Phase 4 Definition of Done

- [ ] Framework is documented and has examples
- [ ] Benchmark numbers published
- [ ] KEP draft submitted to kubeflow/trainer

---

## Sprint Capacity

| Story | Points | Phase | Depends on |
|-------|--------|-------|------------|
| 1.1 Scope and non-goals | 2 | 1 | — |
| 1.2 Core abstractions | 5 | 1 | 1.1 |
| 1.3 JobSet template design | 3 | 1 | 1.1 |
| 2.1 RLEngine (main loop) | 8 | 2 | 1.2 |
| 2.2 GRPO algorithm module | 5 | 2 | 1.2 |
| 2.3 vLLM rollout client | 5 | 2 | — |
| 2.4 Weight sync manager | 5 | 2 | — |
| 2.5 Single-node validation | 5 | 2 | 2.1, 2.2, 2.3, 2.4 |
| 3.1 ClusterTrainingRuntime | 5 | 3 | 2.5 |
| 3.2 Python SDK | 5 | 3 | 3.1 |
| 3.3 Multi-node validation | 8 | 3 | 3.1, 3.2 |
| 3.4 Reward plugins | 3 | 3 | 2.1 |
| 4.1 Error handling + observability | 5 | 4 | 3.3 |
| 4.2 Docs and examples | 3 | 4 | 3.3 |
| 4.3 Benchmark | 3 | 4 | 3.3 |
| 4.4 KEP draft | 3 | 4 | 4.2, 4.3 |
| **Total** | **73** | | |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Scope creep — framework becomes too ambitious | High | Critical | Phase 1 design doc locks scope. GRPO first, nothing else. |
| NCCL weight sync with FSDP-sharded models is hard | Medium | High | Phase 2.4 tackles this early. Fall back to full model gather before broadcast. |
| vLLM log-prob API doesn't return what we need for advantage computation | Medium | High | Verify in Phase 2.3. Worst case: compute log-probs on trainer side. |
| Multi-role JobSet networking is flaky | Medium | High | Phase 3.3 validates this. Fall back to separate Deployment for vLLM. |
| TRL sprint already proves sufficient — this sprint is unnecessary | Medium | Low | Good outcome! This sprint becomes an optional enhancement, not a necessity. |
| Upstream KFT community doesn't want an RL runtime | Low | Medium | Framework works standalone (raw JobSet). KEP is a bonus, not a blocker. |
| Algorithm correctness bugs (wrong advantage, wrong clipping) | Medium | High | Unit tests in 2.2 compare against TRL/veRL reference outputs. |

---

## What This Framework Provides That TRL Alone Doesn't

| Capability | TRL on KFT | trainer-rl |
|------------|-----------|------------|
| Multi-role job (trainer + vLLM) | Manual Deployment + TrainJob | Single JobSet, gang-scheduled |
| vLLM rollout generation | AsyncGRPOTrainer (new, limited) | Built-in, configurable |
| NCCL weight sync | AsyncGRPOTrainer manages it | Explicit WeightSyncManager, observable |
| Reward model server | Not supported | Pluggable (function, HTTP, sidecar) |
| Python SDK for RL jobs | KFT SDK + custom glue | `RLClient.train()` with RL-specific config |
| Reusable K8s runtime | None | ClusterTrainingRuntime `rl-grpo` |
| Algorithm extensibility | GRPOTrainer only | Base `Algorithm` class, add PPO/REINFORCE++ |
| Observability | HF Trainer logging | Prometheus metrics, structured logs |

---

## What This Framework Does NOT Replace

- **TRL's algorithm code:** We may vendor or depend on TRL for GRPO math. The value is orchestration, not algorithm reimplementation.
- **vLLM:** We depend on vLLM for inference. No custom inference engine.
- **KFT/JobSet:** We build on top of KFT, not replace it.
- **Kueue:** We use Kueue for scheduling, not our own scheduler.

---

## Key Dependencies

| Dependency | Version | Why |
|------------|---------|-----|
| PyTorch | ≥ 2.5 | FSDP2, torch.distributed |
| vLLM | ≥ 0.17.1 | Rollout engine + weight transfer APIs |
| kubeflow-trainer SDK | latest | TrainJob submission, runtime management |
| jobset API | v1alpha2 | Multi-role job orchestration |
| Kueue | ≥ 0.8.3 | Gang scheduling, quota management |
| transformers | ≥ 5.2.0 | Model loading, tokenization |
| CUDA | ≥ 12.x | BF16, NCCL |

---

## Success Criteria

This sprint succeeds if a user can run:

```python
from trainer_rl import RLClient, GRPOConfig, RolloutConfig

client = RLClient()
job = client.train(
    model="meta-llama/Llama-3.2-1B-Instruct",
    algorithm=GRPOConfig(num_generations=8, kl_coef=0.001),
    rollout=RolloutConfig(engine="vllm", num_instances=2),
    reward_fn=gsm8k_reward,
    dataset="gsm8k",
    trainer_nodes=2,
    gpus_per_trainer_node=2,
)
```

...and get a completed RL training job on KFT with improving rewards, vLLM rollouts, NCCL weight sync, and zero Ray.
