# vLLM Weight Synchronization — Deep Reference

> **PR:** https://github.com/vllm-project/vllm/pull/31943 (merged Feb 5, 2026)  
> **RFC:** https://github.com/vllm-project/vllm/issues/31848  
> **Docs:** https://docs.vllm.ai/en/latest/training/weight_transfer/

---

## What It Solves

Before PR #31943, every RL framework (SkyRL, veRL, TRL) had to implement its own weight syncing infrastructure between training and inference. This PR standardizes native weight transfer APIs in vLLM.

---

## Transport Backends

| Backend | Use Case |
|---------|----------|
| **NCCL** | Separate GPU pools, TP>1 rollouts, NVLink/IB for bandwidth |
| **IPC** | Colocated GPU scenarios (same node) |

---

## The Four-Phase Protocol

1. **`init_weight_transfer_engine`** — join NCCL group
2. **`start_weight_update`** — prepare engine for sync
3. **`update_weights`** — metadata for this sync; weight bytes move via NCCL
4. **`finish_weight_update`** — finalize

Optional: `POST /pause` and `POST /resume` coordinate generation around in-flight updates.

**Important:** HTTP weight-transfer routes require `VLLM_SERVER_DEV_MODE=1`.

---

## API Details

### `POST /init_weight_transfer_engine`

Sets up NCCL process group between trainer and vLLM.

```json
{
  "init_info": {
    "master_address": "10.0.0.5",
    "master_port": 29500,
    "rank_offset": 1,
    "world_size": 3
  }
}
```

| Field | Meaning |
|-------|---------|
| `master_address` | Rendezvous host reachable by all ranks |
| `master_port` | Rendezvous TCP port |
| `rank_offset` | First rank index for inference workers (trainer = rank 0) |
| `world_size` | Total ranks = 1 (trainer) + number of inference workers |

### `GET /get_world_size`

Returns inference worker count so trainer can compute `world_size = response + 1`.

```json
{"world_size": 2}
```

### `POST /start_weight_update`

```json
{"is_checkpoint_format": true}
```

### `POST /update_weights`

Metadata about tensors to receive via NCCL:

```json
{
  "update_info": {
    "names": ["model.layers.0.self_attn.q_proj.weight", ...],
    "dtype_names": ["bfloat16", ...],
    "shapes": [[4096, 4096], ...],
    "packed": true
  }
}
```

| Field | Meaning |
|-------|---------|
| `names` | Parameter names, same order as `model.named_parameters()` |
| `dtype_names` | Dtype strings per parameter |
| `shapes` | Shape per parameter |
| `packed` | Use packed tensor broadcasting (fewer, larger NCCL ops) |

### `POST /finish_weight_update`

```json
{}
```

### `POST /pause` / `POST /resume`

Quiesce generation before weight update, resume after.

---

## Server Configuration

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve <model> \
  --weight-transfer-config '{"backend": "nccl"}'
```

Programmatic:
```python
from vllm.config import WeightTransferConfig
weight_transfer_config = WeightTransferConfig(backend="nccl")
```

---

## Complete HTTP + NCCL Example

```python
import threading
import requests
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

BASE_URL = "http://vllm-service:8000"

# 1. Get inference world size
inference_ranks = requests.get(f"{BASE_URL}/get_world_size").json()["world_size"]
world_size = inference_ranks + 1
master_address = "<trainer-ip>"
master_port = 29500

# 2. Init weight transfer (threaded — blocks until group forms)
def server_init():
    requests.post(f"{BASE_URL}/init_weight_transfer_engine", json={
        "init_info": {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
    }).raise_for_status()

t = threading.Thread(target=server_init)
t.start()

# 3. Trainer joins as rank 0
grp = NCCLWeightTransferEngine.trainer_init({
    "master_address": master_address,
    "master_port": master_port,
    "world_size": world_size,
})
t.join()

# 4. Pause generation
requests.post(f"{BASE_URL}/pause").raise_for_status()

# 5. Start update
requests.post(f"{BASE_URL}/start_weight_update",
              json={"is_checkpoint_format": True}).raise_for_status()

# 6. Send weights via NCCL (concurrent with update_weights metadata call)
names = [n for n, _ in model.named_parameters()]
shapes = [list(p.shape) for _, p in model.named_parameters()]
dtype_names = [str(p.dtype).replace("torch.", "") for _, p in model.named_parameters()]

def server_update():
    requests.post(f"{BASE_URL}/update_weights", json={
        "update_info": {
            "names": names, "dtype_names": dtype_names,
            "shapes": shapes, "packed": True,
        }
    }).raise_for_status()

ut = threading.Thread(target=server_update)
ut.start()

NCCLWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=NCCLTrainerSendWeightsArgs(group=grp, packed=True),
)
ut.join()

# 7. Finish and resume
requests.post(f"{BASE_URL}/finish_weight_update", json={}).raise_for_status()
requests.post(f"{BASE_URL}/resume").raise_for_status()
```

---

## Integration with RL Frameworks

### TRL AsyncGRPOTrainer

- Decouples rollout (vLLM server) from training
- After every `weight_sync_steps`, weights go to vLLM via NCCL
- Requirements: `vllm>=0.17.1`, `transformers>=5.2.0`, FSDP2 only
- Dedicated GPU split: training GPUs vs vLLM GPUs

### veRL Fully Async

- NCCL-based parameter sync between Trainer and Rollouter
- Megatron/FSDP + vLLM in server mode
- ~2.35-2.67x end-to-end speedup reported

### Custom Loop

Use HTTP control plane + NCCL data plane as shown above. Works with any training framework.

---

## vLLM vs HF .generate() for Rollouts

| Feature | vLLM | HF .generate() |
|---------|------|-----------------|
| Continuous batching | Yes | No |
| PagedAttention (KV cache) | Yes | No |
| Tensor parallelism | Native | Limited |
| Throughput (high concurrency) | Up to ~24x higher | Baseline |
| Integration overhead | Server + networking | None |

---

## Kubernetes Deployment Considerations

| Concern | Implication |
|---------|-------------|
| **HTTP traffic** | Standard ClusterIP Service DNS for `vllm serve` |
| **NCCL traffic** | Does NOT use Service for collectives — needs real Pod/Node IPs |
| **Rendezvous** | `master_address` must be reachable by all vLLM workers + trainer |
| **Firewall** | NetworkPolicy must allow NCCL `master_port` traffic |
| **Multi-node NCCL** | Often requires IB/RoCE, `NCCL_*` env vars, sometimes `hostNetwork` |
| **GPU isolation** | Trainer and vLLM must see distinct CUDA devices |
| **Server flags** | `VLLM_SERVER_DEV_MODE=1` + `--weight-transfer-config '{"backend":"nccl"}'` |

---

## Architecture: Async GRPO with Weight Sync

```
Trainer Pod/Job              vLLM Rollout Service
─────────────────           ──────────────────────
1. Sample prompts     ◄──── fast rollout generation
2. Get completions          (PagedAttention, batched)
3. Score rewards
4. GRPO advantage
5. Policy gradient
6. AllReduce (NCCL)
7. POST /pause
8. POST /start_weight_update
9. NCCL broadcast weights ──► vLLM receives new weights
10. POST /finish_weight_update
11. POST /resume
12. Repeat ↺
```

---

## References

- [Weight transfer overview](https://docs.vllm.ai/en/latest/training/weight_transfer/)
- [NCCL engine docs](https://docs.vllm.ai/en/latest/training/weight_transfer/nccl/)
- [RLHF HTTP NCCL example](https://docs.vllm.ai/en/latest/examples/rl/rlhf_http_nccl/)
- [RLHF NCCL (Ray) example](https://docs.vllm.ai/en/latest/examples/rl/rlhf_nccl/)
- [Async new APIs example](https://docs.vllm.ai/en/latest/examples/rl/rlhf_async_new_apis/)
- [RFC #31848](https://github.com/vllm-project/vllm/issues/31848)
- [TRL Async GRPO](https://huggingface.co/docs/trl/async_grpo_trainer)
