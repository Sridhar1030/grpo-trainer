# GPU Baseline: TRL GRPO on KFT v2 (no vLLM)

**Date:** 2026-05-13  
**TrainJob:** `zaf19f91a904`  
**Node:** `ip-10-0-29-68.us-west-1.compute.internal` (g4dn.12xlarge)  
**GPUs:** 2× Tesla T4 (15.6 GiB each)  
**Runtime:** `torch-distributed` ClusterTrainingRuntime  
**Model:** Qwen/Qwen2.5-0.5B-Instruct (494M params, bf16)  
**Dataset:** GSM8K (256 train, 64 eval)  
**GRPO config:** G=4, per_device_bs=2, grad_accum=2, global_batch=8, lr=5e-6

## nvidia-smi during training (step ~32/128)

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  Tesla T4                       On  |   00000000:00:1B.0 Off |                    0 |
| N/A   52C    P0             71W /   70W |    6049MiB /  15360MiB |    100%      Default |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla T4                       On  |   00000000:00:1C.0 Off |                    0 |
| N/A   52C    P0             45W /   70W |    6051MiB /  15360MiB |    100%      Default |
+-----------------------------------------+------------------------+----------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                    GPU Memory Usage      |
|   0   N/A  N/A         58      C   python3.12                           6046MiB          |
|   1   N/A  N/A         59      C   python3.12                           6048MiB          |
+-----------------------------------------------------------------------------------------+
```

## Key metrics (TRL GRPO, no vLLM)

| Metric | GPU 0 | GPU 1 |
|--------|-------|-------|
| **VRAM used** | 6049 MiB (39%) | 6051 MiB (39%) |
| **GPU util** | 100% | 100% |
| **Power** | 71W / 70W | 45W / 70W |
| **Temp** | 52C | 52C |
| **Process** | python3.12 (PID 58) | python3.12 (PID 59) |

## Training speed (from completed run `s9a413668016`)

| Phase | Time |
|-------|------|
| Model load | 3.26s |
| Dataset prep | 2.39s |
| Step 1 (cold) | ~6.8s |
| Steps 20+ (warm) | ~1.2s |
| **Total 128 steps** | **252s (4m12s)** |
| train_loss | 0.0354 → 0.0 |

## Notes

- Each rank runs its own `GRPOTrainer` process (DDP via torchrun)
- TRL does generation on **each GPU separately** (no vLLM offload)
- VRAM is ~39% of T4 capacity — generation buffers dominate for this small model
- With vLLM: expect generation offloaded to a separate inference server,
  freeing GPU memory on the training ranks and potentially faster generation throughput

## Compare with vLLM run

_Fill in after vLLM-enabled GRPO run:_

| Metric | TRL GRPO (this run) | TRL GRPO + vLLM |
|--------|---------------------|-----------------|
| VRAM per GPU (training) | 6050 MiB (39%) | _TBD_ |
| GPU util (training) | 100% | _TBD_ |
| Step time (warm) | ~1.2s | _TBD_ |
| Total train wall time | **252s (4m12s)** | _TBD_ |
| Generation throughput | inline (TRL, no vLLM) | _TBD (vLLM server)_ |
| Eval | TRL reshape bug on rank1 | _TBD_ |
