# GRPO + Kubeflow Trainer v2

TRL GRPO fine-tuning on Kubeflow Trainer v2 with multi-GPU DDP.

## What's here

| Path | What |
|------|------|
| `notebooks/grpo_kft_v2.ipynb` | End-to-end notebook — submits a multi-GPU TrainJob to the cluster |
| `docs/spike-findings.md` | Spike checklist, known issues, mitigations |
| `runs_log/gpu-baseline-trl-grpo.md` | GPU consumption baseline (TRL GRPO, no vLLM) |
| `k8s/rbac-trainer-access.yaml` | RBAC for Kubeflow Trainer service account |
## How to run

1. Open `notebooks/grpo_kft_v2.ipynb` in JupyterLab on the cluster.
2. Run all cells — it submits a 2×GPU TrainJob via KFT v2 `CustomTrainer`.
3. Logs stream in the notebook; results go to `runs_log/`.
