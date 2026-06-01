#!/usr/bin/env python3
"""Benchmark vLLM weight sync using same-node local path vs shared PVC path.

This benchmark reuses one existing checkpoint from PVC and compares two sync-only modes:
1) localtmp: copy checkpoint to /tmp inside vLLM pod, then --sync-only from /tmp
2) pvc:      --sync-only directly from /mnt/checkpoint
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


CHECKPOINT_PVC = "/mnt/checkpoint/grpo-trained"
CHECKPOINT_TMP = "/tmp/grpo-local-grpo-trained"
SYNC_SCRIPT_REMOTE = "/tmp/grpo_vllm_train_sync.py"
STATE_SCRIPT_REMOTE = "/tmp/create_sync_state.py"
STATE_PATH = "/tmp/grpo-sync-state.json"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def oc(ns: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["oc", "-n", ns, *args], check=check)


def discover_vllm_pod(ns: str, label: str) -> str:
    out = oc(ns, ["get", "pods", "-l", label, "-o", "jsonpath={.items[0].metadata.name}"])
    pod = out.stdout.strip()
    if not pod:
        raise RuntimeError(f"No vLLM pod for label {label} in {ns}")
    return pod


def copy_to_pod(ns: str, local: Path, pod: str, remote: str) -> None:
    run(["oc", "-n", ns, "cp", str(local), f"{pod}:{remote}"])


def ensure_scripts(ns: str, pod: str) -> None:
    root = Path(__file__).resolve().parent
    copy_to_pod(ns, root / "grpo_vllm_train_sync.py", pod, SYNC_SCRIPT_REMOTE)
    copy_to_pod(ns, root / "create_sync_state.py", pod, STATE_SCRIPT_REMOTE)


def ensure_deps(ns: str, pod: str) -> None:
    oc(
        ns,
        ["exec", pod, "--", "pip", "install", "-q", "datasets", "trl", "transformers", "accelerate", "openai"],
        check=False,
    )


def checkpoint_size(ns: str, pod: str, path: str) -> tuple[str, int]:
    human = oc(ns, ["exec", pod, "--", "du", "-sh", path]).stdout.strip()
    raw = oc(ns, ["exec", pod, "--", "du", "-sb", path]).stdout.strip().split()
    return human, (int(raw[0]) if raw else 0)


def prepare_local_copy(ns: str, pod: str) -> None:
    cmd = (
        f"rm -rf {CHECKPOINT_TMP} && cp -a {CHECKPOINT_PVC} {CHECKPOINT_TMP} "
        f"&& test -f {CHECKPOINT_TMP}/model.safetensors && test -f {CHECKPOINT_TMP}/.ready"
    )
    r = oc(ns, ["exec", pod, "--", "bash", "-lc", cmd], check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Failed preparing local checkpoint copy:\n{r.stderr}\n{r.stdout}")


def create_state(ns: str, pod: str) -> None:
    r = oc(
        ns,
        ["exec", pod, "--", "python3", "-u", STATE_SCRIPT_REMOTE, "4", "0.0", "0.0", "0.0", STATE_PATH],
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Failed creating sync state:\n{r.stderr}\n{r.stdout}")


def parse_results(raw: str) -> dict:
    marker = "__RESULTS_JSON__"
    for line in reversed(raw.splitlines()):
        if marker in line:
            return json.loads(line.split(marker, 1)[1].strip())
    raise RuntimeError("Missing __RESULTS_JSON__ in sync output")


def run_sync(ns: str, pod: str, checkpoint: str) -> dict:
    r = oc(
        ns,
        [
            "exec",
            pod,
            "--",
            "env",
            "CUDA_VISIBLE_DEVICES=0,1",
            "python3",
            "-u",
            SYNC_SCRIPT_REMOTE,
            "--sync-only",
            "--checkpoint",
            checkpoint,
            "--state",
            STATE_PATH,
        ],
        check=False,
    )
    merged = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"Sync failed for {checkpoint} ({r.returncode}):\n{merged[-5000:]}")
    return parse_results(merged)


def single_run(ns: str, pod: str) -> dict:
    create_state(ns, pod)
    local = run_sync(ns, pod, CHECKPOINT_TMP)
    create_state(ns, pod)
    pvc = run_sync(ns, pod, CHECKPOINT_PVC)
    return {"localtmp": local, "pvc": pvc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sync-only localtmp vs pvc")
    parser.add_argument("--namespace", "-n", default="grpoxtrainer")
    parser.add_argument("--vllm-label", default="app=grpo-vllm-rollout")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-json", default="baseline_sync_paths.json")
    args = parser.parse_args()

    pod = discover_vllm_pod(args.namespace, args.vllm_label)
    print(f"vLLM pod: {pod}")
    ensure_scripts(args.namespace, pod)
    ensure_deps(args.namespace, pod)

    prepare_local_copy(args.namespace, pod)
    pvc_h, pvc_b = checkpoint_size(args.namespace, pod, CHECKPOINT_PVC)
    loc_h, loc_b = checkpoint_size(args.namespace, pod, CHECKPOINT_TMP)

    runs: list[dict] = []
    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs} ...")
        runs.append(single_run(args.namespace, pod))

    report = {
        "namespace": args.namespace,
        "vllm_pod": pod,
        "checkpoint_sizes": {
            "pvc_human": pvc_h,
            "pvc_bytes": pvc_b,
            "localtmp_human": loc_h,
            "localtmp_bytes": loc_b,
        },
        "runs": runs,
    }
    out = Path(args.output_json)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved report: {out}")

    l_sync = [r["localtmp"]["sync_elapsed_s"] for r in runs]
    p_sync = [r["pvc"]["sync_elapsed_s"] for r in runs]
    l_lat = [r["localtmp"]["post_eval"]["avg_latency_s"] for r in runs]
    p_lat = [r["pvc"]["post_eval"]["avg_latency_s"] for r in runs]
    l_acc = [r["localtmp"]["post_eval"]["accuracy"] for r in runs]
    p_acc = [r["pvc"]["post_eval"]["accuracy"] for r in runs]

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    print("\n=== Sync Path Baseline (same checkpoint) ===")
    print(f"Checkpoint size pvc/localtmp: {pvc_h} / {loc_h}")
    print(f"Sync elapsed (s)      localtmp={mean(l_sync):.3f}  pvc={mean(p_sync):.3f}")
    print(f"Post latency (s)      localtmp={mean(l_lat):.3f}  pvc={mean(p_lat):.3f}")
    print(f"Post accuracy         localtmp={mean(l_acc):.4f}  pvc={mean(p_acc):.4f}")


if __name__ == "__main__":
    main()
