#!/usr/bin/env python3
"""Benchmark localhost sync vs PVC sync baselines.

Method A (localhost):
  Run grpo_vllm_train_sync.py inside vLLM pod (single-pod training + sync),
  checkpoint at /tmp/grpo-vllm-train.

Method B (PVC):
  Submit KFT multi-node GRPO train job, then run post_train_sync.py to perform
  sync from shared PVC checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SYNC_SCRIPT_LOCAL = ROOT / "grpo_vllm_train_sync.py"
SUBMIT_SCRIPT = "/opt/app-root/src/scripts/submit_kft_grpo_multinode.py"
POST_SYNC_SCRIPT = ROOT / "post_train_sync.py"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def oc(ns: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["oc", "-n", ns, *args], check=check)


def discover_vllm_pod(ns: str, label: str) -> str:
    out = oc(ns, ["get", "pods", "-l", label, "-o", "jsonpath={.items[0].metadata.name}"])
    pod = out.stdout.strip()
    if not pod:
        raise RuntimeError(f"No vLLM pod found for label {label} in namespace {ns}")
    return pod


def copy_to_pod(ns: str, local_path: Path, pod: str, remote_path: str) -> None:
    run(["oc", "-n", ns, "cp", str(local_path), f"{pod}:{remote_path}"])


def parse_results_json(raw: str) -> dict:
    marker = "__RESULTS_JSON__"
    for line in reversed(raw.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            return json.loads(payload)
    raise RuntimeError("Missing __RESULTS_JSON__ in output")


def parse_last_json_line(raw: str) -> dict:
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("No JSON line found in output")


def checkpoint_size(ns: str, pod: str, path: str) -> tuple[str, int]:
    human = oc(ns, ["exec", pod, "--", "du", "-sh", path]).stdout.strip()
    raw = oc(ns, ["exec", pod, "--", "du", "-sb", path]).stdout.strip().split()
    bytes_used = int(raw[0]) if raw else 0
    return human, bytes_used


def run_localhost(ns: str, vllm_pod: str) -> dict:
    oc(
        ns,
        [
            "exec",
            vllm_pod,
            "--",
            "pip",
            "install",
            "-q",
            "datasets",
            "trl",
            "transformers",
            "accelerate",
            "openai",
        ],
        check=False,
    )
    copy_to_pod(ns, SYNC_SCRIPT_LOCAL, vllm_pod, "/tmp/grpo_vllm_train_sync.py")
    out = oc(
        ns,
        [
            "exec",
            vllm_pod,
            "--",
            "env",
            "CUDA_VISIBLE_DEVICES=0,1",
            "python3",
            "-u",
            "/tmp/grpo_vllm_train_sync.py",
        ],
        check=False,
    )
    merged = (out.stdout or "") + (out.stderr or "")
    if out.returncode != 0:
        raise RuntimeError(f"localhost run failed ({out.returncode}):\n{merged[-4000:]}")
    results = parse_results_json(merged)
    human, bytes_used = checkpoint_size(ns, vllm_pod, "/tmp/grpo-vllm-train")
    return {
        "mode": "localhost",
        "checkpoint_path": "/tmp/grpo-vllm-train",
        "checkpoint_size_human": human,
        "checkpoint_size_bytes": bytes_used,
        "sync_results": results,
    }


def submit_kft_job(ns: str, workbench_pod: str, workbench_container: str) -> str:
    out = run(
        [
            "oc",
            "-n",
            ns,
            "exec",
            workbench_pod,
            "-c",
            workbench_container,
            "--",
            "python3",
            SUBMIT_SCRIPT,
        ],
        check=False,
    )
    merged = (out.stdout or "") + (out.stderr or "")
    if out.returncode != 0:
        raise RuntimeError(f"KFT submit failed ({out.returncode}):\n{merged}")
    candidates: list[str] = []
    for line in merged.splitlines():
        s = line.strip()
        if re.fullmatch(r"[a-z0-9]{8,20}", s):
            candidates.append(s)
    if not candidates:
        raise RuntimeError(f"Failed to parse TrainJob name from submit output:\n{merged}")
    return candidates[-1]


def run_pvc(ns: str, vllm_label: str, workbench_pod: str, workbench_container: str) -> dict:
    job = submit_kft_job(ns, workbench_pod, workbench_container)
    out = run(
        [
            sys.executable,
            str(POST_SYNC_SCRIPT),
            job,
            "--namespace",
            ns,
            "--vllm-label",
            vllm_label,
            "--json",
        ],
        check=False,
    )
    merged = (out.stdout or "") + (out.stderr or "")
    if out.returncode != 0:
        raise RuntimeError(f"post_train_sync failed ({out.returncode}):\n{merged[-4000:]}")
    payload = parse_last_json_line(merged)
    payload["trainjob"] = job
    return payload


def summarize(mode: str, runs: list[dict]) -> dict:
    train_times = [r["sync_results"]["train_elapsed_s"] for r in runs]
    sync_times = [r["sync_results"]["sync_elapsed_s"] for r in runs]
    pre_lat = [r["sync_results"]["pre_eval"]["avg_latency_s"] for r in runs]
    post_lat = [r["sync_results"]["post_eval"]["avg_latency_s"] for r in runs]
    pre_acc = [r["sync_results"]["pre_eval"]["accuracy"] for r in runs]
    post_acc = [r["sync_results"]["post_eval"]["accuracy"] for r in runs]
    sizes = [r["checkpoint_size_bytes"] for r in runs]

    def ms(vals: list[float]) -> dict:
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return {"mean": mean, "std": std}

    return {
        "mode": mode,
        "n": len(runs),
        "train_elapsed_s": ms(train_times),
        "sync_elapsed_s": ms(sync_times),
        "pre_latency_s": ms(pre_lat),
        "post_latency_s": ms(post_lat),
        "pre_accuracy": ms(pre_acc),
        "post_accuracy": ms(post_acc),
        "checkpoint_size_bytes": ms(sizes),
    }


def print_table(report: dict) -> None:
    print("\n=== Baseline Comparison ===")
    print("metric                           localhost                 pvc")
    print("-" * 74)
    l = report["summary"]["localhost"]
    p = report["summary"]["pvc"]

    def row(name: str, lval: str, pval: str) -> None:
        print(f"{name:<32} {lval:<24} {pval:<24}")

    row("runs", str(l["n"]), str(p["n"]))
    row(
        "checkpoint_size_mb(mean)",
        f"{l['checkpoint_size_bytes']['mean'] / (1024**2):.1f}",
        f"{p['checkpoint_size_bytes']['mean'] / (1024**2):.1f}",
    )
    row(
        "train_elapsed_s(mean±std)",
        f"{l['train_elapsed_s']['mean']:.2f}±{l['train_elapsed_s']['std']:.2f}",
        f"{p['train_elapsed_s']['mean']:.2f}±{p['train_elapsed_s']['std']:.2f}",
    )
    row(
        "sync_elapsed_s(mean±std)",
        f"{l['sync_elapsed_s']['mean']:.2f}±{l['sync_elapsed_s']['std']:.2f}",
        f"{p['sync_elapsed_s']['mean']:.2f}±{p['sync_elapsed_s']['std']:.2f}",
    )
    row(
        "pre_latency_s(mean±std)",
        f"{l['pre_latency_s']['mean']:.3f}±{l['pre_latency_s']['std']:.3f}",
        f"{p['pre_latency_s']['mean']:.3f}±{p['pre_latency_s']['std']:.3f}",
    )
    row(
        "post_latency_s(mean±std)",
        f"{l['post_latency_s']['mean']:.3f}±{l['post_latency_s']['std']:.3f}",
        f"{p['post_latency_s']['mean']:.3f}±{p['post_latency_s']['std']:.3f}",
    )
    row(
        "pre_accuracy(mean)",
        f"{l['pre_accuracy']['mean']:.4f}",
        f"{p['pre_accuracy']['mean']:.4f}",
    )
    row(
        "post_accuracy(mean)",
        f"{l['post_accuracy']['mean']:.4f}",
        f"{p['post_accuracy']['mean']:.4f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark localhost vs PVC sync baselines")
    parser.add_argument("--namespace", "-n", default="grpoxtrainer")
    parser.add_argument("--vllm-label", default="app=grpo-vllm-rollout")
    parser.add_argument("--workbench-pod", default="grpoxtrainerwb1-0")
    parser.add_argument("--workbench-container", default="grpoxtrainerwb1")
    parser.add_argument("--runs", type=int, default=1, help="Runs per mode")
    parser.add_argument("--output-json", default="baseline_results.json")
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    vllm_pod = discover_vllm_pod(args.namespace, args.vllm_label)
    print(f"vLLM pod: {vllm_pod}")

    localhost_runs: list[dict] = []
    pvc_runs: list[dict] = []

    for i in range(args.runs):
        print(f"\n[localhost] run {i + 1}/{args.runs}")
        localhost_runs.append(run_localhost(args.namespace, vllm_pod))

    for i in range(args.runs):
        print(f"\n[pvc] run {i + 1}/{args.runs}")
        pvc_runs.append(run_pvc(args.namespace, args.vllm_label, args.workbench_pod, args.workbench_container))

    report = {
        "namespace": args.namespace,
        "vllm_pod": vllm_pod,
        "runs": args.runs,
        "localhost": localhost_runs,
        "pvc": pvc_runs,
        "summary": {
            "localhost": summarize("localhost", localhost_runs),
            "pvc": summarize("pvc", pvc_runs),
        },
    }

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved raw report: {out_path}")
    print_table(report)


if __name__ == "__main__":
    main()
