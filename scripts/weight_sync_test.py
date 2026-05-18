#!/usr/bin/env python3
"""
Weight sync test — matches the official vLLM rlhf_http_nccl.py pattern.
Run this INSIDE the vLLM pod (oc exec) so trainer and vLLM share localhost.

vLLM uses GPU 0, trainer uses GPU 1.

Usage:
    oc exec <vllm-pod> -- python3 /tmp/weight_sync_test.py
"""

import threading
import time

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    inference_world_size = requests.get(
        f"{BASE_URL}/get_world_size", timeout=10
    ).json()["world_size"]
    world_size = inference_world_size + 1

    device = f"cuda:{inference_world_size}"
    torch.cuda.set_device(int(device.split(":")[1]))
    print(f"Trainer on {device}, vLLM world_size={inference_world_size}, total={world_size}")

    print(f"Loading model: {MODEL_NAME}")
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="dummy")

    print("=" * 50)
    print("BEFORE weight sync:")
    print("=" * 50)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=16,
    )
    print(f"  Answer: {resp.choices[0].message.content}")

    master_address = get_ip()
    master_port = get_open_port()
    rank_offset = 1
    print(f"\nNCCL init: master={master_address}:{master_port} world_size={world_size}")

    init_thread = threading.Thread(
        target=lambda: requests.post(
            f"{BASE_URL}/init_weight_transfer_engine",
            json={"init_info": dict(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
            )},
            timeout=120,
        ).raise_for_status(),
    )
    init_thread.start()

    model_update_group = NCCLWeightTransferEngine.trainer_init(dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    ))
    init_thread.join()
    print("NCCL group established")

    requests.post(f"{BASE_URL}/pause", timeout=30).raise_for_status()
    print("vLLM paused")

    requests.post(
        f"{BASE_URL}/start_weight_update",
        json={"is_checkpoint_format": True},
        timeout=60,
    ).raise_for_status()

    names, dtype_names, shapes = [], [], []
    for name, p in train_model.named_parameters():
        names.append(name)
        dtype_names.append(str(p.dtype).split(".")[-1])
        shapes.append(list(p.shape))

    update_thread = threading.Thread(
        target=lambda: requests.post(
            f"{BASE_URL}/update_weights",
            json={"update_info": dict(
                names=names, dtype_names=dtype_names, shapes=shapes, packed=True,
            )},
            timeout=300,
        ).raise_for_status(),
    )
    update_thread.start()

    print("Broadcasting weights via NCCL...")
    NCCLWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
        trainer_args=NCCLTrainerSendWeightsArgs(group=model_update_group, packed=True),
    )
    update_thread.join()
    print("Weights transferred")

    requests.post(f"{BASE_URL}/finish_weight_update", json={}, timeout=60).raise_for_status()
    requests.post(f"{BASE_URL}/resume", timeout=30).raise_for_status()
    print("vLLM resumed with fresh weights")

    print("=" * 50)
    print("AFTER weight sync:")
    print("=" * 50)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=16,
    )
    print(f"  Answer: {resp.choices[0].message.content}")
    print("\n=== WEIGHT SYNC COMPLETE ===")


if __name__ == "__main__":
    main()
