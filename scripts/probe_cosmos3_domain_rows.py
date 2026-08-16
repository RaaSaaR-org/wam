#!/usr/bin/env python3
"""E-02 probe: which Cosmos3 embodiment domain rows are actually trained, and how wide.

Reads ONLY the safetensors header + the 8 MB `action_proj_out.fc.weight` tensor via
HTTP Range requests. Does not download the 6.7 GB checkpoint.

Method: DomainAwareLinear stores fc as Embedding(num_domains, out*in), viewed as
(D, input_size, output_size) at forward time
(diffusers/models/transformers/transformer_cosmos3.py:162,177).
For action_proj_out that is (32, 2048, 64). Take the per-output-channel L2 norm of
each domain row and divide by the mean of the same statistic over domain rows that
no embodiment name maps to (pure random init). Trained channels stand out; untrained
ones sit at ratio ~1.00. The index of the last channel above threshold is the
embodiment's trained action width.

Usage: python probe_cosmos3_domain_rows.py nvidia/Cosmos3-Edge
"""

import json
import struct
import subprocess
import sys

import numpy as np

SHARD = "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
TENSOR = "action_proj_out.fc.weight"
# Rows no name maps to in cosmos_framework .../action/utils/domain_utils.py
UNASSIGNED = [10, 11, 14, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31]
NAMES = {
    0: "no_action", 1: "av(9)", 2: "camera_pose(9)", 3: "hand_pose(57)", 4: "pusht(2)",
    5: "libero(var)", 6: "umi(10)", 7: "bridge(10)", 8: "droid/franka(10)",
    9: "embodiment_b(30)", 12: "franka-dual(20)", 13: "robomind-ur(10)",
    15: "agibotworld(29)", 16: "xdof_yam(20)", 20: "fractal(10)",
    21: "drawanything(3)", 22: "behavior1k(23)", 23: "maniparena(20)",
}


def fetch(url: str, start: int, end: int) -> bytes:
    return subprocess.run(
        ["curl", "-sL", "-r", f"{start}-{end}", url], capture_output=True, check=True
    ).stdout


def main(repo: str) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/{SHARD}"
    hdr_len = struct.unpack("<Q", fetch(url, 0, 7))[0]
    header = json.loads(fetch(url, 8, hdr_len + 7))
    meta = header[TENSOR]
    assert meta["dtype"] == "BF16", meta["dtype"]
    n_dom, flat = meta["shape"]
    base = 8 + hdr_len
    lo, hi = meta["data_offsets"]
    raw = fetch(url, base + lo, base + hi - 1)

    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    w = u16.view(np.float32).reshape(n_dom, -1, 64)  # (D, hidden, action_dim)
    noise = np.mean([np.linalg.norm(w[i], axis=0) for i in UNASSIGNED], axis=0)

    print(f"{repo}  {TENSOR} {meta['shape']}  (flat {flat} = hidden x action_dim)")
    for d in range(n_dom):
        ratio = np.linalg.norm(w[d], axis=0) / noise
        above = np.where(ratio > 1.15)[0]
        width = int(above.max()) + 1 if above.size else 0
        tag = "UNTRAINED (init)" if width == 0 else f"trained width = {width}"
        print(f"  d{d:2d} {NAMES.get(d, '-'):18s} {tag}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nvidia/Cosmos3-Edge")
