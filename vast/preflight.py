"""Checks before notebooks/run_eval_bundle.py on the rented machine: data files, RAM, GPU, disk.

Exits non-zero on a failed check. No API key is checked: KGEnv builds no LLM client at start-up (the OpenAI
client is created inside each LLM call, and load_dotenv() accepts a missing .env), and the bundle never
calls an LLM.

    python vast/preflight.py [--datasets family fb15k_237 wikidata5m]
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

MANIFEST = Path("vast/data_manifest.txt")
MIN_RAM_BYTES = 64 * 10**9       # wikidata5m: ~19 GB for KGInterface plus pandas and GNN peaks
MIN_GPU_BYTES = 12 * 10**9
MIN_DISK_BYTES = 20 * 10**9      # local-push PPR cache and results


def cgroup_limit():
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit():
            return int(value)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["family", "fb15k_237", "wikidata5m"])
    args, _ = parser.parse_known_args()
    errors, warnings = [], []

    # 1. data files, against the manifest
    if not MANIFEST.exists():
        errors.append(f"{MANIFEST} missing (run from the repo root)")
    else:
        n = 0
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            size, _, path = line.split(maxsplit=2)
            n += 1
            if not os.path.isfile(path):
                errors.append(f"missing data file {path}")
            elif os.path.getsize(path) != int(size):
                errors.append(f"size mismatch {path}: {os.path.getsize(path)} != {size}")
        print(f"manifest: {n} files checked")

    # 2. RAM, the smaller of the host total and the container's cgroup limit
    import psutil
    total = psutil.virtual_memory().total
    limit = cgroup_limit()
    ram = min(total, limit) if limit else total
    print(f"RAM: {ram / 10**9:.1f} GB (host {total / 10**9:.1f} GB, cgroup {limit and limit / 10**9})")
    if "wikidata5m" in args.datasets and ram < MIN_RAM_BYTES:
        errors.append(f"RAM {ram / 10**9:.1f} GB < 64 GB needed for wikidata5m")

    # 3. GPU
    import torch
    if not torch.cuda.is_available():
        errors.append("no CUDA GPU visible to torch")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 10**9:.1f} GB, torch CUDA {torch.version.cuda}")
        if props.total_memory < MIN_GPU_BYTES:
            warnings.append(f"GPU memory {props.total_memory / 10**9:.1f} GB < 12 GB recommended")

    # 4. disk
    free = shutil.disk_usage(".").free
    print(f"disk free: {free / 10**9:.1f} GB")
    if free < MIN_DISK_BYTES:
        warnings.append(f"disk free {free / 10**9:.1f} GB < 20 GB recommended")

    for warning in warnings:
        print("WARNING:", warning)
    if errors:
        print("\nPREFLIGHT FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print("preflight OK")


if __name__ == "__main__":
    main()
