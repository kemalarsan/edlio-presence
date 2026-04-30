#!/usr/bin/env python3
"""
hello-gpu.py — Container GPU verification script
Run inside the edlio-presence container to confirm the environment is sane.

Exit 0 = all checks passed (GPU ready)
Exit 1 = one or more checks failed
"""

import sys
import subprocess

QUIET = "--quiet" in sys.argv
PASS = "✅"
FAIL = "❌"

def log(msg):
    if not QUIET:
        print(msg)

def check(label, fn):
    try:
        result = fn()
        log(f"{PASS}  {label}: {result}")
        return True
    except Exception as e:
        print(f"{FAIL}  {label}: {e}", file=sys.stderr)
        return False

failures = []

# 1. PyTorch importable
def torch_import():
    import torch
    return f"torch {torch.__version__}"
if not check("PyTorch import", torch_import):
    failures.append("torch import")

# 2. CUDA available
def cuda_available():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() == False")
    return f"CUDA {torch.version.cuda}"
if not check("CUDA available", cuda_available):
    failures.append("cuda available")

# 3. GPU device name
def gpu_name():
    import torch
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{name} ({mem_gb:.1f} GB VRAM)"
if not check("GPU device", gpu_name):
    failures.append("gpu device")

# 4. Trivial GPU tensor op
def gpu_tensor():
    import torch
    x = torch.ones(256, 256).cuda()
    y = (x @ x).sum().item()
    return f"matmul ok, sum={y:.0f}"
if not check("GPU tensor op", gpu_tensor):
    failures.append("gpu tensor")

# 5. ffmpeg present
def ffmpeg_check():
    result = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    first_line = result.stdout.split("\n")[0]
    return first_line
if not check("ffmpeg", ffmpeg_check):
    failures.append("ffmpeg")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"❌  GPU NOT READY — {len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
else:
    log("━" * 50)
    log("✅  GPU READY — all checks passed")
    log("━" * 50)
    sys.exit(0)
